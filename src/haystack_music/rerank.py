"""The second stage (ADR 0011): a logistic regression trained on the listener's Verdicts orders the top of the pool.

Stage 1 is the active variant's score (vectors.compute_scores). Stage 2 takes the top `rerank.pool_top` by it and
orders them by a blend of the model's chance that the listener keeps the track and the stage-1 score. The model
is refitted at every scoring (it takes milliseconds) on the judged candidates and is used only when it has at
least `rerank.min_labels` labels and its leave-one-out AUC beats stage 1 alone over the same labels by
`rerank.min_gain`. Exploration picks are not re-ranked (scoring.select_batch keeps the band in stage-1 order).

Whenever it can fit (two yes and two no labels), the model's chance is computed for every scored pool row, used or
not: the Calibrate queue (ADR 0015) asks about the candidates it gives even odds. Soft labels (ADR 0016) join the
fit; the gate and the leave-one-out AUCs count the hard ones.

Every feature is computed the same way for a pool candidate and for a judged one; a judged track is left out of
its own positives, style positives and negatives (its own verdict must not explain it). Pure numpy; the only
writes are record() and write_pool()."""
from __future__ import annotations

import json
import logging
import math

import numpy as np

from . import baseline, calibrate, db, metadata, traits, vectors
from .evaluate import _chunks, _owner_keys, _track_keys
from .scoring import (GENRE_VARIANTS, Space, hubness, is_centred, is_csls, l2norm, negative_penalty, owner_max,
                      topk_mean, zblend)
from .textnorm import song_key
from .verdicts import stars_in_sd

log = logging.getLogger("haystack_music.rerank")

POS_WEIGHT = {3: 1.0, 4: 1.25, 5: 1.5}     # a Rating of 3 stars or more: a yes, weighted by stars
LOW_RATING_WEIGHT = 0.75                    # 1-2 stars: a weak no ("a little" is not for the batch, ADR 0011)
NEG_WEIGHT = {"pass": 1.0, "dislike": 1.5}
OUTCOME_WEIGHT = 1.0                        # no verdict: an offer liked, or expired after a play / removed by hand
IRLS_STEPS = 50
PCA_ROUND = 5
REASON_MIN = 0.1                            # a reason moves the logit by at least this much

_CURRENT = """SELECT v.spotify_id, v.kind, v.stars, v.queue, v.given_at FROM verdicts v
              WHERE v.id = (SELECT w.id FROM verdicts w WHERE w.spotify_id = v.spotify_id
                            ORDER BY w.given_at DESC, w.received_at DESC, w.rowid DESC LIMIT 1)"""


# ---------------- labels ----------------

def labels(con, cfg: dict | None = None) -> list[dict]:
    """The judged candidates, one per track: {sid, cand_id, deezer_id, y (1 yes, 0 no), w, kind, soft, at}.
    A track counts when it was offered or picked for Calibrate, or carries a current verdict given outside the
    Rate your likes queue (those are all liked songs, so they say nothing about yes against no; they still feed
    the star features). The current verdict decides: a Rating of 3 stars or more is a yes weighted by stars, 1-2
    stars, a Pass or a Dislike a no. With no verdict an offer liked is a yes, and one that expired after a play or
    was removed by hand a no; `unplayed` is not a label. A Calibrate pick hearted in Spotify is a yes as a liked
    offer is. Otherwise, with cfg's rerank.soft_labels, an offered or Calibrate track not in Liked Songs whose
    latest needle play stopped before rerank.soft_pass_ms without finishing is a soft no (ADR 0016). `at` is when
    the label was given: the verdict's given_at, the offer's close, the heart, or the play. The track needs a
    candidate row with a preview vector."""
    rr = (cfg or {}).get("rerank", {})
    cur = {r[0]: r for r in con.execute(_CURRENT)}
    offers: dict[str, tuple] = {}
    for r in con.execute("SELECT spotify_id, candidate_id, outcome, removed_at FROM offers "
                         "WHERE spotify_id IS NOT NULL ORDER BY added_at, id"):
        offers[r[0]] = (r[1], r[2], r[3])                 # the latest offer of the track wins
    picked = calibrate.picked(con)
    asked = set(offers) | set(picked)
    sids = sorted(asked | {s for s, r in cur.items() if r[3] != "likes"})
    cands: dict[str, tuple] = {}
    for ch in _chunks(sids):
        for r in con.execute(f"SELECT id, spotify_id, deezer_id FROM candidates WHERE vector_status='ok' AND "
                             f"spotify_id IN ({','.join('?' * len(ch))}) ORDER BY id", ch):
            sid = r[1]
            if sid not in cands or r[0] == (offers.get(sid, (None,))[0] or picked.get(sid)):
                cands[sid] = (r[0], r[2])                 # the offered (or picked) candidate, else the first one
    liked = {r[0]: r[1] for r in con.execute("SELECT spotify_id, added_at FROM likes WHERE removed_at IS NULL")}
    short: dict[str, str] = {}
    if rr.get("soft_labels"):
        quiet = [s for s in sids if s in asked and s not in cur and s not in liked and offers.get(s, (None, None))[1]
                 not in ("liked", "negative", "manual_removed")]
        for ch in _chunks(quiet):
            last: dict[str, tuple] = {}
            for r in con.execute(f"SELECT spotify_id, played_at, ms_played, completed FROM plays WHERE "
                                 f"context='needle' AND spotify_id IN ({','.join('?' * len(ch))}) "
                                 f"ORDER BY played_at", ch):
                last[r[0]] = r                            # the latest needle play wins
            for sid, r in last.items():
                if r[3] == 0 and r[2] is not None and r[2] < rr["soft_pass_ms"]:
                    short[sid] = r[1]
    out = []
    for sid in sids:
        if sid not in cands:
            continue
        v = cur.get(sid)
        soft, at = False, None
        if v is not None:
            kind, stars, at = v[1], v[2], v[4]
            if kind == "rating":
                if stars is None:
                    continue
                y, w = (1, POS_WEIGHT[min(int(stars), 5)]) if stars >= 3 else (0, LOW_RATING_WEIGHT)
                kind = f"{int(stars)}★"
            elif kind in NEG_WEIGHT:
                y, w = 0, NEG_WEIGHT[kind]
            else:
                continue
        else:
            _, outcome, at = offers.get(sid, (None, None, None))
            if outcome == "liked":
                y, w, kind = 1, OUTCOME_WEIGHT, "liked"
            elif outcome in ("negative", "manual_removed"):
                y, w, kind = 0, OUTCOME_WEIGHT, outcome
            elif sid in picked and sid in liked:
                y, w, kind, at = 1, OUTCOME_WEIGHT, "liked", liked[sid]
            elif sid in short:
                y, w, kind, soft, at = 0, rr["soft_pass_weight"], "soft_pass", True, short[sid]
            else:
                continue
        out.append(dict(sid=sid, cand_id=cands[sid][0], deezer_id=cands[sid][1], y=y, w=float(w), kind=kind,
                        soft=soft, at=at))
    return out


# ---------------- features ----------------

TRAIT_PHRASES = {   # trait: (above the usual range, below it)
    "tempo": ("faster", "slower"), "valence": ("brighter", "darker"), "energy": ("more energetic", "calmer"),
    "danceability": ("more danceable", "less danceable"), "acousticness": ("more acoustic", "less acoustic"),
    "instrumentalness": ("more instrumental", "more vocal"), "loudness": ("louder", "quieter"),
}
TRAIT_NOUN = {"valence": "mood"}           # "<noun> in your usual range"; the Trait's own name otherwise
NO_READING = 2                              # a trait side: the track has no Reading of it
# feature: (phrase when the track is high on it, phrase when low); None: no phrase (the next feature is shown)
PHRASES = {
    "stage1": ("very close to what you like", "further from what you like"),
    "raw_max": ("very close to one track you like", "not close to any one track you like"),
    "topk": ("close to many tracks you like", "close to few tracks you like"),
    "genre": ("in your usual styles", "outside your usual styles"),
    "style_missing": (None, None),
    "hubness": ("sounds like a lot of music", "has a distinctive sound"),
    "neg_excess_max": ("sounds like a track you turned down", "unlike the tracks you turned down"),
    "neg_penalty": ("sounds like a track you turned down", "unlike the tracks you turned down"),
    "near_stars": ("closest to a track you rated high", "closest to a track you rated low"),
    "knn_stars": ("close to your 4-5★ tracks", "close to tracks you rated low"),
    "knn_hi": ("close to your 4-5★ tracks", "few of your 4-5★ tracks nearby"),
    "knn_lib": ("sounds like your music library", "sounds more like your Spotify likes than your library"),
    "traits_missing": (None, None),
    "chart": ("typical of Deezer charts", None),
    "artist_in_positives": ("by an artist you like", "by an artist new to you"),
    "artist_has_dislike": ("by an artist you disliked before", None),
    "log_rank": ("a very popular track", "an obscure track"),
    "rank_missing": (None, None),
    "log_fans": ("by a very popular artist", "by a little-known artist"),
    "age_years": ("an older release", None),
    "compilation": ("from a compilation", None),
    "label_known": ("on the label of a song you like", None),
    "label_kept_rate": ("on a label you like", None),
}
# what Deezer publishes about a track (ADR 0017); `rerank show` compares the model with and without them
META_FEATURES = ("log_rank", "rank_missing", "log_fans", "age_years", "compilation", "label_known", "label_kept_rate")
# the sign a feature's weight is expected to have; a weight with the other sign still orders the batch but is
# never quoted as a reason ("held back: close to your 4-5★ tracks" reads as a bug, whatever the data says)
EXPECTED_SIGN = {"stage1": 1, "raw_max": 1, "topk": 1, "near_stars": 1, "knn_stars": 1, "knn_hi": 1,
                 "neg_excess_max": -1, "neg_penalty": -1, "artist_in_positives": 1, "artist_has_dislike": -1,
                 "label_known": 1, "label_kept_rate": 1, "compilation": -1}


def _artist(artist, title) -> str:
    return song_key(artist, title).split("|", 1)[0]


def _prepare(con, cfg: dict, variant: str, res: dict, labs: list[dict]) -> dict | None:
    """Everything the features need, for the scored pool (res rows, first) and then the labelled tracks."""
    pos, owners, bonus, stars = vectors.positives_with_stars(con, cfg)
    if not len(pos) or not res.get("rows"):
        return None
    starts = np.array([i for i in range(len(owners)) if i == 0 or owners[i] != owners[i - 1]])
    uo = [owners[i] for i in starts]
    pool_ids = [int(r[0]) for r in res["rows"]]
    ids = pool_ids + [x["cand_id"] for x in labs]
    meta = {}
    for ch in _chunks(sorted(set(ids))):
        for r in con.execute(f"SELECT id, deezer_id, source, artist, title, spotify_id FROM candidates WHERE id IN "
                             f"({','.join('?' * len(ch))})", ch):
            meta[r[0]] = dict(r)
    dids = [meta[i]["deezer_id"] for i in ids]
    vec = vectors.preview_vectors(con, cfg, dids)
    st = vectors.get_styles(con, dids)
    keep = [j for j, d in enumerate(dids) if d in vec]
    n_pool = sum(1 for j in keep if j < len(pool_ids))
    if n_pool < len(pool_ids):
        raise RuntimeError("a scored pool candidate has no preview vector")
    ids = [ids[j] for j in keep]
    labs = [labs[j - len(pool_ids)] for j in keep if j >= len(pool_ids)]
    dids = [dids[j] for j in keep]
    nan = np.full(400, np.nan, np.float32)
    X = np.vstack([vec[d] for d in dids]).astype(np.float32)
    S = np.vstack([st.get(d, nan) for d in dids]).astype(np.float32)
    # owners: song keys, artists, stars, library flag
    okeys = _owner_keys(con, uo)
    man = [int(o.split(":", 1)[1]) for o in uo if o.startswith("manual:")]
    for ch in _chunks(man):
        for r in con.execute(f"SELECT id, artist, title FROM candidates WHERE id IN ({','.join('?' * len(ch))})", ch):
            okeys[f"manual:{r[0]}"] = song_key(r[1], r[2])
    okey_arr = np.array([okeys.get(o, "") for o in uo], dtype=object)
    by_artist: dict[str, list[int]] = {}
    for j, k in enumerate(okey_arr):
        if k:
            by_artist.setdefault(k.split("|", 1)[0], []).append(j)
    by_label: dict[str, list[int]] = {}
    for j, lab in enumerate(metadata.owner_labels(con, uo)):
        if lab:
            by_label.setdefault(lab, []).append(j)
    # leave-self-out masks for the labelled tracks
    owner_idx = {o: j for j, o in enumerate(uo)}
    local: dict[str, set] = {}
    for r in con.execute("SELECT spotify_id, local_id FROM liked_map WHERE local_id IS NOT NULL UNION "
                         "SELECT spotify_id, id FROM local_tracks WHERE spotify_id IS NOT NULL"):
        local.setdefault(r[0], set()).add(f"local:{r[1]}")
    cids_of: dict[str, set] = {}
    lab_sids = [x["sid"] for x in labs]
    for ch in _chunks(lab_sids):
        q = f"SELECT spotify_id, id FROM candidates WHERE spotify_id IN ({','.join('?' * len(ch))})"
        for r in con.execute(q, ch):
            cids_of.setdefault(r[0], set()).add(r[1])
    tkeys = _track_keys(con, lab_sids)
    n = len(ids)
    mask = np.zeros((n, len(uo)), bool)
    lkeys: list[set] = []
    for i, x in enumerate(labs):
        r = n_pool + i
        m = meta[x["cand_id"]]
        keys = (tkeys.get(x["sid"], set()) | {song_key(m["artist"], m["title"])}) - {"|"}
        lkeys.append(keys)
        if keys:
            mask[r] = np.isin(okey_arr, list(keys))
        own = {f"like:{x['sid']}", f"discovery:{x['sid']}", f"rated:{x['sid']}"} | local.get(x["sid"], set())
        own |= {f"manual:{c}" for c in cids_of.get(x["sid"], ())}
        for o in own:
            if o in owner_idx:
                mask[r, owner_idx[o]] = True
    ls_labels, LS, lb = vectors.liked_styles(con, cfg)
    gmask = np.zeros((n, len(ls_labels)), bool)
    if len(ls_labels) and labs:
        ls_keys = _track_keys(con, ls_labels)
        for i, x in enumerate(labs):
            gmask[n_pool + i] = [s == x["sid"] or bool(ls_keys.get(s, set()) & lkeys[i]) for s in ls_labels]
    npc = cfg["scoring"]["negative_penalty"]
    negs = vectors.negatives(con, cfg) if npc.get("w") else []
    nmask = np.zeros((n, len(negs)), bool)
    for i, x in enumerate(labs):
        cids = cids_of.get(x["sid"], set()) | {x["cand_id"]}
        nmask[n_pool + i] = [g["candidate_id"] in cids or g["deezer_id"] == x["deezer_id"] for g in negs]
    neg_artist = []
    for g in negs:
        a = None
        if g["candidate_id"] is not None and g["candidate_id"] in meta:
            a = _artist(meta[g["candidate_id"]]["artist"], meta[g["candidate_id"]]["title"])
        elif g["candidate_id"] is not None:
            r = con.execute("SELECT artist, title FROM candidates WHERE id=?", (g["candidate_id"],)).fetchone()
            a = _artist(r[0], r[1]) if r else None
        neg_artist.append(a)
    lib = vectors._library_windows(con)
    pids, pm, _ = vectors.pool_vectors(con, cfg=cfg)
    space = vectors.build_space(cfg, lib, (pids, pm))
    ref_at = {lab: j for j, lab in enumerate(space.ref_owner or [])}
    return dict(con=con, variant=variant, cfg=cfg, n_pool=n_pool, ids=ids, labs=labs, meta=meta, X=X, S=S, dids=dids,
                pos=pos, bonus=np.asarray(bonus, np.float32), stars=np.asarray(stars, np.float32), starts=starts,
                uo=uo, owner_lib=np.array([o.startswith("local:") for o in uo]), by_artist=by_artist, mask=mask,
                by_label=by_label, md=metadata.for_candidates(con, ids),
                ls=LS, lb=np.asarray(lb, np.float32), gmask=gmask, negs=negs, nmask=nmask, neg_artist=neg_artist,
                space=space, self_ref=np.array([ref_at.get(f"cand:{i}", -1) for i in ids]),
                unit=float(res.get("unit", 1.0)))


def _stage1(ctx: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (stage-1 score, unpenalised score, negative penalty) for every row, under the active variant and with each
    labelled track left out of its own positives and negatives. For a pool row it is compute_scores' score."""
    cfg, v, X, S, P = ctx["cfg"], ctx["variant"], ctx["X"], ctx["S"], ctx["n_pool"]
    k = cfg["scoring"]["topk"]
    sp = ctx["space"] if v != "raw" else Space()
    n = len(X)
    unit = ctx["unit"]
    raw_cos = None
    s = np.full(n, np.nan, np.float32)
    if v != "genre":
        cen = is_centred(v)
        Xp, Pp = sp.prep(X, cen), sp.prep(ctx["pos"], cen)
        OM = owner_max(Xp, Pp, ctx["bonus"] * unit, ctx["starts"])
        OM[ctx["mask"]] = -np.inf
        if v == "topk_mean":
            s = topk_mean(OM, k)
        else:
            j = OM.argmax(1)
            s = OM[np.arange(n), j]
            raw_cos = s - ctx["bonus"][ctx["starts"]][j] * unit
            if is_csls(v):
                s = s - hubness(Xp, sp.prep(sp.ref, cen), sp.k, ctx["self_ref"])
    if v in GENRE_VARIANTS:
        has = ~np.isnan(S).any(1)
        g = np.full(n, np.nan, np.float32)
        if len(ctx["ls"]) and has.any():
            LS = l2norm(ctx["ls"])
            G = l2norm(S[has]) @ LS.T
            g_unit = 1.0                                  # as compute_scores: the sd over the pool's own rows
            if stars_in_sd(cfg) and has[:P].any():
                g_unit = float(np.std(topk_mean(l2norm(S[:P][has[:P]]) @ LS.T, k)))
            G = G + (ctx["lb"] * g_unit)[None, :]
            G[ctx["gmask"][has]] = -np.inf
            g[has] = topk_mean(G, k)
        if v == "genre":
            s = g
        else:
            s = np.full(n, np.nan, np.float32)
            ref = has.copy()
            ref[P:] = False
            s[has] = zblend(raw_cos[has], g[has], raw_cos[ref], g[ref])
    s = np.asarray(s, np.float32)
    npc = cfg["scoring"]["negative_penalty"]
    pen = np.zeros(n, np.float32)
    if ctx["negs"]:
        E = vectors.penalty_excess(v, ctx["negs"], X, S, X[:P], S[:P], ctx["space"], npc["margin_quantile"])
        pool = s[:P]
        scale = float(np.nanstd(pool)) if np.isfinite(pool).any() else 0.0
        pen, _ = negative_penalty(E, [g["strength"] for g in ctx["negs"]], scale, npc["w"], ctx["nmask"])
    return s - pen, s, pen


def _features(ctx: dict, s1: np.ndarray, pen: np.ndarray, usual: dict, pca: dict | None,
              sources: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """-> (F [rows, features], feature names, trait values' side [rows, traits]: +1 above the usual range, -1
    below, 0 inside, NO_READING). Every row is computed the same way; labelled rows leave themselves out."""
    cfg, X, S, P = ctx["cfg"], ctx["X"], ctx["S"], ctx["n_pool"]
    rr = cfg["rerank"]
    k = cfg["scoring"]["topk"]
    n = len(X)
    cols: dict[str, np.ndarray] = {"stage1": s1}
    Xn = l2norm(X)
    OM = owner_max(Xn, l2norm(ctx["pos"]), None, ctx["starts"])
    OM[ctx["mask"]] = -np.inf
    cols["raw_max"] = OM.max(1)
    cols["topk"] = topk_mean(OM, k)
    # genre top-k without the star bonus; no style vector -> the pool median plus a flag
    has = ~np.isnan(S).any(1)
    g = np.full(n, np.nan, np.float32)
    if len(ctx["ls"]) and has.any():
        G = l2norm(S[has]) @ l2norm(ctx["ls"]).T
        G[ctx["gmask"][has]] = -np.inf
        g[has] = topk_mean(G, k)
    pool_g = g[:P][np.isfinite(g[:P])]
    cols["genre"] = np.where(np.isfinite(g), g, float(np.median(pool_g)) if len(pool_g) else 0.0)
    cols["style_missing"] = (~np.isfinite(g)).astype(np.float32)
    sp = ctx["space"]
    cols["hubness"] = (hubness(Xn, l2norm(sp.ref), sp.k, ctx["self_ref"]) if sp.ref is not None and len(sp.ref) > 1
                       else np.zeros(n, np.float32))
    if ctx["negs"]:
        E = vectors.penalty_excess("raw", ctx["negs"], X, None, X[:P], None, None,
                                   cfg["scoring"]["negative_penalty"]["margin_quantile"])
        E[ctx["nmask"]] = 0.0
        cols["neg_excess_max"] = E.max(1)
    else:
        cols["neg_excess_max"] = np.zeros(n, np.float32)
    cols["neg_penalty"] = pen
    # star-aware neighbourhood (raw space, owner level): a library file nobody rated counts as prior_stars
    stars = np.where(np.isnan(ctx["stars"][ctx["starts"]]), float(rr["prior_stars"]), ctx["stars"][ctx["starts"]])
    kn = max(1, min(int(rr["neighbours"]), OM.shape[1]))
    near = OM.argmax(1)
    cols["near_stars"] = stars[near]
    idx = np.argpartition(-OM, kn - 1, axis=1)[:, :kn]
    sims = np.take_along_axis(OM, idx, 1)
    w = np.where(np.isfinite(sims), np.maximum(sims, 0.0), 0.0)
    sk = stars[idx]
    wsum = w.sum(1)
    cols["knn_stars"] = np.where(wsum > 0, (w * sk).sum(1) / np.where(wsum > 0, wsum, 1.0), sk.mean(1))
    cols["knn_hi"] = (sk >= 4).mean(1)
    cols["knn_lib"] = ctx["owner_lib"][idx].mean(1)
    # Traits: signed distance outside the listener's usual (weighted 25-75 %) range, in widths of that range
    rd = traits.get(ctx["con"], ctx["dids"])
    side = np.zeros((n, len(traits.TRAITS)), np.int8)
    for t_i, t in enumerate(traits.TRAITS):
        u = usual.get(t)
        if not u or u.get("lo") is None or u.get("hi") is None:
            continue
        lo, hi = float(u["lo"]), float(u["hi"])
        width = (hi - lo) or 1.0
        vals = np.array([rd.get(d, {}).get(t, np.nan) for d in ctx["dids"]], np.float64)
        d_ = np.where(vals > hi, (vals - hi) / width, np.where(vals < lo, (vals - lo) / width, 0.0))
        side[:, t_i] = np.where(np.isnan(vals), NO_READING, np.sign(np.nan_to_num(d_))).astype(np.int8)
        cols[f"trait:{t}"] = np.nan_to_num(d_)
    cols["traits_missing"] = np.array([0.0 if rd.get(d) else 1.0 for d in ctx["dids"]], np.float32)
    meta = ctx["meta"]
    src = [meta[i]["source"] for i in ctx["ids"]]
    for s_ in sources:
        cols[f"source:{s_}"] = np.array([x == s_ for x in src], np.float32)
    charts = set(cfg["eval"]["chart_sources"])
    cols["chart"] = np.array([x in charts for x in src], np.float32)
    arts = [_artist(meta[i]["artist"], meta[i]["title"]) for i in ctx["ids"]]
    cols["artist_in_positives"] = np.array([any(not ctx["mask"][r, j] for j in ctx["by_artist"].get(a, ()))
                                            for r, a in enumerate(arts)], np.float32)
    dis = [(a, j) for j, (a, g_) in enumerate(zip(ctx["neg_artist"], ctx["negs"])) if a and g_["kind"] == "dislike"]
    cols["artist_has_dislike"] = np.array([any(a == b and not ctx["nmask"][r, j] for b, j in dis)
                                           for r, a in enumerate(arts)], np.float32)
    cols.update(_meta_features(ctx))
    if pca:
        Z = (Xn - np.asarray(pca["mean"], np.float32)[None, :]) @ np.asarray(pca["components"], np.float32).T
        for c in range(Z.shape[1]):
            cols[f"pca:{c}"] = Z[:, c]
    names = list(cols)
    F = np.column_stack([np.asarray(cols[c], np.float64) for c in names])
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0), names, side


def _meta_features(ctx: dict) -> dict[str, np.ndarray]:
    """What Deezer publishes about each row's track (ADR 0017). A missing number takes the pool's median; a track
    with no rank also gets rank_missing. label_known: the label is on one of the positives (not the track itself);
    label_kept_rate: see _label_rate."""
    P, ids, md = ctx["n_pool"], ctx["ids"], ctx["md"]
    rows = [md.get(i, {}) for i in ids]

    def num(vals):
        v = np.array([np.nan if x is None else float(x) for x in vals], np.float64)
        pool = v[:P][np.isfinite(v[:P])]
        return np.where(np.isfinite(v), v, float(np.median(pool)) if len(pool) else 0.0), ~np.isfinite(v)

    out: dict[str, np.ndarray] = {}
    rank, missing = num([None if m.get("rank") is None else math.log10(1 + max(0, m["rank"])) for m in rows])
    out["log_rank"], out["rank_missing"] = rank, missing.astype(np.float32)
    out["log_fans"] = num([None if m.get("fans") is None else math.log10(1 + max(0, m["fans"])) for m in rows])[0]
    out["age_years"] = num([m.get("age") for m in rows])[0]
    out["compilation"] = np.array([bool(m.get("compilation")) for m in rows], np.float32)
    labels = [m.get("label") for m in rows]
    out["label_known"] = np.array([bool(lab) and any(not ctx["mask"][r, j] for j in ctx["by_label"].get(lab, ()))
                                   for r, lab in enumerate(labels)], np.float32)
    out["label_kept_rate"] = _label_rate(ctx)
    return out


def _label_rate(ctx: dict, rows=None, drop: int | None = None) -> np.ndarray:
    """label_kept_rate for `rows` (every row by default): the share of the hard labels on the row's record label
    that were kept, the row's own verdict left out, shrunk toward the mean kept rate of every hard label by
    rerank.label_prior_n pseudo-labels (the mean alone with no label). drop: a labelled row whose verdict is in no
    row's rate and not in the mean, the held-out label of a leave-one-out fold (ADR 0017)."""
    P, md = ctx["n_pool"], ctx["md"]
    labels = [md.get(i, {}).get("label") for i in ctx["ids"]]
    own = {P + i: x["y"] for i, x in enumerate(ctx["labs"]) if not x["soft"] and P + i != drop}
    mean = float(np.mean(list(own.values()))) if own else 0.5
    kept: dict[str, list[int]] = {}
    for r, y in own.items():
        if labels[r]:
            kept.setdefault(labels[r], []).append(y)
    n0 = float(ctx["cfg"]["rerank"]["label_prior_n"])
    rows = range(len(labels)) if rows is None else rows
    rate = np.full(len(rows), mean, np.float64)
    for o, r in enumerate(rows):
        if labels[r] in kept:
            ys = kept[labels[r]]
            k, m_ = sum(ys) - own.get(r, 0), len(ys) - (r in own)
            rate[o] = (k + n0 * mean) / (m_ + n0) if m_ + n0 else mean
    return rate


def _fit_pca(ctx: dict, n_comp: int) -> dict:
    """The first n_comp principal components of the preview space, fitted on the pool plus one mean vector per
    positive track."""
    P = ctx["n_pool"]
    pos = l2norm(ctx["pos"])
    ends = list(ctx["starts"][1:]) + [len(pos)]
    owner_mean = l2norm(np.vstack([pos[a:b].mean(0) for a, b in zip(ctx["starts"], ends)]))
    M = np.vstack([l2norm(ctx["X"][:P]), owner_mean]).astype(np.float64)
    mu = M.mean(0)
    _, _, vt = np.linalg.svd(M - mu, full_matrices=False)
    comp = vt[:min(n_comp, len(vt))]
    return {"mean": np.round(mu, PCA_ROUND).tolist(), "components": np.round(comp, PCA_ROUND).tolist()}


# ---------------- model ----------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit(Z: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, steps: int = IRLS_STEPS) -> np.ndarray:
    """L2-regularised logistic regression by IRLS (Newton) with a fixed step cap; the intercept is not penalised.
    It minimises the weighted mean log-loss + l2 / 2 * |beta|^2, so l2 is per unit of label weight: the same
    shrinkage at 40 labels as at 400 (ADR 0011). -> beta [1 + features] (intercept first)."""
    A = np.hstack([np.ones((len(Z), 1)), Z])
    beta = np.zeros(A.shape[1])
    R = np.eye(A.shape[1]) * l2 * float(w.sum())
    R[0, 0] = 0.0
    for _ in range(steps):
        p = _sigmoid(A @ beta)
        H = A.T @ (A * (w * p * (1 - p))[:, None]) + R + 1e-9 * np.eye(A.shape[1])
        grad = A.T @ (w * (y - p)) - R @ beta
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def wauc(score, y, w) -> float | None:
    """Chance that a yes outscores a no, both sides weighted (ties count half)."""
    score, y, w = np.asarray(score, np.float64), np.asarray(y), np.asarray(w, np.float64)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return None
    sp, sn = score[pos][:, None], score[neg][None, :]
    gt = (sp > sn) + 0.5 * (sp == sn)
    wp, wn = w[pos][:, None], w[neg][None, :]
    return float((wp * wn * gt).sum() / (w[pos].sum() * w[neg].sum()))


def loo_logits(Z: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, rows=None, refresh=None) -> np.ndarray:
    """Each label's logit from a model fitted on all the other labels (rows: only these labels, NaN elsewhere).
    refresh(i): the features for label i's fold, when a feature is worked out from the verdicts themselves, so
    the held-out verdict is in no row's features."""
    out = np.full(len(Z), np.nan)
    keep = np.ones(len(Z), bool)
    for i in (range(len(Z)) if rows is None else rows):
        Zi = Z if refresh is None else refresh(i)
        keep[i] = False
        b = fit(Zi[keep], y[keep], w[keep], l2)
        out[i] = b[0] + Zi[i] @ b[1:]
        keep[i] = True
    return out


def _loo_refresh(ctx: dict, Zl: np.ndarray, names: list[str], mu: np.ndarray, sd: np.ndarray, lab_rows):
    """loo_logits' refresh for the labels' features Zl: label_kept_rate worked out again without the held-out
    label's verdict, standardised like the rest."""
    j = names.index("label_kept_rate")

    def refresh(i):
        Zi = Zl.copy()
        Zi[:, j] = (_label_rate(ctx, lab_rows, drop=int(lab_rows[i])) - mu[j]) / sd[j]
        return Zi
    return refresh


def _phrase(name: str, high: bool, side: int | None, up: bool = True) -> str | None:
    """The phrase for a feature that moved a track up (up) or down, with the track high or low on it."""
    if name.startswith("trait:"):
        t = name.split(":", 1)[1]
        if side == 0:                         # inside the usual range, and that beat the pool
            return f"{TRAIT_NOUN.get(t, t)} in your usual range" if up else None
        if side not in (1, -1) or (side > 0) != high:   # e.g. above the pool's average but still below the range
            return None
        return f"{TRAIT_PHRASES[t][0 if side > 0 else 1]} than you usually like"
    if name.startswith("source:"):
        from .review import SOURCE_LABELS
        s = name.split(":", 1)[1]
        return f"found via {SOURCE_LABELS.get(s, s)}" if high else None
    if name.startswith("pca:"):
        return None
    hi, lo = PHRASES.get(name, (None, None))
    return hi if high else lo


def direction(Z: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Each feature's own direction over the labels: the sign of (weighted mean over the yes) - (over the no)."""
    yes, no = y == 1, y == 0
    return np.sign(np.average(Z[yes], 0, w[yes]) - np.average(Z[no], 0, w[no]))


def explain(beta: np.ndarray, z: np.ndarray, names: list[str], side_row: np.ndarray, agree: np.ndarray,
            sign: int = 1, top: int = 2, skip=()) -> list[str]:
    """The `top` phrases for the features that push this track's chance up the most (sign=1) or down the most
    (sign=-1): weight x standardised value, largest first, each at least REASON_MIN. Features without a phrase,
    phrases in `skip`, and features whose weight is not a true reason (agree False: it disagrees with the
    feature's own direction over the labels, so it only corrects for a correlated feature, or with
    EXPECTED_SIGN) are skipped."""
    c = beta[1:] * z * sign
    out: list[str] = []
    t_index = {f"trait:{t}": i for i, t in enumerate(traits.TRAITS)}
    for j in np.argsort(-c):
        if c[j] < REASON_MIN or len(out) >= top:
            break
        if not agree[j]:
            continue
        name = names[j]
        p = _phrase(name, z[j] > 0, int(side_row[t_index[name]]) if name in t_index else None, sign > 0)
        if p and p not in out and p not in skip:
            out.append(p)
    return out


# ---------------- the second stage ----------------

def run(con, cfg: dict, variant: str, res: dict, trigger: str = "batch") -> dict:
    """Fit the re-ranker on the current labels and, when it passes the gate, order the top of the pool.
    res: vectors.compute_scores(con, cfg, variant). -> {used, reason, variant, n_labels, n_pos, n_neg (hard
    labels), n_soft, loo_auc_model, loo_auc_stage1 (over the hard labels), n_features, chance ({candidate id: {p,
    reasons}}, every scored pool row, whenever the model is fitted), ranks ({candidate id: rank in stage-1 score
    units}, every scored pool row, only when used), top ({candidate id: {p, blend, reasons, held_back}}, only when
    used), model}."""
    rr = cfg["rerank"]
    out = {"used": False, "reason": None, "variant": variant, "trigger": trigger, "n_labels": 0, "n_pos": 0,
           "n_neg": 0, "n_soft": 0, "loo_auc_model": None, "loo_auc_stage1": None, "loo_auc_no_meta": None,
           "n_features": 0, "chance": None, "ranks": None, "top": None, "model": None, "at": db.now()}
    if not rr["enabled"]:
        return {**out, "reason": "rerank.enabled is false"}
    labs = labels(con, cfg)
    ctx = _prepare(con, cfg, variant, res, labs)
    if ctx is None:
        return {**out, "reason": "no positives or no scored pool"}
    s1, _, pen = _stage1(ctx)
    P = ctx["n_pool"]
    lab_rows = np.arange(P, len(ctx["ids"]))
    ok = np.isfinite(s1[lab_rows])                     # a genre variant can't score a track with no style vector
    lab_rows = lab_rows[ok]
    labs = [x for x, o in zip(ctx["labs"], ok) if o]
    y = np.array([x["y"] for x in labs], np.int8)
    w = np.array([x["w"] for x in labs], np.float64)
    hard = np.array([not x["soft"] for x in labs], bool)
    out.update(n_labels=int(hard.sum()), n_pos=int(y[hard].sum()), n_neg=int(hard.sum() - y[hard].sum()),
               n_soft=int((~hard).sum()))
    usual = baseline.compute(con, cfg)["traits"]
    sources = sorted({ctx["meta"][ctx["ids"][r]]["source"] for r in lab_rows})
    pca = _fit_pca(ctx, int(rr["pca_components"])) if len(labs) >= rr["min_labels_embed"] else None
    F, names, side = _features(ctx, s1, pen, usual, pca, sources)
    pool_ok = np.isfinite(s1[:P])
    mu = F[:P][pool_ok].mean(0)
    sd = F[:P][pool_ok].std(0)
    sd[sd == 0] = 1.0
    Z = (F - mu) / sd
    out["n_features"] = len(names)
    if out["n_pos"] < 2 or out["n_neg"] < 2:
        return {**out, "reason": f"{out['n_pos']} yes and {out['n_neg']} no labels: too few to fit"}
    Zl = Z[lab_rows]
    h = np.nonzero(hard)[0]
    out["loo_auc_stage1"] = _r(wauc(s1[lab_rows][h], y[h], w[h]))
    refresh = _loo_refresh(ctx, Zl, names, mu, sd, lab_rows)
    out["loo_auc_model"] = _r(wauc(loo_logits(Zl, y, w, rr["l2"], h, refresh)[h], y[h], w[h]))
    meta = np.array([f in META_FEATURES for f in names])
    if Zl[:, meta].std(0).max(initial=0.0) > 0:          # without Deezer's metadata, for `rerank show`
        out["loo_auc_no_meta"] = _r(wauc(loo_logits(Zl[:, ~meta], y, w, rr["l2"], h)[h], y[h], w[h]))
    else:                                                 # none known for any label: the same model
        out["loo_auc_no_meta"] = out["loo_auc_model"]
    beta = fit(Zl, y, w, rr["l2"])
    prior = np.array([EXPECTED_SIGN.get(f, 0) for f in names])
    agree = (np.sign(beta[1:]) == direction(Zl, y, w)) & (beta[1:] != 0) & ((prior == 0) | (np.sign(beta[1:]) == prior))
    out["model"] = {"features": names, "mean": _rl(mu), "sd": _rl(sd), "beta": _rl(beta), "sources": sources,
                    "agree": [bool(x) for x in agree], "pca": pca, "l2": rr["l2"], "alpha": rr["alpha"]}
    pool_rows = np.nonzero(pool_ok)[0]
    p_all = _sigmoid(beta[0] + Z[pool_rows] @ beta[1:])
    out["chance"] = {ctx["ids"][r]: {"p": round(float(pi), 4), "reasons": explain(beta, Z[r], names, side[r], agree)}
                     for r, pi in zip(pool_rows, p_all)}
    gain = None if out["loo_auc_model"] is None or out["loo_auc_stage1"] is None else \
        out["loo_auc_model"] - out["loo_auc_stage1"]
    if out["n_labels"] < rr["min_labels"]:
        return {**out, "reason": f"{out['n_labels']} labels (needs {rr['min_labels']})"}
    if gain is None or gain < rr["min_gain"]:
        return {**out, "reason": f"leave-one-out AUC {out['loo_auc_model']} vs stage 1 {out['loo_auc_stage1']} "
                                 f"(needs +{rr['min_gain']})"}
    # blend over the top of the pool by stage-1 score
    order = pool_rows[np.argsort(-s1[pool_rows], kind="stable")]
    top = order[:int(rr["pool_top"])]
    rest = order[len(top):]
    logit = beta[0] + Z[top] @ beta[1:]
    a = float(rr["alpha"])
    m1, sd1 = float(s1[top].mean()), float(s1[top].std()) or 1.0
    blend = a * _z(logit) + (1 - a) * (s1[top] - m1) / sd1
    rank = m1 + sd1 * blend
    ranks = {ctx["ids"][r]: float(x) for r, x in zip(top, rank)}
    if len(rest):        # everything below the top keeps its stage-1 order, under the re-ranked top
        shift = min(0.0, float(rank.min()) - float(s1[rest].max()) - 1e-6)
        ranks.update({ctx["ids"][r]: float(s1[r]) + shift for r in rest})
    p = _sigmoid(logit)
    out["top"] = {}
    for r, pi, b in zip(top, p, blend):
        why = explain(beta, Z[r], names, side[r], agree)
        out["top"][ctx["ids"][r]] = {"p": round(float(pi), 4), "blend": round(float(b), 4), "reasons": why,
                                     "held_back": explain(beta, Z[r], names, side[r], agree, sign=-1, skip=why)}
    return {**out, "used": True, "reason": f"leave-one-out AUC {out['loo_auc_model']} vs stage 1 "
                                            f"{out['loo_auc_stage1']}", "ranks": ranks}


def _z(x):
    sd = float(np.std(x))
    return (x - float(np.mean(x))) / (sd or 1.0)


def _r(x, nd: int = 4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def _rl(a, nd: int = 6) -> list:
    return [round(float(x), nd) for x in a]


def summary(r: dict) -> dict:
    """The batch summary's `rerank` block (no per-candidate data)."""
    return {k: r.get(k) for k in ("used", "reason", "variant", "n_labels", "n_pos", "n_neg", "n_soft",
                                  "loo_auc_model", "loo_auc_stage1", "loo_auc_no_meta", "n_features")}


def record(con, r: dict) -> None:
    """Keep the fit: kv rerank_model (the last one) and a rerank_runs row."""
    model = None if r.get("model") is None else {**r["model"], **summary(r), "at": r["at"]}
    db.kv_set(con, "rerank_model", json.dumps(model if model else summary(r)))
    con.execute("""INSERT INTO rerank_runs(at, trigger, variant, n_labels, n_pos, n_neg, n_soft, loo_auc_model,
                   loo_auc_stage1, loo_auc_no_meta, used, reason, model_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r["at"], r["trigger"], r["variant"], r["n_labels"], r["n_pos"], r["n_neg"], r.get("n_soft", 0),
                 r["loo_auc_model"], r["loo_auc_stage1"], r.get("loo_auc_no_meta"), int(bool(r["used"])),
                 r["reason"], json.dumps(model) if model else None))


def write_pool(con, r: dict) -> None:
    """Clear the pool's re-ranker columns, write the chance and reasons of every scored row when the model was
    fitted, and the blend (rerank_score) of the top rows when it was used: rerank_score marks a re-ranked pick."""
    con.execute("BEGIN")
    con.execute("UPDATE candidates SET rerank_score=NULL, rerank_p=NULL, rerank_reason=NULL WHERE state='pool'")
    con.executemany("UPDATE candidates SET rerank_p=?, rerank_reason=? WHERE id=?",
                    [(x["p"], json.dumps(x["reasons"]), cid) for cid, x in (r.get("chance") or {}).items()])
    con.executemany("UPDATE candidates SET rerank_score=?, rerank_p=?, rerank_reason=? WHERE id=?",
                    [(x["blend"], x["p"], json.dumps(x["reasons"]), cid) for cid, x in (r.get("top") or {}).items()])
    con.execute("COMMIT")


def last(con) -> dict | None:
    v = db.kv_get(con, "rerank_model")
    return json.loads(v) if v else None


def runs(con, limit: int = 10) -> list[dict]:
    """The last fits, newest first (a database from before 0.6.0 has no loo_auc_no_meta: it reads as None)."""
    out = []
    for x in con.execute("SELECT * FROM rerank_runs ORDER BY id DESC LIMIT ?", (limit,)):
        d = dict(x)
        d.pop("model_json", None)
        out.append({"loo_auc_no_meta": None, **d})
    return out
