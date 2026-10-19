"""Offline eval of the scoring variants (token-free; no Spotify calls; `choose` records the result and may store
the variant, ADR 0009).

Positives are built exactly as for scoring (library windows + liked songs' vectors). For each seed a random
share of the liked songs that have a vector is held out: their rows (and any library file of the same song) are
removed from the positives, and each held-out song is ranked against the harvested candidate pool, which
stands in for tracks nobody chose for the listener.

The objective comes from the offers, each scored exactly like a candidate: the ones the listener kept (a Rating
of 3 stars or more, or liked without a Rating), each with its own rows and same-song library files left out of its
positives, against the ones they turned down (a Pass or a Dislike, or expired after a play or removed by hand).
Both sides were offered, so both were picked from the top of the pool the same way; held-out liked songs were
never picked, and next to offers they make whichever variant did the picking look worst. How well a variant ranks
the kept offers above the turned-down ones:
  auc              chance that a kept offer outscores a turned-down one (0.5 = chance), weighted by stars
  gap              mean percentile of the kept offers minus that of the turned-down ones
  top-50 spread    span of the pool's top 50 scores in pool-score sd; under 0.25 the variant is degenerate (its
                   top picks are near-ties) and is not chosen

A held-out song must be represented exactly like a candidate: its Deezer 30 s preview vector. Liked songs
that only have a local-file vector are reported separately as a diagnostic and never drive the choice: a
local-file window is closer to the (local-file) library positives than any Deezer preview can be, and
centring/CSLS amplify that audio-domain gap, so those queries reward domain rather than taste.

Diagnostics per variant (averaged over seeds), and the objective before 0.2 (eval.objective = "mean_pct", also
the fallback while there are fewer than eval.min_verdicts kept or turned-down offers):
  mean percentile  share of pool candidates the held-out song outscores (100 = top, 50 = chance)
  recall@K         share of held-out songs that would rank in the pool's top K (and top 50)
  chart share      share of the pool's top K (one per artist, like a batch) from Deezer chart sources"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime

import numpy as np

from . import db, embed, status, vectors
from .scoring import (EMBED_VARIANTS, GENRE_VARIANTS, VARIANTS, choose_variant, hubness, is_centred, is_csls, l2norm,
                      negative_penalty, owner_max, topk_mean, zblend)
from .textnorm import song_key

log = logging.getLogger("haystack_music.evaluate")
TOP = 50                 # recall@50 and the top-50 spread
DEGENERATE_SD = 0.25     # a top 50 spanning less than this many pool-score sd is degenerate


def _chunks(seq, n=500):
    for s in range(0, len(seq), n):
        yield seq[s:s + n]


def _first_artist(artists_json) -> str:
    a = json.loads(artists_json or "[]")
    return a[0] if a else ""


def _owner_keys(con, owners) -> dict[str, str]:
    keys: dict[str, str] = {}
    local_ids = [int(o.split(":", 1)[1]) for o in owners if o.startswith("local:")]
    for ch in _chunks(local_ids):
        for r in con.execute(f"SELECT id, artist_raw, title_raw FROM local_tracks WHERE id IN ({','.join('?' * len(ch))})", ch):
            keys[f"local:{r[0]}"] = song_key(r[1], r[2])
    sids = sorted({o.split(":", 1)[1] for o in owners if o.startswith(("like:", "discovery:", "rated:"))})
    for ch in _chunks(sids):
        for r in con.execute(f"SELECT spotify_id, name, artists_json FROM spotify_tracks WHERE spotify_id IN "
                             f"({','.join('?' * len(ch))})", ch):
            k = song_key(_first_artist(r[2]), r[1])
            keys[f"like:{r[0]}"] = keys[f"discovery:{r[0]}"] = keys[f"rated:{r[0]}"] = k
    return keys


def _track_keys(con, sids) -> dict[str, set]:
    """spotify id -> the song keys it goes by (its Spotify name and any candidate's Deezer artist/title)."""
    out: dict[str, set] = {}
    for ch in _chunks(list(sids)):
        ph = ",".join("?" * len(ch))
        for r in con.execute(f"SELECT spotify_id, name, artists_json FROM spotify_tracks WHERE spotify_id IN ({ph})", ch):
            if r[1]:
                out.setdefault(r[0], set()).add(song_key(_first_artist(r[2]), r[1]))
        for r in con.execute(f"SELECT spotify_id, artist, title FROM candidates WHERE spotify_id IN ({ph})", ch):
            if r[2]:
                out.setdefault(r[0], set()).add(song_key(r[1], r[2]))
    return {s: k - {None, "|"} for s, k in out.items()}


def _verdict_queries(con, cfg: dict, liked_q: list[dict], liked_ids: set, known: set) -> tuple[list[dict], list]:
    """The offers the listener decided, as queries by their own Deezer preview vector (ADR 0009). Also sets the
    liked queries' weight (stars; an Unrated like counts as scoring.verdicts.unrated_like_stars).
    kept: weighted by its stars: a Rating of 3 stars or more, or liked with no Rating (also when it is a liked
      song now: then it is a liked-song query as well).
    turned down: a current Pass or Dislike, or an offer that expired after a play or was removed by hand with no
      verdict since. Liked songs, and songs that are liked or in the library under another id, are left out.
    Only offers count: a track rated or passed elsewhere was never picked from the pool (see the module doc)."""
    from . import baseline, verdicts
    cur = verdicts.current(con)
    weights = baseline.liked_weights(con, cfg)
    for x in liked_q:
        x["weight"] = weights.get(x["sid"], 0.0)
    offered = {r[0] for r in con.execute("SELECT spotify_id FROM offers WHERE spotify_id IS NOT NULL")}

    def kept(sid):
        v = cur.get(sid)
        return v.keeps_heart if v is not None else sid in liked_ids
    yes = [s for s in weights if s in offered and kept(s)]
    no = [s for s, v in cur.items() if s in offered and v.kind != "rating" and s not in liked_ids]
    no += sorted({r[0] for r in con.execute("SELECT spotify_id FROM offers WHERE outcome IN ('negative', 'manual_removed') "
                                            "AND spotify_id IS NOT NULL")} - set(cur) - liked_ids)
    tv = vectors.track_vectors(con, yes + no)
    keys = _track_keys(con, tv)
    local: dict[str, set] = {}
    for r in con.execute("SELECT spotify_id, local_id FROM liked_map WHERE local_id IS NOT NULL UNION "
                         "SELECT spotify_id, id FROM local_tracks WHERE spotify_id IS NOT NULL"):
        local.setdefault(r[0], set()).add(f"local:{r[1]}")
    vec = vectors.get_deezer_vectors(con, [t["deezer_id"] for t in tv.values()])
    out, vecs = [], []
    for kind, sids in (("kept", yes), ("turned down", no)):
        for sid in sids:
            t = tv.get(sid)
            if not t or t["deezer_id"] not in vec:
                continue
            k = keys.get(sid, set())
            if kind == "turned down" and k & known:
                continue
            cid = t["candidate_id"]
            owners = {f"like:{sid}", f"discovery:{sid}", f"rated:{sid}"} | local.get(sid, set())
            if cid is not None:
                owners.add(f"manual:{cid}")
            out.append(dict(sid=sid, owner=f"cand:{cid}", owners=owners, kind=kind, keys=k, did=t["deezer_id"],
                            label=t["label"], weight=weights[sid] if kind == "kept" else 1.0))
            vecs.append(vec[t["deezer_id"]])
    return out, vecs


def load(con, cfg: dict) -> dict:
    """Everything the eval needs, loaded once (vectors in raw space)."""
    pos, owners, bonus = vectors.positives(con, cfg)
    lib = vectors._library_windows(con)
    pool_ids, pool_m, pool_rows = vectors.pool_vectors(con)
    space = vectors.build_space(cfg, lib, (pool_ids, pool_m))
    # contiguous owner runs -> owner-level max via reduceat
    starts = [i for i in range(len(owners)) if i == 0 or owners[i] != owners[i - 1]]
    uo = [owners[i] for i in starts]
    assert len(set(uo)) == len(uo), "positives are not grouped by owner"
    run_of = {o: (s, (starts[j + 1] if j + 1 < len(starts) else len(owners))) for j, (o, s) in enumerate(zip(uo, starts))}
    okeys = _owner_keys(con, uo)
    liked = con.execute("""SELECT m.spotify_id, m.use_local, m.local_id, m.deezer_id, m.vector_status,
                           s.name, s.artists_json
                           FROM likes l JOIN liked_map m ON m.spotify_id=l.spotify_id
                           LEFT JOIN spotify_tracks s ON s.spotify_id=l.spotify_id
                           WHERE l.removed_at IS NULL""").fetchall()
    liked_ids = {r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL")}
    liked_keys = {song_key(_first_artist(r["artists_json"]), r["name"]) for r in liked if r["name"]}
    dzv = vectors.get_deezer_vectors(con, [r["deezer_id"] for r in liked if r["vector_status"] == "ok"])
    queries, qvecs = [], []
    for r in liked:
        owner = f"local:{r['local_id']}" if r["use_local"] else f"like:{r['spotify_id']}"
        if r["vector_status"] == "ok" and r["deezer_id"] in dzv:
            v, kind = dzv[r["deezer_id"]], "preview"
        elif owner in run_of:
            s, e = run_of[owner]
            v, kind = pos[s + (e - s) // 2], "local file"      # middle 30 s window (diagnostic only)
        else:
            continue
        keys = {okeys.get(owner)} | ({song_key(_first_artist(r["artists_json"]), r["name"])} if r["name"] else set())
        liked_keys |= keys
        queries.append(dict(sid=r["spotify_id"], owner=owner, kind=kind, keys=keys - {None, "|"},
                            did=r["deezer_id"] if kind == "preview" else None,
                            label=f"{_first_artist(r['artists_json'])} - {r['name']}"))
        qvecs.append(v)
    lib_keys = {song_key(r[0], r[1]) for r in con.execute("SELECT artist_raw, title_raw FROM local_tracks WHERE present=1")}
    known = liked_keys | lib_keys
    keep = [i for i, r in enumerate(pool_rows)
            if r["spotify_id"] not in liked_ids and song_key(r["artist"], r["title"]) not in known]
    vq, vvecs = _verdict_queries(con, cfg, queries, liked_ids, known)
    queries += vq
    qvecs += vvecs
    try:
        from . import feedback
        jrows = feedback.judged(con, cfg)
    except Exception:  # noqa: BLE001
        jrows = []
    jv = vectors.get_deezer_vectors(con, [r["deezer_id"] for r in jrows])
    js = vectors.get_styles(con, [r["deezer_id"] for r in jrows])
    judged = [dict(label=f"{r['artist']} - {r['title']}", verdict=r["verdict"], weight=r["weight"],
                   owner=f"manual:{r['candidate_id']}", candidate_id=r["candidate_id"], vec=jv[r["deezer_id"]],
                   style=js.get(r["deezer_id"]))
              for r in jrows if r["deezer_id"] in jv]
    try:
        negs = vectors.negatives(con, cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("cannot load negatives: %r", e)
        negs = []
    qst = vectors.get_styles(con, [x["did"] for x in queries if x["did"]])
    pst = vectors.get_styles(con, [pool_rows[i]["deezer_id"] for i in keep])
    nan = np.full(400, np.nan, np.float32)
    return dict(pos=pos, owners=owners, bonus=bonus, starts=np.array(starts), uo=uo, okeys=okeys, space=space, judged=judged,
                negs=negs,
                q_style=np.vstack([qst.get(x["did"], nan) for x in queries]) if queries else np.zeros((0, 400), np.float32),
                pool_style=np.vstack([pst.get(pool_rows[i]["deezer_id"], nan) for i in keep]) if keep
                else np.zeros((0, 400), np.float32),
                queries=queries, q=np.vstack(qvecs).astype(np.float32) if qvecs else np.zeros((0, 512), np.float32),
                pool_ids=[pool_ids[i] for i in keep], pool_m=pool_m[keep] if keep else np.zeros((0, 512), np.float32),
                pool_rows=[pool_rows[i] for i in keep], pool_total=len(pool_ids))


def _owner_max(x, p, bonus, starts, chunk=256):
    return owner_max(x, p, bonus, starts, chunk)


def liked_genres(con, cfg: dict | None = None, top: int = 15) -> dict:
    """Genre makeup of the liked songs that have a style vector: mean activation per style, top-1 style counts,
    and parent-genre shares (activation mass per parent genre)."""
    _, st, _ = vectors.liked_styles(con, cfg)
    if not len(st):
        return {"liked_with_styles": 0}
    classes = embed.style_classes()
    mean = st.mean(0)
    top1 = np.bincount(st.argmax(1), minlength=len(classes))
    parent = {}
    for i, c in enumerate(classes):
        parent[c.split("---")[0]] = parent.get(c.split("---")[0], 0.0) + float(mean[i])
    tot = sum(parent.values()) or 1.0
    return {"liked_with_styles": int(len(st)),
            "top_styles_mean_activation": [[classes[i].replace("---", " / "), round(float(mean[i]), 3)]
                                           for i in np.argsort(-mean)[:top]],
            "top1_style_counts": [[classes[i].replace("---", " / "), int(top1[i])] for i in np.argsort(-top1)[:top] if top1[i]],
            "parent_genre_share": [[g, round(v / tot, 3)] for g, v in sorted(parent.items(), key=lambda kv: -kv[1])]}


def _judged_check(d, variants, mats, geo_extra, GL, GC, kt, npc: dict | None = None) -> dict:
    """Sanity check only (not a selection criterion): percentile of each manually judged track among the pool
    under each variant, with all positives except the judged tracks themselves. With a negative penalty
    configured (npc['w'] > 0) also the percentile after the penalty ('pct_pen'), applied to the pool and the
    judged tracks alike; a rejected track is not penalised by its own verdict (leave-one-out)."""
    J = d["judged"]
    sp = d["space"]
    jv = np.vstack([x["vec"] for x in J]).astype(np.float32)
    keepc = ~np.isin(np.array(d["uo"], dtype=object), [x["owner"] for x in J])
    has_style = all(x["style"] is not None for x in J)
    out = {"tracks": [dict(label=x["label"], verdict=x["verdict"], weight=x["weight"]) for x in J], "variants": {}}
    for v in variants:
        if v in ("genre", "genre_blend") and (GL is None or not has_style):
            continue
        parts = {}
        for geo, agg in ((("raw", "max"),) if v in ("raw", "genre_blend") else ((v, "max"),) if v in
                         ("centred", "csls", "csls_centred") else (("raw", "topk"),) if v == "topk_mean" else ()):
            SC, _, hc, _ = mats[geo]
            P, R, cen, b = geo_extra[geo]
            Jp = sp.prep(jv, cen)
            SJ = _owner_max(Jp, P, b, d["starts"])
            hj = hubness(Jp, R, sp.k) if R is not None else np.zeros(len(J), np.float32)
            f = (lambda M: M.max(1)) if agg == "max" else (lambda M: topk_mean(M, kt))
            parts["emb"] = (f(SC[:, keepc]) - hc, f(SJ[:, keepc]) - hj)
        if v in ("genre", "genre_blend"):
            GJ = l2norm(np.vstack([x["style"] for x in J])) @ GL.T
            parts["genre"] = (topk_mean(GC, kt), topk_mean(GJ, kt))
        if v == "genre":
            pool, js = parts["genre"]
        elif v == "genre_blend":
            (rp, rj), (gp, gj) = parts["emb"], parts["genre"]
            pool, js = zblend(rp, gp), zblend(rj, gj, rp, gp)
        else:
            pool, js = parts["emb"]
        def pcts(pool, js):
            pct = [round(100.0 * float((pool < x).mean() + 0.5 * (pool == x).mean()), 1) for x in js]
            yes = [p_ for p_, x in zip(pct, J) if x["verdict"] == "yes"]
            no = [p_ for p_, x in zip(pct, J) if x["verdict"] == "no"]
            return pct, (round(float(np.mean(yes)), 1) if yes else None), (round(float(np.mean(no)), 1) if no else None)
        pct, my, mn = pcts(pool, js)
        out["variants"][v] = {"pct": pct, "mean_yes": my, "mean_no": mn}
        negs = d.get("negs") or []
        if npc and npc.get("w") and negs:
            gen = v in ("genre", "genre_blend")
            jst = np.vstack([x["style"] for x in J]).astype(np.float32) if gen else None
            pst = d["pool_style"] if gen else None
            q = npc["margin_quantile"]
            Ep = vectors.penalty_excess(v, negs, d["pool_m"], pst, d["pool_m"], pst, sp, q)
            Ej = vectors.penalty_excess(v, negs, jv, jst, d["pool_m"], pst, sp, q)
            strength = [x["strength"] for x in negs]
            scale = float(np.nanstd(pool))
            self_mask = np.array([[x.get("candidate_id") == n_["candidate_id"] for n_ in negs] for x in J])
            pp, _ = negative_penalty(Ep, strength, scale, npc["w"])
            pj, _ = negative_penalty(Ej, strength, scale, npc["w"], self_mask)
            pct, my, mn = pcts(pool - pp, js - pj)
            out["variants"][v].update(pct_pen=pct, mean_yes_pen=my, mean_no_pen=mn)
    if npc and npc.get("w") and d.get("negs"):
        out["penalty"] = {"w": npc["w"], "margin_quantile": npc["margin_quantile"], "negatives": len(d["negs"])}
    return out


def _pct(sorted_pool: np.ndarray, x) -> np.ndarray:
    """Percentile of each x among the pool (100 = above all of it, ties count half)."""
    lo = np.searchsorted(sorted_pool, x, "left")
    hi = np.searchsorted(sorted_pool, x, "right")
    return 100.0 * (lo + 0.5 * (hi - lo)) / len(sorted_pool)


def auc(pos, weights, neg) -> float | None:
    """Weighted chance that a positive outscores a negative (ties count half); the weights are on the positive side."""
    pos, w, neg = np.asarray(pos, np.float64), np.asarray(weights, np.float64), np.sort(np.asarray(neg, np.float64))
    if not len(pos) or not len(neg) or w.sum() <= 0:
        return None
    lo = np.searchsorted(neg, pos, "left")
    hi = np.searchsorted(neg, pos, "right")
    return float((w * (lo + 0.5 * (hi - lo))).sum() / (w.sum() * len(neg)))


def run(con, cfg: dict, data: dict | None = None, variants=VARIANTS) -> dict:
    ev = cfg["eval"]
    if ev["objective"] not in ("auc", "mean_pct"):
        raise ValueError(f"eval.objective {ev['objective']!r} is not auc or mean_pct")
    K = ev["recall_k"]
    kt = cfg["scoring"]["topk"]
    d = data or load(con, cfg)
    kinds = np.array([x["kind"] for x in d["queries"]])
    q_has = ~np.isnan(d["q_style"]).any(1) if len(d["q_style"]) else np.zeros(0, bool)
    c_has = ~np.isnan(d["pool_style"]).any(1) if len(d["pool_style"]) else np.zeros(0, bool)
    prev_all = np.nonzero(kinds == "preview")[0]
    cov_q = float(q_has[prev_all].mean()) if len(prev_all) else 0.0
    cov_c = float(c_has.mean()) if len(c_has) else 0.0
    variants = list(variants)
    genre_note = None
    use_genre = any(v in GENRE_VARIANTS for v in variants)
    if use_genre and min(cov_q, cov_c) < ev["min_style_coverage"]:
        genre_note = (f"genre variants skipped: style vectors cover {cov_q:.0%} of preview queries and {cov_c:.0%} of "
                      f"the pool (need {ev['min_style_coverage']:.0%})")
        variants = [v for v in variants if v not in GENRE_VARIANTS]
        use_genre = False
    # paired comparison: with genre variants every variant uses the songs/candidates that have style vectors
    ckeep = np.nonzero(c_has)[0] if use_genre else np.arange(len(d["pool_ids"]))
    prev_idx = [int(i) for i in prev_all if (q_has[i] or not use_genre)]
    loc_idx = [int(i) for i in np.nonzero(kinds == "local file")[0]]
    yes_idx = [int(i) for i in np.nonzero(kinds == "kept")[0] if (q_has[i] or not use_genre)]
    neg_idx = [int(i) for i in np.nonzero(kinds == "turned down")[0] if (q_has[i] or not use_genre)]
    nq, nc = len(d["queries"]), len(ckeep)
    base = {"liked_with_vectors": int(len(prev_all) + len(loc_idx)),
            "queries_by_kind": {"preview": len(prev_idx), "local file": len(loc_idx)},
            "verdict_queries": {"kept": len(yes_idx), "turned down": len(neg_idx)},
            "pool": nc, "pool_total": d["pool_total"], "positive_tracks": len(d["uo"]),
            "style_coverage": {"preview_queries": round(cov_q, 3), "pool": round(cov_c, 3)}, "genre_note": genre_note,
            "seeds": list(ev["seeds"]), "holdout": ev["holdout"], "recall_k": K, "topk": kt,
            "min_queries": ev["min_queries"], "objective": ev["objective"], "min_verdicts": ev["min_verdicts"],
            "auc_tie": ev["auc_tie"], "degenerate_sd": DEGENERATE_SD, "objective_used": None,
            "at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")}
    if len(prev_idx) < 5 or nc < 50:
        return {**base, "skipped": True, "chosen": None,
                "reason": f"not enough data for an eval ({len(prev_idx)} liked songs with a preview vector, "
                          f"{nc} pool candidates)"}
    rows = [d["pool_rows"][i] for i in ckeep]
    is_chart = np.array([r["source"] in set(ev["chart_sources"]) for r in rows])
    artist = [r["artist_key"] for r in rows]
    pool_ids = [d["pool_ids"][i] for i in ckeep]
    sp = d["space"]
    ref_at = {lab: j for j, lab in enumerate(sp.ref_owner or [])}
    q_self = np.array([ref_at.get(x["owner"], -1) for x in d["queries"]])
    c_self = np.array([ref_at.get(f"cand:{i}", -1) for i in pool_ids])
    owner_idx = {o: j for j, o in enumerate(d["uo"])}
    okey_arr = np.array([d["okeys"].get(o, "") for o in d["uo"]], dtype=object)
    # leave-one-out for the kept offers: each one's own rows and same-song files are left out of its positives
    ymask = np.zeros((len(yes_idx), len(d["uo"])), bool)
    for r_, i in enumerate(yes_idx):
        x = d["queries"][i]
        if x["keys"]:
            ymask[r_] = np.isin(okey_arr, list(x["keys"]))
        for o in x.get("owners", ()):
            if o in owner_idx:
                ymask[r_, owner_idx[o]] = True
    yes_w = np.array([d["queries"][i].get("weight", 1.0) for i in yes_idx], np.float64)
    # embedding-space matrices (owner level), one set per geometry
    mats, geo_extra = {}, {}
    for geo in sorted({("raw" if v in ("raw", "topk_mean", "genre_blend") else v) for v in variants if v != "genre"}):
        cen = is_centred(geo)
        P = sp.prep(d["pos"], cen)
        C = sp.prep(d["pool_m"][ckeep], cen)
        Q = sp.prep(d["q"], cen)
        b = d["bonus"] * vectors.star_unit(cfg, C, P)      # star_weight_sd: in this geometry's pool-score sd
        SC, SQ = _owner_max(C, P, b, d["starts"]), _owner_max(Q, P, b, d["starts"])
        R = None
        if is_csls(geo):
            R = sp.prep(sp.ref, cen)
            hc, hq = hubness(C, R, sp.k, c_self), hubness(Q, R, sp.k, q_self)
        else:
            hc, hq = np.zeros(nc, np.float32), np.zeros(nq, np.float32)
        mats[geo] = (SC, SQ, hc, hq)
        geo_extra[geo] = (P, R, cen, b)
    if use_genre:   # style positives = the liked preview queries' own style vectors, plus the other kept offers
        liked_pre = {d["queries"][i]["sid"] for i in prev_idx}
        gl_idx = prev_idx + [i for i in yes_idx if d["queries"][i]["sid"] not in liked_pre]
        GL = l2norm(d["q_style"][gl_idx])
        GC = l2norm(d["pool_style"][ckeep]) @ GL.T
        g_rows = prev_idx + yes_idx + neg_idx
        GQ = l2norm(d["q_style"][g_rows]) @ GL.T
        grow = {q: j for j, q in enumerate(g_rows)}
        pos_in_gl = {q: j for j, q in enumerate(gl_idx)}
        gl_keys = [d["queries"][q]["keys"] for q in gl_idx]
        gl_sid = [d["queries"][q]["sid"] for q in gl_idx]
        ygmask = np.zeros((len(yes_idx), len(gl_idx)), bool)
        for r_, i in enumerate(yes_idx):
            x = d["queries"][i]
            ygmask[r_] = [gl_sid[c] == x["sid"] or bool(gl_keys[c] & x["keys"]) for c in range(len(gl_idx))]
    nh_p = max(1, int(round(ev["holdout"] * len(prev_idx))))
    nh_l = int(round(ev["holdout"] * len(loc_idx)))
    acc = {v: dict(p=[], h=[], h_top=[], l=[], charts=[], seed_means=[], bands=[], auc=[], pct_yes=[], pct_no=[],
                   spread=[]) for v in variants}
    for seed in ev["seeds"]:
        rnd = random.Random(seed)
        held_p = rnd.sample(prev_idx, nh_p)
        held_l = rnd.sample(loc_idx, nh_l) if nh_l else []
        held = held_p + held_l
        hkeys = set().union(*(d["queries"][i]["keys"] for i in held))
        mask = np.isin(okey_arr, list(hkeys)) if hkeys else np.zeros(len(d["uo"]), bool)
        for i in held:
            j = owner_idx.get(d["queries"][i]["owner"])
            if j is not None:
                mask[j] = True
        keepc = ~mask
        ykeep = ymask[:, keepc]
        scores = {}

        def emb(geo, agg):
            SC, SQ, hc, hq = mats[geo]
            f = (lambda M: M.max(1)) if agg == "max" else (lambda M: topk_mean(M, kt))
            Y = SQ[yes_idx][:, keepc]
            Y[ykeep] = -np.inf
            return dict(pool=f(SC[:, keepc]) - hc, p=f(SQ[held_p][:, keepc]) - hq[held_p],
                        l=(f(SQ[held_l][:, keepc]) - hq[held_l]) if held_l else None,
                        y=f(Y) - hq[yes_idx], n=f(SQ[neg_idx][:, keepc]) - hq[neg_idx])
        for v in variants:
            if v in ("raw", "centred", "csls", "csls_centred"):
                scores[v] = emb(v, "max")
            elif v == "topk_mean":
                scores[v] = emb("raw", "topk")
        if use_genre:
            lkeep = np.array([not (k_ & hkeys) for k_ in gl_keys])
            for i in held_p:
                lkeep[pos_in_gl[i]] = False

            def g(idx):
                return topk_mean(GQ[[grow[i] for i in idx]][:, lkeep], kt)
            GY = GQ[[grow[i] for i in yes_idx]][:, lkeep]
            GY[ygmask[:, lkeep]] = -np.inf
            gs = dict(pool=topk_mean(GC[:, lkeep], kt), p=g(held_p), l=None, y=topk_mean(GY, kt), n=g(neg_idx))
            if "genre" in variants:
                scores["genre"] = gs
            if "genre_blend" in variants:
                r = emb("raw", "max")
                scores["genre_blend"] = {k: None if gs[k] is None else zblend(r[k], gs[k], r["pool"], gs["pool"])
                                         for k in gs}
        for v, s in scores.items():
            a = acc[v]
            pool = s["pool"]
            ps = np.sort(pool)
            pct = _pct(ps, s["p"])
            rank = nc - np.searchsorted(ps, s["p"], "right") + 1
            a["p"] += list(pct); a["h"] += list(rank <= K); a["h_top"] += list(rank <= TOP)
            a["seed_means"].append(float(pct.mean()))
            if s["l"] is not None and len(s["l"]):
                a["l"] += list(_pct(ps, s["l"]))
            a["auc"].append(auc(s["y"], yes_w, s["n"]))
            if len(s["y"]) and yes_w.sum() > 0:
                a["pct_yes"].append(float(np.average(_pct(ps, s["y"]), weights=yes_w)))
            if len(s["n"]):
                a["pct_no"].append(float(_pct(ps, s["n"]).mean()))
            sd = float(pool.std())
            head = ps[::-1][:TOP]
            a["spread"].append(float((head[0] - head[-1]) / sd) if sd > 0 else 0.0)
            top, seen = [], set()
            for j in np.argsort(-pool):
                if artist[j] in seen:
                    continue
                seen.add(artist[j]); top.append(j)
                if len(top) >= K:
                    break
            a["charts"].append(float(is_chart[top].mean()))
            a["bands"].append((float(pool[top].min()), float(pool[top].max())))
    res = {}
    for v in variants:
        a = acc[v]
        aucs = [x for x in a["auc"] if x is not None]
        yes_m = float(np.mean(a["pct_yes"])) if a["pct_yes"] else None
        no_m = float(np.mean(a["pct_no"])) if a["pct_no"] else None
        spread = float(np.mean(a["spread"]))
        res[v] = {"auc": round(float(np.mean(aucs)), 4) if aucs else None,
                  "gap": round(yes_m - no_m, 2) if yes_m is not None and no_m is not None else None,
                  "pct_yes": None if yes_m is None else round(yes_m, 2), "pct_no": None if no_m is None else round(no_m, 2),
                  f"top{TOP}_band": round(spread, 3), "degenerate": spread < DEGENERATE_SD,
                  "mean_pct": round(float(np.mean(a["p"])), 2), "seed_sd": round(float(np.std(a["seed_means"])), 2),
                  "median_pct": round(float(np.median(a["p"])), 2), f"recall_at_{K}": round(float(np.mean(a["h"])), 4),
                  f"recall_at_{TOP}": round(float(np.mean(a["h_top"])), 4),
                  "mean_pct_local_file_queries": round(float(np.mean(a["l"])), 2) if a["l"] else None,
                  f"chart_share_top{K}": round(float(np.mean(a["charts"])), 3),
                  f"top{K}_score_band": [round(min(x for x, _ in a["bands"]), 4), round(max(y for _, y in a["bands"]), 4)]}
        log.info("eval %s: %s", v, res[v])
    if "raw" in res:   # paired: the same held-out songs under every variant (se is optimistic: seeds overlap)
        r0 = np.array(acc["raw"]["p"])
        for v in variants:
            diff = np.array(acc[v]["p"]) - r0
            res[v]["paired_vs_raw"] = [round(float(diff.mean()), 2),
                                       round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 2) if len(diff) > 1 else None]
    # the sanity check keeps its old style positives (liked songs only): it leaves the judged tracks out all at once
    nl = len(prev_idx)
    judged_out = _judged_check(d, variants, mats, geo_extra, GL[:nl] if use_genre else None,
                               GC[:, :nl] if use_genre else None, kt, cfg["scoring"].get("negative_penalty")) \
        if d.get("judged") else None
    out = {**base, "held_out_per_seed": {"preview": nh_p, "local file": nh_l}, "judged_check": judged_out,
           "pool_chart_share": round(float(is_chart.mean()), 3), "variants": res, "tie_tolerance": ev["tie_tolerance"],
           "n_pos": len(yes_idx), "n_neg": len(neg_idx)}
    n_min = min(len(yes_idx), len(neg_idx))
    if len(prev_idx) < ev["min_queries"]:
        out["chosen"] = None
        out["reason"] = (f"only {len(prev_idx)} liked songs have a preview vector (need {ev['min_queries']}); "
                         "keeping the configured / default variant")
    elif ev["objective"] == "auc" and n_min >= ev["min_verdicts"] and all(x["auc"] is not None for x in res.values()):
        out["objective_used"] = "auc"
        out["chosen"] = choose_variant({v: x["auc"] for v, x in res.items()}, ev["auc_tie"],
                                       exclude={v for v, x in res.items() if x["degenerate"]})
    else:
        out["objective_used"] = "mean_pct"
        out["objective_note"] = ("eval.objective is mean_pct" if ev["objective"] == "mean_pct" else
                                 f"auc needs {ev['min_verdicts']} kept and {ev['min_verdicts']} turned-down offers "
                                 f"with a preview vector, there are {len(yes_idx)} and {len(neg_idx)}")
        out["chosen"] = choose_variant({v: x["mean_pct"] for v, x in res.items()}, ev["tie_tolerance"])
    try:
        out["liked_genres"] = liked_genres(con, cfg)
    except Exception as e:  # noqa: BLE001
        out["liked_genres"] = {"error": repr(e)}
    return out


def _hysteresis(ev: dict, current: str, margin: float) -> tuple[str, str]:
    """The weekly rule: -> (variant, why). Switch only on the AUC objective, and only when the current variant
    is degenerate or was not evaluated, or the winner's AUC is more than `margin` above the current one's."""
    if ev.get("objective_used") != "auc":
        return current, f"kept: no switch on the mean-percentile fallback ({ev.get('objective_note')})"
    win = ev["chosen"]
    if win == current:
        return current, "kept: still the best"
    cur = ev["variants"].get(current) or {}
    if cur.get("auc") is None:
        return win, f"switched: {current} was not evaluated"
    if cur.get("degenerate"):
        return win, f"switched: {current} is degenerate"
    gain = ev["variants"][win]["auc"] - cur["auc"]
    if gain > margin:
        return win, f"switched: AUC {gain:+.3f} over {current} (more than {margin})"
    return current, f"kept: {win} is AUC {gain:+.3f} over {current}, within the switch margin {margin}"


def choose(con, cfg: dict, trigger: str = "set", *, write: bool = True, data: dict | None = None) -> dict:
    """Run the eval and settle the scoring variant (ADR 0009). A pinned scoring.variant is never changed.
    trigger 'set' (the first batch, `eval --choose`): the eval's choice becomes the variant.
    trigger 'weekly' (every batch): the hysteresis rule above.
    trigger 'report' (`haystack eval`): the choice is reported and the variant stays.
    With write the run is recorded (eval_runs, status.json scoring_eval) and the variant stored; without
    (`batch --plan`, `batch --dry-run`) nothing is written and 'used' is the variant the batch would use."""
    if trigger not in ("set", "weekly", "report"):
        raise ValueError(f"unknown trigger {trigger!r}")
    ev = run(con, cfg, data=data)
    pinned = cfg["scoring"].get("variant", "auto")
    current = vectors.active_variant(con, cfg)
    used, why = current, None
    if pinned != "auto":
        why = "pinned in config"
    elif not ev.get("chosen"):
        why = f"kept: {ev.get('reason') or 'no choice'}"
    elif trigger == "set":
        used, why = ev["chosen"], "set by the eval"
    elif trigger == "weekly":
        used, why = _hysteresis(ev, current, cfg["eval"]["switch_margin"])
    else:
        why = "kept: report only"
    ev.update(pinned=pinned if pinned != "auto" else None, previous=current, used=used, switched=used != current,
              decision=why, trigger=trigger)
    log.info("scoring variant: eval chose %s by %s; using %s (%s)", ev.get("chosen"), ev.get("objective_used"), used, why)
    if write:
        if pinned == "auto" and (ev["switched"] or (trigger == "set" and ev.get("chosen"))):
            db.kv_set(con, "scoring_variant", used)
        record(con, ev)
        status.update(None, scoring_eval=ev)
    return ev


def record(con, ev: dict) -> None:
    con.execute("""INSERT INTO eval_runs(at, trigger, objective, chosen, used, n_pos, n_neg, auc_json, switched)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (db.now(), ev.get("trigger"), ev.get("objective_used"), ev.get("chosen"), ev.get("used"), ev.get("n_pos"),
                 ev.get("n_neg"), json.dumps({v: x.get("auc") for v, x in (ev.get("variants") or {}).items()}),
                 int(bool(ev.get("switched")))))


def summary(ev: dict) -> dict:
    """The batch's record of its variant choice: no track names, a few numbers."""
    used = (ev.get("variants") or {}).get(ev.get("used")) or {}
    return {k: ev.get(k) for k in ("objective_used", "chosen", "previous", "used", "switched", "decision", "n_pos",
                                    "n_neg")} | {"auc_used": used.get("auc")}


def _num(x, fmt: str) -> str:
    return "-" if x is None else format(x, fmt)


def markdown(r: dict, title: str = "Scoring variant eval") -> str:
    lines = [f"## {title} ({r.get('at', '')})", ""]
    if r.get("skipped"):
        return "\n".join(lines + [f"Skipped: {r.get('reason')}"
                                   + (f". Variant used: **{r['used']}**." if r.get("used") else ""), ""])
    K = r["recall_k"]
    qk, hk = r.get("queries_by_kind", {}), r.get("held_out_per_seed", {})
    order = " < ".join(VARIANTS)
    has_auc = any(x.get("auc") is not None for x in r["variants"].values())
    if has_auc:
        lines += [f"Offers: {r['n_pos']} kept (rated 3 stars or more, or liked with no rating; weighted by stars, "
                  f"each left out of its own positives) against {r['n_neg']} turned down (a Pass or a Dislike, or "
                  f"expired after a play or removed by hand). Each is scored like a candidate and ranked among "
                  f"{r['pool']} pool candidates. AUC is the chance a kept offer outscores a turned-down one "
                  f"(0.5 = chance). A variant whose top {TOP} pool scores span less than "
                  f"{r.get('degenerate_sd', DEGENERATE_SD)} pool-score sd is degenerate: its top picks are near-ties.",
                  ""]
        if r.get("genre_note"):
            lines += [f"Note: {r['genre_note']}.", ""]
        lines += [f"| variant | AUC, kept above turned down | gap (kept - turned down) | mean percentile kept / turned "
                  f"down | top-{TOP} spread (sd) "
                  f"| recall@{K} | recall@{TOP} |", "|---|---|---|---|---|---|---|"]
        for v, x in r["variants"].items():
            mark = " **(chosen)**" if v == r.get("chosen") else ""
            spread = f"{x[f'top{TOP}_band']:.2f}" + (" degenerate" if x.get("degenerate") else "")
            lines.append(f"| {v}{mark} | {_num(x.get('auc'), '.3f')} | {_num(x.get('gap'), '+.1f')} | "
                         f"{_num(x.get('pct_yes'), '.1f')} / {_num(x.get('pct_no'), '.1f')} | {spread} | "
                         f"{x[f'recall_at_{K}']:.3f} | {x.get(f'recall_at_{TOP}', 0):.3f} |")
        lines.append("")
    choice = (f"Chosen: **{r['chosen']}**" if r.get("chosen") else f"No choice: {r.get('reason')}")
    used = ""
    if r.get("used"):
        used = f"; used for the batch: **{r['used']}**" + (" (pinned in config)" if r.get("pinned") else "")
    if r.get("objective_used") == "auc":
        rule = (f"Choice: the best AUC among the variants that are not degenerate; within {r['auc_tie']} is a tie and "
                f"the simpler variant wins ({order}).")
    elif r.get("objective_used") == "mean_pct" and r.get("objective_note"):
        rule = (f"Choice: **the mean-percentile fallback** ({r['objective_note']}): the best mean percentile on "
                f"held-out liked songs; within {r['tie_tolerance']} points is a tie and the simpler variant wins "
                f"({order}).")
    else:
        rule = (f"Choice: best mean percentile on preview-vector queries; within {r['tie_tolerance']} points is a "
                f"tie and the simpler variant wins ({order}).")
    lines += [f"{rule} {choice}{used}."]
    if r.get("decision"):
        lines += ["", f"Variant: {r.get('previous')} -> **{r['used']}** ({r['decision']})."]
    lines += ["", f"### Held-out liked songs{' (diagnostic)' if has_auc else ''}", "",
              f"Queries: {qk.get('preview', 0)} liked songs with a Deezer preview vector (the same audio domain as the "
              f"candidates; {hk.get('preview', 0)} held out per seed, {r['holdout']:.0%}, seeds {r['seeds']}). Each "
              f"held-out song is ranked among {r['pool']} pool candidates (of {r['pool_total']}; liked and library "
              f"songs removed). Positives: {r['positive_tracks']} tracks (genre variants: the liked songs' style "
              f"vectors). Chart sources are {r['pool_chart_share']:.0%} of the pool. k = {r.get('topk', 10)} for "
              f"topk_mean and genre. The last column uses {qk.get('local file', 0)} liked songs that only have a "
              f"local-file vector; it is a diagnostic only.", ""]
    if r.get("genre_note") and not has_auc:
        lines += [f"Note: {r['genre_note']}.", ""]
    lines += [f"| variant | mean percentile (50 = chance) | seed sd | vs raw (paired, ±se) | recall@{K} "
              f"| chart share top {K} | top-{K} score band | local-file queries (diagnostic) |",
              "|---|---|---|---|---|---|---|---|"]
    for v, x in r["variants"].items():
        mark = " **(chosen)**" if v == r.get("chosen") else ""
        band = x[f"top{K}_score_band"]
        pv = x.get("paired_vs_raw") or [0.0, None]
        lf = x.get("mean_pct_local_file_queries")
        lines.append(f"| {v}{mark} | {x['mean_pct']:.2f} | {x['seed_sd']:.2f} | {pv[0]:+.2f}"
                     f"{f' ±{pv[1]:.2f}' if pv[1] is not None else ''} | {x[f'recall_at_{K}']:.3f} | "
                     f"{x[f'chart_share_top{K}']:.0%} | {band[0]:.4f}-{band[1]:.4f} | "
                     f"{'-' if lf is None else f'{lf:.2f}'} |")
    lines.append("")
    jc = r.get("judged_check")
    if jc and jc.get("variants"):
        vs = list(jc["variants"])
        lines += ["### Sanity check: the listener's verdicts on the dry-run picks (percentile among the pool; not used for "
                  "the choice)", "", "| track | verdict | " + " | ".join(vs) + " |", "|---|---|" + "---|" * len(vs)]
        for i, t in enumerate(jc["tracks"]):
            verdict = t["verdict"] + (" (strong)" if t["weight"] > 1 else " (mild)" if t["weight"] < 1 else "")
            lines.append(f"| {t['label']} | {verdict} | " + " | ".join(f"{jc['variants'][v]['pct'][i]:.1f}" for v in vs) + " |")
        lines.append("| **mean yes / mean no** | | " + " | ".join(
            f"{jc['variants'][v]['mean_yes']} / {jc['variants'][v]['mean_no']}" for v in vs) + " |")
        pv = [v for v in vs if "pct_pen" in jc["variants"][v]]
        if pv and jc.get("penalty"):
            pn = jc["penalty"]
            lines += ["", f"With the negative penalty (w = {pn['w']}, margin = the pool's {pn['margin_quantile']:.0%} "
                      f"similarity to each of {pn['negatives']} rejected tracks; a rejected track is not penalised by "
                      "its own verdict):", "", "| track | verdict | " + " | ".join(pv) + " |", "|---|---|" + "---|" * len(pv)]
            for i, t in enumerate(jc["tracks"]):
                verdict = t["verdict"] + (" (strong)" if t["weight"] > 1 else " (mild)" if t["weight"] < 1 else "")
                lines.append(f"| {t['label']} | {verdict} | " + " | ".join(
                    f"{jc['variants'][v]['pct_pen'][i]:.1f}" for v in pv) + " |")
            lines.append("| **mean yes / mean no** | | " + " | ".join(
                f"{jc['variants'][v]['mean_yes_pen']} / {jc['variants'][v]['mean_no_pen']}" for v in pv) + " |")
        lines.append("")
    g = r.get("liked_genres") or {}
    if g.get("liked_with_styles"):
        lines += [f"### Genre makeup of the liked songs ({g['liked_with_styles']} with a style vector, genre_discogs400)", "",
                  "Parent genres (share of activation mass): "
                  + ", ".join(f"{n} {v:.0%}" for n, v in g["parent_genre_share"][:10]), "",
                  "| style | mean activation | | style (top-1) | songs |", "|---|---|---|---|---|"]
        a, b = g["top_styles_mean_activation"], g["top1_style_counts"]
        for i in range(max(len(a), len(b))):
            x = a[i] if i < len(a) else ["", ""]
            y = b[i] if i < len(b) else ["", ""]
            lines.append(f"| {x[0]} | {x[1]} | | {y[0]} | {y[1]} |")
        lines.append("")
    return "\n".join(lines)
