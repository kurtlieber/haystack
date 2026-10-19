"""Vector storage + taste (positives) assembly + candidate scoring (token-free)."""
from __future__ import annotations

import json
import logging
import os
import warnings

import numpy as np

from . import config, db, embed, verdicts
from .scoring import (VARIANTS, Space, is_centred, is_csls, l2norm, nearest_positive, negative_excess,
                      negative_penalty, score_variant, style_topk, topk_owner_mean, zblend)

log = logging.getLogger("haystack_music.vectors")


def put_deezer_vector(con, deezer_id: int, e512, e1280, n: int, secs: float, style=None) -> None:
    st = None if style is None else np.asarray(style, np.float16).tobytes()
    con.execute("""INSERT INTO deezer_vectors(deezer_id, e512, e1280, n_patches, seconds, created_at, style400)
                   VALUES(?, ?, ?, ?, ?, ?, ?) ON CONFLICT(deezer_id) DO UPDATE SET e512=excluded.e512,
                   e1280=excluded.e1280, n_patches=excluded.n_patches, seconds=excluded.seconds,
                   style400=COALESCE(excluded.style400, deezer_vectors.style400)""",
                (int(deezer_id), np.asarray(e512, np.float32).tobytes(), np.asarray(e1280, np.float16).tobytes(),
                 int(n), float(secs), db.now(), st))


def put_style(con, deezer_id: int, style) -> None:
    con.execute("UPDATE deezer_vectors SET style400=? WHERE deezer_id=?",
                (np.asarray(style, np.float16).tobytes(), int(deezer_id)))


def get_styles(con, ids) -> dict[int, np.ndarray]:
    out = {}
    ids = [int(i) for i in ids if i is not None]
    for s in range(0, len(ids), 500):
        chunk = ids[s:s + 500]
        q = (f"SELECT deezer_id, style400 FROM deezer_vectors WHERE style400 IS NOT NULL AND deezer_id IN "
             f"({','.join('?' * len(chunk))})")
        for r in con.execute(q, chunk):
            out[r[0]] = np.frombuffer(r[1], dtype=np.float16).astype(np.float32)
    return out


def get_deezer_vectors(con, ids) -> dict[int, np.ndarray]:
    out = {}
    ids = [int(i) for i in ids if i is not None]
    for s in range(0, len(ids), 500):
        chunk = ids[s:s + 500]
        q = f"SELECT deezer_id, e512 FROM deezer_vectors WHERE deezer_id IN ({','.join('?' * len(chunk))})"
        for r in con.execute(q, chunk):
            out[r[0]] = np.frombuffer(r[1], dtype=np.float32)
    return out


def _library_windows(con) -> tuple[list[int], list[np.ndarray]]:
    """Per present local track: its window matrix. Uses a consolidated cache keyed by the set of hashes."""
    rows = con.execute("""SELECT t.id, t.content_hash, e.file FROM local_tracks t JOIN local_embeddings e
                          ON e.content_hash = t.content_hash AND e.status = 'ok' WHERE t.present = 1""").fetchall()
    cache_p = config.EMB_DIR / "library_windows.cache.npz"
    by_hash: dict[str, np.ndarray] = {}
    hashes = sorted({r[1] for r in rows})
    if cache_p.exists():
        try:
            with np.load(cache_p, allow_pickle=False) as z:
                ch, counts, mat = list(z["hashes"]), z["counts"], z["windows"]
            off = np.concatenate([[0], np.cumsum(counts)])
            by_hash = {h: mat[off[i]:off[i + 1]] for i, h in enumerate(ch)}
        except Exception:  # noqa: BLE001
            by_hash = {}
    missing = [r for r in rows if r[1] not in by_hash]
    for r in missing:
        try:
            by_hash[r[1]] = embed.load_windows(r[2])
        except Exception as e:  # noqa: BLE001
            log.warning("cannot load %s: %r", r[2], e)
    if missing or set(by_hash) != set(hashes):
        keep = [h for h in hashes if h in by_hash]
        if keep:
            tmp = cache_p.with_name("library_windows.tmp.npz")
            np.savez(tmp, hashes=np.array(keep), counts=np.array([len(by_hash[h]) for h in keep]),
                     windows=np.vstack([by_hash[h] for h in keep]).astype(np.float32))
            os.replace(tmp, cache_p)
    ids, mats = [], []
    for r in rows:
        if r[1] in by_hash:
            ids.append(r[0]); mats.append(by_hash[r[1]])
    return ids, mats


def track_vectors(con, sids=None) -> dict[str, dict]:
    """spotify id -> {deezer_id, candidate_id, label} for tracks with a stored preview vector: candidates, liked
    songs and on-demand mappings (spotify_deezer), in that order of preference."""
    want = None if sids is None else set(sids)
    out: dict[str, dict] = {}

    def put(sid, did, cid, label):
        if sid and did is not None and sid not in out and (want is None or sid in want):
            out[sid] = {"deezer_id": int(did), "candidate_id": cid, "label": label}

    for r in con.execute("SELECT id, spotify_id, deezer_id, artist, title FROM candidates "
                         "WHERE spotify_id IS NOT NULL AND vector_status='ok'"):
        put(r["spotify_id"], r["deezer_id"], r["id"], f"{r['artist']} - {r['title']}")
    for r in con.execute("""SELECT m.spotify_id, m.deezer_id, s.name, s.artists_json FROM liked_map m
                            LEFT JOIN spotify_tracks s ON s.spotify_id=m.spotify_id WHERE m.vector_status='ok'
                            UNION ALL
                            SELECT d.spotify_id, d.deezer_id, s.name, s.artists_json FROM spotify_deezer d
                            LEFT JOIN spotify_tracks s ON s.spotify_id=d.spotify_id WHERE d.status='ok'"""):
        arts = json.loads(r[3] or "[]")
        put(r[0], r[1], None, f"{arts[0] if arts else '?'} - {r[2] or '?'}")
    return out


def positives(con, cfg: dict):
    """-> (matrix [m,512], owner labels [m], bonus [m]).
    Positives = the local library (one row per window) + Liked Songs + liked discoveries + every Rated track
    (ADR 0002). A liked track uses its local vector only if liked_map.use_local (confident, same version, <=2 s),
    otherwise its Deezer-preview vector. The bonus comes from the verdict (stars) or, for an Unrated like, counts
    as 3 stars; library tracks the listener hasn't liked get bonus_library (0 with star_weight_sd). A track whose
    current verdict is a Pass or a Dislike is never a positive. One owner per track. With star_weight_sd the
    bonus is in pool-score sd (verdicts.stars_in_sd): the scorer resolves it (star_unit)."""
    return positives_with_stars(con, cfg)[:3]


def positives_with_stars(con, cfg: dict):
    """positives() plus each row's stars [m] (NaN: a library file the listener hasn't liked or rated)."""
    sc = cfg["scoring"]
    in_sd = verdicts.stars_in_sd(cfg)
    cur = verdicts.current(con)
    liked = {r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL")}
    disc_liked = {r[0] for r in con.execute("SELECT spotify_id FROM offers WHERE outcome='liked'")}

    def rejected(sid):
        v = cur.get(sid)
        return v is not None and v.kind != "rating"

    def bonus_of(sid, is_liked):
        return verdicts.positive_bonus(cfg, cur.get(sid), liked=is_liked)

    def stars_of(sid, is_liked):
        s = verdicts.positive_stars(cfg, cur.get(sid), liked=is_liked)
        return np.nan if s is None else float(s)

    local_like = {}
    for r in con.execute("""SELECT m.spotify_id, m.local_id FROM liked_map m JOIN likes l ON l.spotify_id=m.spotify_id
                            WHERE l.removed_at IS NULL AND m.use_local=1 AND m.local_id IS NOT NULL"""):
        local_like[r[1]] = r[0]
    tagged = {r[0]: r[1] for r in con.execute("SELECT id, spotify_id FROM local_tracks WHERE present=1 AND spotify_id IS NOT NULL")}
    mats, owners, bonus, stars = [], [], [], []
    covered: set[str] = set()
    ids, wins = _library_windows(con)
    lib_bonus = 0.0 if in_sd else sc["bonus_library"]
    for i, w in zip(ids, wins):
        sid = local_like.get(i) or tagged.get(i)
        if sid and rejected(sid):
            continue
        is_liked = sid in local_like.values() or sid in liked or sid in disc_liked
        b = bonus_of(sid, is_liked) if sid else None
        if sid in local_like.values():
            covered.add(sid)
        mats.append(w); owners += [f"local:{i}"] * len(w); bonus += [lib_bonus if b is None else b] * len(w)
        stars += [stars_of(sid, is_liked) if sid else np.nan] * len(w)
    # liked tracks represented by preview vectors
    rows = con.execute("""SELECT m.spotify_id, m.deezer_id FROM liked_map m JOIN likes l ON l.spotify_id=m.spotify_id
                          WHERE l.removed_at IS NULL AND m.use_local=0 AND m.vector_status='ok'""").fetchall()
    vec = get_deezer_vectors(con, [r[1] for r in rows])
    for sid, did in rows:
        if did in vec and sid not in covered and not rejected(sid):
            mats.append(vec[did][None, :]); owners.append(f"like:{sid}"); bonus.append(bonus_of(sid, True))
            stars.append(stars_of(sid, True)); covered.add(sid)
    # liked discoveries: the candidate's own preview vector
    rows = con.execute("""SELECT o.spotify_id, c.deezer_id FROM offers o JOIN candidates c ON c.id=o.candidate_id
                          WHERE o.outcome='liked'""").fetchall()
    vec = get_deezer_vectors(con, [r[1] for r in rows])
    for sid, did in rows:
        if did in vec and sid not in covered and not rejected(sid):
            mats.append(vec[did][None, :]); owners.append(f"discovery:{sid}"); bonus.append(bonus_of(sid, True))
            stars.append(stars_of(sid, True)); covered.add(sid)
    # every other Rated track: its preview vector
    rated = [sid for sid, v in cur.items() if v.kind == "rating" and sid not in covered]
    tv = track_vectors(con, rated)
    vec = get_deezer_vectors(con, [x["deezer_id"] for x in tv.values()])
    for sid in rated:
        t = tv.get(sid)
        if t and t["deezer_id"] in vec:
            mats.append(vec[t["deezer_id"]][None, :])
            owners.append(f"manual:{t['candidate_id']}" if t["candidate_id"] is not None else f"rated:{sid}")
            bonus.append(bonus_of(sid, True)); stars.append(stars_of(sid, True))
    if not mats:
        return np.zeros((0, 512), np.float32), [], np.zeros(0, np.float32), np.zeros(0, np.float32)
    return (np.vstack(mats).astype(np.float32), owners, np.asarray(bonus, np.float32),
            np.asarray(stars, np.float32))


def pool_vectors(con, states=("pool",)) -> tuple[list[int], np.ndarray, list]:
    """-> (candidate ids, preview matrix, candidate rows) for embedded candidates in `states`."""
    q = (f"SELECT * FROM candidates WHERE vector_status='ok' AND state IN ({','.join('?' * len(states))}) "
         "ORDER BY id")
    rows = con.execute(q, list(states)).fetchall()
    vec = get_deezer_vectors(con, [r["deezer_id"] for r in rows])
    rows = [r for r in rows if r["deezer_id"] in vec]
    m = np.vstack([vec[r["deezer_id"]] for r in rows]).astype(np.float32) if rows else np.zeros((0, 512), np.float32)
    return [r["id"] for r in rows], m, rows


def build_space(cfg: dict, lib: tuple[list[int], list[np.ndarray]], pool: tuple[list[int], np.ndarray]) -> Space:
    """Centring mean + fixed CSLS reference set from library tracks (one vector each) and the candidate pool.
    Library tracks count once each (mean of their windows for mu, one seeded-random window for the reference
    set) so the 9 windows per file don't outweigh the single-preview candidates."""
    sc = cfg["scoring"]
    lib_ids, mats = lib
    cand_ids, cm = pool
    rng = np.random.default_rng(sc["ref_seed"])
    parts, labels, ref_rows = [], [], []
    if lib_ids:
        parts.append(l2norm(np.vstack([l2norm(w).mean(0) for w in mats])))
        ref_rows += [w[rng.integers(len(w))] for w in mats]
        labels += [f"local:{i}" for i in lib_ids]
    if len(cand_ids):
        parts.append(l2norm(cm))
        ref_rows += list(cm)
        labels += [f"cand:{i}" for i in cand_ids]
    if not parts:
        return Space(k=sc["csls_k"])
    mu = np.vstack(parts).mean(0).astype(np.float32)
    ref = np.vstack(ref_rows).astype(np.float32)
    if len(ref) > sc["ref_size"]:
        pick = np.sort(rng.choice(len(ref), sc["ref_size"], replace=False))
        ref, labels = ref[pick], [labels[i] for i in pick]
    return Space(mu=mu, ref=ref, ref_owner=labels, k=sc["csls_k"])


def liked_styles(con, cfg: dict | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Style vectors of the positives for the genre variants, with their bonus: liked songs' previews (every
    liked song with a preview, locally matched or not), liked discoveries and Rated tracks; Passes and Dislikes
    are left out. The library has no style vectors (that would need a second full backbone pass over every
    file). -> (labels, styles [n,400], bonus [n])."""
    cfg = cfg or config.load()
    cur = verdicts.current(con)
    rows = con.execute("""SELECT m.spotify_id, m.deezer_id FROM liked_map m JOIN likes l ON l.spotify_id=m.spotify_id
                          WHERE l.removed_at IS NULL AND m.vector_status='ok'
                          UNION SELECT o.spotify_id, c.deezer_id FROM offers o JOIN candidates c ON c.id=o.candidate_id
                          WHERE o.outcome='liked'""").fetchall()
    seen = {r[0] for r in rows}
    rated = [sid for sid, v in cur.items() if v.kind == "rating" and sid not in seen]
    rows = [(r[0], r[1]) for r in rows] + [(sid, t["deezer_id"]) for sid, t in track_vectors(con, rated).items()]
    st = get_styles(con, [r[1] for r in rows])
    labels, mats, bonus, done = [], [], [], set()
    for sid, did in rows:
        v = cur.get(sid)
        b = verdicts.positive_bonus(cfg, v, liked=True)
        if did in st and sid not in done and b is not None:
            done.add(sid); labels.append(sid); mats.append(st[did]); bonus.append(b)
    if not mats:
        return [], np.zeros((0, 400), np.float32), np.zeros(0, np.float32)
    return labels, np.vstack(mats), np.asarray(bonus, np.float32)


def active_variant(con, cfg: dict) -> str:
    """Pinned scoring.variant, else (auto) the variant the eval chose (kv scoring_variant), else raw."""
    v = cfg["scoring"].get("variant", "auto")
    if v == "auto":
        v = db.kv_get(con, "scoring_variant") or "raw"
    if v not in VARIANTS:
        raise ValueError(f"scoring.variant {v!r} is not one of auto, {', '.join(VARIANTS)}")
    return v


def negatives(con, cfg: dict) -> list[dict]:
    """Tracks the listener rejected, with their strength: Passes and Dislikes on any track (scoring.verdicts),
    discovery offers that expired after being played without a like (outcome 'negative') and offers taken off
    Discoveries by hand (outcome 'manual_removed') at the configured weights. One entry per track (the strongest
    wins); only tracks with a preview vector can be compared, so others are dropped."""
    npc = cfg["scoring"]["negative_penalty"]
    by: dict = {}

    def add(key, cid, did, label, strength, kind):
        if did is None or not strength:
            return
        if key not in by or strength > by[key]["strength"]:
            by[key] = {"candidate_id": cid, "deezer_id": did, "label": label, "strength": float(strength), "kind": kind}

    cur = verdicts.current(con)
    rejected = {sid: v for sid, v in cur.items() if v.kind != "rating"}
    for sid, t in track_vectors(con, list(rejected)).items():
        v = rejected[sid]
        key = t["candidate_id"] if t["candidate_id"] is not None else f"sid:{sid}"
        add(key, t["candidate_id"], t["deezer_id"], t["label"], verdicts.negative_strength(cfg, v), v.kind)
    for r in con.execute("""SELECT o.outcome, c.id, c.deezer_id, c.artist, c.title FROM offers o
                            JOIN candidates c ON c.id=o.candidate_id WHERE o.outcome IN ('negative','manual_removed')"""):
        w = npc["weight_expired"] if r[0] == "negative" else npc["weight_removed"]
        add(r[1], r[1], r[2], f"{r[3]} - {r[4]}", w, "expired_played" if r[0] == "negative" else "removed")
    out = list(by.values())
    vec = get_deezer_vectors(con, [x["deezer_id"] for x in out])
    st = get_styles(con, [x["deezer_id"] for x in out])
    res = []
    for x in out:
        if x["deezer_id"] in vec:
            res.append({**x, "vec": vec[x["deezer_id"]], "style": st.get(x["deezer_id"])})
    return res


def penalty_excess(variant: str, negs: list[dict], cand_vec: np.ndarray, cand_style: np.ndarray | None,
                   ref_vec: np.ndarray, ref_style: np.ndarray | None, space: Space | None, q: float) -> np.ndarray:
    """[n, m] 0..1 excess similarity of candidates to each negative, measured in the variant's own space:
    the 512-d preview embedding (centred for the centred variants) for the embedding variants, the 400-style
    vector for genre, and the average of the two for genre_blend. Margins come from the reference pool.
    Rows with no style vector (NaN) count only on the embedding side; negatives without one likewise."""
    m = len(negs)
    n = len(cand_vec)
    if not m or not n:
        return np.zeros((n, m), np.float32)
    parts = []
    if variant != "genre":
        cen = is_centred(variant) and space is not None and space.mu is not None
        prep = (lambda x: space.prep(x, True)) if cen else l2norm
        N = prep(np.vstack([x["vec"] for x in negs]))
        parts.append(negative_excess(prep(cand_vec) @ N.T, prep(ref_vec) @ N.T, q))
    if variant in ("genre", "genre_blend") and cand_style is not None:
        has_n = np.array([x["style"] is not None for x in negs])
        E = np.full((n, m), np.nan, np.float32)
        if has_n.any():
            N = l2norm(np.vstack([x["style"] for x, h in zip(negs, has_n) if h]))
            ok_c = ~np.isnan(cand_style).any(1)
            ok_r = ~np.isnan(ref_style).any(1)
            sub = np.full((n, int(has_n.sum())), np.nan, np.float32)
            if ok_c.any() and ok_r.any():
                sub[ok_c] = negative_excess(l2norm(cand_style[ok_c]) @ N.T, l2norm(ref_style[ok_r]) @ N.T, q)
            E[:, has_n] = sub
        parts.append(E)
    if not parts:
        return np.zeros((n, m), np.float32)
    with warnings.catch_warnings():               # all-NaN columns (no style on either side) -> 0
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(np.stack(parts), axis=0)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def star_unit(cfg: dict, cands: np.ndarray, pos: np.ndarray) -> float:
    """What one unit of a star_weight_sd bonus is worth in this space: the sd over the scored rows of their
    nearest-positive similarity with no bonus (both sides already prepared). 1.0 with the fixed star_bonus."""
    if not verdicts.stars_in_sd(cfg) or not len(cands) or not len(pos):
        return 1.0
    return float(np.std(nearest_positive(cands, pos)[0]))


def compute_scores(con, cfg: dict, variant: str, only_unscored: bool = False) -> dict | None:
    """Score the embedded pool under `variant` without writing anything.
    -> dict(rows, score (penalised), base (unpenalised), raw, idx, owners, penalty, neg_idx, negs, n_pos,
    unit), or None if there are no positives / candidates. The negative penalty (scoring.negative_penalty) is
    applied after the variant's score: final = score - w * sd(score over the pool) * max_j(strength_j * excess_j).
    With star_weight_sd the positives' bonus is multiplied by star_unit in the variant's space (unit)."""
    pos, owners, bonus = positives(con, cfg)
    if not len(pos):
        return {"rows": [], "n_pos": 0}
    q = """SELECT c.id, c.deezer_id FROM candidates c WHERE c.state='pool' AND c.vector_status='ok'"""
    if only_unscored:
        q += " AND c.score IS NULL"
    rows = con.execute(q).fetchall()
    vec = get_deezer_vectors(con, [r[1] for r in rows])
    rows = [r for r in rows if r[1] in vec]
    if not rows:
        return {"rows": [], "n_pos": len(pos)}
    m = np.vstack([vec[r[1]] for r in rows])
    k = cfg["scoring"]["topk"]
    space = None
    cst_m = None
    if variant in ("raw", "centred", "csls", "csls_centred"):
        self_idx = None
        if variant != "raw":
            pool_ids, pool_m, _ = pool_vectors(con)
            space = build_space(cfg, _library_windows(con), (pool_ids, pool_m))
            if is_csls(variant):
                where = {lab: j for j, lab in enumerate(space.ref_owner)}
                self_idx = np.array([where.get(f"cand:{r[0]}", -1) for r in rows])
        cen = is_centred(variant)
        sp_ = space or Space()
        unit = star_unit(cfg, sp_.prep(m, cen), sp_.prep(pos, cen))
        s, raw, idx = score_variant(variant, m, pos, bonus * unit, space, self_idx)
    else:
        unit = star_unit(cfg, l2norm(m), l2norm(pos))
        s, raw, idx = nearest_positive(m, pos, bonus * unit)   # nearest row is still recorded for review
        if variant == "topk_mean":
            starts = np.array([i for i in range(len(owners)) if i == 0 or owners[i] != owners[i - 1]])
            s = topk_owner_mean(m, pos, bonus * unit, starts, k)
        else:                                                   # genre / genre_blend
            _, ls, lb = liked_styles(con, cfg)
            cst = get_styles(con, [r[1] for r in rows])
            has = np.array([r[1] in cst for r in rows])
            if not len(ls) or not has.any():
                raise RuntimeError(f"variant {variant} needs style vectors (liked: {len(ls)}, pool: {int(has.sum())})")
            cst_m = np.full((len(rows), 400), np.nan, np.float32)
            cst_m[has] = np.vstack([cst[r[1]] for r, h in zip(rows, has) if h])
            g = np.full(len(rows), np.nan, np.float32)
            g_unit = float(np.std(style_topk(cst_m[has], ls, k))) if verdicts.stars_in_sd(cfg) else 1.0
            g[has] = style_topk(cst_m[has], ls, k, lb * g_unit)
            if variant == "genre":
                s = g
            else:
                s = np.full(len(rows), np.nan, np.float32)
                s[has] = zblend(raw[has], g[has])
            log.info("%s: %d of %d pool candidates have style vectors (the rest stay unscored)",
                     variant, int(has.sum()), len(rows))
    s = np.asarray(s, np.float32)
    npc = cfg["scoring"]["negative_penalty"]
    negs = negatives(con, cfg) if npc.get("w") else []
    pen = np.zeros(len(rows), np.float32)
    nidx = np.full(len(rows), -1, np.int64)
    if negs:
        E = penalty_excess(variant, negs, m, cst_m, m, cst_m, space, npc["margin_quantile"])
        scale = float(np.nanstd(s)) if np.isfinite(s).any() else 0.0
        pen, nidx = negative_penalty(E, [x["strength"] for x in negs], scale, npc["w"])
    return {"rows": rows, "score": s - pen, "base": s, "raw": raw, "idx": idx, "owners": owners, "penalty": pen,
            "neg_idx": nidx, "negs": negs, "n_pos": len(pos), "unit": unit}


def score_candidates(con, cfg: dict, only_unscored: bool = False, variant: str | None = None,
                     result: dict | None = None) -> dict:
    """Score the pool and store the scores. result: compute_scores' output for this variant, when the caller
    already has it (the batch re-ranks from the same scores)."""
    variant = variant or active_variant(con, cfg)
    if only_unscored and db.kv_get(con, "scored_variant") != variant:
        only_unscored = False      # scores from another variant are not comparable: rescore everything
    r = result if result is not None else compute_scores(con, cfg, variant, only_unscored)
    rows = r["rows"]
    if not rows:
        return {"scored": 0, "positives": r["n_pos"], "variant": variant}
    s, raw, idx, owners, pen, nidx, negs = r["score"], r["raw"], r["idx"], r["owners"], r["penalty"], r["neg_idx"], r["negs"]
    ts = db.now()
    con.execute("BEGIN")
    for (cid, _), sc_, rc, i, p, j in zip(rows, s, raw, idx, pen, nidx):
        con.execute("UPDATE candidates SET score=?, raw_cos=?, nearest=?, scored_at=?, neg_penalty=?, nearest_negative=? "
                    "WHERE id=?", (None if np.isnan(sc_) else float(sc_), float(rc), owners[int(i)], ts,
                                   float(p) if p > 0 else None, negs[int(j)]["label"] if j >= 0 else None, cid))
    con.execute("COMMIT")
    db.kv_set(con, "scored_variant", variant)
    npc = cfg["scoring"]["negative_penalty"]
    return {"scored": len(rows), "positives": r["n_pos"], "positive_tracks": len(set(owners)), "variant": variant,
            "score_p10_p50_p90": [round(float(x), 4) for x in np.nanpercentile(s, [10, 50, 90])],
            "negative_penalty": {"w": npc.get("w", 0), "margin_quantile": npc.get("margin_quantile"),
                                 "negatives": len(negs), "penalised": int((pen > 0).sum()),
                                 "max": round(float(pen.max()), 4) if len(pen) else 0.0}}
