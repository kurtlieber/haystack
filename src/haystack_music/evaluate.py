"""Offline eval of the scoring variants (token-free; no Spotify calls, no writes except status/kv by callers).

Positives are built exactly as for scoring (library windows + liked songs' vectors). For each seed a random
share of the liked songs that have a vector is held out: their rows (and any library file of the same song) are
removed from the positives, and each held-out song is ranked against the harvested candidate pool, which
stands in for tracks nobody chose for the listener.

A held-out song must be represented exactly like a candidate: its Deezer 30 s preview vector. Liked songs
that only have a local-file vector are reported separately as a diagnostic and never drive the choice: a
local-file window is closer to the (local-file) library positives than any Deezer preview can be, and
centring/CSLS amplify that audio-domain gap, so those queries reward domain rather than taste.

Metrics per variant (averaged over seeds):
  mean percentile  share of pool candidates the held-out song outscores (100 = top, 50 = chance)
  recall@K         share of held-out songs that would rank in the pool's top K
  chart share      share of the pool's top K (one per artist, like a batch) from Deezer chart sources"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime

import numpy as np

from . import vectors
from . import embed
from .scoring import (EMBED_VARIANTS, GENRE_VARIANTS, VARIANTS, choose_variant, hubness, is_centred, is_csls, l2norm,
                      negative_penalty, owner_max, topk_mean, zblend)
from .textnorm import song_key

log = logging.getLogger("haystack_music.evaluate")


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
    try:
        from . import feedback
        jrows = feedback.judged(con)
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


def liked_genres(con, top: int = 15) -> dict:
    """Genre makeup of the liked songs that have a style vector: mean activation per style, top-1 style counts,
    and parent-genre shares (activation mass per parent genre)."""
    _, st, _ = vectors.liked_styles(con)
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
            P, R, cen = geo_extra[geo]
            Jp = sp.prep(jv, cen)
            SJ = _owner_max(Jp, P, d["bonus"], d["starts"])
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


def run(con, cfg: dict, data: dict | None = None, variants=VARIANTS) -> dict:
    ev = cfg["eval"]
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
    nq, nc = len(d["queries"]), len(ckeep)
    base = {"liked_with_vectors": nq, "queries_by_kind": {"preview": len(prev_idx), "local file": len(loc_idx)},
            "pool": nc, "pool_total": d["pool_total"], "positive_tracks": len(d["uo"]),
            "style_coverage": {"preview_queries": round(cov_q, 3), "pool": round(cov_c, 3)}, "genre_note": genre_note,
            "seeds": list(ev["seeds"]), "holdout": ev["holdout"], "recall_k": K, "topk": kt,
            "min_queries": ev["min_queries"], "at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")}
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
    # embedding-space matrices (owner level), one set per geometry
    mats, geo_extra = {}, {}
    for geo in sorted({("raw" if v in ("raw", "topk_mean", "genre_blend") else v) for v in variants if v != "genre"}):
        cen = is_centred(geo)
        P = sp.prep(d["pos"], cen)
        C = sp.prep(d["pool_m"][ckeep], cen)
        Q = sp.prep(d["q"], cen)
        SC, SQ = _owner_max(C, P, d["bonus"], d["starts"]), _owner_max(Q, P, d["bonus"], d["starts"])
        R = None
        if is_csls(geo):
            R = sp.prep(sp.ref, cen)
            hc, hq = hubness(C, R, sp.k, c_self), hubness(Q, R, sp.k, q_self)
        else:
            hc, hq = np.zeros(nc, np.float32), np.zeros(nq, np.float32)
        mats[geo] = (SC, SQ, hc, hq)
        geo_extra[geo] = (P, R, cen)
    if use_genre:   # style positives = the preview queries' own style vectors (liked songs)
        GL = l2norm(d["q_style"][prev_idx])
        GC = l2norm(d["pool_style"][ckeep]) @ GL.T
        GQ = l2norm(d["q_style"][prev_idx]) @ GL.T          # row j = query prev_idx[j]
        pos_in_gl = {q: j for j, q in enumerate(prev_idx)}
        gl_keys = [d["queries"][q]["keys"] for q in prev_idx]
    nh_p = max(1, int(round(ev["holdout"] * len(prev_idx))))
    nh_l = int(round(ev["holdout"] * len(loc_idx)))
    acc = {v: dict(p=[], h=[], l=[], charts=[], seed_means=[], bands=[]) for v in variants}
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
        scores = {}

        def emb(geo, agg):
            SC, SQ, hc, hq = mats[geo]
            f = (lambda M: M.max(1)) if agg == "max" else (lambda M: topk_mean(M, kt))
            pool = f(SC[:, keepc]) - hc
            qp = f(SQ[held_p][:, keepc]) - hq[held_p]
            ql = (f(SQ[held_l][:, keepc]) - hq[held_l]) if held_l else np.zeros(0)
            return pool, qp, ql
        for v in variants:
            if v in ("raw", "centred", "csls", "csls_centred"):
                scores[v] = emb(v, "max")
            elif v == "topk_mean":
                scores[v] = emb("raw", "topk")
        if use_genre:
            lkeep = np.array([not (k_ & hkeys) for k_ in gl_keys])
            for i in held_p:
                lkeep[pos_in_gl[i]] = False
            g_pool = topk_mean(GC[:, lkeep], kt)
            g_q = topk_mean(GQ[[pos_in_gl[i] for i in held_p]][:, lkeep], kt)
            if "genre" in variants:
                scores["genre"] = (g_pool, g_q, None)
            if "genre_blend" in variants:
                r_pool, r_q, _ = emb("raw", "max")
                scores["genre_blend"] = (zblend(r_pool, g_pool), zblend(r_q, g_q, r_pool, g_pool), None)
        for v, (pool, qp, ql) in scores.items():
            a = acc[v]
            pct = 100.0 * ((pool[None, :] < qp[:, None]).sum(1) + 0.5 * (pool[None, :] == qp[:, None]).sum(1)) / nc
            rank = (pool[None, :] > qp[:, None]).sum(1) + 1
            a["p"] += list(pct); a["h"] += list(rank <= K); a["seed_means"].append(float(pct.mean()))
            if ql is not None and len(ql):
                a["l"] += list(100.0 * ((pool[None, :] < ql[:, None]).sum(1) + 0.5 * (pool[None, :] == ql[:, None]).sum(1)) / nc)
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
        res[v] = {"mean_pct": round(float(np.mean(a["p"])), 2), "seed_sd": round(float(np.std(a["seed_means"])), 2),
                  "median_pct": round(float(np.median(a["p"])), 2), f"recall_at_{K}": round(float(np.mean(a["h"])), 4),
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
    judged_out = _judged_check(d, variants, mats, geo_extra, GL if use_genre else None,
                               GC if use_genre else None, kt, cfg["scoring"].get("negative_penalty")) \
        if d.get("judged") else None
    out = {**base, "held_out_per_seed": {"preview": nh_p, "local file": nh_l}, "judged_check": judged_out,
           "pool_chart_share": round(float(is_chart.mean()), 3), "variants": res, "tie_tolerance": ev["tie_tolerance"]}
    if len(prev_idx) < ev["min_queries"]:
        out["chosen"] = None
        out["reason"] = (f"only {len(prev_idx)} liked songs have a preview vector (need {ev['min_queries']}); "
                         "keeping the configured / default variant")
    else:
        out["chosen"] = choose_variant({v: r["mean_pct"] for v, r in res.items()}, ev["tie_tolerance"])
    try:
        out["liked_genres"] = liked_genres(con)
    except Exception as e:  # noqa: BLE001
        out["liked_genres"] = {"error": repr(e)}
    return out


def markdown(r: dict, title: str = "Scoring variant eval") -> str:
    lines = [f"## {title} ({r.get('at', '')})", ""]
    if r.get("skipped"):
        return "\n".join(lines + [f"Skipped: {r.get('reason')}"
                                   + (f". Variant used: **{r['used']}**." if r.get("used") else ""), ""])
    K = r["recall_k"]
    qk, hk = r.get("queries_by_kind", {}), r.get("held_out_per_seed", {})
    lines += [f"Queries: {qk.get('preview', 0)} liked songs with a Deezer preview vector (the same audio domain as the "
              f"candidates; {hk.get('preview', 0)} held out per seed, {r['holdout']:.0%}, seeds {r['seeds']}). Each "
              f"held-out song is ranked among {r['pool']} pool candidates (of {r['pool_total']}; liked and library "
              f"songs removed). Positives: {r['positive_tracks']} tracks (genre variants: the liked songs' style "
              f"vectors). Chart sources are {r['pool_chart_share']:.0%} of the pool. k = {r.get('topk', 10)} for "
              f"topk_mean and genre. The last column uses {qk.get('local file', 0)} liked songs that only have a "
              f"local-file vector; it is a diagnostic only.", ""]
    if r.get("genre_note"):
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
    choice = (f"Chosen: **{r['chosen']}**" if r.get("chosen") else f"No choice: {r.get('reason')}")
    used = ""
    if r.get("used"):
        used = f"; used for the batch: **{r['used']}**" + (" (pinned in config)" if r.get("pinned") else "")
    lines += ["", f"Choice: best mean percentile on preview-vector queries; within {r['tie_tolerance']} points is a "
              f"tie and the simpler variant wins ({' < '.join(VARIANTS)}). {choice}{used}.", ""]
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
