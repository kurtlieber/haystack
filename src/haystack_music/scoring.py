"""Pure scoring + batch selection.

score(candidate) = max over positive rows of (cosine(candidate preview 512-d, positive row) + row bonus).
Library tracks contribute one row per 30 s window (best window wins); liked tracks without a confident local
match contribute their Deezer-preview vector."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np


def l2norm(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def nearest_positive(cands: np.ndarray, pos: np.ndarray, bonus: np.ndarray | None = None,
                     chunk: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (score [n], raw cosine of the winning row [n], winning row index [n])."""
    c = l2norm(cands)
    p = l2norm(pos)
    b = np.zeros(len(p), np.float32) if bonus is None else np.asarray(bonus, np.float32)
    scores = np.empty(len(c), np.float32); raw = np.empty(len(c), np.float32); idx = np.empty(len(c), np.int64)
    for s in range(0, len(c), chunk):
        sims = c[s:s + chunk] @ p.T
        adj = sims + b[None, :]
        i = adj.argmax(1)
        idx[s:s + chunk] = i
        scores[s:s + chunk] = adj[np.arange(len(i)), i]
        raw[s:s + chunk] = sims[np.arange(len(i)), i]
    return scores, raw, idx


# simplest first (ties go to the earlier one)
VARIANTS = ("raw", "topk_mean", "centred", "csls", "csls_centred", "genre", "genre_blend")
EMBED_VARIANTS = ("raw", "topk_mean", "centred", "csls", "csls_centred")
GENRE_VARIANTS = ("genre", "genre_blend")


@dataclass
class Space:
    """Shared geometry for the scoring variants.
    mu: mean of the L2-normalised library + candidate-pool vectors (centring).
    ref: fixed random reference set of library + pool vectors (CSLS hubness), raw space; ref_owner labels each row."""
    mu: np.ndarray | None = None
    ref: np.ndarray | None = None
    ref_owner: list | None = None
    k: int = 10

    def prep(self, m: np.ndarray, centred: bool) -> np.ndarray:
        x = l2norm(m)
        if centred and self.mu is not None:
            x = l2norm(x - self.mu[None, :])
        return x


def is_centred(variant: str) -> bool:
    return variant in ("centred", "csls_centred")


def is_csls(variant: str) -> bool:
    return variant in ("csls", "csls_centred")


def hubness(q: np.ndarray, ref: np.ndarray, k: int = 10, self_idx: np.ndarray | None = None,
            chunk: int = 1024, same: float = 0.9999) -> np.ndarray:
    """Mean cosine of each (already prepared) query row to its k nearest reference rows.
    The query's own reference row (self_idx >= 0) and identical vectors (cos >= `same`) are excluded."""
    out = np.empty(len(q), np.float32)
    k = max(1, min(k, len(ref) - 1))
    for s in range(0, len(q), chunk):
        sims = q[s:s + chunk] @ ref.T
        sims[sims >= same] = -np.inf
        if self_idx is not None:
            si = np.asarray(self_idx[s:s + chunk])
            rows = np.nonzero(si >= 0)[0]
            sims[rows, si[rows]] = -np.inf
        top = -np.partition(-sims, k - 1, axis=1)[:, :k]
        top[~np.isfinite(top)] = np.nan
        out[s:s + chunk] = np.nan_to_num(np.nanmean(top, axis=1))
    return out


def score_variant(variant: str, cands: np.ndarray, pos: np.ndarray, bonus: np.ndarray | None,
                  space: Space | None = None, self_ref_idx: np.ndarray | None = None):
    """-> (score [n], cosine of the winning row in the variant's space [n], winning row index [n]).
    raw: max cosine (+ bonus). centred: same after subtracting the library+pool mean and re-normalising.
    csls / csls_centred: nearest-positive score minus the candidate's mean cosine to its k nearest reference
    vectors, which penalises hubs (tracks close to everything)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown scoring variant {variant!r} (one of {', '.join(VARIANTS)})")
    if variant not in ("raw", "centred", "csls", "csls_centred"):
        raise ValueError(f"score_variant handles the embedding-space variants only, not {variant}")
    space = space or Space()
    if variant != "raw" and (space.mu is None or (is_csls(variant) and space.ref is None)):
        raise ValueError(f"variant {variant} needs a Space with mu{' and ref' if is_csls(variant) else ''}")
    cen = is_centred(variant)
    c = space.prep(cands, cen)
    s, cos, idx = nearest_positive(c, space.prep(pos, cen), bonus)
    if is_csls(variant):
        s = s - hubness(c, space.prep(space.ref, cen), space.k, self_ref_idx)
    return s, cos, idx


def owner_max(x: np.ndarray, p: np.ndarray, bonus: np.ndarray | None, starts, chunk: int = 256) -> np.ndarray:
    """[n, owners]: best (cosine + bonus) over each owner's rows (rows of an owner are contiguous from starts)."""
    b = np.zeros(len(p), np.float32) if bonus is None else np.asarray(bonus, np.float32)
    out = np.empty((len(x), len(starts)), np.float32)
    for s in range(0, len(x), chunk):
        out[s:s + chunk] = np.maximum.reduceat(x[s:s + chunk] @ p.T + b[None, :], starts, axis=1)
    return out


def topk_mean(S: np.ndarray, k: int = 10) -> np.ndarray:
    """Row-wise mean of the k largest entries."""
    k = max(1, min(k, S.shape[1]))
    return -np.partition(-S, k - 1, axis=1)[:, :k].mean(1)


def topk_owner_mean(cands, pos, bonus, starts, k: int = 10) -> np.ndarray:
    """Mean over the k nearest positive TRACKS (best window each) instead of the single nearest row."""
    return topk_mean(owner_max(l2norm(cands), l2norm(pos), bonus, starts), k)


def style_topk(cand_styles: np.ndarray, pos_styles: np.ndarray, k: int = 10,
               bonus: np.ndarray | None = None) -> np.ndarray:
    """Genre-profile similarity: mean (cosine + the positive's bonus) between a candidate's 400-style vector and
    its k most similar positives' style vectors."""
    S = l2norm(cand_styles) @ l2norm(pos_styles).T
    if bonus is not None:
        S = S + np.asarray(bonus, np.float32)[None, :]
    return topk_mean(S, k)


def zblend(a: np.ndarray, b: np.ndarray, ref_a: np.ndarray | None = None, ref_b: np.ndarray | None = None,
           w: float = 0.5) -> np.ndarray:
    """w * z(a) + (1 - w) * z(b), z-scored with the reference sets' mean/sd (default: a and b themselves)."""
    ra = a if ref_a is None else ref_a
    rb = b if ref_b is None else ref_b
    za = (a - ra.mean()) / (ra.std() or 1.0)
    zb = (b - rb.mean()) / (rb.std() or 1.0)
    return (w * za + (1 - w) * zb).astype(np.float32)


def negative_excess(S: np.ndarray, ref_S: np.ndarray, q: float = 0.95) -> np.ndarray:
    """How far each scored item is past the pool's typical similarity to each negative, on a 0..1 scale.
    S [n, m]: similarity of the scored items to the m negatives; ref_S [r, m]: the same for the reference pool.
    margin_j = q-quantile of ref_S[:, j] (per negative, so it adapts to each space's spread);
    excess = (S - margin) / (1 - margin), clipped to [0, 1]: 0 for anything no more similar to the negative than
    the pool's q-quantile, 1 for an exact copy (cosine 1)."""
    S = np.asarray(S, np.float32)
    if S.ndim != 2 or not S.shape[1]:
        return np.zeros((len(S), 0), np.float32)
    ref = np.asarray(ref_S, np.float32)
    m = np.quantile(ref, q, axis=0) if len(ref) else np.zeros(S.shape[1], np.float32)
    m = np.minimum(m, 1.0 - 1e-6).astype(np.float32)
    return np.clip((S - m[None, :]) / (1.0 - m[None, :]), 0.0, 1.0).astype(np.float32)


def negative_penalty(excess: np.ndarray, strength, scale: float, w: float,
                     self_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """-> (penalty [n], index of the negative that set it [n], -1 if none).
    penalty = w * scale * max_j(strength_j * excess_ij): the most damaging negative counts (a strong no is not
    masked by a nearer mild one). scale = the spread (sd) of the variant's scores over the pool, so w is in
    score-sd units whatever the variant. self_mask [n, m] (True = ignore) leaves a negative out for itself."""
    n = len(excess)
    if not w or excess.size == 0 or not np.isfinite(scale) or scale <= 0:
        return np.zeros(n, np.float32), np.full(n, -1, np.int64)
    E = excess * np.asarray(strength, np.float32)[None, :]
    if self_mask is not None:
        E = np.where(self_mask, 0.0, E)
    j = E.argmax(1)
    best = E[np.arange(n), j]
    return (w * scale * best).astype(np.float32), np.where(best > 0, j, -1)


def choose_variant(score: dict[str, float], tol: float = 0.5, exclude=()) -> str:
    """Best score (the eval's mean percentile or AUC); anything within `tol` of the best counts as a tie -> simplest
    wins. Variants in `exclude` (degenerate ones) are left out, unless that leaves none."""
    score = {v: s for v, s in score.items() if v not in exclude} or score
    best = max(score.values())
    for v in VARIANTS:
        if v in score and score[v] >= best - tol:
            return v
    return max(score, key=score.get)


@dataclass
class Pick:
    cand_id: int
    artist_key: str
    score: float
    source: str
    exploration: bool = False
    spotify_id: str | None = None
    nearest: str | None = None      # owner label of the positive it scored closest to (candidates.nearest)
    style: int | None = None        # top-1 Discogs style index (None: no style vector)
    percentile: float | None = None # its score's percentile in the batch's scored pool (offers.score_percentile)


def select_batch(pool: list[Pick], size: int, explore_fraction: float = 0.2,
                 band: tuple[float, float] = (0.4, 0.75), max_per_artist: int = 1,
                 exclude_artists: set[str] | None = None, rng: random.Random | None = None, *,
                 taken: list[Pick] = (), vectors: dict[int, np.ndarray] | None = None, diversity: float = 0.0,
                 max_style_share: float = 1.0, max_per_nearest: int = 0) -> list[Pick]:
    """Top-scored picks plus ~explore_fraction mid-similarity picks, at most max_per_artist per artist (ADR 0010).
    taken: picks already in this batch (an earlier round); they count toward every cap and the diversity.
    diversity: after each pick the remaining top picks are re-ranked by
      score - diversity * sd(pool score) * max(0, max cosine to the picks so far),
      with vectors {candidate id: unit vector}; a candidate without one is never marked down. 0 = by score alone.
    max_style_share: at most this share of the batch's top (exploit) picks may share a top-1 style (Pick.style;
      None is unconstrained). max_per_nearest: at most this many picks may share a nearest positive (0 = no cap).
    The caps give way when they would leave the batch short: the rest is filled without the style cap, then
    without either."""
    rng = rng or random.Random()
    counts: dict[str, int] = {a: max_per_artist for a in (exclude_artists or ())}
    near: Counter = Counter()
    styles: Counter = Counter()
    ranked = sorted(pool, key=lambda p: p.score, reverse=True)
    n = len(ranked)
    n_explore = int(round(size * explore_fraction))
    n_exploit = len(taken) + size - sum(p.exploration for p in taken) - n_explore
    style_cap = max(1, int(max_style_share * n_exploit + 1e-9)) if max_style_share < 1 else None
    chosen: list[Pick] = []
    chosen_ids: set[int] = set()
    caps = 2   # 2: both caps, 1: the nearest cap only, 0: none

    def ok(p, explore):
        if p.cand_id in chosen_ids or counts.get(p.artist_key, 0) >= max_per_artist:
            return False
        if caps and max_per_nearest and p.nearest is not None and near[p.nearest] >= max_per_nearest:
            return False
        return caps < 2 or explore or style_cap is None or p.style is None or styles[p.style] < style_cap

    def count(p):
        counts[p.artist_key] = counts.get(p.artist_key, 0) + 1
        if p.nearest is not None:
            near[p.nearest] += 1
        if not p.exploration and p.style is not None:
            styles[p.style] += 1

    def take(p, explore):
        q = replace(p, exploration=explore)
        chosen.append(q)
        chosen_ids.add(p.cand_id)
        count(q)

    for p in taken:
        count(p)
    # exploration first so the exploit pass cannot starve the band of artists
    if n_explore and n:
        lo, hi = int(n * (1 - band[1])), int(n * (1 - band[0]))   # ranks of the percentile band
        mid = ranked[lo:max(hi, lo + 1)]
        rng.shuffle(mid)
        for p in mid:
            if len([c for c in chosen if c.exploration]) >= n_explore:
                break
            if ok(p, True):
                take(p, True)
    diverse = diversity > 0 and bool(vectors) and n > 0
    if diverse:
        score = np.array([p.score for p in ranked], np.float64)
        V = np.zeros((n, len(next(iter(vectors.values())))), np.float32)
        for j, p in enumerate(ranked):
            if p.cand_id in vectors:
                V[j] = vectors[p.cand_id]
        sim = np.zeros(n, np.float32)
        lam = diversity * float(score.std())
        for p in (*taken, *chosen):
            if p.cand_id in vectors:
                sim = np.maximum(sim, V @ vectors[p.cand_id])

    def fill():
        nonlocal sim
        if not diverse:
            for p in ranked:
                if len(chosen) >= size:
                    break
                if ok(p, False):
                    take(p, False)
            return
        while len(chosen) < size:
            for j in np.argsort(-(score - lam * sim), kind="stable"):
                if ok(ranked[j], False):
                    take(ranked[j], False)
                    if ranked[j].cand_id in vectors:
                        sim = np.maximum(sim, V @ vectors[ranked[j].cand_id])
                    break
            else:
                return

    fill()
    if len(chosen) < size and style_cap is not None:
        caps = 1
        fill()
    if len(chosen) < size and max_per_nearest:
        caps = 0
        fill()
    chosen.sort(key=lambda p: (p.exploration, -p.score))
    return chosen


def adapt_exploration(prior: float, explore: tuple[int, int], exploit: tuple[int, int], min_judged: int = 8,
                      bounds: tuple[float, float] = (0.1, 0.4), gain: float = 0.5) -> dict:
    """This batch's exploration share (ADR 0010). explore / exploit: (kept, judged) offers of each kind over the
    recent batches. With at least min_judged on both sides: clip(prior + gain * (rate_explore - rate_exploit),
    *bounds); otherwise the prior."""
    (ke, ne), (kx, nx) = explore, exploit
    rate_e = ke / ne if ne else None
    rate_x = kx / nx if nx else None
    out = {"prior": prior, "explore_rate": None if rate_e is None else round(rate_e, 3), "explore_judged": ne,
           "exploit_rate": None if rate_x is None else round(rate_x, 3), "exploit_judged": nx}
    if ne < min_judged or nx < min_judged:
        return out | {"fraction": prior, "why": f"prior: needs {min_judged} judged offers of each kind"}
    f = min(max(prior + gain * (rate_e - rate_x), bounds[0]), bounds[1])
    return out | {"fraction": round(f, 3), "why": "moved by the recent kept rates"}


def like_rates(offers: list[dict]) -> dict:
    """offers: dicts with source, outcome (None while active), played (bool).
    like rate = liked / decided offers; also liked / played-decided."""
    def summarise(rows):
        decided = [r for r in rows if r.get("outcome")]
        liked = sum(1 for r in decided if r["outcome"] == "liked")
        played = [r for r in decided if r.get("played") or r["outcome"] == "liked"]
        return {"offered": len(rows), "decided": len(decided), "liked": liked,
                "like_rate": round(liked / len(decided), 3) if decided else None,
                "like_rate_of_played": round(liked / len(played), 3) if played else None}
    by: dict[str, list] = {}
    for r in offers:
        by.setdefault(r.get("source") or "unknown", []).append(r)
    return {"overall": summarise(offers), "per_source": {k: summarise(v) for k, v in sorted(by.items())}}


def can_retire(source_stats: dict, min_offers: int = 30) -> bool:
    return (source_stats or {}).get("offered", 0) >= min_offers
