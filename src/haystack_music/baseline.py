"""The Baseline: the listener's typical Reading per Trait.

Over the tracks the listener likes (Liked Songs and every Rated track; Passes and Dislikes left out), each
weighted by its stars, an Unrated like counting as 3 (5 stars count 5x a 1-star). Per Trait: the weighted mean,
which Concept A draws as "your baseline", and the weighted 25th-75th percentile, the usual range that decides
which Traits the summary sentence mentions."""
from __future__ import annotations

import numpy as np

from . import traits, verdicts


def weighted_quantile(values, weights, q: float) -> float:
    """Quantile of a weighted sample: the value where the cumulative weight (centred on each point) reaches q."""
    v = np.asarray(values, np.float64)
    w = np.asarray(weights, np.float64)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), cum, v))


def stats(values, weights) -> dict | None:
    if not len(values):
        return None
    w = np.asarray(weights, np.float64)
    return {"mean": float(np.average(values, weights=w)), "lo": weighted_quantile(values, w, 0.25),
            "hi": weighted_quantile(values, w, 0.75), "n": int(len(values))}


def liked_weights(con, cfg: dict) -> dict[str, float]:
    """spotify id -> weight for every track the listener likes."""
    cur = verdicts.current(con)
    unrated = cfg["scoring"]["verdicts"]["unrated_like_stars"]
    out = {}
    for (sid,) in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL"):
        v = cur.get(sid)
        if v is None:
            out[sid] = float(unrated)
        elif v.kind == "rating":
            out[sid] = float(v.stars)
    for sid, v in cur.items():
        if v.kind == "rating":
            out[sid] = float(v.stars)
    return out


def compute(con, cfg: dict) -> dict:
    """-> {"tracks": n with Readings, "traits": {trait: {mean, lo, hi, n} | None}}."""
    from .vectors import track_vectors
    w = liked_weights(con, cfg)
    tv = track_vectors(con, list(w))
    rd = traits.get(con, [t["deezer_id"] for t in tv.values()])
    per = {t: ([], []) for t in traits.TRAITS}
    n = 0
    for sid, weight in w.items():
        t = tv.get(sid)
        r = rd.get(t["deezer_id"]) if t else None
        if not r:
            continue
        n += 1
        for trait, val in r.items():
            if trait in per:
                per[trait][0].append(val)
                per[trait][1].append(weight)
    return {"tracks": n, "traits": {t: stats(v, ws) for t, (v, ws) in per.items()}}
