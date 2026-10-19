"""How each candidate source does for this listener (ADR 0013): the kept rate by source and by the seed a candidate
was found from, and the per-run cap each source earns from it.

kept and decided are the report's (report.offer_rows): an offer is decided once it has left Discoveries, and kept
when the track's current verdict is a Rating of 3 stars or more, or it has no verdict and closed as liked."""
from __future__ import annotations

import re

from . import report

SEED_TAG = re.compile(r"\bseed=(\w+)")
SEED_ORDER = ["5", "4", "3", "discovery", "like", "library", "broad"]
CAP_BOUNDS = (0.25, 2.0)


def seed_label(tag: str | None) -> str:
    """'5' -> '5★'; other tags as they are; None (found before seeds were tagged, or not from a seed) -> 'untagged'."""
    if tag is None:
        return "untagged"
    return f"{tag}★" if tag.isdigit() else tag


def _seed_tags(con) -> dict[tuple[int, str], str | None]:
    """{(candidate id, source): the seed tag of the first sighting by that source} for the offered candidates. A
    later sighting doesn't count: a candidate found before seeds were tagged stays untagged when it is seen again."""
    out: dict[tuple[int, str], str | None] = {}
    for cid, src, detail in con.execute("""SELECT candidate_id, source, detail FROM candidate_sources
                                           WHERE candidate_id IN (SELECT candidate_id FROM offers)
                                           ORDER BY seen_at, rowid"""):
        if (cid, src) not in out:
            m = SEED_TAG.search(detail or "")
            out[(cid, src)] = m.group(1) if m else None
    return out


def offer_rows(con) -> list[dict]:
    """report.offer_rows plus the seed each offer's candidate was found from (None when untagged)."""
    tags = _seed_tags(con)
    rows = report.offer_rows(con)
    for r in rows:
        r["seed"] = tags.get((r["candidate"], r["source"]))
    return rows


def _rate(rows: list[dict]) -> dict:
    d = [r for r in rows if r["decided"]]
    kept = sum(r["kept"] for r in d)
    return {"offered": len(rows), "decided": len(d), "kept": kept,
            "kept_rate": round(kept / len(d), 3) if d else None}


def rates(con) -> dict:
    """{"overall": ..., "per_source": {source: ...}, "per_seed": {label: ...}}, each offered, decided, kept and
    kept_rate (kept / decided)."""
    rows = offer_rows(con)
    by_src: dict[str, list] = {}
    by_seed: dict[str, list] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
        by_seed.setdefault(seed_label(r["seed"]), []).append(r)
    order = {seed_label(t): i for i, t in enumerate(SEED_ORDER)}
    return {"overall": _rate(rows),
            "per_source": {s: _rate(v) for s, v in sorted(by_src.items())},
            "per_seed": {s: _rate(by_seed[s]) for s in sorted(by_seed, key=lambda s: (order.get(s, len(order)), s))}}


def multiplier(rate: float, mean: float) -> float:
    """A source's cap multiplier: its kept rate over the mean, clipped to CAP_BOUNDS."""
    if not mean:
        return 1.0
    return round(min(max(rate / mean, CAP_BOUNDS[0]), CAP_BOUNDS[1]), 3)


def caps(cfg: dict, sources: list[str], r: dict) -> dict[str, dict]:
    """This run's cap per source: {source: {cap, base, multiplier, why}}. The base is harvest.caps[source], else
    harvest.max_new_per_source. With harvest.adaptive_caps, a source with at least sources.min_offers_before_retire
    decided offers has its base scaled by multiplier(its kept rate, the kept rate over all decided offers); below
    that it keeps the base. r: rates(con)."""
    h = cfg["harvest"]
    need = cfg["sources"]["min_offers_before_retire"]
    mean = r["overall"]["kept_rate"]
    out = {}
    for s in sources:
        base = int(h.get("caps", {}).get(s, h["max_new_per_source"]))
        st = r["per_source"].get(s, {})
        m, why = 1.0, "adaptive_caps off"
        if h.get("adaptive_caps", True):
            if st.get("decided", 0) < need:
                why = f"{st.get('decided', 0)} decided offers; scaled from {need}"
            elif not mean:
                why = "no kept offers yet"
            else:
                m = multiplier(st["kept_rate"], mean)
                why = f"kept rate {st['kept_rate']:.0%} against {mean:.0%} over all sources"
        out[s] = {"cap": max(1, round(base * m)), "base": base, "multiplier": m, "why": why}
    return out
