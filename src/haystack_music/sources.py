"""How each candidate source does for this listener (ADR 0013): the kept rate by source and by the seed a candidate
was found from, and the per-run cap each source earns from it.

kept and decided are the report's (report.offer_rows): an offer is decided once it has left Discoveries, and kept
when the track's current verdict is a Rating of 3 stars or more, or it has no verdict and closed as liked.

A source's already-known share (ADR 0026) is read from the harvest runs' summaries: of what it brought in, the share
that was already a candidate, a liked song or a library file. And a pool candidate stays eligible for a batch and
Calibrate only while one of the sources that found it is live (live_row)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

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


KNOWN_DAYS = 14
KNOWN_KEYS = ("dupe", "in_library", "liked")      # a harvest run's per-source counts of intake already known


def _share(c: dict) -> dict:
    k = sum(c.get(x, 0) for x in KNOWN_KEYS)
    n = k + c.get("new", 0)
    return {"known": k, "intake": n, "known_share": round(k / n, 3) if n else None}


def run_known_share(per_source: dict) -> dict:
    """{source: {known, intake, known_share}} for one harvest run's per_source counts."""
    return {s: _share(c) for s, c in per_source.items() if isinstance(c, dict) and "new" in c}


def known_share(con, days: int = KNOWN_DAYS) -> dict:
    """{"per_source": {source: {known, intake, known_share}}, "charts": {chart id: ...}} over the harvest runs
    (daily, broad and their shards) of the last `days` days: of each source's intake, the share that was already a
    candidate, a liked song or a library file. charts: deezer_editorial's own charts."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    tot: dict[str, dict] = {}
    charts: dict[str, dict] = {}

    def add(into: dict, c: dict) -> None:
        for k in (*KNOWN_KEYS, "new"):
            into[k] = into.get(k, 0) + int(c.get(k) or 0)

    for (summary,) in con.execute("SELECT summary FROM runs WHERE job LIKE 'harvest%' AND ok=1 AND started_at >= ?",
                                  (since,)):
        try:
            per = (json.loads(summary or "{}") or {}).get("per_source") or {}
        except ValueError:
            continue
        for src, c in per.items():
            if not isinstance(c, dict):
                continue
            add(tot.setdefault(src, {}), c)
            for eid, cc in (c.get("charts") or {}).items():
                add(charts.setdefault(str(eid), {}), cc)
    return {"per_source": {s: _share(c) for s, c in sorted(tot.items())},
            "charts": {e: _share(c) for e, c in sorted(charts.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit()
                                                                        else 0)}}


def live_row(cfg: dict, source: str | None, detail: str | None = None) -> bool:
    """A candidate_sources row (or, with no detail, a candidate's own source) that still counts: its source is not
    in harvest.retired_sources, and a Deezer editorial chart or new-release selection (detail chart/<id> or
    editorial/<id>/...) is still in harvest.deezer_editorial_ids."""
    h = cfg["harvest"]
    if source in h.get("retired_sources", []):
        return False
    d = detail or ""
    prefix = {"deezer_editorial": "chart/", "deezer_new_releases": "editorial/"}.get(source or "")
    if prefix and d.startswith(prefix):
        eid = d[len(prefix):].split("/", 1)[0].split()[0] if d[len(prefix):] else ""
        if eid.lstrip("-").isdigit():
            return int(eid) in {int(x) for x in h.get("deezer_editorial_ids", [])}
    return True


def retired_candidates(con, cfg: dict) -> set[int]:
    """Pool candidates no live source holds (ADR 0026, #343's rule): every one of their candidate_sources rows is
    retired, or, for a candidate without rows, its own source is. A song another live source also found stays."""
    live: dict[int, bool] = {}
    for cid, src, detail in con.execute("""SELECT cs.candidate_id, cs.source, cs.detail FROM candidate_sources cs
                                           JOIN candidates c ON c.id=cs.candidate_id WHERE c.state='pool'"""):
        if not live.get(cid):
            live[cid] = live_row(cfg, src, detail)
    out = {cid for cid, ok in live.items() if not ok}
    for cid, src in con.execute("SELECT id, source FROM candidates WHERE state='pool'"):
        if cid not in live and not live_row(cfg, src):
            out.add(cid)
    return out


def last_skips(con, days: int = KNOWN_DAYS) -> dict[str, str]:
    """{source: why} for the sources the latest harvest run that ran them skipped (not configured: no key, no
    user name, not this week)."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    seen: dict[str, str | None] = {}
    for (summary,) in con.execute("""SELECT summary FROM runs WHERE job LIKE 'harvest%' AND ok=1 AND started_at >= ?
                                     ORDER BY started_at DESC, id DESC""", (since,)):
        try:
            per = (json.loads(summary or "{}") or {}).get("per_source") or {}
        except ValueError:
            continue
        for src, c in per.items():
            if src not in seen and isinstance(c, dict):
                seen[src] = c.get("skipped")
    return {s: why for s, why in seen.items() if why}
