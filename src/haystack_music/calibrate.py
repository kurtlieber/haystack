"""The Calibrate queue (ADR 0015): needle asks the listener about the pool candidates haystack is least sure of.

The pool is split into bands by stage-1 percentile (calibrate.bands, with shares). Within a band the candidates
the re-ranker gives the most even odds come first (|chance - 0.5|, when a model is fitted), then the ones the
ways of measuring closeness disagree about most (the standard deviation of a candidate's percentile across the
embedding variants and `genre`). One per artist per ISO week, none by an artist with a current Dislike, none
already liked, offered, asked about, judged or in the library, chart sources within harvest.chart_max_share of the
week.

select() is pure over rows, so `haystack calibrate plan` (token-free, no writes) and the hourly top-up (sync.run,
which maps the picks to Spotify as the batch does) share it. A pick stays in the pool; its Verdict is an ordinary
Verdict and moves it on, as for an offer."""
from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
from collections import Counter
from datetime import datetime

import numpy as np

from . import db, verdicts

log = logging.getLogger("haystack_music.calibrate")

EVEN_ODDS = 0.15           # |chance - 0.5| at most this: "even odds"
SPLIT = 10.0               # percentile points of standard deviation across the variants: "disagree"
DISAGREE_VARIANTS = ("raw", "topk_mean", "centred", "csls", "csls_centred", "genre")
REASONS = {"odds": "the re-ranker gives it even odds",
           "split": "the ways haystack measures closeness disagree about it",
           "far": "from the part of the ranking the batch rarely reaches",
           "close": "one of the closest calls at the top of the ranking"}


def week(ts: str | None = None) -> str:
    """The ISO week of a db.now() timestamp (default now), e.g. 2026-W43."""
    y, w, _ = datetime.fromisoformat((ts or db.now()).replace("Z", "+00:00")).isocalendar()
    return f"{y}-W{w:02d}"


def picked(con) -> dict[str, int]:
    """{Spotify id: candidate id} of every Calibrate pick, judged or not; {} on a database from before 0.5.0."""
    try:
        return {r[0]: r[1] for r in con.execute("SELECT spotify_id, candidate_id FROM calibrations ORDER BY id")}
    except sqlite3.OperationalError:
        return {}


def known(con) -> set[str]:
    """Spotify ids never to offer or ask about: liked, offered before, asked about in Calibrate, in the library."""
    out = {r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL")}
    out |= {r[0] for r in con.execute("SELECT spotify_id FROM offers")}
    out |= set(picked(con))
    out |= {r[0] for r in con.execute("SELECT spotify_id FROM local_tracks WHERE present=1 AND spotify_id IS NOT NULL")}
    return out


def judged(con) -> set[str]:
    """Spotify ids with any Verdict: Calibrate never asks about a track the listener has already judged."""
    return {r[0] for r in con.execute("SELECT DISTINCT spotify_id FROM verdicts")}


def disliked_artists(con) -> set[str]:
    """Artist keys of the tracks whose current Verdict is a Dislike."""
    from .verdicts import current
    sids = [s for s, v in current(con).items() if v.kind == "dislike"]
    out: set[str] = set()
    for q in ("SELECT artist_key FROM spotify_tracks WHERE spotify_id=?", "SELECT artist_key FROM candidates "
              "WHERE spotify_id=?", "SELECT artist_key FROM local_tracks WHERE spotify_id=?"):
        for s in sids:
            out |= {r[0] for r in con.execute(q, (s,)) if r[0]}
    return out


def bands(cfg: dict) -> list[tuple[str, float, float, float]]:
    """calibrate.bands as [(name, lo, hi, share)], lowest first; "50-75" is 50 <= percentile < 75 (100 included)."""
    out = []
    for name, share in cfg["calibrate"]["bands"].items():
        lo, hi = (float(x) for x in name.split("-"))
        out.append((name, lo, hi, float(share)))
    return sorted(out, key=lambda b: b[1])


def band_of(pct: float | None, bs) -> str | None:
    if pct is None or math.isnan(pct):
        return None
    for name, lo, hi, _ in bs:
        if lo <= pct < hi or (hi >= 100 and pct >= lo):
            return name
    return None


def reason(row: dict, top_band: bool) -> str:
    """The card's phrase: why haystack is unsure about this candidate."""
    p, d = row.get("rerank_p"), row.get("disagreement")
    if p is not None and abs(p - 0.5) <= EVEN_ODDS:
        return REASONS["odds"]
    if d is not None and d >= SPLIT:
        return REASONS["split"]
    return REASONS["close"] if top_band else REASONS["far"]


def order(rows: list[dict], rng: random.Random) -> list[dict]:
    """Most uncertain first: with a chance by |chance - 0.5|, then without one by disagreement (largest first,
    none last). Ties random."""
    rows = list(rows)
    rng.shuffle(rows)

    def k(r):
        if r.get("rerank_p") is not None:
            return (0, abs(r["rerank_p"] - 0.5))
        d = r.get("disagreement")
        return (1, -d) if d is not None else (2, 0.0)
    return sorted(rows, key=k)


def select(rows: list[dict], n: int, bs, rng: random.Random, *, have: dict[str, int] | None = None,
           artists: set[str] = frozenset(), chart_sources=frozenset(), max_chart: int | None = None,
           chart_have: int = 0) -> list[dict]:
    """Up to n picks from eligible rows ({id, artist_key, source, percentile, rerank_p, disagreement}).
    have: picks already made this week, by band; the n go to the bands furthest below their share of the week's
    total, one at a time, skipping a band with nothing left. artists: taken this week (one per artist).
    max_chart: chart-source picks allowed this week (chart_have already made). Each pick gains band and reason."""
    have = Counter(have or {})
    by_band: dict[str, list[dict]] = {b[0]: [] for b in bs}
    for r in rows:
        b = band_of(r.get("percentile"), bs)
        if b is not None:
            by_band[b].append(r)
    queues = {b: order(rs, rng) for b, rs in by_band.items()}
    share = {b[0]: b[3] for b in bs}
    top_band = bs[-1][0] if bs else None
    taken = set(artists)
    charts = chart_have
    out: list[dict] = []
    total = sum(have.values()) + n

    def next_in(b):
        q = queues[b]
        while q:
            r = q[0]
            if r["artist_key"] in taken or (max_chart is not None and r["source"] in chart_sources
                                            and charts >= max_chart):
                q.pop(0)
                continue
            return r
        return None
    while len(out) < n:
        open_ = [b for b in queues if next_in(b) is not None]
        if not open_:
            break
        b = max(open_, key=lambda x: (share[x] * total - have[x], share[x]))
        r = queues[b].pop(0)
        taken.add(r["artist_key"])
        charts += r["source"] in chart_sources
        have[b] += 1
        out.append({**r, "band": b, "reason": reason(r, b == top_band)})
    return out


# ---------------- reading the pool ----------------

def state(con, cfg: dict, ts: str | None = None) -> dict:
    """How many picks are open, made this week (by band, artist, chart), and how many more the queue takes."""
    c = cfg["calibrate"]
    wk = week(ts)
    try:
        rows = con.execute("SELECT k.band, k.judged_at, k.week, c.artist_key, c.source FROM calibrations k "
                           "LEFT JOIN candidates c ON c.id=k.candidate_id").fetchall()
    except sqlite3.OperationalError:            # a database from before 0.5.0, opened read-only
        rows = []
    outstanding = sum(r["judged_at"] is None for r in rows)
    this = [r for r in rows if r["week"] == wk]
    chart = set(cfg["harvest"]["chart_sources"])
    need = max(0, min(int(c["outstanding"]) - outstanding, int(c["per_week"]) - len(this))) if c["enabled"] else 0
    return {"week": wk, "outstanding": outstanding, "this_week": len(this), "need": need,
            "have": Counter(r["band"] for r in this), "artists": {r["artist_key"] for r in this if r["artist_key"]},
            "chart_have": sum(r["source"] in chart for r in this),
            "max_chart": int(cfg["harvest"]["chart_max_share"] * int(c["per_week"]) + 1e-9)}


def candidates(con, cfg: dict, variant: str | None = None) -> dict:
    """The eligible pool rows with their percentile, chance and disagreement, scored in memory (seconds).
    -> {variant, scored, rows, rerank (the fit's summary)}."""
    from . import rerank, vectors
    variant = variant or vectors.active_variant(con, cfg)
    res = vectors.compute_scores(con, cfg, variant) or {"rows": []}
    out = {"variant": variant, "scored": 0, "rows": [], "rerank": None}
    if not res.get("rows"):
        return out
    pct = {variant: _percentiles(res)}
    for v in DISAGREE_VARIANTS:
        if v != variant:
            pct[v] = _percentiles(vectors.compute_scores(con, cfg, v) or {"rows": []})
    rr = rerank.run(con, cfg, variant, res, "calibrate")
    out["rerank"] = rerank.summary(rr)
    chance = rr.get("chance") or {}
    retired = set(cfg["harvest"].get("retired_sources", []))
    skip = known(con) | judged(con)
    asked = set(picked(con).values())
    bad = disliked_artists(con)
    out["scored"] = len(pct[variant])
    for r in con.execute("SELECT id, deezer_id, artist, artist_key, title, source, spotify_id, isrc, duration "
                         "FROM candidates WHERE state='pool' AND vector_status='ok' "
                         "AND COALESCE(spotify_status,'') != 'not_found'"):
        cid = r["id"]
        if cid not in pct[variant] or cid in asked or r["source"] in retired or r["artist_key"] in bad \
                or (r["spotify_id"] and r["spotify_id"] in skip):
            continue
        ps = [p[cid] for p in pct.values() if cid in p]
        ch = chance.get(cid)
        out["rows"].append({**dict(r), "percentile": pct[variant][cid],
                            "rerank_p": None if ch is None else ch["p"],
                            "rerank_reason": None if ch is None else ch["reasons"],
                            "disagreement": round(float(np.std(ps)), 2) if len(ps) >= 2 else None})
    return out


def _percentiles(res: dict) -> dict[int, float]:
    """{candidate id: percentile of its score in the scored pool}, as the batch computes it."""
    rows = res.get("rows") or []
    sc = np.asarray(res.get("score", ()), np.float64)
    ok = np.isfinite(sc)
    ps = np.sort(sc[ok])
    return {int(rows[i][0]): round(100.0 * float(np.searchsorted(ps, sc[i], "left")) / len(ps), 2)
            for i in np.nonzero(ok)[0]}


# ---------------- plan and top-up ----------------

def plan(con, cfg: dict, rng: random.Random | None = None, n: int | None = None) -> dict:
    """`calibrate plan`: what the next top-up would pick, without Spotify and without writes. n overrides the
    number the queue takes now (default: what it needs; with none needed, a full week's worth to show)."""
    st = state(con, cfg)
    want = n if n is not None else (st["need"] or int(cfg["calibrate"]["per_week"]))
    pool = candidates(con, cfg)
    picks = select(pool["rows"], want, bands(cfg), rng or random.Random(), have=st["have"],
                   artists=st["artists"], chart_sources=frozenset(cfg["harvest"]["chart_sources"]),
                   max_chart=st["max_chart"], chart_have=st["chart_have"])
    return {"enabled": cfg["calibrate"]["enabled"], "week": st["week"], "outstanding": st["outstanding"],
            "this_week": st["this_week"], "need": st["need"], "variant": pool["variant"], "scored": pool["scored"],
            "eligible": len(pool["rows"]), "rerank": pool["rerank"],
            "fitted": any(r["rerank_p"] is not None for r in pool["rows"]), "by_band": dict(Counter(p["band"] for p in picks)),
            "picks": picks}


def top_up(con, cfg: dict, sp, dz=None, rng: random.Random | None = None) -> dict:
    """The hourly top-up (sync.run): pick what the queue needs, map each pick to Spotify as the batch does (an
    unmapped one is skipped and counted, and the next in line tried), and store it. -> {picked, outstanding,
    by_band, unmapped}."""
    from .batch import map_to_spotify
    from .deezer import Deezer
    from .sync import upsert_track
    verdicts.judge_picks(con)                   # picks given a Verdict or a heart since: their places are free
    st = state(con, cfg)
    out = {"picked": 0, "outstanding": st["outstanding"], "by_band": {}, "unmapped": 0, "known": 0}
    if not st["need"]:
        return out
    rng = rng or random.Random()
    pool = candidates(con, cfg)
    rows = {r["id"]: r for r in pool["rows"]}
    dz = dz or Deezer(cfg["harvest"]["deezer_min_interval"])
    tol = cfg["matching"]["duration_tolerance"]
    skip = known(con) | judged(con)
    bs, have, artists, charts = bands(cfg), Counter(st["have"]), set(st["artists"]), st["chart_have"]
    chart = frozenset(cfg["harvest"]["chart_sources"])
    attempts, limit = 0, st["need"] * int(cfg["batch"]["map_attempts_factor"])
    ts = db.now()
    while out["picked"] < st["need"] and rows and attempts < limit:
        picks = select(list(rows.values()), st["need"] - out["picked"], bs, rng, have=have, artists=artists,
                       chart_sources=chart, max_chart=st["max_chart"], chart_have=charts)
        if not picks:
            break
        for p in picks:
            attempts += 1
            rows.pop(p["id"])
            sid = p["spotify_id"] or map_to_spotify(con, sp, dz, p, tol)
            if not sid:
                con.execute("UPDATE candidates SET spotify_status='not_found' WHERE id=?", (p["id"],))
                out["unmapped"] += 1
                continue
            con.execute("UPDATE candidates SET spotify_id=?, spotify_status='ok' WHERE id=?", (sid, p["id"]))
            if sid in skip or sp.library_contains([sid]).get(sid):
                con.execute("UPDATE candidates SET state='excluded', exclude_reason='known on spotify' WHERE id=?",
                            (p["id"],))
                out["known"] += 1
                continue
            if con.execute("SELECT 1 FROM spotify_tracks WHERE spotify_id=? AND image_url IS NOT NULL",
                           (sid,)).fetchone() is None:
                upsert_track(con, sp.track(sid))            # the card's cover, year and album type
            con.execute("""INSERT INTO calibrations(candidate_id, spotify_id, picked_at, week, band, percentile,
                           rerank_p, disagreement, reason, rerank_reason) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (p["id"], sid, ts, st["week"], p["band"], p["percentile"], p["rerank_p"],
                         p["disagreement"], p["reason"],
                         None if p["rerank_reason"] is None else json.dumps(p["rerank_reason"])))
            skip.add(sid)
            have[p["band"]] += 1
            artists.add(p["artist_key"])
            charts += p["source"] in chart
            out["picked"] += 1
            out["by_band"][p["band"]] = out["by_band"].get(p["band"], 0) + 1
            if out["picked"] >= st["need"]:
                break
    out["outstanding"] = st["outstanding"] + out["picked"]
    log.info("calibrate: picked %d (%s), %d outstanding, %d unmapped", out["picked"], out["by_band"],
             out["outstanding"], out["unmapped"])
    return out
