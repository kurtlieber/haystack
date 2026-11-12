"""The Taste brief (ADR 0028): what the listener's Verdicts say about styles, tempo and instrumental level, in a form
a client can write song ideas from. `GET /v1/taste` serves it; spindle reads it.

Read-only and model-free: SQLite and numpy over the stored Verdicts, styles and Readings. It is not the Taste profile
(llm.py), which summarises the listener for haystack's own language model, and it imports none of that code.

One Song per ISRC (a track with none is a Song of its own). Its Verdict is the newest by (given_at, received_at),
whichever release it was given on. "Down" is a current Pass or Dislike. "New" means the Verdict came from the
Calibrate or Discoveries queue, so the song was new to the listener. "Measured" means it has styles."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

from . import baseline, traits

DAYS = 45                    # the window of the recent lists
MIN_N = 8                    # the smallest style group the tables show
STYLE_ON = 0.10              # a style counts as present at this activation or more
NEW_QUEUES = ("calibrate", "discoveries")
INSTRUMENTAL_BANDS = ((0, 0.05), (0.05, 0.3), (0.3, 0.8), (0.8, 1))       # the last band includes 1
TEMPO_BANDS = ((0, 95), (95, 110), (110, 132), (132, 145), (145, 165), (165, 200), (200, None))
RECENT_FIVE_STARS, RECENT_DISLIKES, MOST_PLAYED = 25, 15, 15
STYLES_SHOWN = 3


def _deezer_ids(con) -> dict[str, int]:
    """spotify id -> the Deezer track whose preview was measured: a candidate's, else an on-demand mapping's,
    else a like's."""
    out: dict[str, int] = {}
    for r in con.execute("SELECT spotify_id, deezer_id FROM spotify_deezer "
                         "WHERE deezer_id IS NOT NULL AND status='ok'"):
        out[r[0]] = r[1]
    for r in con.execute("SELECT spotify_id, deezer_id FROM liked_map WHERE deezer_id IS NOT NULL"):
        out.setdefault(r[0], r[1])
    for r in con.execute("SELECT spotify_id, deezer_id FROM candidates "
                         "WHERE spotify_id IS NOT NULL AND deezer_id IS NOT NULL"):
        out[r[0]] = r[1]
    return out


def _styles(con, deezer_ids) -> dict[int, np.ndarray]:
    ids = sorted({int(i) for i in deezer_ids if i is not None})
    out = {}
    for s in range(0, len(ids), 500):
        ch = ids[s:s + 500]
        for r in con.execute(f"SELECT deezer_id, style400 FROM deezer_vectors WHERE style400 IS NOT NULL "
                             f"AND deezer_id IN ({','.join('?' * len(ch))})", ch):
            out[r[0]] = np.frombuffer(r[1], dtype=np.float16).astype(np.float32)
    return out


def songs(con) -> list[dict]:
    """Every Song with a Verdict: its newest Verdict, the track it was given on, its Readings, styles and plays."""
    latest: dict[str, dict] = {}
    for r in con.execute("SELECT spotify_id, kind, stars, queue, given_at, received_at FROM verdicts "
                         "ORDER BY given_at, received_at"):
        latest[r["spotify_id"]] = dict(r)
    isrc = {r[0]: r[1] for r in con.execute("SELECT spotify_id, isrc FROM spotify_tracks WHERE isrc IS NOT NULL")}
    best: dict[str, tuple] = {}
    for sid, v in latest.items():
        key = isrc.get(sid) or sid
        order = (v["given_at"] or "", v["received_at"] or "")
        if key not in best or order > best[key][0]:
            best[key] = (order, v)
    dz = _deezer_ids(con)
    names = {r[0]: (r[1], r[2]) for r in con.execute("SELECT spotify_id, name, artists_json FROM spotify_tracks")}
    cand = {r[0]: (r[1], r[2]) for r in con.execute(
        "SELECT spotify_id, artist, title FROM candidates WHERE spotify_id IS NOT NULL")}
    wanted = {dz.get(v["spotify_id"]) for _, v in best.values()}
    readings = traits.get(con, wanted)
    styles = _styles(con, wanted)
    plays = {r[0]: r[1] for r in con.execute("SELECT spotify_id, COUNT(*) FROM plays GROUP BY 1")}
    out = []
    for _, v in best.values():
        sid = v["spotify_id"]
        name, artists_json = names.get(sid) or (None, None)
        c_artist, c_title = cand.get(sid) or (None, None)
        d = dz.get(sid)
        out.append({"id": sid, "kind": v["kind"], "stars": v["stars"], "queue": v["queue"],
                    "given_at": v["given_at"] or "",
                    "artist": ", ".join(json.loads(artists_json or "[]")) or c_artist or None,
                    "title": name or c_title or None,
                    "readings": readings.get(d), "style": styles.get(d), "plays": plays.get(sid, 0)})
    return out


def tally(group: list[dict]) -> dict:
    """-> {n, five (5 stars), four_five (4 or 5 stars), down (a Pass or a Dislike)}: counts, not rates."""
    return {"n": len(group), "five": sum(1 for s in group if s["stars"] == 5),
            "four_five": sum(1 for s in group if (s["stars"] or 0) >= 4),
            "down": sum(1 for s in group if s["kind"] in ("pass", "dislike"))}


def _style_table(group: list[dict], classes: list[str], min_n: int) -> list[dict]:
    by: dict[int, list] = defaultdict(list)
    for s in group:
        for i in np.nonzero(s["style"] >= STYLE_ON)[0]:
            by[int(i)].append(s)
    keep = [(i, g) for i, g in by.items() if len(g) >= min_n and i < len(classes)]
    keep.sort(key=lambda ig: (-tally(ig[1])["five"] / len(ig[1]), -len(ig[1])))
    rows = []
    for i, g in keep:
        genre, _, style = classes[i].rpartition("---")
        rows.append({"style": style, "genre": genre or None, **tally(g)})
    return rows


def _bands(group: list[dict], trait: str, bands) -> list[dict]:
    """A tally per band, lo <= value < hi; the last band includes its hi. Songs without the Reading are in none."""
    values = [(s, (s["readings"] or {}).get(trait)) for s in group]
    rows = []
    for k, (lo, hi) in enumerate(bands):
        last = k == len(bands) - 1
        g = [s for s, v in values if v is not None and lo <= v and (hi is None or v < hi or (last and v == hi))]
        rows.append({"lo": lo, "hi": hi, **tally(g)})
    return rows


def _entry(s: dict, classes: list[str]) -> dict:
    top = []
    if s["style"] is not None and classes:
        top = [[classes[i].rpartition("---")[2], round(float(s["style"][i]), 2)]
               for i in np.argsort(-s["style"])[:STYLES_SHOWN] if i < len(classes)]
    return {"id": s["id"], "title": s["title"], "artist": s["artist"], "kind": s["kind"], "stars": s["stars"],
            "given_at": s["given_at"], "plays": s["plays"], "readings": s["readings"] or None, "styles": top}


def brief(con, cfg: dict, classes: list[str], now: datetime | None = None, days: int = DAYS,
          min_n: int = MIN_N) -> dict:
    """The Taste brief as docs/api.md describes `GET /v1/taste`. classes: the style class names
    ('Genre---Style'); without them the style tables are empty."""
    now = now or datetime.now(timezone.utc)
    rows = songs(con)
    measured = [s for s in rows if s["style"] is not None]
    new = [s for s in measured if s["queue"] in NEW_QUEUES]
    since = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    recent5 = sorted((s for s in new if s["stars"] == 5 and s["given_at"] >= since),
                     key=lambda s: s["given_at"], reverse=True)
    down = sorted((s for s in new if s["kind"] == "dislike" and s["given_at"] >= since),
                  key=lambda s: s["given_at"], reverse=True)
    played = sorted((s for s in rows if s["stars"] == 5 and s["plays"]), key=lambda s: -s["plays"])
    groups = {"all": measured, "new": new}
    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "days": days, "min_n": min_n, "style_on": STYLE_ON,
        "counts": {"songs": len(rows), "measured": len(measured), "new": len(new),
                   "recent_five_stars": len(recent5), "recent_dislikes": len(down)},
        "totals": {"all": tally(rows), "new": tally(new)},
        "styles": {k: _style_table(g, classes, min_n) for k, g in groups.items()},
        "instrumentalness": {k: _bands(g, "instrumentalness", INSTRUMENTAL_BANDS) for k, g in groups.items()},
        "tempo": {k: _bands(g, "tempo", TEMPO_BANDS) for k, g in groups.items()},
        "recent": {"five_stars": [_entry(s, classes) for s in recent5[:RECENT_FIVE_STARS]],
                   "dislikes": [_entry(s, classes) for s in down[:RECENT_DISLIKES]]},
        "most_played": [_entry(s, classes) for s in played[:MOST_PLAYED]],
        "traits": baseline.compute(con, cfg)["traits"],
    }
