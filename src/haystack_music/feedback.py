"""The `feedback yes|no` command, kept as a thin alias of verdicts on candidates (ADR 0005).
yes -> a 3-star Rating; no -> a Pass (mild, normal) or a Dislike (strong). The verdict itself, its weight in
scoring and its Spotify writes all live in verdicts.py; this module only finds the candidate by artist/title."""
from __future__ import annotations

import logging

from . import verdicts
from .textnorm import song_key

log = logging.getLogger("haystack_music.feedback")


def find_candidate(con, artist: str, title: str):
    sk = song_key(artist, title)
    return con.execute("SELECT * FROM candidates WHERE song_key=? ORDER BY spotify_id IS NOT NULL DESC, "
                       "vector_status='ok' DESC, id LIMIT 1", (sk,)).fetchone()


def as_verdict(verdict: str, strength: str = "normal") -> tuple[str, int | None]:
    if verdict == "yes":
        return "rating", 3
    if verdict == "no":
        return ("dislike" if strength == "strong" else "pass"), None
    raise ValueError("verdict must be yes or no")


def record(con, verdict: str, artist: str, title: str, strength: str = "normal", note: str | None = None,
           source: str = "cli") -> dict:
    kind, stars = as_verdict(verdict, strength)
    c = find_candidate(con, artist, title)
    if c is None:
        raise LookupError(f"no candidate for {artist} - {title}")
    if not c["spotify_id"]:
        raise LookupError(f"{artist} - {title} has no Spotify id yet (it was never offered)")
    r = verdicts.record(con, c["spotify_id"], kind, stars, source=source)
    return {"candidate_id": c["id"], "artist": c["artist"], "title": c["title"], "spotify_id": c["spotify_id"],
            "kind": kind, "stars": stars, **r}


def judged(con, cfg: dict | None = None) -> list[dict]:
    """Candidates with a current verdict, in the old feedback shape the eval's sanity check reads:
    verdict yes (a Rating) / no (Pass, Dislike), weight = the negative strength (1 for a Rating)."""
    if cfg is None:
        from . import config
        cfg = config.load()
    cur = verdicts.current(con)
    out = []
    for r in con.execute("SELECT id, spotify_id, deezer_id, artist, title FROM candidates WHERE spotify_id IS NOT NULL"):
        v = cur.get(r["spotify_id"])
        if v is None:
            continue
        neg = verdicts.negative_strength(cfg, v)
        out.append({"candidate_id": r["id"], "spotify_id": r["spotify_id"], "deezer_id": r["deezer_id"],
                    "artist": r["artist"], "title": r["title"], "verdict": "no" if neg else "yes",
                    "weight": neg or 1.0, "kind": v.kind, "stars": v.stars})
    return out
