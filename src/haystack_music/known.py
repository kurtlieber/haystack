"""The songs the listener already has, on any release (ADR 0021): the check the batch and the Calibrate queue make
on each pool candidate, and the sync's step that takes such candidates out of the pool.

The harvest matches a candidate against the library and Liked Songs once, when it arrives (harvest.offer), and
the batch and Calibrate then check Spotify ids (calibrate.known). A song liked or ripped after its candidate arrived
stays in the pool, and a remaster, a deluxe edition or a best-of of it is another Spotify track, so the id check
can't see it. The same audio as a positive scores about 0.99, so such candidates sit at the top of the ranking.

Known matches a pool row against three sets, the library's files, the liked songs, and the tracks with a verdict,
an offer or a Calibrate pick: by ISRC where the candidate has one, else by the names (same_song). The names rule is
not the harvest's intake rule (matching.compare within 3 s): Deezer's length for the same recording differs from
Spotify's or the file's by 4 to 30 s often enough, and another master or a longer cut of a song the listener has
is not a discovery, so the length may differ by up to TOL seconds. In exchange the titles must agree exactly once
the version markers are stripped, or differ only by a decoration ('I Don't Want to Miss a Thing - From
"Armageddon" Soundtrack'): compare's fuzzy title test, fine within 3 s, pairs FATE with Faded and Panama (2015
Remaster) with Jump - 2015 Remaster once the length is free. A remix, a live take, an acoustic version or a cover
is a different recording and stays a candidate (the version check)."""
from __future__ import annotations

import json
import logging
import re
import sqlite3

from . import library
from .matching import Index, TrackRef, _sim
from .textnorm import key, version_check, version_signature

log = logging.getLogger("haystack_music.known")

TOL = 90.0          # seconds the length may differ by (the harvest's intake rule allows 3)
MIN_ARTIST = 0.8    # artist similarity needed when the names, not the ISRC, make the match
PREVIEW_SECONDS = 31  # a Deezer length this short is the preview's, not the song's: unknown
# what a title may carry past the other's, besides a version marker: 'Happy - From "Despicable Me 2"', not
# 'help_urself 2' (a sequel) or 'Wish You Were Here (Take 1)' (another take)
DECORATION = re.compile(r"\b(from|soundtrack|theme|ost|remaster(ed)?|version|edit|mix|original|single|album|"
                        r"deluxe|anniversary|mono|stereo|explicit|clean|\d{4})\b")


def _refs(con, sql: str) -> list[TrackRef]:
    out = []
    for r in con.execute(sql):
        arts = json.loads(r["artists_json"] or "[]")
        if r["name"] and arts:
            out.append(TrackRef(r["spotify_id"], arts[0], r["name"], r["duration"], r["spotify_id"], r["isrc"], arts))
    return out


def liked_refs(con) -> list[TrackRef]:
    """The liked songs, as the harvest matches them."""
    return _refs(con, """SELECT s.spotify_id, s.name, s.artists_json, s.duration, s.isrc FROM spotify_tracks s
                         JOIN likes l ON l.spotify_id=s.spotify_id WHERE l.removed_at IS NULL""")


def judged_refs(con) -> list[TrackRef]:
    """The tracks with a verdict, an offer or a Calibrate pick: judged, or waiting to be."""
    sql = """SELECT s.spotify_id, s.name, s.artists_json, s.duration, s.isrc FROM spotify_tracks s
             WHERE s.spotify_id IN (SELECT spotify_id FROM verdicts UNION SELECT spotify_id FROM offers{})"""
    try:
        return _refs(con, sql.format(" UNION SELECT spotify_id FROM calibrations"))
    except sqlite3.OperationalError:            # a database from before 0.5.0, opened read-only
        return _refs(con, sql.format(""))


def _val(row, k: str):
    try:
        return row[k]
    except (KeyError, IndexError):
        return None


def same_song(q: TrackRef, c: TrackRef) -> bool:
    """Is c the same song as q, on whatever release: the artists agree (MIN_ARTIST), the lengths within TOL, no
    hard version marker between them, and the base titles equal once the version markers are stripped, or one
    the other plus a decoration (DECORATION) that carries no version marker."""
    if q.duration and c.duration and abs(float(q.duration) - float(c.duration)) > TOL:
        return False
    if max((_sim(a, b) for a in q.artist_keys for b in c.artist_keys), default=0.0) < MIN_ARTIST:
        return False
    (qb, qs), (cb, cs) = q.base_suffix, c.base_suffix
    if version_check(qs, cs) == "mismatch":
        return False
    kq, kc = key(qb), key(cb)
    if kq == kc:
        return bool(kq)
    short, long = sorted((kq, kc), key=len)
    if not short or not long.startswith(short + " "):
        return False
    rest = long[len(short):]
    return bool(DECORATION.search(rest)) and not version_signature(rest)[0]


class Known:
    """What the listener already has. why(row) says which set a candidate row (id, artist, title, duration, isrc)
    is the same song as: 'in_library', 'liked' or 'judged', else None."""

    def __init__(self, con, lib: Index | None = None):
        self.sets = (("in_library", lib if lib is not None else Index(library.library_refs(con))),
                     ("liked", Index(liked_refs(con))), ("judged", Index(judged_refs(con))))

    def why(self, row) -> str | None:
        dur = _val(row, "duration")
        if dur is not None and dur <= PREVIEW_SECONDS:     # Deezer's length for a track it has only the preview of
            dur = None
        q = TrackRef(_val(row, "id"), _val(row, "artist") or "", _val(row, "title") or "", dur, None, _val(row, "isrc"))
        for name, idx in self.sets:
            m = idx.match(q, TOL)                      # the ISRC, else the best fuzzy match
            if m and m.method != "fuzzy":
                return name
            if any(same_song(q, c) for c in idx.candidates(q)):
                return name
        return None


def exclude_known(con, cfg: dict, lib: Index | None = None) -> dict:
    """The sync's step: a pool candidate that is now the same song as a liked song or a library file leaves the
    pool as excluded, with the harvest's reason (liked, in_library). A judged song's other releases stay in the
    pool and are skipped at selection: the verdict's state is on the track that got it.
    -> {checked, in_library, liked}."""
    have = Known(con, lib)
    c = {"checked": 0, "in_library": 0, "liked": 0}
    for r in con.execute("SELECT id, artist, title, duration, isrc FROM candidates WHERE state='pool'").fetchall():
        c["checked"] += 1
        why = have.why(r)
        if why in ("in_library", "liked"):
            con.execute("UPDATE candidates SET state='excluded', exclude_reason=? WHERE id=? AND state='pool'",
                        (why, r["id"]))
            c[why] += 1
    if c["in_library"] or c["liked"]:
        log.info("%d pool candidates are songs the listener has now (%d in the library, %d liked): excluded",
                 c["in_library"] + c["liked"], c["in_library"], c["liked"])
    return c
