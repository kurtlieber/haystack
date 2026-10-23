"""Possible covers and live versions, and the Swap that puts the Original in their place (ADR 0022).

A track card says when a track may stand in for someone else's recording, which is its Original. `check` looks for four
signs, cheapest first:
  * marker: the title's version suffix (in brackets or after a dash) says cover, karaoke, tribute, "originally
    performed by", "made famous by" or "in the style of", or an artist's name says karaoke or tribute;
  * live: the suffix says live, unplugged or BBC, and the Original is the same artist's studio recording;
  * isrc: Deezer files the track's ISRC under another artist's track of the same title (a cover that borrowed it);
  * rank: on Deezer, another artist's track of the same title is at least RANK_GAP times as popular.
On 62 covers the listener had removed, 19 well-known covers they keep and 200 other songs of theirs (2026-10-23), the
signs flagged 57, 0 and 14% of them. A sign is a hint, never a verdict: needle shows it and the listener decides.

A Swap is that decision. It takes back the cover's Verdicts (kept in swap_retracted, so they can be put back), and the
background writes put the Original where the cover was in each playlist the listener owns, unless its Song is there
already, take the cover out and move the heart. This is the one place haystack edits the listener's own playlists."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone

from . import db, playlists, verdicts
from .textnorm import key, split_title, version_check

log = logging.getLogger("haystack_music.originals")
RANK_GAP = 3                    # the other artist's Deezer rank, as a multiple of the track's own
RANK_PLACEHOLDER = 100000       # the rank Deezer gives a track it has no figure for
RETRY = timedelta(hours=24)     # a check that failed is tried again after this
REASONS = ("marker", "live", "isrc", "rank")

SCHEMA = """
CREATE TABLE IF NOT EXISTS original_checks (
    spotify_id TEXT PRIMARY KEY,
    checked_at TEXT NOT NULL,
    reason TEXT,                         -- NULL (no sign) | marker | live | isrc | rank
    of_title TEXT, of_artist TEXT,       -- the Original: the recording the track may stand in for
    of_spotify_id TEXT,                  -- the Original on Spotify, when a search found it
    own_rank INTEGER, other_rank INTEGER,    -- Deezer's ranks, when looked up
    error TEXT                           -- the check failed: tried again after RETRY
);
CREATE TABLE IF NOT EXISTS swaps (
    id TEXT PRIMARY KEY,                 -- client-made
    cover TEXT NOT NULL,                 -- the track swapped out
    original TEXT NOT NULL,              -- the track swapped in
    given_at TEXT NOT NULL, received_at TEXT NOT NULL,
    effects TEXT NOT NULL DEFAULT 'pending',   -- pending | done
    effects_at TEXT, effects_error TEXT,
    result TEXT                          -- JSON: what the Spotify writes did
);
CREATE INDEX IF NOT EXISTS swaps_cover ON swaps(cover);
CREATE TABLE IF NOT EXISTS swap_retracted (
    verdict_id TEXT PRIMARY KEY,         -- a Verdict a Swap took back: a resend of it is dropped
    swap_id TEXT NOT NULL,
    spotify_id TEXT NOT NULL,
    given_at TEXT NOT NULL,
    row TEXT NOT NULL                    -- the whole verdicts row as JSON, to put it back by hand
);
"""

_SUFFIX = r"(?:[\(\[][^\)\]]*|\s[-–—/]\s?.*)"        # in brackets, or after a dash: "Cover Me" is a title
MARKER = re.compile(_SUFFIX + r"\b(cover(ed)?|karaoke|tribute|originally performed|made famous|in the style of)\b",
                    re.I)
LIVE = re.compile(r"(?:[\(\[][^\)\]]*|\s[-–—]\s.*)\b(live|unplugged|bbc)\b", re.I)
ARTIST_MARKER = re.compile(r"\b(karaoke|tribute)\b", re.I)
BY = re.compile(r"\b(?:originally performed by|made famous by|in the style of|tribute to|cover of)\s+"
                r"([^\)\]]+?)\s*(?:[\)\]]|\s[-–—]\s|$)", re.I)


def ensure(con) -> None:
    con.executescript(SCHEMA)


def bare(title: str | None) -> str:
    """The title without a dash suffix or anything in brackets: 'Zombie - Live at Rockpalast' -> 'Zombie'."""
    t = re.sub(r"\s+[-–—]\s+.*$", "", title or "")
    t = re.sub(r"\s*[\(\[].*?[\)\]]", "", t).strip()
    return t or (title or "").strip()


def _akey(name: str | None) -> str:
    return re.sub(r"^the ", "", key(name))


def same_artist(a: str | None, b: str | None) -> bool:
    """One name is the other, or holds it as whole words: 'Bob Marley' and 'Bob Marley & The Wailers'."""
    a, b = _akey(a), _akey(b)
    return bool(a and b) and (a == b or f" {a} " in f" {b} " or f" {b} " in f" {a} ")


def _ours(name: str | None, artists: list[str]) -> bool:
    """One of these artists, leaving out a karaoke or tribute act: "Foo Fighters Tribute Band" isn't Foo Fighters."""
    return any(same_artist(name, a) for a in artists if not ARTIST_MARKER.search(a))


def _most_popular(dz, title: str, artists: list[str]) -> dict | None:
    """Deezer's most popular track of this title by someone other than these artists, and not itself a karaoke or
    tribute track (a track:"…" search sorted by rank, then plain text when that finds nothing)."""
    q = title.replace('"', "")
    hits = dz.search(f'track:"{q}"', limit=25, order="RANKING") or dz.search(q, limit=25, order="RANKING")
    best = None
    for h in hits:
        who = (h.get("artist") or {}).get("name")
        if key(bare(h.get("title"))) == key(title) and who and not _ours(who, artists) \
                and not ARTIST_MARKER.search(who) and not MARKER.search(h.get("title") or "") \
                and (best is None or (h.get("rank") or 0) > (best.get("rank") or 0)):
            best = h
    return best


def signs(title: str, artists: list[str], isrc: str | None, dz) -> dict:
    """The first sign that a track stands in for another recording (the module docstring), from its own title and
    artists, then from Deezer. -> {"reason" (None: no sign), "title", "artist" (None: unknown), "own_rank",
    "other_rank"}."""
    base = bare(title)
    out = {"reason": None, "title": base, "artist": None, "own_rank": None, "other_rank": None}
    if MARKER.search(title) or any(ARTIST_MARKER.search(a) for a in artists):
        m = BY.search(title)
        out.update(reason="marker", artist=m.group(1).strip() if m else None)
        if out["artist"] is None:                         # "(Karaoke Version)": the best-known one is the Original
            best = _most_popular(dz, base, artists)
            if best:
                out.update(artist=best["artist"]["name"], other_rank=best.get("rank"))
        return out
    if LIVE.search(title):
        return {**out, "reason": "live", "artist": artists[0] if artists else None}
    own = dz.track_by_isrc(isrc) if isrc else None
    own = own if isinstance(own, dict) and own.get("id") else None
    if own:
        out["own_rank"] = own.get("rank")
        theirs = [(own.get("artist") or {}).get("name")] + [c.get("name") for c in own.get("contributors") or []]
        if not any(_ours(n, artists) for n in theirs if n) and key(bare(own.get("title"))) == key(base):
            return {**out, "reason": "isrc", "artist": theirs[0]}
    best = _most_popular(dz, base, artists)
    if best:
        out["other_rank"] = best.get("rank")
        if out["own_rank"] is not None and best.get("rank") not in (None, RANK_PLACEHOLDER) \
                and best["rank"] >= RANK_GAP * max(out["own_rank"], 1):
            out.update(reason="rank", artist=best["artist"]["name"])
    return out


def find_original(sp, title: str, artist: str, exclude: str) -> dict | None:
    """The Original on Spotify: the first search result with the same title and no version marker (live, remix,
    karaoke, …) by that artist and no karaoke or tribute act. None when the search finds none."""
    t, a = title.replace('"', ""), artist.replace('"', "")
    for q in (f'track:"{t}" artist:"{a}"', f"{t} {a}"):
        for r in sp.search_tracks(q, limit=10):
            if not r or not r.get("id") or r["id"] == exclude:
                continue
            name = r.get("name") or ""
            if key(bare(name)) != key(title) or MARKER.search(name) or LIVE.search(name) \
                    or version_check(split_title(name)[1], "") == "mismatch":
                continue
            names = [x.get("name") or "" for x in r.get("artists") or []]
            if any(same_artist(artist, n) for n in names) and not any(ARTIST_MARKER.search(n) for n in names):
                return r
    return None


def due(con, sids: list[str]) -> list[str]:
    """The tracks to check: never checked, or the last check failed more than RETRY ago."""
    cutoff = (datetime.now(timezone.utc) - RETRY).strftime("%Y-%m-%dT%H:%M:%SZ")
    done = set()
    for i in range(0, len(sids), 500):
        ch = sids[i:i + 500]
        done |= {r[0] for r in con.execute(f"SELECT spotify_id FROM original_checks WHERE spotify_id IN "
                                           f"({','.join('?' * len(ch))}) AND (error IS NULL OR checked_at > ?)",
                                           (*ch, cutoff))}
    return [s for s in sids if s not in done]


def check(con, sid: str, dz, sp, force: bool = False) -> dict | None:
    """Look for the signs on one track and store what was found. A track checked already is skipped (one whose check
    failed, until RETRY has passed) unless force. -> the stored row, or None when skipped."""
    from .sync import upsert_track
    if not force and not due(con, [sid]):
        return None
    row = {"spotify_id": sid, "checked_at": db.now(), "reason": None, "of_title": None, "of_artist": None,
           "of_spotify_id": None, "own_rank": None, "other_rank": None, "error": None}
    try:
        t = con.execute("SELECT name, artists_json, isrc FROM spotify_tracks WHERE spotify_id=?", (sid,)).fetchone()
        if t is None:
            upsert_track(con, sp.track(sid))
            t = con.execute("SELECT name, artists_json, isrc FROM spotify_tracks WHERE spotify_id=?", (sid,)).fetchone()
        artists = json.loads(t["artists_json"] or "[]")
        s = signs(t["name"] or "", artists, t["isrc"], dz)
        row.update(reason=s["reason"], own_rank=s["own_rank"], other_rank=s["other_rank"])
        if s["reason"]:
            row.update(of_title=s["title"], of_artist=s["artist"])
            orig = find_original(sp, s["title"], s["artist"], sid) if s["artist"] else None
            if orig:
                upsert_track(con, orig)
                row["of_spotify_id"] = orig["id"]
    except Exception as e:  # noqa: BLE001 - a failed check is stored and tried again later
        log.warning("possible-cover check of %s failed: %r", sid, e)
        row.update(reason=None, of_title=None, of_artist=None, of_spotify_id=None, error=repr(e)[:300])
    con.execute(f"INSERT OR REPLACE INTO original_checks({', '.join(row)}) VALUES({', '.join('?' * len(row))})",
                tuple(row.values()))
    return row


def card_fields(con, sids: list[str], checking=()) -> dict[str, dict]:
    """Each track's `original` and `original_check` for its card (docs/api.md, Track card)."""
    rows = {}
    for i in range(0, len(sids), 500):
        ch = sids[i:i + 500]
        for r in con.execute(f"SELECT * FROM original_checks WHERE spotify_id IN ({','.join('?' * len(ch))})", ch):
            rows[r["spotify_id"]] = r
    out = {}
    for s in sids:
        r = rows.get(s)
        done = r is not None and r["error"] is None
        out[s] = {"original": {"title": r["of_title"], "artist": r["of_artist"], "id": r["of_spotify_id"],
                               "reason": r["reason"]} if done and r["reason"] else None,
                  "original_check": "done" if done else "pending" if s in checking else "unknown"}
    return out


# ---------------- Swaps ----------------

def record_swap(con, swap_id: str, cover: str, original: str, given_at: str) -> dict:
    """Store one Swap and take back the cover's Verdicts given up to it, which are kept in swap_retracted. Idempotent
    by id. -> {"id", "status": accepted | duplicate, "retracted": how many Verdicts}."""
    con.execute("BEGIN IMMEDIATE")
    try:
        n = 0
        new = con.execute("INSERT OR IGNORE INTO swaps(id, cover, original, given_at, received_at) VALUES(?,?,?,?,?)",
                          (swap_id, cover, original, given_at, db.now())).rowcount
        if new:
            for r in con.execute("SELECT * FROM verdicts WHERE spotify_id=? AND given_at<=?", (cover, given_at)).fetchall():
                con.execute("INSERT OR IGNORE INTO swap_retracted(verdict_id, swap_id, spotify_id, given_at, row) "
                            "VALUES(?,?,?,?,?)", (r["id"], swap_id, cover, r["given_at"], json.dumps(dict(r))))
            n = con.execute("DELETE FROM verdicts WHERE spotify_id=? AND given_at<=?", (cover, given_at)).rowcount
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return {"id": swap_id, "status": "accepted" if new else "duplicate", "retracted": n}


def swallowed(con, verdict_id: str, sid: str, given_at: str) -> bool:
    """True for a Verdict a Swap has taken back, sent again (by its id), and for a Verdict on a swapped cover given
    before the Swap that arrives after it: the server drops both."""
    return con.execute("SELECT 1 FROM swap_retracted WHERE verdict_id=?", (verdict_id,)).fetchone() is not None or \
        con.execute("SELECT 1 FROM swaps WHERE cover=? AND given_at>=? LIMIT 1", (sid, given_at)).fetchone() is not None


def pending(con) -> list[str]:
    """Swaps whose Spotify writes are still to make."""
    return [r[0] for r in con.execute("SELECT id FROM swaps WHERE effects='pending' ORDER BY received_at, rowid")]


def swapping(con) -> set[str]:
    """Covers whose Swap is still to be made on Spotify."""
    return {r[0] for r in con.execute("SELECT cover FROM swaps WHERE effects='pending'")}


def pending_writes(con) -> tuple[int, str | None]:
    """-> (Swaps with writes to make, the latest error among them), for the snapshot's `spotify`."""
    n = con.execute("SELECT COUNT(*) FROM swaps WHERE effects='pending'").fetchone()[0]
    err = con.execute("SELECT effects_error FROM swaps WHERE effects='pending' AND effects_error IS NOT NULL "
                      "ORDER BY received_at DESC LIMIT 1").fetchone()
    return n, err[0] if err else None


def apply_swaps(con, cfg: dict, sp, limit: int = 50) -> dict:
    """Make the Spotify writes of every pending Swap, oldest first. In each playlist the listener owns that holds the
    cover, the Original goes where the cover first was (unless its Song is there already) and the cover comes out. A
    liked cover's heart moves to the Original, unless the Original's Song has a Verdict below 3 stars, and the cover
    leaves To Rip if haystack put it there. Playlists the listener follows are left alone (the Web API can't edit
    them): the result names them. Each step adds before it removes and reads Spotify's state first, so a Swap that
    failed half-way is simply made again. -> counts."""
    rows = con.execute("SELECT * FROM swaps WHERE effects='pending' ORDER BY received_at, rowid LIMIT ?",
                       (limit,)).fetchall()
    out = {"swaps": len(rows), "applied": 0, "pending": 0}
    fresh: set[str] = set()                  # playlists checked against Spotify in this run
    for s in rows:
        try:
            res = _apply(con, sp, s, fresh)
            con.execute("UPDATE swaps SET effects='done', effects_at=?, effects_error=NULL, result=? WHERE id=?",
                        (db.now(), json.dumps(res), s["id"]))
            out["applied"] += 1
        except Exception as e:  # noqa: BLE001 - retried by the next run
            log.warning("swap %s (%s -> %s) failed: %r", s["id"], s["cover"], s["original"], e)
            con.execute("UPDATE swaps SET effects_error=? WHERE id=?", (repr(e)[:300], s["id"]))
            fresh.clear()                    # a half-made edit: read the playlists again
            out["pending"] += 1
    return out


def _apply(con, sp, s, fresh: set[str]) -> dict:
    from .sync import bot_remove, upsert_track
    cover, orig = s["cover"], s["original"]
    if not con.execute("SELECT 1 FROM spotify_tracks WHERE spotify_id=?", (orig,)).fetchone():
        upsert_track(con, sp.track(orig))
    res = {"playlists": [], "already_there": [], "followed": [], "heart": None, "to_rip": False}
    song = verdicts.songs(con)
    k = verdicts.song_of(song, orig)
    for p in con.execute("""SELECT DISTINCT p.id, p.mine, p.position FROM playlists p JOIN playlist_tracks t
                            ON t.playlist_id=p.id WHERE p.removed_at IS NULL AND t.spotify_id=? ORDER BY p.position""",
                         (cover,)).fetchall():
        pid = p["id"]
        if not p["mine"]:
            res["followed"].append(pid)
            continue
        if pid not in fresh:
            snap = sp.playlist(pid).get("snapshot_id")
            if snap != con.execute("SELECT tracks_snapshot FROM playlists WHERE id=?", (pid,)).fetchone()[0]:
                playlists.read_web(con, sp, pid, snap)
                con.execute("UPDATE playlists SET snapshot_id=? WHERE id=?", (snap, pid))
            fresh.add(pid)
        rows = [tuple(r) for r in con.execute("SELECT spotify_id, added_at FROM playlist_tracks WHERE playlist_id=? "
                                              "ORDER BY position", (pid,))]
        ids = [r[0] for r in rows]
        if cover not in ids:
            continue
        there = any(x != cover and (x == orig or verdicts.song_of(song, x) == k) for x in ids)
        snap = None if there else sp.add_items(pid, [orig], position=ids.index(cover))
        snap = sp.remove_items(pid, [cover], snap)
        kept, placed = [], there               # the stored songs follow, so the next sync has nothing to re-read
        for x, added in rows:
            if x == cover:
                if not placed:
                    kept.append((orig, db.now()))
                    placed = True
                continue
            kept.append((x, added))
        playlists.store(con, pid, snap, kept, "web")
        con.execute("UPDATE playlists SET snapshot_id=? WHERE id=?", (snap, pid))
        res["already_there" if there else "playlists"].append(pid)
    liked = sp.library_contains([cover, orig])
    if liked.get(cover):
        v = verdicts.current_by_song(con, [orig]).get(orig)
        now = db.now()
        if not liked.get(orig) and (v is None or v.keeps_heart):
            sp.library_save([orig])
            con.execute("""INSERT INTO likes(spotify_id, added_at, first_seen, last_seen) VALUES(?,?,?,?)
                           ON CONFLICT(spotify_id) DO UPDATE SET removed_at=NULL, last_seen=excluded.last_seen""",
                        (orig, now, now, now))
            res["heart"] = "moved"
        else:
            res["heart"] = "removed"
        sp.library_remove([cover])
        con.execute("UPDATE likes SET removed_at=? WHERE spotify_id=? AND removed_at IS NULL", (now, cover))
    to_rip = verdicts._bot_on(con, "to_rip", cover)          # only if haystack put it there
    if to_rip and bot_remove(con, sp, "to_rip", to_rip, [cover], "swap"):
        con.execute("UPDATE to_rip SET removed_at=? WHERE spotify_id=? AND removed_at IS NULL", (db.now(), cover))
        res["to_rip"] = True
    return res
