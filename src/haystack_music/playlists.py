"""The listener's Spotify playlists, for needle's Playlist review: the picker's wall of covers and the songs it
plays.

The server lists every playlist in the listener's Spotify library, the ones they own and the ones they follow,
except its own (Discoveries and To Rip), and keeps each one's songs, re-read only when the playlist's snapshot_id
changes. The Web API lists the songs of the playlists they own. For a Followed playlist the 2026 dev mode answers 403, so needle's
engine reads its songs through librespot and sends them here (ADR 0008). Nothing here writes to Spotify."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter

from . import db, verdicts

log = logging.getLogger("haystack_music.playlists")
PLAYLIST_ID = re.compile(r"^[0-9A-Za-z]{22}$")
MAX_TRACKS = 10000          # Spotify's own limit for one playlist
ARTISTS_SHOWN = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS playlists (
    id TEXT PRIMARY KEY,
    name TEXT, owner_id TEXT, owner_name TEXT,
    mine INTEGER NOT NULL DEFAULT 0,     -- the listener owns it (else it is a Followed playlist)
    image_url TEXT,
    total INTEGER,                       -- Spotify's own song count
    snapshot_id TEXT,                    -- the current version, from /me/playlists
    position INTEGER,                    -- its place in the listener's library
    tracks_snapshot TEXT,                -- the version the stored songs belong to
    tracks_source TEXT,                  -- web | engine (needle's engine read them: ADR 0008)
    tracks_at TEXT,
    refused_snapshot TEXT,               -- the Web API refused this version's songs (dev mode: not the listener's)
    first_seen TEXT, last_seen TEXT,
    removed_at TEXT                      -- no longer in the library
);
CREATE TABLE IF NOT EXISTS playlist_tracks (
    playlist_id TEXT NOT NULL,
    position INTEGER NOT NULL,           -- read from the Web API: its place on Spotify, counting the local files left out
    spotify_id TEXT NOT NULL,
    added_at TEXT,
    PRIMARY KEY (playlist_id, position)
);
CREATE INDEX IF NOT EXISTS playlist_tracks_track ON playlist_tracks(spotify_id);
"""


def ensure(con) -> None:
    con.executescript(SCHEMA)


def cover_url(images) -> str | None:
    """The smallest image at least 300 px wide (a mosaic comes in 640, 300 and 60 px), else the first one: a
    playlist's own picture comes in one size, with no width given."""
    imgs = [i for i in (images or []) if i and i.get("url")]
    if not imgs:
        return None
    wide = sorted((i for i in imgs if (i.get("width") or 0) >= 300), key=lambda i: i["width"])
    return (wide[0] if wide else imgs[0])["url"]


def _reserved(con, cfg: dict, me: str):
    """The server's own playlists, which have a Review of their own (Discoveries) or are a to-do list (To Rip)."""
    ids = {v for v in (db.kv_get(con, "playlist_discoveries"), db.kv_get(con, "playlist_to_rip")) if v}
    from .sync import slots
    names = {name for _, name in slots(cfg)}

    def reserved(p: dict) -> bool:
        return p["id"] in ids or (not ids and (p.get("owner") or {}).get("id") == me and p.get("name") in names)
    return reserved


def sync(con, sp, cfg: dict, me: str | None = None) -> dict:
    """List the listener's playlists, then re-read the songs of each one whose snapshot changed. One call when nothing
    changed. -> counts."""
    me = me or db.kv_get(con, "spotify_user") or sp.me().get("id")
    reserved = _reserved(con, cfg, me)
    ts = db.now()
    seen = []
    for p in sp.my_playlists():
        if not p or not p.get("id") or reserved(p):
            continue
        owner = p.get("owner") or {}
        total = (p.get("items") or p.get("tracks") or {}).get("total")
        con.execute("""INSERT INTO playlists(id, name, owner_id, owner_name, mine, image_url, total, snapshot_id, position,
                       first_seen, last_seen) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET name=excluded.name, owner_id=excluded.owner_id,
                       owner_name=excluded.owner_name, mine=excluded.mine, image_url=excluded.image_url,
                       total=excluded.total, snapshot_id=excluded.snapshot_id, position=excluded.position,
                       last_seen=excluded.last_seen, removed_at=NULL""",
                    (p["id"], p.get("name"), owner.get("id"), owner.get("display_name"), int(owner.get("id") == me),
                     cover_url(p.get("images")), total, p.get("snapshot_id"), len(seen), ts, ts))
        seen.append(p["id"])
    gone = con.execute(f"""UPDATE playlists SET removed_at=? WHERE removed_at IS NULL
                           AND id NOT IN ({','.join('?' * len(seen)) or "''"})""", (ts, *seen)).rowcount
    out = {"playlists": len(seen), "removed": gone, "read": 0, "unchanged": 0, "for_engine": 0, "errors": 0}
    for r in con.execute("SELECT id, snapshot_id, tracks_snapshot, refused_snapshot FROM playlists WHERE removed_at IS NULL "
                         "ORDER BY position").fetchall():
        if r["tracks_snapshot"] == r["snapshot_id"]:
            out["unchanged"] += 1
        elif r["refused_snapshot"] == r["snapshot_id"]:
            out["for_engine"] += 1
        else:
            try:
                out["read" if read_web(con, sp, r["id"], r["snapshot_id"]) is not None else "for_engine"] += 1
            except Exception as e:  # noqa: BLE001 - one playlist must not stop the rest
                log.warning("reading playlist %s failed: %r", r["id"], e)
                out["errors"] += 1
    db.kv_set(con, "playlists_synced_at", ts)
    return out


def read_web(con, sp, pid: str, snapshot: str | None) -> int | None:
    """Store a playlist's songs from the Web API, with their track metadata. -> the number stored, or None when
    Spotify refuses (a Followed playlist in dev mode): needle's engine reads it instead."""
    from .spotify import SpotifyError
    from .sync import upsert_track
    try:
        items = list(sp.playlist_tracks(pid))
    except SpotifyError as e:
        if e.status in (403, 404):
            con.execute("UPDATE playlists SET refused_snapshot=? WHERE id=?", (snapshot, pid))
            log.info("Spotify won't list playlist %s (%s): left to needle's engine", pid, e.status)
            return None
        raise
    for _pos, _added, t in items:
        upsert_track(con, t)
    return store(con, pid, snapshot, [(t["id"], added, pos) for pos, added, t in items], "web")


def store(con, pid: str, snapshot: str | None, rows: list[tuple], source: str) -> int:
    """Replace a playlist's songs with this version of them, in order: [(spotify id, added_at)], or from the Web API
    [(spotify id, added_at, position)], where position is the song's place on Spotify, which counts the local files
    and episodes left out (a Swap adds at it)."""
    con.execute("BEGIN")
    try:
        con.execute("DELETE FROM playlist_tracks WHERE playlist_id=?", (pid,))
        con.executemany("INSERT INTO playlist_tracks(playlist_id, position, spotify_id, added_at) VALUES(?,?,?,?)",
                        [(pid, r[2] if len(r) > 2 else i, r[0], r[1]) for i, r in enumerate(rows)])
        con.execute("UPDATE playlists SET tracks_snapshot=?, tracks_source=?, tracks_at=? WHERE id=?",
                    (snapshot, source, db.now(), pid))
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return len(rows)


# ---------------- songs read by needle's engine (ADR 0008) ----------------

def _text(v, what: str, limit: int = 500, required: bool = False):
    if v is None and not required:
        return None
    if not isinstance(v, str) or not v or len(v) > limit:
        raise ValueError(f"{what} must be text (at most {limit} characters)")
    return v


def _image(v, what: str):
    v = _text(v, what, 300)
    if v is not None and not v.startswith("https://"):
        raise ValueError(f"{what} must be an https URL")
    return v


def parse_engine_track(t: dict) -> dict:
    """One song as needle's engine read it -> a Web API-shaped track object for sync.upsert_track. Raises
    ValueError with a reason."""
    if not isinstance(t, dict):
        raise ValueError("a track is an object")
    if not isinstance(t.get("id"), str) or not verdicts.SPOTIFY_ID.match(t["id"]):
        raise ValueError("id must be a 22-character Spotify track id")
    artists = t.get("artists")
    if not isinstance(artists, list) or not all(isinstance(a, str) and 0 < len(a) <= 300 for a in artists):
        raise ValueError("artists must be a list of names")
    dur = t.get("duration_ms")
    if dur is not None and (isinstance(dur, bool) or not isinstance(dur, int) or not 0 < dur < 24 * 3600 * 1000):
        raise ValueError("duration_ms must be a positive whole number")
    album_type = t.get("album_type")
    if album_type not in (None, "album", "single", "compilation"):
        raise ValueError("album_type must be album, single or compilation")
    release = _text(t.get("release_date"), "release_date", 10)
    if release is not None and not re.match(r"^\d{4}(-\d\d(-\d\d)?)?$", release):
        raise ValueError("release_date must be YYYY, YYYY-MM or YYYY-MM-DD")
    isrc = _text(t.get("isrc"), "isrc", 20)
    images = [{"url": u, "width": w} for u, w in ((_image(t.get("cover"), "cover"), 640),
                                                   (_image(t.get("cover_small"), "cover_small"), 64)) if u]
    return {"id": t["id"], "name": _text(t.get("title"), "title", required=True),
            "artists": [{"name": a} for a in artists], "duration_ms": dur,
            "external_ids": {"isrc": isrc} if isrc else {},
            "album": {"name": _text(t.get("album"), "album"), "album_type": album_type, "release_date": release,
                      "images": images}}


def from_engine(con, pid: str, body) -> dict:
    """Store the songs needle's engine read for a playlist the server can't list. Raises LookupError for a playlist
    the server doesn't know, ValueError for a malformed body (nothing is stored then)."""
    from .sync import upsert_track
    if not con.execute("SELECT 1 FROM playlists WHERE id=? AND removed_at IS NULL", (pid,)).fetchone():
        raise LookupError(pid)
    if not isinstance(body, dict):
        raise ValueError("expected {\"snapshot_id\": ..., \"tracks\": [...]}")
    snapshot = _text(body.get("snapshot_id"), "snapshot_id", 100, required=True)
    tracks = body.get("tracks")
    if not isinstance(tracks, list) or len(tracks) > MAX_TRACKS:
        raise ValueError(f"tracks must be a list (at most {MAX_TRACKS})")
    parsed = []
    for i, t in enumerate(tracks):
        try:
            parsed.append(parse_engine_track(t))
        except ValueError as e:
            raise ValueError(f"track {i}: {e}") from None
    for t in parsed:
        upsert_track(con, t)
    n = store(con, pid, snapshot, [(t["id"], None) for t in parsed], "engine")
    return {"stored": n, "snapshot_id": snapshot}


# ---------------- what needle shows ----------------

def _tracks(con) -> dict[str, list[str]]:
    """playlist id -> its songs in order; a song listed twice counts once."""
    out: dict[str, list[str]] = {}
    seen: dict[str, set] = {}
    for r in con.execute("""SELECT t.playlist_id, t.spotify_id FROM playlist_tracks t JOIN playlists p ON p.id=t.playlist_id
                            WHERE p.removed_at IS NULL ORDER BY t.playlist_id, t.position"""):
        s = seen.setdefault(r[0], set())
        if r[1] not in s:
            s.add(r[1])
            out.setdefault(r[0], []).append(r[1])
    return out


def _first_artists(con, sids) -> dict[str, str]:
    out = {}
    sids = list(sids)
    for i in range(0, len(sids), 500):
        ch = sids[i:i + 500]
        for r in con.execute(f"SELECT spotify_id, artists_json FROM spotify_tracks WHERE spotify_id IN "
                             f"({','.join('?' * len(ch))})", ch):
            a = json.loads(r[1] or "[]")
            if a:
                out[r[0]] = a[0]
    return out


def _summary(r, ids: list[str] | None, artists: dict[str, str]) -> dict:
    current = r["tracks_snapshot"] is not None and r["tracks_snapshot"] == r["snapshot_id"]
    refused = r["refused_snapshot"] is not None and r["refused_snapshot"] == r["snapshot_id"]
    top = Counter(artists[s] for s in ids or [] if s in artists).most_common(ARTISTS_SHOWN)
    return {"id": r["id"], "name": r["name"], "cover": r["image_url"], "mine": bool(r["mine"]),
            "owner": r["owner_name"] or r["owner_id"], "total": r["total"], "snapshot_id": r["snapshot_id"],
            "tracks": ids if r["tracks_snapshot"] is not None else None, "current": current,
            # who reads its songs: the server (Web API), or the client's engine when Spotify refuses it (ADR 0008)
            "reader": "engine" if refused or (r["tracks_source"] == "engine" and not current) else "server",
            "artists": [a for a, _n in top]}


def summaries(con) -> dict:
    """The picker's data, for /v1/state: every playlist with its songs' ids (no cards) and the current verdict of
    each of those songs that has one, so needle can count progress for any selection, offline. A verdict is the
    Song's: one on another release of a song counts (ADR 0021)."""
    tracks = _tracks(con)
    every = set().union(*tracks.values()) if tracks else set()
    artists = _first_artists(con, every)
    cur = verdicts.current_by_song(con, every)
    rows = con.execute("SELECT * FROM playlists WHERE removed_at IS NULL ORDER BY position").fetchall()
    return {"synced_at": db.kv_get(con, "playlists_synced_at"),
            "items": [_summary(r, tracks.get(r["id"], []), artists) for r in rows],
            "verdicts": {s: {"kind": cur[s].kind, "stars": cur[s].stars, "given_at": cur[s].given_at}
                         for s in sorted(every) if s in cur}}


def _in_library(con, sids: list[str]) -> set[str]:
    """Songs certainly in the music library already: by the Spotify-ID tag, the ISRC, or a liked song's mapping.
    (The To Rip check also tries a fuzzy match, so this can miss a few; it never claims one that isn't.)"""
    have = {r[0] for r in con.execute("SELECT spotify_id FROM local_tracks WHERE present=1 AND spotify_id IS NOT NULL")}
    have |= {r[0] for r in con.execute("SELECT spotify_id FROM liked_map WHERE local_id IS NOT NULL")}
    isrcs = {r[0] for r in con.execute("SELECT isrc FROM local_tracks WHERE present=1 AND isrc IS NOT NULL")}
    out = {s for s in sids if s in have}
    rest = [s for s in sids if s not in out]
    for i in range(0, len(rest), 500):
        ch = rest[i:i + 500]
        for r in con.execute(f"SELECT spotify_id, isrc FROM spotify_tracks WHERE spotify_id IN ({','.join('?' * len(ch))})", ch):
            if r[1] and r[1] in isrcs:
                out.add(r[0])
    return out


def cards(con, cards_builder, pid: str) -> dict | None:
    """One playlist with a full track card for each of its songs, in its order (/v1/playlists/{id}). Each card also
    says which of the listener's playlists the song is on and whether it is in Liked Songs, both by its Song: any
    release of it counts (ADR 0021), and whether the track is in the music library already (its own: a file with its
    Spotify id or its ISRC, or matched to its like). None for a playlist the server doesn't know."""
    r = con.execute("SELECT * FROM playlists WHERE id=? AND removed_at IS NULL", (pid,)).fetchone()
    if r is None:
        return None
    tracks = _tracks(con)
    sids = tracks.get(pid, [])
    order = [x[0] for x in con.execute("SELECT id FROM playlists WHERE removed_at IS NULL ORDER BY position")]
    song = verdicts.songs(con)
    on: dict[str, list[str]] = {}
    for p in order:
        for k in dict.fromkeys(verdicts.song_of(song, s) for s in tracks.get(p, [])):
            on.setdefault(k, []).append(p)
    liked = {verdicts.song_of(song, x[0]) for x in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL")}
    lib = _in_library(con, sids)
    built = cards_builder.build(sids)
    for c in built:
        k = verdicts.song_of(song, c["id"])
        c["playlists"] = on.get(k, [pid])
        c["liked"] = k in liked
        c["in_library"] = c["id"] in lib
    return {**_summary(r, sids, _first_artists(con, sids)), "items": built}
