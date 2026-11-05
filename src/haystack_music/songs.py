"""The Song as a row of its own (ADR 0025): one row per song, by its ISRC where it has one, else by its names.

Every candidate, Spotify track and library file joins a Song (`song_id`). A row is found or made by the ISRC; a
track without one joins the Song of the same names: the first artist, the base title and the hard version markers
(a remix, a live take, an acoustic version stay apart; a remaster or an edit does not). When a name-keyed Song's
track later gets an ISRC, the Song takes that code, or folds into the Song that already has it (one UPDATE per
table, logged).

A remaster often gets an ISRC of its own, so two Songs can share their names. The Song key is what the harvest's
dupe check and `known.Known` try first; Known still runs the names rule (`known.same_song`) on every row, so a
remaster with a code of its own is still a song the listener has. A Verdict's Song (ADR 0021) stays the ISRC.

A library file's ISRC is its own tag, else the ISRC of the Spotify track its Spotify id tag names.
"""
from __future__ import annotations

import json
import logging

from . import db
from .textnorm import song_key, split_title, version_signature

log = logging.getLogger("haystack_music.songs")

TABLES = ("candidates", "spotify_tracks", "local_tracks")


def ensure(con) -> None:
    con.executescript("""
    CREATE TABLE IF NOT EXISTS songs (
        id INTEGER PRIMARY KEY,
        isrc TEXT UNIQUE,                    -- NULL: known by its names only
        name_key TEXT NOT NULL,              -- first artist | base title | hard version markers
        mbid TEXT,                           -- MusicBrainz recording id, when one is known (WP17)
        created_at TEXT
    );
    CREATE INDEX IF NOT EXISTS songs_name ON songs(name_key);
    """)
    for t in TABLES:
        db._add_column(con, t, "song_id", "INTEGER")
        con.execute(f"CREATE INDEX IF NOT EXISTS {t}_song_id ON {t}(song_id)")
    db._add_column(con, "candidates", "isrc_checked_at", "TEXT")   # Deezer asked; NULL isrc after it: none


def norm_isrc(x) -> str | None:
    x = (x or "").strip().upper()
    return x if len(x) == 12 and x.isalnum() else None


def name_key(artist: str | None, title: str | None) -> str:
    """The names a Song without an ISRC goes by: song_key plus the hard version markers, so 'Hells Bells - Live'
    is not 'Hells Bells', while 'Hells Bells (2009 Remaster)' is."""
    hard, _ = version_signature(split_title(title)[1])
    return song_key(artist, title) + "|" + ",".join(sorted(hard))


def find_or_make(con, isrc: str | None, nkey: str) -> int:
    """The Song for this ISRC (or these names, without one). A name-keyed Song of the same names takes a new ISRC."""
    if isrc:
        r = con.execute("SELECT id FROM songs WHERE isrc=?", (isrc,)).fetchone()
        if r:
            return r[0]
        r = con.execute("SELECT id FROM songs WHERE isrc IS NULL AND name_key=? ORDER BY id LIMIT 1",
                        (nkey,)).fetchone()
        if r:
            con.execute("UPDATE songs SET isrc=? WHERE id=?", (isrc, r[0]))
            return r[0]
    else:
        r = con.execute("SELECT id FROM songs WHERE name_key=? ORDER BY isrc IS NOT NULL, id LIMIT 1",
                        (nkey,)).fetchone()
        if r:
            return r[0]
    return con.execute("INSERT INTO songs(isrc, name_key, created_at) VALUES(?,?,?)",
                       (isrc, nkey, db.now())).lastrowid


def merge(con, from_id: int, into_id: int) -> None:
    """Fold one Song into another: every row that joined it moves, and the row goes."""
    for t in TABLES:
        con.execute(f"UPDATE {t} SET song_id=? WHERE song_id=?", (into_id, from_id))
    con.execute("DELETE FROM songs WHERE id=?", (from_id,))
    log.info("song %d folded into song %d (an ISRC joined them)", from_id, into_id)


# Each table's rows as (key, isrc, artist, title), for rows that have no Song yet or whose ISRC is not their Song's.
_TODO = {
    "candidates": """SELECT c.id, c.isrc, c.artist, c.title FROM candidates c LEFT JOIN songs s ON s.id=c.song_id
                     WHERE c.song_id IS NULL OR s.id IS NULL OR (c.isrc IS NOT NULL AND s.isrc IS NOT c.isrc)""",
    "spotify_tracks": """SELECT t.spotify_id, t.isrc, t.artists_json, t.name FROM spotify_tracks t
                         LEFT JOIN songs s ON s.id=t.song_id
                         WHERE t.song_id IS NULL OR s.id IS NULL OR (t.isrc IS NOT NULL AND s.isrc IS NOT t.isrc)""",
    "local_tracks": """SELECT l.id, COALESCE(l.isrc, st.isrc), l.artist_raw, l.title_raw FROM local_tracks l
                       LEFT JOIN spotify_tracks st ON st.spotify_id=l.spotify_id
                       LEFT JOIN songs s ON s.id=l.song_id
                       WHERE l.song_id IS NULL OR s.id IS NULL
                          OR (COALESCE(l.isrc, st.isrc) IS NOT NULL AND s.isrc IS NOT COALESCE(l.isrc, st.isrc))""",
}
_KEY = {"candidates": "id", "spotify_tracks": "spotify_id", "local_tracks": "id"}


def _artist(table: str, v) -> str:
    if table == "spotify_tracks":
        try:
            arts = json.loads(v or "[]")
        except ValueError:
            arts = []
        return arts[0] if arts else ""
    return v or ""


def assign(con) -> dict:
    """Give every row without a Song (or whose ISRC moved on) its Song. Cheap when there is nothing to do; the
    harvest, the sync, the batch and `songs backfill` run it. -> {table: rows assigned, 'merged': n}."""
    out = {t: 0 for t in TABLES} | {"merged": 0}
    own = not con.in_transaction                 # inside a caller's transaction, it is the caller's to commit
    if own:
        con.execute("BEGIN IMMEDIATE")
    try:
        for t in TABLES:
            for ref, isrc, artist, title in con.execute(_TODO[t]).fetchall():
                isrc = norm_isrc(isrc)
                nkey = name_key(_artist(t, artist), title)
                cur = con.execute(f"SELECT s.id, s.isrc FROM {t} x JOIN songs s ON s.id=x.song_id "
                                  f"WHERE x.{_KEY[t]}=?", (ref,)).fetchone()
                if cur and (isrc is None or cur[1] == isrc):
                    continue                     # a code that isn't one (or another spelling of its Song's): done
                if cur and cur[1] is None:
                    # a name-keyed Song's track has its code now: the Song takes it, or folds into the one with it
                    have = con.execute("SELECT id FROM songs WHERE isrc=?", (isrc,)).fetchone()
                    if have:
                        merge(con, cur[0], have[0]); out["merged"] += 1
                    else:
                        con.execute("UPDATE songs SET isrc=? WHERE id=?", (isrc, cur[0]))
                    out[t] += 1
                    continue
                sid = find_or_make(con, isrc, nkey)
                con.execute(f"UPDATE {t} SET song_id=? WHERE {_KEY[t]}=?", (sid, ref))
                out[t] += 1
        if own:
            con.execute("COMMIT")
    except BaseException:
        if own:
            con.execute("ROLLBACK")
        raise
    if any(out.values()):
        log.info("songs: %s", out)
    return out


def note_isrc(con, deezer_id: int, track: dict | None) -> str | None:
    """Keep the ISRC of a Deezer track object fetched for another reason, on the candidate that has none yet."""
    isrc = norm_isrc((track or {}).get("isrc"))
    if isrc and deezer_id:
        con.execute("UPDATE candidates SET isrc=?, isrc_checked_at=? WHERE deezer_id=? AND isrc IS NULL",
                    (isrc, db.now(), deezer_id))
    return isrc


def backfill_order(con, limit: int) -> list[tuple[int, int]]:
    """Candidates Deezer hasn't been asked for an ISRC: the pool first (best scored first), then the judged and
    offered ones, excluded rows last."""
    return [tuple(r) for r in con.execute("""
        SELECT id, deezer_id FROM candidates
        WHERE isrc IS NULL AND isrc_checked_at IS NULL AND deezer_id IS NOT NULL
        ORDER BY CASE state WHEN 'pool' THEN 0 WHEN 'excluded' THEN 2 ELSE 1 END, score IS NULL, score DESC, id
        LIMIT ?""", (limit,)).fetchall()]


def backfill(con, dz, limit: int = 300) -> dict:
    """Ask Deezer (/track/{id}, one throttled request each) for the ISRC of up to `limit` candidates, then join them
    to their Songs. A track Deezer has no code for, or no longer has, is not asked again; an error is."""
    c = dict(todo=0, found=0, none=0, errors=0)
    rows = backfill_order(con, limit) if limit > 0 else []
    c["todo"] = len(rows)
    for cid, did in rows:
        try:
            t = dz.track(did)
        except Exception as e:  # noqa: BLE001 - asked again next run
            c["errors"] += 1
            log.warning("isrc for deezer %s failed: %r", did, e)
            continue
        isrc = norm_isrc((t or {}).get("isrc"))
        con.execute("UPDATE candidates SET isrc=COALESCE(isrc, ?), isrc_checked_at=? WHERE id=?",
                    (isrc, db.now(), cid))
        c["found" if isrc else "none"] += 1
    return {**c, "songs": assign(con), "coverage": coverage(con)}


def coverage(con) -> dict:
    """Candidates with an ISRC and with a Song (pool and all), and Songs with two or more candidates: the
    re-releases the dupe check folds."""
    def one(sql: str) -> int:
        return con.execute(sql).fetchone()[0] or 0
    cols = {r[1] for r in con.execute("PRAGMA table_info(candidates)")}
    has_song = "song_id" in cols
    out = {}
    for name, where in (("pool", "state='pool'"), ("all", "1")):
        n = one(f"SELECT COUNT(*) FROM candidates WHERE {where}")
        k = one(f"SELECT COUNT(*) FROM candidates WHERE {where} AND isrc IS NOT NULL")
        s = one(f"SELECT COUNT(*) FROM candidates WHERE {where} AND song_id IS NOT NULL") if has_song else 0
        out[name] = {"n": n, "with_isrc": k, "isrc_share": round(k / n, 3) if n else None, "with_song": s}
    out["songs_with_2_or_more_candidates"] = one(
        "SELECT COUNT(*) FROM (SELECT song_id FROM candidates WHERE song_id IS NOT NULL GROUP BY song_id "
        "HAVING COUNT(*) > 1)") if has_song else 0
    return out


def show(con, ref: str) -> dict | None:
    """A Song by its ISRC or its id, with every candidate, Spotify track and library file that joined it."""
    ref = ref.strip()
    r = (con.execute("SELECT * FROM songs WHERE id=?", (int(ref),)).fetchone() if ref.isdigit()
         else con.execute("SELECT * FROM songs WHERE isrc=?", (ref.upper(),)).fetchone())
    if not r:
        return None
    sid = r["id"]
    return {"song": dict(r),
            "candidates": [dict(x) for x in con.execute(
                "SELECT id, deezer_id, artist, title, isrc, state, source FROM candidates WHERE song_id=? ORDER BY id",
                (sid,))],
            "spotify_tracks": [dict(x) for x in con.execute(
                "SELECT spotify_id, name, artists_json, album, isrc FROM spotify_tracks WHERE song_id=?", (sid,))],
            "library_files": [dict(x) for x in con.execute(
                "SELECT id, path, artist_raw, title_raw, isrc, spotify_id FROM local_tracks WHERE song_id=? "
                "AND present=1", (sid,))]}
