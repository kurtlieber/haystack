"""What Deezer publishes about a track (ADR 0017): its album's label, release date, record type and fans, the track's
Deezer rank and its artist's fans. The re-ranker learns from them (rerank.py) and the track card shows the label.

The harvest stores a new candidate's album id and rank from the track object it already has (no request).
backfill() fetches the rest through the shared Deezer throttle: /album/{id} once per album, /artist/{id} once per
artist, and /track/{id} first for a track whose album id isn't known (candidates from before 0.6.0, liked songs)."""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone

from . import db
from .evaluate import _chunks
from .textnorm import key

log = logging.getLogger("haystack_music.metadata")

_JUDGED = "(c.state IN ('offered','liked','negative','retired') OR c.id IN (SELECT candidate_id FROM calibrations))"


def _album_todo(col: str) -> str:
    """SQL: the row's album isn't known yet (no album id, or no album row for it)."""
    return f"({col} IS NULL OR ({col} > 0 AND {col} NOT IN (SELECT album_id FROM deezer_albums)))"


def _has_album(col: str) -> str:
    """SQL: the row's album is known and Deezer gave it a label or a date."""
    return (f"{col} > 0 AND {col} IN (SELECT album_id FROM deezer_albums "
            f"WHERE label IS NOT NULL OR release_date IS NOT NULL)")


def store_album(con, album_id: int, a: dict | None) -> None:
    a = a or {}
    genres = [g.get("name") for g in ((a.get("genres") or {}).get("data") or []) if g.get("name")]
    date = a.get("release_date")
    con.execute("""INSERT OR REPLACE INTO deezer_albums(album_id, label, release_date, record_type, genres_json, fans,
                   nb_tracks, fetched_at) VALUES(?,?,?,?,?,?,?,?)""",
                (album_id, a.get("label") or None, date if date and not date.startswith("0000") else None,
                 a.get("record_type"), json.dumps(genres) if a else None, a.get("fans"), a.get("nb_tracks"), db.now()))


def _store_artist(con, artist_id: int, a: dict | None) -> None:
    con.execute("INSERT OR REPLACE INTO deezer_artist_fans(artist_id, nb_fan, fetched_at) VALUES(?,?,?)",
                (artist_id, (a or {}).get("nb_fan"), db.now()))


def _todo(con, limit: int) -> list[tuple]:
    """-> [(table, row key, deezer_id, album_id, artist_id)]: the judged candidates first (they train the
    re-ranker), then the liked songs mapped to Deezer, then the pool, oldest first."""
    if limit <= 0:
        return []
    artist_todo = ("(c.deezer_artist_id IS NOT NULL"
                   " AND c.deezer_artist_id NOT IN (SELECT artist_id FROM deezer_artist_fans))")
    rows = [("candidates",) + tuple(r) for r in con.execute(f"""
        SELECT c.id, c.deezer_id, c.album_id, c.deezer_artist_id FROM candidates c
        WHERE c.vector_status='ok' AND c.state != 'excluded' AND ({_album_todo('c.album_id')}
              OR {artist_todo})
        ORDER BY {_JUDGED} DESC, c.harvested_at, c.id LIMIT ?""", (limit,))]
    judged = {r[0] for r in con.execute(f"SELECT c.id FROM candidates c WHERE {_JUDGED}")}
    first = [r for r in rows if r[1] in judged]
    rest = [r for r in rows if r[1] not in judged]
    liked = [("liked_map",) + tuple(r) + (None,) for r in con.execute(f"""
        SELECT m.spotify_id, m.deezer_id, m.album_id FROM liked_map m JOIN likes l ON l.spotify_id=m.spotify_id
        WHERE l.removed_at IS NULL AND m.deezer_id IS NOT NULL AND {_album_todo('m.album_id')}
        ORDER BY l.added_at DESC LIMIT ?""", (limit,))]
    return (first + liked + rest)[:limit]


def backfill(con, dz, limit: int = 150) -> dict:
    """Fetch the metadata of up to `limit` tracks (judged candidates, liked songs, then the pool, oldest first).
    One bad answer never stops the rest: it counts as an error and is tried again next time."""
    c = dict(todo=0, tracks=0, albums=0, artists=0, gone=0, errors=0)
    rows = _todo(con, limit)
    c["todo"] = len(rows)
    albums_seen = {r[0] for r in con.execute("SELECT album_id FROM deezer_albums")}
    artists_seen = {r[0] for r in con.execute("SELECT artist_id FROM deezer_artist_fans")}
    for table, ref, did, album_id, artist_id in rows:
        try:
            if album_id is None:
                t = dz.track(did)
                c["tracks"] += 1
                album_id = int(((t or {}).get("album") or {}).get("id") or 0)   # 0: Deezer has no such track now
                if not t:
                    c["gone"] += 1
                if table == "candidates":
                    con.execute("UPDATE candidates SET album_id=?, deezer_rank=COALESCE(?, deezer_rank), "
                                "deezer_artist_id=COALESCE(deezer_artist_id, ?) WHERE id=?",
                                (album_id, (t or {}).get("rank"), ((t or {}).get("artist") or {}).get("id"), ref))
                    artist_id = artist_id or ((t or {}).get("artist") or {}).get("id")
                else:
                    con.execute("UPDATE liked_map SET album_id=? WHERE spotify_id=?", (album_id, ref))
            if album_id and album_id not in albums_seen:
                store_album(con, album_id, dz.album(album_id))
                albums_seen.add(album_id); c["albums"] += 1
            if artist_id and artist_id not in artists_seen:
                _store_artist(con, artist_id, dz.artist(artist_id))
                artists_seen.add(artist_id); c["artists"] += 1
        except Exception as e:  # noqa: BLE001
            c["errors"] += 1
            log.warning("metadata for deezer %s failed: %r", did, e)
    return {**c, "coverage": coverage(con)}


def coverage(con) -> dict:
    """The share of the pool, the judged candidates and the liked songs mapped to Deezer whose album is known."""
    def share(total_sql: str, done_sql: str) -> dict:
        n = con.execute(total_sql).fetchone()[0]
        k = con.execute(done_sql).fetchone()[0]
        return {"n": n, "with_album": k, "share": round(k / n, 3) if n else None}
    pool = "FROM candidates WHERE state='pool' AND vector_status='ok'"
    judged = f"FROM candidates c WHERE c.vector_status='ok' AND {_JUDGED}"
    liked = ("FROM liked_map m JOIN likes l ON l.spotify_id=m.spotify_id "
             "WHERE l.removed_at IS NULL AND m.deezer_id IS NOT NULL")
    return {"pool": share(f"SELECT COUNT(*) {pool}", f"SELECT COUNT(*) {pool} AND {_has_album('album_id')}"),
            "judged": share(f"SELECT COUNT(*) {judged}", f"SELECT COUNT(*) {judged} AND {_has_album('c.album_id')}"),
            "liked": share(f"SELECT COUNT(*) {liked}", f"SELECT COUNT(*) {liked} AND {_has_album('m.album_id')}")}


# ---------------- lookups ----------------

def _age_years(release_date: str | None, now: datetime | None = None) -> float | None:
    try:
        d = datetime.strptime(release_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return max(0.0, ((now or datetime.now(timezone.utc)) - d).days / 365.25)


def for_candidates(con, ids, now: datetime | None = None) -> dict[int, dict]:
    """candidate id -> {rank, fans, age, compilation, label (its comparison key)}; None for a fact Deezer didn't
    give. {} on a database from before 0.6.0."""
    out: dict[int, dict] = {}
    try:
        for ch in _chunks(sorted(set(ids))):
            for r in con.execute(f"""SELECT c.id, c.deezer_rank, f.nb_fan, a.release_date, a.record_type, a.label,
                                     a.album_id FROM candidates c
                                     LEFT JOIN deezer_albums a ON a.album_id=c.album_id
                                     LEFT JOIN deezer_artist_fans f ON f.artist_id=c.deezer_artist_id
                                     WHERE c.id IN ({','.join('?' * len(ch))})""", ch):
                out[r[0]] = {"rank": r[1], "fans": r[2], "age": _age_years(r[3], now),
                             "compilation": None if r[6] is None or r[4] is None else r[4] == "compile",
                             "label": key(r[5]) or None}
    except sqlite3.OperationalError:            # a database from before 0.6.0, opened read-only
        return {}
    return out


def labels_of_tracks(con, sids) -> dict[str, str]:
    """spotify id -> its label's comparison key, from a candidate or a liked song with an album row."""
    out: dict[str, str] = {}
    try:
        for ch in _chunks(sorted(set(sids))):
            ph = ",".join("?" * len(ch))
            for r in con.execute(f"""SELECT x.spotify_id, a.label FROM (
                                       SELECT spotify_id, album_id FROM candidates WHERE spotify_id IN ({ph})
                                       UNION ALL SELECT spotify_id, album_id FROM liked_map
                                       WHERE spotify_id IN ({ph})) x
                                     JOIN deezer_albums a ON a.album_id=x.album_id WHERE a.label IS NOT NULL""",
                                 ch + ch):
                out.setdefault(r[0], key(r[1]))
    except sqlite3.OperationalError:            # a database from before 0.6.0, opened read-only
        return {}
    return {s: k for s, k in out.items() if k}


def owner_labels(con, owners: list[str]) -> list[str | None]:
    """The label of each positive's owner (vectors.positives: like:, discovery:, rated:, local:, manual:)."""
    sid_of: dict[str, str] = {}
    for o in owners:
        kind, ref = o.split(":", 1)
        if kind in ("like", "discovery", "rated"):
            sid_of[o] = ref
    local = [int(o.split(":", 1)[1]) for o in owners if o.startswith("local:")]
    for ch in _chunks(local):
        ph = ",".join("?" * len(ch))
        for r in con.execute(f"SELECT local_id, spotify_id FROM liked_map WHERE local_id IN ({ph}) UNION ALL "
                             f"SELECT id, spotify_id FROM local_tracks WHERE spotify_id IS NOT NULL AND id IN ({ph})",
                             ch + ch):
            sid_of.setdefault(f"local:{r[0]}", r[1])
    by_sid = labels_of_tracks(con, sid_of.values())
    manual = [int(o.split(":", 1)[1]) for o in owners if o.startswith("manual:")]
    by_cand = {f"manual:{c}": m["label"] for c, m in for_candidates(con, manual).items()}
    return [by_cand.get(o) if o.startswith("manual:") else by_sid.get(sid_of.get(o, "")) for o in owners]


def labels_for_cards(con, dids) -> dict[int, str]:
    """Deezer track id -> its label as Deezer writes it (the track card's `label`)."""
    out: dict[int, str] = {}
    try:
        for ch in _chunks(sorted({d for d in dids if d is not None})):
            ph = ",".join("?" * len(ch))
            for r in con.execute(f"""SELECT x.deezer_id, a.label FROM (
                                       SELECT deezer_id, album_id FROM candidates WHERE deezer_id IN ({ph})
                                       UNION ALL SELECT deezer_id, album_id FROM liked_map WHERE deezer_id IN ({ph})) x
                                     JOIN deezer_albums a ON a.album_id=x.album_id WHERE a.label IS NOT NULL""",
                                 ch + ch):
                out.setdefault(r[0], r[1])
    except sqlite3.OperationalError:            # a database from before 0.6.0, opened read-only
        return {}
    return out
