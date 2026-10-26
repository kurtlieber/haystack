"""Empty the Discoveries playlist before a batch.

Every track is removed. Outcomes are committed before the Spotify delete, so the hourly sync cannot see a
track missing from the playlist whose offer is still active (that path sets outcome 'manual_removed', which
the negative penalty treats as a rejection).

A track in Liked Songs keeps that positive: the offer is closed as liked and no verdict is recorded.
Any other track on the playlist for at least batch.clear_min_age_hours (default 24) splits three ways on the play
log (ADR 0023; the clear polls plays first, so the log reaches the moment of the clear):

  played        a play since it was added (Spotify's or needle's): a Pass (verdict source 'playlist', effects
                skipped: the clear makes its own Spotify write) and the offer is closed as negative
  never_played  no play, and no gap in the log (play_gaps) over its time on the playlist: closed as unplayed, back
                in the pool with no negative, and unplayed_expiries goes up as at the 14-day expiry; it retires at
                batch.max_unplayed_expiries
  unknown       no play, but a gap overlaps: closed as unplayed without counting

A track the listener already gave a verdict (needle, the CLI) keeps it: its heart may simply not be written yet.
Younger tracks follow batch.clear_recent:

  remove  take them off and close the offer as unplayed (back in the pool, no negative). The weekly unit pins this.
  keep    leave them in Discoveries until a later run

--no-penalty closes every removed offer except likes the same way: unplayed, back in the pool, no 'no' verdict.
Logs are counts only.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from . import config, db, feedback, verdicts
from .spotify import Spotify

log = logging.getLogger("haystack_music.clear")

RECENT_MODES = ("keep", "remove")

PENDING_SCHEMA = """
CREATE TABLE IF NOT EXISTS clear_pending (
    playlist_id TEXT NOT NULL,
    spotify_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    verdict TEXT,
    candidate_id INTEGER,
    added_at TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (playlist_id, spotify_id)
);
"""

_REMOVED_REASON = {"liked": "liked", "unliked": "clear_no", "recent": "unplayed", "unplayed": "unplayed",
                   "never_played": "never_played", "unknown": "unplayed"}
_UNPLAYED = ("recent", "unplayed", "never_played", "unknown")      # the reasons that close as unplayed
_COUNTS = ("feedback", "unlinked", "liked", "no", "recent_removed", "unplayed", "never_played", "unknown", "retired")


def ensure(con) -> None:
    con.execute(PENDING_SCHEMA)


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _options(cfg: dict, min_age_hours: float | None, recent: str | None,
             no_penalty: bool | None) -> tuple[float, str, bool]:
    if min_age_hours is None:
        min_age_hours = cfg["batch"]["clear_min_age_hours"]
    if recent is None:
        recent = cfg["batch"]["clear_recent"]
    if no_penalty is None:
        no_penalty = bool(cfg["batch"].get("clear_no_penalty", False))
    min_age_hours = float(min_age_hours)
    if min_age_hours < 0:
        raise ValueError("clear_min_age_hours must be >= 0")
    if recent not in RECENT_MODES:
        raise ValueError(f"clear_recent must be one of {', '.join(RECENT_MODES)}")
    return min_age_hours, recent, bool(no_penalty)


def _summary(min_age: float, recent: str, dry_run: bool, no_penalty: bool) -> dict:
    return {
        "dry_run": dry_run,
        "min_age_hours": min_age,
        "recent": recent,
        "no_penalty": no_penalty,
        "playlist_found": False,
        "items": 0,
        "tracks": 0,
        "liked": 0,
        "no": 0,
        "unplayed": 0,
        "never_played": 0,
        "unknown": 0,
        "retired": 0,
        "recent_removed": 0,
        "recent_kept": 0,
        "removed": 0,
        "would_remove": 0,
        "feedback": 0,
        "unlinked": 0,
        "reconciled": 0,
        "age_unknown": 0,
    }


def _candidate(con, sid: str, item: dict):
    row = con.execute("""SELECT c.* FROM offers o JOIN candidates c ON c.id=o.candidate_id
                         WHERE o.spotify_id=? ORDER BY (o.removed_at IS NULL) DESC, o.id DESC LIMIT 1""",
                      (sid,)).fetchone()
    if row:
        return row
    row = con.execute("SELECT * FROM candidates WHERE spotify_id=? ORDER BY id DESC LIMIT 1", (sid,)).fetchone()
    if row:
        return row
    artist = next((a for a in (item.get("artists") or []) if a), None)
    title = item.get("name")
    if artist and title:
        found = feedback.find_candidate(con, artist, title)
        if found:
            return found
    t = con.execute("SELECT artists_json, name FROM spotify_tracks WHERE spotify_id=?", (sid,)).fetchone()
    if t and t["name"]:
        arts = json.loads(t["artists_json"] or "[]")
        if arts:
            return feedback.find_candidate(con, arts[0], t["name"])
    return None


def _added_at(con, sid: str, item_added: str | None) -> str | None:
    if _parse(item_added):
        return item_added
    for q in (
        "SELECT MIN(added_at) FROM offers WHERE spotify_id=? AND removed_at IS NULL",
        "SELECT MIN(added_at) FROM playlist_adds WHERE playlist='discoveries' AND spotify_id=? AND removed_at IS NULL",
    ):
        r = con.execute(q, (sid,)).fetchone()
        if r and _parse(r[0]):
            return r[0]
    return None


def _collapse(items: list[dict]) -> list[dict]:
    """One row per Spotify id. Age uses the earliest added_at we were given."""
    by: dict[str, dict] = {}
    for it in items:
        sid = it.get("spotify_id")
        if not sid:
            continue
        cur = by.get(sid)
        if cur is None:
            by[sid] = dict(it)
            continue
        a, b = _parse(cur.get("added_at")), _parse(it.get("added_at"))
        if b and (a is None or b < a):
            cur["added_at"] = it.get("added_at")
        if not cur.get("name") and it.get("name"):
            cur["name"] = it.get("name")
        if not cur.get("artists") and it.get("artists"):
            cur["artists"] = it.get("artists")
    return list(by.values())


def _heard(con, sid: str, added_at: str | None, now: datetime) -> str:
    """'played' when the play log has a play of the track since it was added (Spotify's or needle's), 'never' when it
    has none and no gap (play_gaps) overlaps its time on the playlist, 'unknown' when one does. With no known
    added_at, any play counts, and no play is unknown."""
    if added_at is None:
        return "played" if con.execute("SELECT 1 FROM plays WHERE spotify_id=? LIMIT 1", (sid,)).fetchone() \
            else "unknown"
    if con.execute("SELECT 1 FROM plays WHERE spotify_id=? AND played_at>=? LIMIT 1", (sid, added_at)).fetchone():
        return "played"
    added_ms, now_ms = int(_parse(added_at).timestamp() * 1000), int(now.timestamp() * 1000)
    gap = con.execute("SELECT 1 FROM play_gaps WHERE to_ms>? AND from_ms<? LIMIT 1", (added_ms, now_ms)).fetchone()
    return "unknown" if gap else "never"


def _classify(added_at: str | None, liked: bool, min_age: float, recent_mode: str, now: datetime,
              no_penalty: bool, heard: str = "played") -> dict:
    if liked:
        return {"reason": "liked", "verdict": None, "keep": False, "age_unknown": False, "recent": False}
    parsed = _parse(added_at)
    unknown = parsed is None
    is_recent = parsed is not None and (now - parsed) < timedelta(hours=min_age)
    if is_recent and recent_mode == "keep":
        return {"reason": "keep", "verdict": None, "keep": True, "age_unknown": False, "recent": True}
    # Recent removals, and every non-like when --no-penalty is set, close as unplayed.
    if is_recent or no_penalty:
        return {"reason": "unplayed", "verdict": None, "keep": False, "age_unknown": False, "recent": is_recent}
    # Old enough to judge: a no only if it was heard (ADR 0023).
    if heard == "never":
        return {"reason": "never_played", "verdict": None, "keep": False, "age_unknown": unknown, "recent": False}
    if heard == "unknown":
        return {"reason": "unknown", "verdict": None, "keep": False, "age_unknown": unknown, "recent": False}
    return {"reason": "unliked", "verdict": "no", "keep": False, "age_unknown": unknown, "recent": False}


def _tx(con, fn):
    con.execute("BEGIN")
    try:
        out = fn()
        con.execute("COMMIT")
        return out
    except Exception:
        con.execute("ROLLBACK")
        raise


def _row_recent(r) -> bool:
    try:
        return bool(r["recent"])
    except (IndexError, KeyError):
        return False


def _close_unplayed(con, sid: str, cid: int | None, ts: str, max_unplayed: int | None = None) -> bool:
    """Close like an unplayed expiry: the candidate returns to the pool and carries no negative weight.

    Without max_unplayed (a recent track, --no-penalty, or plays unknown for a gap) unplayed_expiries is left alone,
    so the close never retires the track. With it (surely never played, ADR 0023) closing the active offer counts as
    the 14-day expiry does: unplayed_expiries goes up and the candidate retires at max_unplayed; a retry with the
    offer already closed counts nothing. True when it retired.
    A Pass from an earlier clear is dropped; a verdict the listener gave is left alone.
    """
    verdicts.retract_playlist_no(con, sid)
    closed = con.execute("""UPDATE offers SET removed_at=?, outcome='unplayed'
                            WHERE spotify_id=? AND removed_at IS NULL""", (ts, sid)).rowcount
    retired = False
    if cid and not verdicts.judged_elsewhere(con, sid):
        if max_unplayed is None:
            con.execute("UPDATE candidates SET state='pool' WHERE id=? AND state!='liked'", (cid,))
        elif closed:
            row = con.execute("SELECT unplayed_expiries FROM candidates WHERE id=? AND state!='liked'",
                              (cid,)).fetchone()
            if row:
                n = (row[0] or 0) + 1
                retired = n >= max_unplayed
                con.execute("UPDATE candidates SET unplayed_expiries=?, state=? WHERE id=?",
                            (n, "retired" if retired else "pool", cid))
        else:       # no active offer left: a retry after the Spotify delete failed, and the first pass counted it
            con.execute("UPDATE candidates SET state='pool' WHERE id=? AND state NOT IN ('liked', 'retired')", (cid,))
        con.execute("""UPDATE candidates SET exclude_reason=NULL WHERE id=? AND (exclude_reason LIKE 'verdict:%'
                       OR exclude_reason LIKE 'playlist feedback:%')""", (cid,))
    return retired


def _finalize(con, playlist_id: str, rows, max_unplayed: int) -> dict:
    """Commit verdicts and offer outcomes. Idempotent. Call this before the Spotify delete."""
    counts = dict.fromkeys(_COUNTS, 0)
    if not rows:
        return counts
    ts = db.now()

    def apply():
        for r in rows:
            sid, reason, cid = r["spotify_id"], r["reason"], r["candidate_id"]
            if reason == "liked":
                counts["liked"] += 1
                verdicts.retract_playlist_no(con, sid)
                if cid and not verdicts.judged_elsewhere(con, sid):
                    con.execute("UPDATE candidates SET state='liked' WHERE id=?", (cid,))
                con.execute("""UPDATE offers SET removed_at=?, outcome='liked'
                               WHERE spotify_id=? AND removed_at IS NULL""", (ts, sid))
            elif reason in _UNPLAYED:
                counts["unplayed"] += 1
                if reason == "recent" or _row_recent(r):
                    counts["recent_removed"] += 1
                if reason in ("never_played", "unknown"):
                    counts[reason] += 1
                counts["retired"] += _close_unplayed(con, sid, cid, ts,
                                                     max_unplayed if reason == "never_played" else None)
            else:
                counts["no"] += 1
                con.execute("""UPDATE offers SET removed_at=?, outcome='negative'
                               WHERE spotify_id=? AND removed_at IS NULL""", (ts, sid))
                cand = con.execute("SELECT * FROM candidates WHERE id=?", (cid,)).fetchone() if cid else None
                if not cand:
                    counts["unlinked"] += 1
                elif not verdicts.judged_elsewhere(con, sid):
                    verdicts.record(con, sid, "pass", id=f"playlist-{sid}", source=verdicts.SOURCE_PLAYLIST,
                                    queue="discoveries", effects="skipped")
                    counts["feedback"] += 1
            removed_reason = _REMOVED_REASON.get(reason, reason)
            con.execute("""UPDATE playlist_adds SET removed_at=?, removed_reason=?
                           WHERE playlist='discoveries' AND spotify_id=? AND removed_at IS NULL""",
                        (ts, removed_reason, sid))
            con.execute("DELETE FROM clear_pending WHERE playlist_id=? AND spotify_id=?", (playlist_id, sid))
        return counts

    return _tx(con, apply)


def reconcile_pending(con, playlist_id: str, present_ids: set[str], max_unplayed: int | None = None) -> dict:
    """Finish bookkeeping for a clear whose Spotify delete already happened."""
    ensure(con)
    if max_unplayed is None:
        max_unplayed = config.DEFAULTS["batch"]["max_unplayed_expiries"]
    empty = {"reconciled": 0, **dict.fromkeys(_COUNTS, 0)}
    if not playlist_id:
        return empty
    rows = con.execute("SELECT * FROM clear_pending WHERE playlist_id=?", (playlist_id,)).fetchall()
    due = [r for r in rows if r["spotify_id"] not in present_ids]
    if not due:
        return empty
    counts = _finalize(con, playlist_id, due, max_unplayed)
    counts["reconciled"] = len(due)
    log.info("clear discoveries: reconciled %d track(s) already removed (liked=%d no=%d unplayed=%d never_played=%d "
             "unknown=%d retired=%d recent_removed=%d feedback=%d)", counts["reconciled"], counts["liked"],
             counts["no"], counts["unplayed"], counts["never_played"], counts["unknown"], counts["retired"],
             counts["recent_removed"], counts["feedback"])
    return counts


def _bump(summary: dict, extra: dict, keys: tuple[str, ...]) -> None:
    for k in keys:
        summary[k] = summary.get(k, 0) + extra.get(k, 0)


def clear_discoveries(con, cfg: dict, sp: Spotify | None = None, *, playlist_id: str | None = None,
                      resolve_playlist: bool = True, min_age_hours: float | None = None,
                      recent: str | None = None, no_penalty: bool | None = None, dry_run: bool = False,
                      now: datetime | None = None) -> dict:
    """Remove every Discoveries track. Liked Songs stay positives; older unliked tracks get a normal no if they
    were played, and close as unplayed if not (never_played counts toward retiring them; unknown does not).

    recent='keep' leaves tracks younger than min_age_hours in the playlist. recent='remove' (default)
    takes those tracks off and closes them as unplayed. no_penalty closes every non-like the same way.
    Outcomes are committed before the Spotify delete. dry_run reads Spotify and writes nothing (nor polls plays).
    """
    min_age, recent_mode, no_penalty = _options(cfg, min_age_hours, recent, no_penalty)
    max_unplayed = cfg["batch"]["max_unplayed_expiries"]
    now = now or datetime.now(timezone.utc)
    summary = _summary(min_age, recent_mode, dry_run, no_penalty)
    ensure(con)
    sp = sp or Spotify(cfg)
    if playlist_id is None and resolve_playlist:
        from .sync import ensure_playlists
        playlist_id = ensure_playlists(con, sp, cfg, create=not dry_run).get("discoveries")
    if not playlist_id:
        log.info("clear discoveries: no Discoveries playlist yet; nothing to remove (dry_run=%s)", dry_run)
        return summary
    summary["playlist_found"] = True

    items = list(sp.playlist_items(playlist_id))
    present = {it["spotify_id"] for it in items if it.get("spotify_id")}
    if not dry_run:
        _bump(summary, reconcile_pending(con, playlist_id, present, max_unplayed), ("reconciled", *_COUNTS))
        # Plays up to now, so a song heard since the last 15-minute poll is not read as never played.
        from .sync import poll_plays
        poll_plays(con, sp)

    tracks = _collapse(items)
    summary["items"] = len(items)
    summary["tracks"] = len(tracks)
    if tracks:
        saved = sp.library_contains([t["spotify_id"] for t in tracks])
    else:
        saved = {}

    plan = []
    for t in tracks:
        sid = t["spotify_id"]
        added = _added_at(con, sid, t.get("added_at"))
        decision = _classify(added, bool(saved.get(sid)), min_age, recent_mode, now, no_penalty,
                             _heard(con, sid, added, now))
        cand = None if decision["keep"] else _candidate(con, sid, t)
        if decision["keep"]:
            summary["recent_kept"] += 1
        elif decision["reason"] == "liked":
            summary["liked"] += 1
            summary["would_remove"] += 1
        elif decision["reason"] in _UNPLAYED:
            summary["unplayed"] += 1
            summary["would_remove"] += 1
            if decision["recent"]:
                summary["recent_removed"] += 1
            if decision["reason"] in ("never_played", "unknown"):
                summary[decision["reason"]] += 1
            if decision["age_unknown"]:
                summary["age_unknown"] += 1
        else:
            summary["no"] += 1
            summary["would_remove"] += 1
            if decision["age_unknown"]:
                summary["age_unknown"] += 1
            if cand is None:
                summary["unlinked"] += 1
        plan.append((sid, added, decision, cand))

    log.info("clear discoveries: items=%d tracks=%d liked=%d no=%d unplayed=%d never_played=%d unknown=%d "
             "recent_removed=%d recent_kept=%d would_remove=%d unlinked=%d age_unknown=%d reconciled=%d dry_run=%s "
             "min_age_hours=%s recent=%s no_penalty=%s",
             summary["items"], summary["tracks"], summary["liked"], summary["no"], summary["unplayed"],
             summary["never_played"], summary["unknown"], summary["recent_removed"], summary["recent_kept"],
             summary["would_remove"], summary["unlinked"], summary["age_unknown"], summary["reconciled"], dry_run,
             min_age, recent_mode, no_penalty)
    if dry_run:
        return summary

    # Commit the close before Spotify deletes. Sync only penalises an active offer that is already
    # missing from the playlist; once removed_at is set, that branch cannot see the row.
    keep_ids = [sid for sid, _, decision, _ in plan if decision["keep"]]
    if keep_ids:
        def drop_pending():
            for sid in keep_ids:
                con.execute("DELETE FROM clear_pending WHERE playlist_id=? AND spotify_id=?", (playlist_id, sid))
        _tx(con, drop_pending)
    rows = [{"spotify_id": sid, "reason": decision["reason"], "verdict": decision["verdict"],
             "candidate_id": None if cand is None else cand["id"], "recent": decision["recent"]}
            for sid, _, decision, cand in plan if not decision["keep"]]
    done = _finalize(con, playlist_id, rows, max_unplayed)
    summary["feedback"] += done["feedback"]
    summary["retired"] += done["retired"]
    remove_ids = [r["spotify_id"] for r in rows]
    if not remove_ids:
        return summary
    snap = sp.playlist(playlist_id).get("snapshot_id")
    sp.remove_items(playlist_id, remove_ids, snap)
    summary["removed"] = len(remove_ids)
    log.info("clear discoveries: removed=%d feedback=%d unplayed=%d never_played=%d unknown=%d retired=%d "
             "no_penalty=%s", summary["removed"], summary["feedback"], summary["unplayed"], summary["never_played"],
             summary["unknown"], summary["retired"], no_penalty)
    return summary
