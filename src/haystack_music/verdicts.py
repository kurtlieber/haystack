"""Verdicts: the listener's Ratings (1-5 stars), Passes and Dislikes on any Spotify track (ADR 0002, ADR 0005).

The store is append-only: every verdict needle (or the CLI) sends is a row with a client-made id, so an outbox
replay is a no-op. A track's current verdict is its latest by `given_at`; older rows are history. The Spotify
writes a verdict implies (the heart, Discoveries, To Rip) are applied here, by the server only, and only for the
current verdict, so a burst of re-ratings ends in one consistent state."""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass

from . import db

log = logging.getLogger("haystack_music.verdicts")

KINDS = ("rating", "pass", "dislike")
SOURCE_PLAYLIST = "playlist"   # a Discoveries-clear no (clear.py): a Pass the next clear may take back
SPOTIFY_ID = re.compile(r"^[0-9A-Za-z]{22}$")
LIBRARY_SCOPE = "user-library-modify"

SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (
    id TEXT PRIMARY KEY,                 -- client-made UUID
    spotify_id TEXT NOT NULL,
    kind TEXT NOT NULL,                  -- rating | pass | dislike
    stars INTEGER,                       -- 1..5 for a rating
    given_at TEXT NOT NULL,              -- the client's clock: orders the history
    received_at TEXT NOT NULL,
    source TEXT NOT NULL,                -- needle | cli | feedback | playlist (Discoveries clear)
    queue TEXT,                          -- discoveries | likes
    effects TEXT NOT NULL DEFAULT 'pending',   -- pending | done | skipped
    effects_at TEXT, effects_error TEXT
);
CREATE INDEX IF NOT EXISTS verdicts_track ON verdicts(spotify_id, given_at);
CREATE INDEX IF NOT EXISTS verdicts_pending ON verdicts(effects);
"""


@dataclass(frozen=True)
class Verdict:
    spotify_id: str
    kind: str
    stars: int | None
    given_at: str

    @property
    def keeps_heart(self) -> bool:
        """A Rating of 3 stars or more: belongs in Liked Songs and, if not ripped yet, To Rip."""
        return self.kind == "rating" and (self.stars or 0) >= 3


def ensure(con) -> None:
    """Schema plus the one-time feedback migration (db.connect runs this)."""
    con.executescript(SCHEMA)
    _migrate_feedback(con)


def _migrate_feedback(con) -> None:
    """One-time copy of the old yes/no `feedback` rows: yes -> 3 stars, no -> Pass (mild, normal) or Dislike
    (strong). Their Spotify writes were made by hand (or by the clear) back then, so effects are 'skipped'.
    A Discoveries-clear no keeps source 'playlist', so a later clear can still take it back."""
    if db.kv_get(con, "verdicts_migrated_feedback"):
        return
    if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='feedback'").fetchone():
        return
    for r in con.execute("SELECT * FROM feedback WHERE spotify_id IS NOT NULL ORDER BY id").fetchall():
        if r["verdict"] == "yes":
            kind, stars = "rating", 3
        else:
            kind, stars = ("dislike" if r["weight"] > 1.0 else "pass"), None
        given = r["created_at"] or db.now()
        source = "playlist" if "source" in r.keys() and r["source"] == "playlist" else "feedback"
        con.execute("""INSERT OR IGNORE INTO verdicts(id, spotify_id, kind, stars, given_at, received_at, source,
                       effects, effects_at) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (f"feedback-{r['id']}", r["spotify_id"], kind, stars, given, given, source, "skipped",
                     db.now()))
    db.kv_set(con, "verdicts_migrated_feedback", db.now())


def validate(spotify_id: str, kind: str, stars) -> int | None:
    if not isinstance(spotify_id, str) or not SPOTIFY_ID.match(spotify_id):
        raise ValueError("track must be a 22-character Spotify track id")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if kind == "rating":
        if isinstance(stars, bool) or not isinstance(stars, int) or not 1 <= stars <= 5:
            raise ValueError("a rating needs stars from 1 to 5")
        return stars
    if stars is not None:
        raise ValueError(f"a {kind} has no stars")
    return None


def record(con, spotify_id: str, kind: str, stars: int | None = None, *, given_at: str | None = None,
           id: str | None = None, source: str = "needle", queue: str | None = None,
           effects: str = "pending") -> dict:
    """Store one verdict. -> {"id", "status": accepted | duplicate}. Raises ValueError on a malformed one.
    effects='skipped' for a verdict whose Spotify writes the caller makes itself (the Discoveries clear)."""
    stars = validate(spotify_id, kind, stars)
    vid = id or str(uuid.uuid4())
    now = db.now()
    cur = con.execute("""INSERT OR IGNORE INTO verdicts(id, spotify_id, kind, stars, given_at, received_at, source,
                         queue, effects, effects_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                      (vid, spotify_id, kind, stars, given_at or now, now, source, queue, effects,
                       None if effects == "pending" else now))
    if cur.rowcount:
        # the server's own state follows at once (the Spotify writes may wait): a judged candidate is never offered
        # again, and its offer is closed so the sync can't mistake the server's playlist removal for the listener's.
        _close_offer(con, spotify_id, current(con, [spotify_id])[spotify_id])
    return {"id": vid, "status": "accepted" if cur.rowcount else "duplicate"}


def judged_elsewhere(con, spotify_id: str) -> bool:
    """True when the track has a verdict the listener gave (needle, the CLI, old feedback), not just a clear's no."""
    return con.execute("SELECT 1 FROM verdicts WHERE spotify_id=? AND source!=? LIMIT 1",
                       (spotify_id, SOURCE_PLAYLIST)).fetchone() is not None


def retract_playlist_no(con, spotify_id: str) -> bool:
    """Drop the clear's own Pass on a track (it turned out liked, or unplayed). The listener's verdicts are untouched."""
    return con.execute("DELETE FROM verdicts WHERE spotify_id=? AND source=?",
                       (spotify_id, SOURCE_PLAYLIST)).rowcount > 0


def current(con, ids=None) -> dict[str, Verdict]:
    q = """SELECT v.spotify_id, v.kind, v.stars, v.given_at FROM verdicts v
           WHERE v.id = (SELECT w.id FROM verdicts w WHERE w.spotify_id = v.spotify_id
                         ORDER BY w.given_at DESC, w.received_at DESC, w.rowid DESC LIMIT 1)"""
    out = {r[0]: Verdict(r[0], r[1], r[2], r[3]) for r in con.execute(q)}
    return out if ids is None else {i: out[i] for i in ids if i in out}


def history(con, spotify_id: str) -> list[dict]:
    return [dict(r) for r in con.execute("SELECT * FROM verdicts WHERE spotify_id=? ORDER BY given_at, received_at",
                                         (spotify_id,))]


def pending(con) -> list[str]:
    """Tracks whose verdicts still have Spotify writes to make."""
    return [r[0] for r in con.execute("SELECT DISTINCT spotify_id FROM verdicts WHERE effects='pending' ORDER BY spotify_id")]


# ---------------- export / import (a portable copy of every verdict) ----------------

FORMAT = "haystack-verdicts"
FORMAT_VERSION = 1


def export_doc(con) -> dict:
    """Every verdict, oldest first, with the track's title, artists, album, ISRC and duration for people and for
    re-matching. The current verdict of a track is its latest given_at."""
    import json
    rows = con.execute("""SELECT v.id, v.spotify_id, v.kind, v.stars, v.given_at, v.source, v.queue,
                                 st.name, st.artists_json, st.album, st.isrc, st.duration,
                                 (SELECT c.title FROM candidates c WHERE c.spotify_id = v.spotify_id LIMIT 1) AS c_title,
                                 (SELECT c.artist FROM candidates c WHERE c.spotify_id = v.spotify_id LIMIT 1) AS c_artist
                          FROM verdicts v LEFT JOIN spotify_tracks st ON st.spotify_id = v.spotify_id
                          ORDER BY v.given_at, v.id""").fetchall()
    out = []
    for r in rows:
        artists = json.loads(r["artists_json"]) if r["artists_json"] else ([r["c_artist"]] if r["c_artist"] else [])
        out.append({"id": r["id"], "track": r["spotify_id"], "kind": r["kind"], "stars": r["stars"],
                    "given_at": r["given_at"], "source": r["source"], "queue": r["queue"],
                    "title": r["name"] or r["c_title"], "artists": artists, "album": r["album"],
                    "isrc": r["isrc"], "duration": r["duration"]})
    return {"format": FORMAT, "version": FORMAT_VERSION, "exported_at": db.now(),
            "note": "Every Verdict, oldest first; the current one per track is the latest given_at. "
                    "Keyed by Spotify track id; title/artists/album/isrc are for people and re-matching.",
            "verdicts": out}


def import_doc(con, doc: dict, effects: str = "skipped") -> dict:
    """Add an export's verdicts. By id, so importing the same file twice adds nothing; all or none, so a bad
    entry stops the import with its index. effects: skipped (the account already shows them, e.g. a restore)
    or pending (make the hearts and playlist writes, e.g. a new Spotify account). A track's title and artists
    are kept for `verdict list` until the sync reads the track from Spotify."""
    from .sync import upsert_track
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} file")
    if doc.get("version") != FORMAT_VERSION:
        raise ValueError(f"{FORMAT} version {doc.get('version')!r} (this haystack reads {FORMAT_VERSION})")
    if effects not in ("skipped", "pending"):
        raise ValueError("effects must be skipped or pending")
    out = {"verdicts": 0, "added": 0, "duplicate": 0, "tracks_added": 0}
    con.execute("BEGIN")
    try:
        for i, v in enumerate(doc.get("verdicts") or []):
            try:
                if not (isinstance(v, dict) and isinstance(v.get("id"), str) and v["id"]
                        and isinstance(v.get("given_at"), str) and v["given_at"]):
                    raise ValueError("an id and given_at are required")
                got = record(con, v.get("track"), v.get("kind"), v.get("stars"), given_at=v["given_at"],
                             id=v["id"], source=v.get("source") or "import", queue=v.get("queue"), effects=effects)
            except ValueError as e:
                raise ValueError(f"verdicts[{i}]: {e}") from None
            out["verdicts"] += 1
            out["added" if got["status"] == "accepted" else "duplicate"] += 1
            if v.get("title") and not con.execute("SELECT 1 FROM spotify_tracks WHERE spotify_id=?",
                                                  (v["track"],)).fetchone():
                upsert_track(con, {"id": v["track"], "name": v["title"],
                                   "artists": [{"name": a} for a in v.get("artists") or []],
                                   "duration_ms": round((v.get("duration") or 0) * 1000),
                                   "external_ids": {"isrc": v.get("isrc")}, "album": {"name": v.get("album")}})
                out["tracks_added"] += 1
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return out


# ---------------- weights (scoring.verdicts) ----------------

def positive_bonus(cfg: dict, v: Verdict | None, liked: bool = False) -> float | None:
    """The additive bonus a track gets as a discovery positive, or None if it is not a positive.
    A Rating's bonus grows with its stars; an Unrated like counts as `unrated_like_stars` (3)."""
    w = cfg["scoring"]["verdicts"]
    if v is None:
        return float(w["star_bonus"][w["unrated_like_stars"] - 1]) if liked else None
    if v.kind != "rating":
        return None
    return float(w["star_bonus"][v.stars - 1])


def negative_strength(cfg: dict, v: Verdict | None) -> float | None:
    w = cfg["scoring"]["verdicts"]
    if v is None or v.kind == "rating":
        return None
    return float(w["pass_strength"] if v.kind == "pass" else w["dislike_strength"])


# ---------------- effects (ADR 0002) ----------------

def plan(v: Verdict, *, liked: bool, in_library: bool, on_to_rip: bool, in_discoveries: bool,
         to_rip: bool = True) -> list[str]:
    """The Spotify writes that bring a track in line with its current verdict.
    on_to_rip / in_discoveries mean "the bot added it there and it's still there". to_rip: the To Rip playlist
    is on (spotify.to_rip); a track the bot put there before it was turned off still comes off."""
    out = []
    if in_discoveries:
        out.append("discoveries_remove")
    if v.keeps_heart:
        if not liked:
            out.append("heart_add")
        if to_rip and not in_library and not on_to_rip:
            out.append("to_rip_add")
    else:
        if liked:
            out.append("heart_remove")
        if on_to_rip:
            out.append("to_rip_remove")
    return out


OUTCOME = {"rating": "rated", "pass": "passed", "dislike": "disliked"}


def _bot_on(con, slot: str, sid: str) -> str | None:
    r = con.execute("SELECT playlist_id FROM playlist_adds WHERE playlist=? AND spotify_id=? AND removed_at IS NULL "
                    "ORDER BY id DESC LIMIT 1", (slot, sid)).fetchone()
    return r[0] if r else None


def _in_library(con, cfg: dict, sp, sid: str, lib) -> bool:
    """Spotify-ID tag, ISRC, then fuzzy, as for To Rip everywhere else."""
    from .sync import spotify_ref, upsert_track
    if con.execute("SELECT 1 FROM local_tracks WHERE present=1 AND spotify_id=?", (sid,)).fetchone():
        return True
    ref = spotify_ref(con, sid)
    if ref is None:
        upsert_track(con, sp.track(sid))
        ref = spotify_ref(con, sid)
    m = lib.match(ref, cfg["matching"]["duration_tolerance"]) if ref else None
    return bool(m and m.confidence in ("high", "medium"))


def _close_offer(con, sid: str, v: Verdict) -> None:
    for o in con.execute("SELECT id, candidate_id FROM offers WHERE spotify_id=? AND removed_at IS NULL", (sid,)).fetchall():
        con.execute("UPDATE offers SET removed_at=?, outcome=? WHERE id=?", (db.now(), OUTCOME[v.kind], o["id"]))
    state = "liked" if v.kind == "rating" else "negative"
    con.execute("UPDATE candidates SET state=?, exclude_reason=? WHERE spotify_id=? AND state IN ('pool','offered','liked','negative')",
                (state, f"verdict: {v.kind}{'' if v.stars is None else f' {v.stars}'}", sid))


def apply_effects(con, cfg: dict, sp, library=None, limit: int = 200) -> dict:
    """Apply the current verdict of every track with pending effects. Each step is idempotent, so a partial
    failure is simply retried later. A missing `user-library-modify` scope leaves only the heart pending."""
    from . import library as libmod
    from .matching import Index
    from .spotify import SpotifyError
    from .sync import ensure_playlists, slots
    sids = pending(con)[:limit]
    out = {"tracks": len(sids), "applied": 0, "pending": 0, "errors": 0, "actions": {}}
    if not sids:
        return out
    cur = current(con, sids)
    lib = Index(library if library is not None else libmod.library_refs(con))
    to_rip = cfg["spotify"]["to_rip"]
    pl = {slot: db.kv_get(con, f"playlist_{slot}") for slot, _ in slots(cfg)}
    if not all(pl.values()):
        pl = ensure_playlists(con, sp, cfg)
    liked = {}
    for i in range(0, len(sids), 40):
        liked.update(sp.library_contains(sids[i:i + 40]))
    for sid in sids:
        v = cur[sid]
        error = None
        try:
            _close_offer(con, sid, v)
            steps = plan(v, liked=bool(liked.get(sid)),
                         in_library=v.keeps_heart and to_rip and _in_library(con, cfg, sp, sid, lib),
                         on_to_rip=_bot_on(con, "to_rip", sid) is not None,
                         in_discoveries=_bot_on(con, "discoveries", sid) is not None, to_rip=to_rip)
            for step in steps:
                try:
                    _do(con, sp, pl, sid, step)
                    out["actions"][step] = out["actions"].get(step, 0) + 1
                except SpotifyError as e:
                    if step.startswith("heart") and isinstance(e, SpotifyError) and e.status in (401, 403):
                        error = (f"{step}: Spotify refused ({e.status}); the token needs the {LIBRARY_SCOPE} scope: "
                                 "run `haystack auth`")
                    else:
                        raise
        except Exception as e:  # noqa: BLE001
            log.warning("verdict effects for %s failed: %r", sid, e)
            error = repr(e)[:300]
            out["errors"] += 1
        if error:
            con.execute("UPDATE verdicts SET effects_error=? WHERE spotify_id=? AND effects='pending'", (error, sid))
            out["pending"] += 1
        else:
            con.execute("UPDATE verdicts SET effects='done', effects_at=?, effects_error=NULL WHERE spotify_id=? "
                        "AND effects='pending'", (db.now(), sid))
            out["applied"] += 1
    return out


def _do(con, sp, pl: dict, sid: str, step: str) -> None:
    from .sync import bot_add, bot_remove
    now = db.now()
    if step == "heart_add":
        sp.library_save([sid])
        con.execute("""INSERT INTO likes(spotify_id, added_at, first_seen, last_seen) VALUES(?,?,?,?)
                       ON CONFLICT(spotify_id) DO UPDATE SET removed_at=NULL, last_seen=excluded.last_seen""",
                    (sid, now, now, now))
    elif step == "heart_remove":
        sp.library_remove([sid])
        con.execute("UPDATE likes SET removed_at=? WHERE spotify_id=? AND removed_at IS NULL", (now, sid))
    elif step == "discoveries_remove":
        bot_remove(con, sp, "discoveries", _bot_on(con, "discoveries", sid), [sid], "verdict")
    elif step == "to_rip_add":
        bot_add(con, sp, "to_rip", pl["to_rip"], [sid])
        con.execute("""INSERT INTO to_rip(spotify_id, added_at) VALUES(?,?) ON CONFLICT(spotify_id)
                       DO UPDATE SET added_at=excluded.added_at, removed_at=NULL""", (sid, now))
    elif step == "to_rip_remove":
        bot_remove(con, sp, "to_rip", _bot_on(con, "to_rip", sid), [sid], "verdict")
        con.execute("UPDATE to_rip SET removed_at=? WHERE spotify_id=? AND removed_at IS NULL", (now, sid))
    else:
        raise ValueError(step)
