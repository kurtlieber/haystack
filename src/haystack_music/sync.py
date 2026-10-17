"""Hourly sync (needs Spotify auth): recently-played, Liked Songs, liked->library mapping, liked-track preview
vectors, Discoveries outcomes (like / 14-day expiry / manual removal), To Rip upkeep, the listener's playlists (for
needle's Playlist review), token-expiry warning.

Recently-played only returns the last 50 tracks, so more than ~50 plays between two hourly syncs are not
visible; such offers count as "unheard" at expiry (the conservative direction, by design). On an always-on
server that only happens during a long outage or after more than ~50 plays in one hour."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from . import alert, db, library, playlists, status, traits, verdicts
from .deezer import Deezer
from .matching import Index, TrackRef, use_local_vector
from .spotify import Spotify
from .textnorm import key, split_title

log = logging.getLogger("haystack_music.sync")


def _iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _images(album: dict) -> tuple[str | None, str | None]:
    """-> (the largest cover, the smallest at least 64 px wide): needle's cover and up-next thumbnail."""
    imgs = sorted((i for i in (album.get("images") or []) if i.get("url")), key=lambda i: -(i.get("width") or 0))
    if not imgs:
        return None, None
    small = [i for i in imgs if (i.get("width") or 0) >= 64]
    return imgs[0]["url"], (small[-1] if small else imgs[-1])["url"]


def upsert_track(con, t: dict) -> None:
    arts = [a.get("name", "") for a in t.get("artists", [])]
    base, suffix = split_title(t.get("name", ""))
    album = t.get("album") or {}
    big, small = _images(album)
    con.execute("""INSERT INTO spotify_tracks(spotify_id, name, artists_json, artist_key, title_key, version_suffix,
                   duration, isrc, album, album_type, release_date, image_url, image_url_small, updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(spotify_id) DO UPDATE SET name=excluded.name, artists_json=excluded.artists_json,
                   artist_key=excluded.artist_key, title_key=excluded.title_key, version_suffix=excluded.version_suffix,
                   duration=excluded.duration, isrc=COALESCE(excluded.isrc, spotify_tracks.isrc), album=excluded.album,
                   album_type=COALESCE(excluded.album_type, spotify_tracks.album_type),
                   release_date=COALESCE(excluded.release_date, spotify_tracks.release_date),
                   image_url=COALESCE(excluded.image_url, spotify_tracks.image_url),
                   image_url_small=COALESCE(excluded.image_url_small, spotify_tracks.image_url_small),
                   updated_at=excluded.updated_at""",
                (t["id"], t.get("name"), json.dumps(arts, ensure_ascii=False), key(arts[0]) if arts else "",
                 key(base), suffix, (t.get("duration_ms") or 0) / 1000.0 or None,
                 ((t.get("external_ids") or {}).get("isrc") or "").upper() or None,
                 album.get("name"), album.get("album_type"), album.get("release_date"), big, small, db.now()))


def spotify_ref(con, sid: str) -> TrackRef | None:
    r = con.execute("SELECT * FROM spotify_tracks WHERE spotify_id=?", (sid,)).fetchone()
    if not r:
        return None
    arts = json.loads(r["artists_json"] or "[]")
    return TrackRef(sid, arts[0] if arts else "", r["name"] or "", r["duration"], sid, r["isrc"], arts)


# ---------------- playlists ----------------

def slots(cfg: dict) -> list[tuple[str, str]]:
    """(slot, playlist name) of the bot's playlists: Discoveries, and To Rip when spotify.to_rip is on."""
    sp = cfg["spotify"]
    return [("discoveries", sp["discoveries_name"])] + ([("to_rip", sp["to_rip_name"])] if sp["to_rip"] else [])


def ensure_playlists(con, sp: Spotify, cfg: dict, create: bool = True) -> dict:
    """Find or create exactly ONE 'Discoveries' and (spotify.to_rip) ONE 'To Rip' playlist (ids cached in kv).
    New playlists are created private. An existing playlist the listener owns with the same name is adopted as-is
    (its visibility and existing tracks are never changed). create=False (dry runs) never writes."""
    me = db.kv_get(con, "spotify_user") or sp.me().get("id")
    db.kv_set(con, "spotify_user", me)
    out, mine = {}, None
    for slot, name in slots(cfg):
        pid = db.kv_get(con, f"playlist_{slot}")
        if pid:
            try:
                sp.playlist(pid); out[slot] = pid; continue
            except Exception as e:  # noqa: BLE001
                log.warning("cached playlist %s (%s) not readable: %r", slot, pid, e)
        if mine is None:
            mine = [p for p in sp.my_playlists() if (p.get("owner") or {}).get("id") == me]
        same = [p for p in mine if p.get("name") == name]
        if len(same) > 1:
            log.warning("%d playlists named %r; using the first, never creating another", len(same), name)
        if same:
            pid = same[0]["id"]
            log.info("adopted existing playlist %r (%s, %s); its existing tracks are never touched", name, pid,
                     "public" if same[0].get("public") else "private")
        elif not create:
            out[slot] = None
            continue
        else:
            pid = sp.create_playlist(name, f"Managed by haystack ({slot.replace('_', ' ')}). "
                                           "Manual additions are never touched.", public=False)["id"]
            log.info("created private playlist %r (%s)", name, pid)
        db.kv_set(con, f"playlist_{slot}", pid)
        out[slot] = pid
    return out


def bot_add(con, sp: Spotify, slot: str, pid: str, sids: list[str]) -> str | None:
    if not sids:
        return None
    snap = sp.add_items(pid, sids)
    ts = db.now()
    for s in sids:
        con.execute("INSERT INTO playlist_adds(playlist, playlist_id, spotify_id, added_at, snapshot_id) VALUES(?,?,?,?,?)",
                    (slot, pid, s, ts, snap))
    return snap


def bot_remove(con, sp: Spotify, slot: str, pid: str, sids: list[str], reason: str) -> list[str]:
    """Remove only tracks the bot itself added (recorded in playlist_adds). Others are left alone."""
    mine = [s for s in sids if con.execute(
        "SELECT 1 FROM playlist_adds WHERE playlist=? AND playlist_id=? AND spotify_id=? AND removed_at IS NULL",
        (slot, pid, s)).fetchone()]
    if not mine:
        return []
    snap = sp.playlist(pid).get("snapshot_id")
    sp.remove_items(pid, mine, snap)
    ts = db.now()
    for s in mine:
        con.execute("""UPDATE playlist_adds SET removed_at=?, removed_reason=? WHERE playlist=? AND playlist_id=?
                       AND spotify_id=? AND removed_at IS NULL""", (ts, reason, slot, pid, s))
    return mine


def fill_metadata(con, sp: Spotify) -> dict:
    """Covers, years and album types for the tracks needle shows (the Review queues) that lack them: one pass
    over Liked Songs (50 per call) for likes, GET /tracks/{id} for the rest."""
    need = {r[0] for r in con.execute("""SELECT l.spotify_id FROM likes l LEFT JOIN spotify_tracks s
                                         ON s.spotify_id=l.spotify_id WHERE l.removed_at IS NULL AND s.image_url IS NULL""")}
    c = {"likes": 0, "tracks": 0}
    if need:
        for _added, t in sp.saved_tracks():
            upsert_track(con, t)
            if t["id"] in need:
                need.discard(t["id"]); c["likes"] += 1
            if not need:
                break
    rest = [r[0] for r in con.execute("""SELECT DISTINCT o.spotify_id FROM offers o LEFT JOIN spotify_tracks s
                                         ON s.spotify_id=o.spotify_id WHERE (o.removed_at IS NULL
                                         OR o.batch_id=(SELECT MAX(id) FROM batches)) AND s.image_url IS NULL""")]
    for sid in rest:
        upsert_track(con, sp.track(sid)); c["tracks"] += 1
    return c


# ---------------- polling ----------------

def poll_plays(con, sp: Spotify) -> int:
    after = db.kv_get(con, "recently_played_after_ms")
    items = sp.recently_played(int(after) if after else None)
    n, mx = 0, int(after or 0)
    for it in items:
        t = it.get("track") or it.get("item")
        if not t or not t.get("id"):
            continue
        upsert_track(con, t)
        cur = con.execute("INSERT OR IGNORE INTO plays(spotify_id, played_at, context) VALUES(?,?,?)",
                          (t["id"], it["played_at"], ((it.get("context") or {}).get("uri"))))
        n += cur.rowcount
        mx = max(mx, int(_iso(it["played_at"]).timestamp() * 1000))
    if mx:
        db.kv_set(con, "recently_played_after_ms", mx)
    return n


def poll_likes(con, sp: Spotify, cfg: dict) -> dict:
    last_full = db.kv_get(con, "likes_full_sync_at")
    full = not last_full or datetime.now(timezone.utc) - _iso(last_full) > timedelta(
        hours=cfg["spotify"]["full_likes_resync_hours"])
    known = {r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL")}
    seen, new, ts = set(), 0, db.now()
    streak = 0
    for added_at, t in sp.saved_tracks():
        sid = t["id"]; seen.add(sid)
        if sid in known:
            streak += 1
            con.execute("UPDATE likes SET last_seen=? WHERE spotify_id=?", (ts, sid))
            if full:
                upsert_track(con, t)        # keeps covers, years and album types current for needle
            if not full and streak >= 50:     # a full page of already-known likes: incremental pass done
                break
            continue
        streak = 0
        upsert_track(con, t)
        con.execute("""INSERT INTO likes(spotify_id, added_at, first_seen, last_seen) VALUES(?,?,?,?)
                       ON CONFLICT(spotify_id) DO UPDATE SET removed_at=NULL, added_at=excluded.added_at,
                       last_seen=excluded.last_seen""", (sid, added_at, ts, ts))
        new += 1
    removed = 0
    if full:
        for sid in known - seen:
            con.execute("UPDATE likes SET removed_at=? WHERE spotify_id=?", (ts, sid)); removed += 1
        db.kv_set(con, "likes_full_sync_at", ts)
    return {"new_likes": new, "unliked": removed, "full_resync": full,
            "total_likes": con.execute("SELECT COUNT(*) FROM likes WHERE removed_at IS NULL").fetchone()[0]}


# ---------------- mapping ----------------

def map_likes(con, cfg: dict, lib: Index | None = None, remap_unmatched: bool = False) -> dict:
    """Liked songs -> local library: Spotify-ID tag, ISRC, then fuzzy with the version check.
    Uncertain matches go to the review list (`haystack review`)."""
    lib = lib or Index(library.library_refs(con))
    tol, cdt = cfg["matching"]["duration_tolerance"], cfg["matching"]["confident_dt"]
    q = """SELECT l.spotify_id FROM likes l LEFT JOIN liked_map m ON m.spotify_id=l.spotify_id
           WHERE l.removed_at IS NULL AND (m.spotify_id IS NULL"""
    q += " OR (m.local_id IS NULL AND COALESCE(m.review,'') != 'rejected'))" if remap_unmatched else ")"
    c = dict(mapped=0, certain=0, review=0, none=0)
    for (sid,) in con.execute(q).fetchall():
        ref = spotify_ref(con, sid)
        if not ref:
            continue
        m = lib.match(ref, tol, cdt)
        c["mapped"] += 1
        if m is None:
            c["none"] += 1
            con.execute("""INSERT INTO liked_map(spotify_id, local_id, confidence, use_local, updated_at)
                           VALUES(?, NULL, 'none', 0, ?) ON CONFLICT(spotify_id) DO UPDATE SET local_id=NULL,
                           confidence='none', use_local=0, updated_at=excluded.updated_at""", (sid, db.now()))
            continue
        review = None if m.certain else "pending"
        c["certain" if m.certain else "review"] += 1
        con.execute("""INSERT INTO liked_map(spotify_id, local_id, method, confidence, dt, version, reason, review,
                       use_local, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(spotify_id) DO UPDATE SET
                       local_id=excluded.local_id, method=excluded.method, confidence=excluded.confidence, dt=excluded.dt,
                       version=excluded.version, reason=excluded.reason, review=excluded.review,
                       use_local=excluded.use_local, updated_at=excluded.updated_at""",
                    (sid, m.ref, m.method, m.confidence, m.dt, m.version, m.reason, review,
                     1 if use_local_vector(m, cdt) else 0, db.now()))
    return c


def liked_preview_vectors(con, cfg: dict, limit: int) -> dict:
    """Deezer preview vectors for liked tracks (throttled, capped per run). Tracks without a confident local
    match come first: their preview vector is a scoring positive. With eval.preview_all_likes the rest follow:
    those previews are only used as eval queries (same audio domain as the candidate pool); scoring keeps
    using the local file for them."""
    all_likes = 1 if cfg.get("eval", {}).get("preview_all_likes", True) else 0
    rows = con.execute("""SELECT m.spotify_id FROM liked_map m JOIN likes l ON l.spotify_id=m.spotify_id
                          WHERE l.removed_at IS NULL AND m.vector_status IS NULL AND (m.use_local=0 OR ?)
                          ORDER BY m.use_local, l.added_at DESC LIMIT ?""", (all_likes, limit)).fetchall()
    if not rows:
        return {"embedded": 0, "not_found": 0}
    dz = Deezer(cfg["harvest"]["deezer_min_interval"])
    tol = cfg["matching"]["duration_tolerance"]
    readings = not traits.missing_models()
    c = dict(embedded=0, not_found=0, errors=0)
    for (sid,) in rows:
        ref = spotify_ref(con, sid)
        t, found_any, _method = traits.find_deezer(dz, ref, tol) if ref else (None, False, None)
        if not t:
            st = "no_preview" if found_any else "not_found"
            c[st] = c.get(st, 0) + 1
            con.execute("UPDATE liked_map SET vector_status=? WHERE spotify_id=?", (st, sid)); continue
        try:
            traits.store(con, t["id"], traits.analyse_preview(dz, t["preview"], readings=readings, bpm_hint=t.get("bpm")))
            con.execute("UPDATE liked_map SET deezer_id=?, vector_status='ok' WHERE spotify_id=?", (t["id"], sid))
            c["embedded"] += 1
        except Exception as e:  # noqa: BLE001
            c["errors"] += 1
            log.warning("liked preview %s: %r", sid, e)
            con.execute("UPDATE liked_map SET vector_status='error' WHERE spotify_id=?", (sid,))
    return c


# ---------------- outcomes ----------------

def process_offers(con, sp: Spotify, cfg: dict, pl: dict, lib: Index) -> dict:
    now = datetime.now(timezone.utc)
    expiry = timedelta(days=cfg["batch"]["expiry_days"])
    max_unplayed = cfg["batch"]["max_unplayed_expiries"]
    liked = {r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL")}
    active = con.execute("SELECT * FROM offers WHERE removed_at IS NULL").fetchall()
    c = dict(active=len(active), liked=0, expired_played=0, expired_unplayed=0, retired=0, manual_removed=0,
             to_rip_added=0)
    pid = pl.get("discoveries")
    pending = pid and con.execute("SELECT 1 FROM clear_pending WHERE playlist_id=? LIMIT 1", (pid,)).fetchone()
    if not active and not pending:
        return c
    in_playlist = {i["spotify_id"] for i in sp.playlist_items(pid)}
    if pending:
        # A clear already deleted these tracks and crashed before recording verdicts. Finish that,
        # so the absence is not read as the listener removing them by hand.
        from .clear import reconcile_pending
        reconcile_pending(con, pid, in_playlist)
        active = con.execute("SELECT * FROM offers WHERE removed_at IS NULL").fetchall()
        c["active"] = len(active)
        if not active:
            return c
    to_remove: dict[str, list] = {}
    for o in active:
        sid = o["spotify_id"]
        p = con.execute("SELECT MIN(played_at) FROM plays WHERE spotify_id=? AND played_at>=?", (sid, o["added_at"])).fetchone()[0]
        if p and not o["played"]:
            con.execute("UPDATE offers SET played=1, first_played_at=? WHERE id=?", (p, o["id"]))
        played = bool(p or o["played"])
        if sid in liked:
            to_remove.setdefault("liked", []).append(o)
        elif sid not in in_playlist:
            # the listener removed it: an explicit dismissal; never re-offered.
            # Clear commits liked / no / unplayed before it deletes, so those offers are already
            # closed and never reach this branch.
            con.execute("UPDATE offers SET removed_at=?, outcome='manual_removed' WHERE id=?", (db.now(), o["id"]))
            con.execute("UPDATE candidates SET state='negative' WHERE id=?", (o["candidate_id"],))
            con.execute("""UPDATE playlist_adds SET removed_at=?, removed_reason='manual' WHERE playlist='discoveries'
                           AND spotify_id=? AND removed_at IS NULL""", (db.now(), sid))
            c["manual_removed"] += 1
        elif now - _iso(o["added_at"]) >= expiry:
            to_remove.setdefault("expired_played" if played else "expired_unplayed", []).append(o)
    for reason, offers in to_remove.items():
        bot_remove(con, sp, "discoveries", pl["discoveries"], [o["spotify_id"] for o in offers], reason)
        for o in offers:
            if reason == "liked":
                con.execute("UPDATE offers SET removed_at=?, outcome='liked' WHERE id=?", (db.now(), o["id"]))
                con.execute("UPDATE candidates SET state='liked' WHERE id=?", (o["candidate_id"],))
                c["liked"] += 1
                ref = spotify_ref(con, o["spotify_id"])
                m = lib.match(ref, cfg["matching"]["duration_tolerance"]) if ref else None
                if cfg["spotify"]["to_rip"] and not (m and m.confidence in ("high", "medium")) and not con.execute(
                        "SELECT 1 FROM to_rip WHERE spotify_id=? AND removed_at IS NULL", (o["spotify_id"],)).fetchone():
                    bot_add(con, sp, "to_rip", pl["to_rip"], [o["spotify_id"]])
                    con.execute("""INSERT INTO to_rip(spotify_id, added_at) VALUES(?,?) ON CONFLICT(spotify_id)
                                   DO UPDATE SET added_at=excluded.added_at, removed_at=NULL""", (o["spotify_id"], db.now()))
                    c["to_rip_added"] += 1
            elif reason == "expired_played":
                con.execute("UPDATE offers SET removed_at=?, outcome='negative' WHERE id=?", (db.now(), o["id"]))
                con.execute("UPDATE candidates SET state='negative' WHERE id=?", (o["candidate_id"],))
                c["expired_played"] += 1
            else:
                con.execute("UPDATE offers SET removed_at=?, outcome='unplayed' WHERE id=?", (db.now(), o["id"]))
                n = con.execute("SELECT unplayed_expiries FROM candidates WHERE id=?", (o["candidate_id"],)).fetchone()[0] + 1
                st = "retired" if n >= max_unplayed else "pool"
                con.execute("UPDATE candidates SET unplayed_expiries=?, state=? WHERE id=?", (n, st, o["candidate_id"]))
                c["expired_unplayed"] += 1
                c["retired"] += st == "retired"
    return c


def torip_check(con, cfg: dict, sp: Spotify | None, lib: Index | None = None) -> dict:
    """When a To Rip track shows up in the library (ID tag, ISRC, fuzzy + version check): remove it from To Rip
    and switch the liked track to the local file's vector."""
    lib = lib or Index(library.library_refs(con))
    rows = con.execute("SELECT spotify_id FROM to_rip WHERE removed_at IS NULL").fetchall()
    c = dict(open=len(rows), found_locally=0, removed=0)
    found = []
    for (sid,) in rows:
        ref = spotify_ref(con, sid)
        m = lib.match(ref, cfg["matching"]["duration_tolerance"]) if ref else None
        if m and m.confidence in ("high", "medium"):
            found.append((sid, m))
    c["found_locally"] = len(found)
    if found and sp is not None:
        pid = db.kv_get(con, "playlist_to_rip")
        if pid:
            bot_remove(con, sp, "to_rip", pid, [s for s, _ in found], "ripped")
        for sid, m in found:
            con.execute("UPDATE to_rip SET removed_at=?, local_id=?, match_method=? WHERE spotify_id=?",
                        (db.now(), m.ref, m.method, sid))
            c["removed"] += 1
    map_likes(con, cfg, lib, remap_unmatched=True)
    return c


def token_check(cfg: dict, meta: dict, con) -> dict:
    info = status.token_expiry_info(meta.get("issued at"), cfg["spotify"]["token_warn_days"],
                                    cfg["spotify"]["token_lifetime_days"])
    if info.get("expiry_warning"):
        if alert.once_a_day(con, "token_warning_notified_on"):
            from .auth import reauth_hint
            msg = (f"re-auth soon: the Spotify refresh token was issued {info['issued_at'][:10]} and expires around "
                   f"{info['expires_estimate']}. {reauth_hint(cfg)}")
            alert.send(cfg, "auth", msg, "warn")
            status.notify("haystack: re-auth soon", msg, "normal")
    return info


def run(con, cfg: dict, sp: Spotify | None = None) -> dict:
    sp = sp or Spotify(cfg)
    pl = ensure_playlists(con, sp, cfg)
    out = {"plays_new": poll_plays(con, sp)}
    out["likes"] = poll_likes(con, sp, cfg)
    lib = Index(library.library_refs(con))
    out["mapping"] = map_likes(con, cfg, lib)
    out["liked_vectors"] = liked_preview_vectors(con, cfg, cfg["spotify"]["liked_preview_vectors_per_run"])
    out["offers"] = process_offers(con, sp, cfg, pl, lib)
    out["to_rip"] = torip_check(con, cfg, sp, lib)
    out["verdict_effects"] = verdicts.apply_effects(con, cfg, sp)     # retries any the API couldn't make
    try:                                                              # needle's Playlist review
        out["playlists"] = playlists.sync(con, sp, cfg)
    except Exception as e:  # noqa: BLE001 - the picker can wait an hour; the rest of the sync can't
        log.warning("playlists: %r", e)
        out["playlists"] = {"error": repr(e)[:300]}
    out["token"] = token_check(cfg, sp.token_meta, con)
    return out
