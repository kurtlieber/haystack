"""Weekly discovery batch (needs Spotify auth): clear Discoveries, then score the pool, pick ~15
(20 % exploration, 1 per artist), map to Spotify (ISRC search, then artist/title + duration + version check),
drop anything already liked, in the library or offered before, and add to the Discoveries playlist."""
from __future__ import annotations

import json
import logging
import random

from . import db, library, vectors
from .deezer import Deezer
from .matching import Index, TrackRef, compare
from .scoring import Pick, select_batch
from .spotify import Spotify
from .clear import clear_discoveries
from .sync import bot_add, ensure_playlists, upsert_track
from .textnorm import key, split_title, version_check

log = logging.getLogger("haystack_music.batch")


def map_to_spotify(con, sp: Spotify, dz: Deezer, cand, tol: float = 3.0) -> str | None:
    """ISRC first (fetched from Deezer's track object), then artist/title search with duration + version check."""
    isrc = cand["isrc"]
    if not isrc and cand["deezer_id"]:
        t = dz.track(cand["deezer_id"]) or {}
        isrc = (t.get("isrc") or "").upper() or None
        if isrc:
            con.execute("UPDATE candidates SET isrc=? WHERE id=?", (isrc, cand["id"]))
    q = TrackRef(cand["id"], cand["artist"], cand["title"], cand["duration"])
    if isrc:
        for t in sp.search_tracks(f"isrc:{isrc}", 10):
            dt = abs((t.get("duration_ms") or 0) / 1000 - (cand["duration"] or 0))
            if dt <= tol and version_check(split_title(cand["title"])[1], split_title(t.get("name", ""))[1]) != "mismatch":
                upsert_track(con, t)
                return t["id"]
    base, _ = split_title(cand["title"])
    best = None
    for t in sp.search_tracks(f'track:"{base}" artist:"{cand["artist"]}"', 10):
        r = TrackRef(t["id"], (t.get("artists") or [{}])[0].get("name", ""), t.get("name", ""),
                     (t.get("duration_ms") or 0) / 1000, artists=[a.get("name", "") for a in t.get("artists", [])])
        m = compare(q, r, tol)
        if m and m.confidence in ("high", "medium") and (best is None or m.title_sim + m.artist_sim > best[0]):
            best = (m.title_sim + m.artist_sim, t)
    if best:
        upsert_track(con, best[1])
        return best[1]["id"]
    return None


def embedding_progress(con, cfg: dict) -> tuple[bool, str, dict]:
    """Is the full-library embedding complete enough for a real batch? (positives are thin otherwise)
    Without a library (library.root unset) there is nothing to wait for."""
    if not cfg["library"]["root"]:
        return True, "no local library: the positives are Liked Songs and Ratings", {
            "embedded": 0, "total": 0, "unembeddable": 0, "fraction": None}
    total = con.execute("SELECT COUNT(DISTINCT content_hash) FROM local_tracks WHERE present=1").fetchone()[0]
    ok = con.execute("""SELECT COUNT(DISTINCT t.content_hash) FROM local_tracks t JOIN local_embeddings e
                        ON e.content_hash=t.content_hash AND e.status='ok' WHERE t.present=1""").fetchone()[0]
    gave_up = con.execute("""SELECT COUNT(DISTINCT t.content_hash) FROM local_tracks t JOIN local_embeddings e
                             ON e.content_hash=t.content_hash AND e.status='error' AND e.attempts>=?
                             WHERE t.present=1""", (cfg["library"]["max_attempts"],)).fetchone()[0]
    need = cfg["batch"]["min_embedded_fraction"]
    frac = ok / total if total else 0.0
    done = total > 0 and (ok + gave_up) >= total * need
    msg = (f"library embedding {'complete' if done else 'incomplete'}: {ok}/{total} embedded ({frac:.1%}), "
           f"{gave_up} unembeddable; batch needs >= {need:.0%}")
    return done, msg, {"embedded": ok, "total": total, "unembeddable": gave_up, "fraction": round(frac, 4)}


def run(con, cfg: dict, sp: Spotify | None = None, size: int | None = None, dry_run: bool = False,
        rng: random.Random | None = None, force: bool = False, variant: str | None = None,
        clear_min_age_hours: float | None = None, clear_recent: str | None = None,
        no_penalty: bool | None = None) -> dict:
    b = cfg["batch"]
    size = size or b["size"]
    done, msg, prog = embedding_progress(con, cfg)
    if not done and not dry_run and not force:
        log.warning("batch skipped: %s", msg)
        return {"skipped": True, "reason": msg, **prog}
    if not done:
        log.warning("%s (continuing: %s)", msg, "dry run" if dry_run else "--force")
    sp = sp or Spotify(cfg)
    pl = ensure_playlists(con, sp, cfg, create=not dry_run)
    cleared = clear_discoveries(con, cfg, sp, playlist_id=pl.get("discoveries"), resolve_playlist=False,
                                min_age_hours=clear_min_age_hours, recent=clear_recent, no_penalty=no_penalty,
                                dry_run=dry_run)
    variant = variant or vectors.active_variant(con, cfg)
    scored = vectors.score_candidates(con, cfg, variant=variant)
    retired = set(cfg["harvest"].get("retired_sources", []))
    rows = {r["id"]: r for r in con.execute(
        "SELECT * FROM candidates WHERE state='pool' AND vector_status='ok' AND score IS NOT NULL "
        "AND COALESCE(spotify_status,'') != 'not_found'")}
    rows = {k: v for k, v in rows.items() if v["source"] not in retired}
    liked_ids = {r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL")}
    offered_ids = {r[0] for r in con.execute("SELECT spotify_id FROM offers")}
    lib_ids = {r[0] for r in con.execute("SELECT spotify_id FROM local_tracks WHERE present=1 AND spotify_id IS NOT NULL")}
    active_artists = {r[0] for r in con.execute(
        "SELECT c.artist_key FROM offers o JOIN candidates c ON c.id=o.candidate_id WHERE o.removed_at IS NULL")}
    dz = Deezer(cfg["harvest"]["deezer_min_interval"])
    tol = cfg["matching"]["duration_tolerance"]
    chosen: list[Pick] = []
    attempts = 0
    rng = rng or random.Random()
    while len(chosen) < size and rows and attempts < size * b["map_attempts_factor"]:
        pool = [Pick(r["id"], r["artist_key"], r["score"], r["source"]) for r in rows.values()]
        need = size - len(chosen)
        explore_have = sum(p.exploration for p in chosen)
        frac = max(0.0, (round(size * b["exploration_fraction"]) - explore_have) / need) if need else 0
        picks = select_batch(pool, need, frac, tuple(b["explore_band"]), b["max_per_artist"],
                             active_artists | {p.artist_key for p in chosen}, rng)
        if not picks:
            break
        for p in picks:
            attempts += 1
            c = rows.pop(p.cand_id)
            sid = c["spotify_id"] or map_to_spotify(con, sp, dz, c, tol)
            if not sid:
                con.execute("UPDATE candidates SET spotify_status='not_found' WHERE id=?", (c["id"],)); continue
            con.execute("UPDATE candidates SET spotify_id=?, spotify_status='ok' WHERE id=?", (sid, c["id"]))
            if sid in liked_ids or sid in offered_ids or sid in lib_ids:
                con.execute("UPDATE candidates SET state='excluded', exclude_reason='known on spotify' WHERE id=?", (c["id"],))
                continue
            chosen.append(Pick(p.cand_id, p.artist_key, p.score, p.source, p.exploration, sid))
            if len(chosen) >= size:
                break
    # final guard: Spotify's own saved-state check
    if chosen:
        saved = sp.library_contains([p.spotify_id for p in chosen])
        for p in [p for p in chosen if saved.get(p.spotify_id)]:
            con.execute("UPDATE candidates SET state='excluded', exclude_reason='liked' WHERE id=?", (p.cand_id,))
            chosen.remove(p)
    out = {"scored": scored, "picked": len(chosen), "exploration": sum(p.exploration for p in chosen),
           "by_source": {}, "dry_run": dry_run, "embedding": prog,
           "variant": variant, "clear": cleared}
    for p in chosen:
        out["by_source"][p.source] = out["by_source"].get(p.source, 0) + 1
    if dry_run or not chosen:
        out["tracks"] = _describe(con, chosen)
        return out
    sids = [p.spotify_id for p in chosen]
    snap = bot_add(con, sp, "discoveries", pl["discoveries"], sids)
    ts = db.now()
    bid = con.execute("INSERT INTO batches(created_at, size, n_explore) VALUES(?,?,?)",
                      (ts, len(chosen), out["exploration"])).lastrowid
    for p in chosen:
        con.execute("""INSERT INTO offers(candidate_id, spotify_id, batch_id, source, score, exploration, added_at)
                       VALUES(?,?,?,?,?,?,?)""", (p.cand_id, p.spotify_id, bid, p.source, p.score, int(p.exploration), ts))
        con.execute("UPDATE candidates SET state='offered' WHERE id=?", (p.cand_id,))
    out["batch_id"] = bid; out["snapshot_id"] = snap
    out["tracks"] = _describe(con, chosen)
    return out


def _describe(con, chosen: list[Pick]) -> list[dict]:
    out = []
    for p in chosen:
        c = con.execute("SELECT artist, title FROM candidates WHERE id=?", (p.cand_id,)).fetchone()
        out.append(dict(artist=c["artist"], title=c["title"], source=p.source, score=round(p.score, 4),
                        exploration=p.exploration, spotify_id=p.spotify_id))
    return out
