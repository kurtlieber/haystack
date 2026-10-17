"""Local library index (token-free): read-only scan of library.root + incremental embedding.

Nothing under the library root is ever written: files are opened read-only, tags are only read."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import signal
import struct
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

from . import config, db, status
from .textnorm import key, split_artists, split_title

log = logging.getLogger("haystack_music.library")
SPOTIFY_ID = re.compile(r"^[0-9A-Za-z]{22}$")
SAMPLE = 65536


# ---------- pure helpers ----------

def find_mdat(f, size: int) -> tuple[int | None, int | None]:
    """Walk top-level MP4 atoms; -> (payload offset, payload length) of 'mdat'."""
    pos = 0
    while pos + 8 <= size:
        f.seek(pos)
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        sz, typ = struct.unpack(">I4s", hdr)
        hl = 8
        if sz == 1:
            ext = f.read(8)
            if len(ext) < 8:
                break
            sz = struct.unpack(">Q", ext)[0]; hl = 16
        elif sz == 0:
            sz = size - pos
        if sz < hl:
            break
        if typ == b"mdat":
            return pos + hl, sz - hl
        pos += sz
    return None, None


def audio_hash(f, size: int) -> str:
    """Short audio-content hash: length + 4 samples of the mdat payload (tags/moov edits don't change it)."""
    off, ln = find_mdat(f, size)
    if off is None:
        off, ln = 0, size
    h = hashlib.blake2b(digest_size=10)
    h.update(str(ln).encode())
    for frac in (0.0, 0.33, 0.66):
        f.seek(off + int(max(0, ln - SAMPLE) * frac)); h.update(f.read(SAMPLE))
    f.seek(off + max(0, ln - SAMPLE)); h.update(f.read(SAMPLE))
    return h.hexdigest()


def _first(t, k) -> str:
    v = t.get(k)
    if not v:
        return ""
    v = v[0] if isinstance(v, list) else v
    if hasattr(v, "decode"):
        v = bytes(v).decode("utf-8", "replace")
    return str(v).strip()


def _all(t, k) -> list[str]:
    v = t.get(k) or []
    out = []
    for x in v if isinstance(v, list) else [v]:
        s = bytes(x).decode("utf-8", "replace") if hasattr(x, "decode") else str(x)
        if s.strip():
            out.append(s.strip())
    return out


def parse_tags(t: dict, duration: float) -> dict:
    """Pure: mutagen MP4 tag dict -> normalised row fields."""
    artist = _first(t, "\xa9ART")
    artists = split_artists(artist, _all(t, "----:com.apple.iTunes:ARTISTS"))
    title = _first(t, "\xa9nam")
    base, suffix = split_title(title)
    sid = ""
    # Some rippers store the 22-char Spotify track id in the MusicBrainz Track Id fields (and artist ids in
    # MusicBrainz Artist Id). A real MusicBrainz id is a 36-char UUID, so it never matches.
    for k in ("----:com.apple.iTunes:MusicBrainz Track Id", "----:com.apple.iTunes:MusicBrainz Release Track Id"):
        v = _first(t, k)
        if SPOTIFY_ID.match(v):
            sid = v; break
    art_ids = [v for v in _all(t, "----:com.apple.iTunes:MusicBrainz Artist Id") if SPOTIFY_ID.match(v)]
    isrc = _first(t, "----:com.apple.iTunes:ISRC").upper() or None
    return dict(artist_raw=artist, artists_json=json.dumps(artists, ensure_ascii=False),
                artist_key=key(artists[0]) if artists else key(artist), title_raw=title, title_base=base,
                title_key=key(base), version_suffix=suffix, album=_first(t, "\xa9alb"),
                genre=_first(t, "\xa9gen"), year=_first(t, "\xa9day")[:4], duration=round(duration, 2),
                spotify_id=sid or None, spotify_artist_ids=json.dumps(art_ids) if art_ids else None, isrc=isrc)


# ---------- scan ----------

def iter_files(root: str, exts: list[str]):
    exts = tuple(e.lower() for e in exts)
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            if f.lower().endswith(exts):
                yield os.path.join(dp, f)


class LibraryMissing(RuntimeError):
    """library.root is set but there is nothing there: refuse rather than mark every track gone."""


def scan(con, cfg: dict) -> dict:
    """Incremental read-only scan. Path+size unchanged -> just refresh tags; otherwise hash the audio.
    A new path whose content hash matches a vanished row is treated as a move (keeps its embedding).
    No library.root: nothing to scan. A root that is missing, or has no files while tracks are known, fails."""
    if not cfg["library"]["root"]:
        return {"skipped": True, "reason": "no local library (library.root is not set)"}
    root = os.path.expanduser(cfg["library"]["root"])
    if not os.path.isdir(root):
        raise LibraryMissing(f"library.root {root} is not a directory (not mounted?); no track was marked missing")
    from mutagen.mp4 import MP4
    t0 = time.time(); ts = db.now()
    present = con.execute("SELECT COUNT(*) FROM local_tracks WHERE present=1").fetchone()[0]
    known = {r["path"]: r for r in con.execute("SELECT id, path, size, content_hash FROM local_tracks")}
    seen: set[str] = set()
    c = dict(files=0, new=0, changed=0, moved=0, unchanged=0, missing=0, errors=0)
    new_rows = []
    con.execute("BEGIN")
    try:
        _scan_files(con, root, cfg, known, seen, c, new_rows, ts, MP4)
        if not c["files"] and present:
            raise LibraryMissing(f"no files under library.root {root}, but {present} tracks are known (not mounted?); "
                                 "none was marked missing")
        _apply_new(con, known, seen, c, new_rows, ts)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    c["seconds"] = round(time.time() - t0, 1)
    return c


def _scan_files(con, root, cfg, known, seen, c, new_rows, ts, MP4):
    for p in iter_files(root, cfg["library"]["extensions"]):
        c["files"] += 1; seen.add(p)
        try:
            with open(p, "rb") as f:                      # read-only
                size = os.fstat(f.fileno()).st_size
                m = MP4(f)
                fields = parse_tags(m.tags or {}, m.info.length)
                old = known.get(p)
                if old and old["size"] == size:
                    h = old["content_hash"]; c["unchanged"] += 1
                else:
                    h = audio_hash(f, size)
        except Exception as e:  # noqa: BLE001
            c["errors"] += 1
            log.warning("scan error %s: %r", p, e)
            continue
        if old:
            if old["size"] != size:
                c["changed"] += 1
            con.execute("""UPDATE local_tracks SET size=:size, content_hash=:h, present=1, last_seen=:ts, updated_at=:ts,
                           artist_raw=:a, artists_json=:aj, artist_key=:ak, title_raw=:t, title_base=:tb, title_key=:tk,
                           version_suffix=:vs, album=:al, genre=:g, year=:y, duration=:d, spotify_id=:sid,
                           spotify_artist_ids=:said, isrc=:isrc WHERE id=:id""",
                        dict(size=size, h=h, ts=ts, id=old["id"], **_named(fields)))
        else:
            new_rows.append((p, size, h, fields))


def _apply_new(con, known, seen, c, new_rows, ts):
    # moves: vanished rows whose content hash re-appears under a new path
    vanished = {r["content_hash"]: r for p, r in known.items() if p not in seen}
    for p, size, h, fields in new_rows:
        v = vanished.pop(h, None)
        if v is not None:
            c["moved"] += 1
            con.execute("""UPDATE local_tracks SET path=:p, size=:size, content_hash=:h, present=1, last_seen=:ts,
                           updated_at=:ts, artist_raw=:a, artists_json=:aj, artist_key=:ak, title_raw=:t, title_base=:tb,
                           title_key=:tk, version_suffix=:vs, album=:al, genre=:g, year=:y, duration=:d, spotify_id=:sid,
                           spotify_artist_ids=:said, isrc=:isrc WHERE id=:id""",
                        dict(p=p, size=size, h=h, ts=ts, id=v["id"], **_named(fields)))
        else:
            c["new"] += 1
            con.execute("""INSERT INTO local_tracks(path, size, content_hash, present, first_seen, last_seen, updated_at,
                           artist_raw, artists_json, artist_key, title_raw, title_base, title_key, version_suffix, album,
                           genre, year, duration, spotify_id, spotify_artist_ids, isrc)
                           VALUES(:p, :size, :h, 1, :ts, :ts, :ts, :a, :aj, :ak, :t, :tb, :tk, :vs, :al, :g, :y, :d, :sid,
                           :said, :isrc)""", dict(p=p, size=size, h=h, ts=ts, **_named(fields)))
    gone = [r["id"] for p, r in known.items() if p not in seen and r["content_hash"] in vanished]
    for i in gone:
        con.execute("UPDATE local_tracks SET present=0, updated_at=? WHERE id=? AND present=1", (ts, i))
    c["missing"] = len(gone)


def _named(f: dict) -> dict:
    return dict(a=f["artist_raw"], aj=f["artists_json"], ak=f["artist_key"], t=f["title_raw"], tb=f["title_base"],
                tk=f["title_key"], vs=f["version_suffix"], al=f["album"], g=f["genre"], y=f["year"], d=f["duration"],
                sid=f["spotify_id"], said=f["spotify_artist_ids"], isrc=f["isrc"])


# ---------- embedding ----------

def pending(con, cfg: dict) -> list[tuple[str, str]]:
    """(content_hash, path) for present files without an ok embedding (and not given up on)."""
    rows = con.execute("""
        SELECT t.content_hash, MIN(t.path) AS path FROM local_tracks t
        LEFT JOIN local_embeddings e ON e.content_hash = t.content_hash
        WHERE t.present = 1 AND (e.content_hash IS NULL OR (e.status = 'error' AND e.attempts < ?))
        GROUP BY t.content_hash ORDER BY MIN(t.path)""", (cfg["library"]["max_attempts"],)).fetchall()
    return [(r[0], r[1]) for r in rows]


def _worker_init(model_path: str, threads: int):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from . import embed
    embed.init(model_path, threads)


def _worker(content_hash: str, path: str, n_windows: int, win: float, emb_dir: str):
    from . import embed
    t = time.time()
    try:
        out = embed.npz_path(emb_dir, content_hash)
        if not out.exists():
            data = embed.embed_track(path, n_windows, win)
            data["path"] = path
            embed.save_npz(out, data)
            nw = len(data["starts"])
        else:
            nw = len(embed.load_windows(out))
        return content_hash, "ok", nw, str(out), None, time.time() - t
    except Exception as e:  # noqa: BLE001
        return content_hash, "error", None, None, repr(e)[:500], time.time() - t


def embed_pending(con, cfg: dict, workers: int | None = None, limit: int | None = None) -> dict:
    todo = pending(con, cfg)
    if limit:
        todo = todo[:limit]
    total_files = con.execute("SELECT COUNT(DISTINCT content_hash) FROM local_tracks WHERE present=1").fetchone()[0]
    already = total_files - len(pending(con, cfg)) if todo else total_files
    c = dict(todo=len(todo), ok=0, errors=0)
    if not todo:
        _progress(con, cfg, 0, 0, time.time(), 0, final=True)
        return c
    workers = workers or cfg["library"]["workers"]
    ectx = get_context("spawn")
    emb_dir = str(config.EMB_DIR)
    n_w, win = cfg["embedding"]["windows"], cfg["embedding"]["window_seconds"]
    t0 = time.time(); done = 0
    stop = {"flag": False}

    def _term(*_):
        stop["flag"] = True
        log.info("stop requested; finishing in-flight files (progress is kept per file)")
    old = signal.signal(signal.SIGTERM, _term)
    log.info("embedding %d files with %d workers (%d already embedded)", len(todo), workers, already)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ectx, initializer=_worker_init,
                                 initargs=(cfg["embedding"]["model"], cfg["library"]["threads_per_worker"])) as ex:
            it = iter(todo); inflight = set()
            while True:
                while not stop["flag"] and len(inflight) < workers * 2:
                    nxt = next(it, None)
                    if nxt is None:
                        break
                    inflight.add(ex.submit(_worker, nxt[0], nxt[1], n_w, win, emb_dir))
                if not inflight:
                    break
                fin, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fu in fin:
                    h, st, nw, f, err, secs = fu.result()
                    done += 1
                    if st == "ok":
                        c["ok"] += 1
                        con.execute("""INSERT INTO local_embeddings(content_hash, status, attempts, n_windows, file, error,
                                       seconds, updated_at) VALUES(?, 'ok', 1, ?, ?, NULL, ?, ?)
                                       ON CONFLICT(content_hash) DO UPDATE SET status='ok', attempts=attempts+1,
                                       n_windows=excluded.n_windows, file=excluded.file, error=NULL,
                                       seconds=excluded.seconds, updated_at=excluded.updated_at""",
                                    (h, nw, f, round(secs, 1), db.now()))
                    else:
                        c["errors"] += 1
                        log.warning("embed error %s: %s", h, err)
                        con.execute("""INSERT INTO local_embeddings(content_hash, status, attempts, error, seconds, updated_at)
                                       VALUES(?, 'error', 1, ?, ?, ?) ON CONFLICT(content_hash) DO UPDATE SET
                                       status='error', attempts=attempts+1, error=excluded.error, updated_at=excluded.updated_at""",
                                    (h, err, round(secs, 1), db.now()))
                    if done % 10 == 0 or done == len(todo):
                        _progress(con, cfg, done, len(todo), t0, c["errors"])
                if stop["flag"] and not inflight:
                    break
            if stop["flag"]:
                ex.shutdown(cancel_futures=True)
    finally:
        signal.signal(signal.SIGTERM, old)
    _progress(con, cfg, done, len(todo), t0, c["errors"], final=True)
    c["seconds"] = round(time.time() - t0, 1)
    c["stopped_early"] = stop["flag"]
    return c


def _progress(con, cfg, done, todo, t0, errors, final=False):
    el = time.time() - t0
    rate = done / el * 60 if el > 0 and done else None
    eta = (todo - done) / (done / el) / 60 if done and el > 0 else None
    total = con.execute("SELECT COUNT(DISTINCT content_hash) FROM local_tracks WHERE present=1").fetchone()[0]
    ok = con.execute("""SELECT COUNT(DISTINCT t.content_hash) FROM local_tracks t JOIN local_embeddings e
                        ON e.content_hash=t.content_hash AND e.status='ok' WHERE t.present=1""").fetchone()[0]
    data = {"files": con.execute("SELECT COUNT(*) FROM local_tracks WHERE present=1").fetchone()[0],
            "unique_audio": total, "embedded": ok, "pending": total - ok,
            "run": {"done": done, "todo": todo, "errors": errors, "files_per_min": round(rate, 2) if rate else None,
                    "eta_minutes": round(eta, 1) if eta is not None and not final else (0 if final else None),
                    "eta_at": (time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time() + eta * 60))
                               if eta is not None and not final else None)}}
    status.update("library", data)
    if done:
        log.info("embedded %d/%d (%.1f files/min, eta %s min, errors %d) - library %d/%d", done, todo,
                 rate or 0, f"{eta:.0f}" if eta is not None else "?", errors, ok, total)


def library_refs(con):
    """TrackRefs for present local files (for matching)."""
    from .matching import TrackRef
    out = []
    for r in con.execute("""SELECT id, artist_raw, artists_json, title_raw, duration, spotify_id, isrc
                            FROM local_tracks WHERE present=1"""):
        out.append(TrackRef(r["id"], r["artist_raw"] or "", r["title_raw"] or "", r["duration"], r["spotify_id"],
                            r["isrc"], json.loads(r["artists_json"] or "[]")))
    return out
