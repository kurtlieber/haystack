"""Candidate harvesting (token-free): Deezer (charts, editorial charts, new releases, artist radio, related
artists) and ListenBrainz similar-recordings. Sources only REACH candidates; they never contribute to the score.
Every candidate records its source(s). Already-in-library, already-liked and already-known/offered are excluded."""
from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
from datetime import date

from . import db, library, traits, vectors
from . import embed
from .deezer import Deezer, DeezerError
from .listenbrainz import ListenBrainz
from .matching import Index, TrackRef, compare
from .textnorm import key, song_key, split_title

log = logging.getLogger("haystack_music.harvest")


def _liked_refs(con) -> list[TrackRef]:
    out = []
    for r in con.execute("""SELECT s.spotify_id, s.name, s.artists_json, s.duration, s.isrc FROM spotify_tracks s
                            JOIN likes l ON l.spotify_id=s.spotify_id WHERE l.removed_at IS NULL"""):
        arts = json.loads(r["artists_json"] or "[]")
        out.append(TrackRef(r["spotify_id"], arts[0] if arts else "", r["name"] or "", r["duration"],
                            r["spotify_id"], r["isrc"], arts))
    return out


def stratified_seeds(con, n: int, rng: random.Random) -> list[dict]:
    """~n seed tracks with distinct artists, spread across genre buckets (the library's genre tag; liked songs
    take the genre of their local file, liked songs not in the library form their own bucket). Buckets get
    seeds in proportion to sqrt(size), so small genres are represented and big ones don't dominate; within a
    bucket liked songs are drawn first."""
    liked_local = {r[0] for r in con.execute("""SELECT m.local_id FROM liked_map m JOIN likes l ON l.spotify_id=m.spotify_id
                                                WHERE l.removed_at IS NULL AND m.local_id IS NOT NULL""")}
    buckets: dict[str, list[dict]] = {}
    for r in con.execute("SELECT id, artist_raw, artists_json, title_raw, genre FROM local_tracks WHERE present=1"):
        arts = json.loads(r["artists_json"] or "[]")
        a = arts[0] if arts else (r["artist_raw"] or "")
        if not a or not r["title_raw"]:
            continue
        g = key(r["genre"]) or "untagged"
        buckets.setdefault(g, []).append(dict(artist=a, title=r["title_raw"], liked=r["id"] in liked_local, bucket=g))
    for r in con.execute("""SELECT s.artists_json, s.name FROM likes l JOIN spotify_tracks s ON s.spotify_id=l.spotify_id
                            LEFT JOIN liked_map m ON m.spotify_id=l.spotify_id
                            WHERE l.removed_at IS NULL AND m.local_id IS NULL"""):
        arts = json.loads(r[0] or "[]")
        if arts and r[1]:
            buckets.setdefault("liked (not in library)", []).append(dict(artist=arts[0], title=r[1], liked=True,
                                                                       bucket="liked (not in library)"))
    small = [g for g, v in buckets.items() if len(v) < 5]
    for g in small:
        for x in buckets.pop(g):
            x["bucket"] = "other"
            buckets.setdefault("other", []).append(x)
    if not buckets:
        return []
    names = sorted(buckets)
    w = {g: math.sqrt(len(buckets[g])) for g in names}
    tot = sum(w.values())
    quota = {g: max(1, round(n * w[g] / tot)) for g in names}
    for g in names:
        rng.shuffle(buckets[g])
        buckets[g].sort(key=lambda x: not x["liked"])          # stable: liked first, random within
    out, seen = [], set()
    for g in names:
        got = 0
        for x in buckets[g]:
            if got >= quota[g]:
                break
            k = key(x["artist"])
            if k and k not in seen:
                seen.add(k); out.append(x); got += 1
    rng.shuffle(out)
    return out[:n]


class Harvester:
    def __init__(self, con, cfg: dict, rng: random.Random | None = None, seeds: list[dict] | None = None,
                 liked_seeds: list[tuple[str, str]] | None = None):
        self.con, self.cfg = con, cfg
        self.seeds = seeds                  # broad mode: stratified seed tracks (artist radio / related)
        self.liked_seeds = liked_seeds      # broad mode: liked songs for ListenBrainz
        h = cfg["harvest"]
        self.dz = Deezer(h["deezer_min_interval"])
        self.lb = ListenBrainz(h["listenbrainz_min_interval"], h["listenbrainz_algorithm"],
                               contact=h["listenbrainz_contact"])
        self.rng = rng or random.Random(date.today().toordinal())
        self.lib = Index(library.library_refs(con))
        self.liked = Index(_liked_refs(con))
        self.known_dz = {r[0]: r[1] for r in con.execute("SELECT deezer_id, id FROM candidates WHERE deezer_id IS NOT NULL")}
        self.known_song = {r[0]: r[1] for r in con.execute("SELECT song_key, id FROM candidates")}
        self.tol = cfg["matching"]["duration_tolerance"]
        self.stats: dict[str, dict] = {}

    # ---- seeds (positives: liked songs first, then the library) ----
    def seed_tracks(self, n: int) -> list[tuple[str, str]]:
        liked = self.con.execute("""SELECT s.artists_json, s.name FROM likes l JOIN spotify_tracks s
                                    ON s.spotify_id=l.spotify_id WHERE l.removed_at IS NULL
                                    ORDER BY l.added_at DESC LIMIT 200""").fetchall()
        liked = [(json.loads(a or "[]")[0], t) for a, t in liked if json.loads(a or "[]")]
        lib = [(json.loads(a or "[]")[0] if json.loads(a or "[]") else "", t) for a, t in self.con.execute(
            "SELECT artists_json, title_raw FROM local_tracks WHERE present=1")]
        lib = [x for x in lib if x[0] and x[1]]
        n_like = min(len(liked), int(round(n * 0.7))) if liked else 0
        out = self.rng.sample(liked, n_like) if n_like else []
        out += self.rng.sample(lib, min(len(lib), n - len(out)))
        return out

    def seed_artists(self, n: int) -> list[str]:
        if self.seeds is not None:
            return [x["artist"] for x in self.seeds]
        seen, out = set(), []
        for a, _ in self.seed_tracks(n * 3):
            if key(a) not in seen:
                seen.add(key(a)); out.append(a)
            if len(out) >= n:
                break
        return out

    def deezer_artist_id(self, name: str) -> int | None:
        k = key(name)
        r = self.con.execute("SELECT deezer_artist_id FROM deezer_artists WHERE name_key=?", (k,)).fetchone()
        if r:
            return r[0]
        aid = None
        for a in self.dz.search_artist(name):
            if key(a.get("name")) == k:
                aid = a["id"]; break
        self.con.execute("INSERT OR REPLACE INTO deezer_artists(name_key, deezer_artist_id, name, checked_at) VALUES(?,?,?,?)",
                         (k, aid, name, db.now()))
        return aid

    # ---- intake ----
    def _st(self, source):
        return self.stats.setdefault(source, dict(seen=0, new=0, dupe=0, in_library=0, liked=0, embedded=0,
                                                  no_preview=0, errors=0))

    def offer(self, source: str, detail: str, t: dict, mbid: str | None = None) -> bool:
        """Consider one Deezer track object. Returns True if it became a new candidate."""
        st = self._st(source); st["seen"] += 1
        did = t.get("id"); artist = (t.get("artist") or {}).get("name") or ""
        title = t.get("title") or ""; dur = t.get("duration")
        if not did or not artist or not title:
            return False
        sk = song_key(artist, title)
        cid = self.known_dz.get(did) or self.known_song.get(sk)
        if cid:
            st["dupe"] += 1
            self.con.execute("INSERT OR IGNORE INTO candidate_sources(candidate_id, source, detail, seen_at) VALUES(?,?,?,?)",
                             (cid, source, detail, db.now()))
            return False
        q = TrackRef(None, artist, title, dur)
        reason = None
        m = self.lib.match(q, self.tol)
        if m and m.confidence in ("high", "medium"):
            reason = "in_library"; st["in_library"] += 1
        else:
            m = self.liked.match(q, self.tol)
            if m and m.confidence in ("high", "medium"):
                reason = "liked"; st["liked"] += 1
        base, suffix = split_title(title)
        try:
            cur = self.con.execute(
            """INSERT INTO candidates(deezer_id, artist, artist_key, title, title_key, version_suffix, song_key, duration,
               mbid, deezer_artist_id, source, harvested_at, state, exclude_reason)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (did, artist, key(artist), title, key(base), suffix, sk, dur, mbid, (t.get("artist") or {}).get("id"),
             source, db.now(), "excluded" if reason else "pool", reason))
        except sqlite3.IntegrityError:      # a parallel harvester took this Deezer track first
            st["dupe"] += 1
            return False
        cid = cur.lastrowid
        self.known_dz[did] = cid; self.known_song[sk] = cid
        self.con.execute("INSERT OR IGNORE INTO candidate_sources(candidate_id, source, detail, seen_at) VALUES(?,?,?,?)",
                         (cid, source, detail, db.now()))
        if reason:
            return False
        st["new"] += 1
        url = t.get("preview")
        if not url:
            st["no_preview"] += 1
            self.con.execute("UPDATE candidates SET vector_status='no_preview' WHERE id=?", (cid,))
            return True
        try:
            traits.store(self.con, did, traits.analyse_preview(self.dz, url, bpm_hint=t.get("bpm")))
            self.con.execute("UPDATE candidates SET vector_status='ok' WHERE id=?", (cid,))
            st["embedded"] += 1
        except Exception as e:  # noqa: BLE001
            st["errors"] += 1
            log.warning("preview embed failed for deezer %s: %r", did, e)
            self.con.execute("UPDATE candidates SET vector_status='error' WHERE id=?", (cid,))
        return True

    def _feed(self, source, detail, tracks, cap, per_feed=None):
        added = 0
        for t in tracks:
            if self.stats.get(source, {}).get("new", 0) >= cap or (per_feed and added >= per_feed):
                break
            if self.offer(source, detail, t):
                added += 1
        return added

    # ---- sources ----
    def chart_room(self) -> int:
        """Chart + editorial-chart candidates may make up at most harvest.chart_max_share of the pool."""
        h = self.cfg["harvest"]
        srcs = tuple(h["chart_sources"])
        q = f"SELECT COUNT(*), SUM(source IN ({','.join('?' * len(srcs))})) FROM candidates WHERE state='pool' AND vector_status='ok'"
        pool, chart = self.con.execute(q, srcs).fetchone()
        share = h["chart_max_share"]
        # adding x keeps (chart + x) / (pool + x) <= share
        return max(0, int((share * (pool or 0) - (chart or 0)) / (1 - share)))

    def deezer_chart(self, cap):
        cap = min(cap, self.chart_room())
        if cap > 0:
            self._feed("deezer_chart", "chart/0", self.dz.chart_tracks(0, 100), cap)

    def deezer_editorial(self, cap):
        cap = min(cap, self.chart_room())
        if cap <= 0:
            return
        ids = [i for i in self.cfg["harvest"]["deezer_editorial_ids"] if i != 0]
        self.rng.shuffle(ids)
        per = max(3, cap // max(1, len(ids)))
        for eid in ids:
            self._feed("deezer_editorial", f"chart/{eid}", self.dz.chart_tracks(eid, 50), cap, per)

    def deezer_new_releases(self, cap):
        albums = []
        for eid in self.cfg["harvest"]["deezer_editorial_ids"]:
            albums += [(eid, a) for a in self.dz.editorial_releases(eid, 25)]
        self.rng.shuffle(albums)
        for eid, a in albums:
            if self._st("deezer_new_releases")["new"] >= cap:
                break
            tracks = self.dz.album_tracks(a["id"], 2)
            for t in tracks:
                t.setdefault("artist", a.get("artist") or {})
            self._feed("deezer_new_releases", f"editorial/{eid}/album/{a['id']}", tracks[:1], cap, 1)

    def deezer_artist_radio(self, cap):
        n = len(self.seeds) if self.seeds is not None else self.cfg["harvest"]["seed_artists"]
        per = max(2, cap // max(1, n)) if self.seeds is None else self.cfg["harvest"]["broad_radio_per_seed"]
        for a in self.seed_artists(n):
            if self._st("deezer_artist_radio")["new"] >= cap:
                break
            aid = self.deezer_artist_id(a)
            if aid:
                radio = [t for t in self.dz.artist_radio(aid, 25) if (t.get("artist") or {}).get("id") != aid]
                self._feed("deezer_artist_radio", f"artist/{aid}/radio ({a})", radio, cap, per)

    def deezer_related(self, cap):
        n = len(self.seeds) if self.seeds is not None else self.cfg["harvest"]["seed_artists"]
        per = max(2, cap // max(1, n)) if self.seeds is None else self.cfg["harvest"]["broad_related_per_seed"]
        for a in self.seed_artists(n):
            if self._st("deezer_related")["new"] >= cap:
                break
            aid = self.deezer_artist_id(a)
            if not aid:
                continue
            got = 0
            rel = self.dz.artist_related(aid, 10)
            self.rng.shuffle(rel)
            for ra in rel[:4]:
                if got >= per or self._st("deezer_related")["new"] >= cap:
                    break
                got += self._feed("deezer_related", f"artist/{aid}/related -> {ra.get('name')}",
                                  self.dz.artist_top(ra["id"], 2), cap, 1)

    def listenbrainz_similar(self, cap):
        seeds = self.liked_seeds if self.liked_seeds is not None else self.seed_tracks(self.cfg["harvest"]["seed_recordings"])
        per = max(2, cap // max(1, len(seeds))) if self.liked_seeds is None else self.cfg["harvest"]["broad_lb_per_seed"]
        for a, t in seeds:
            if self._st("listenbrainz_similar")["new"] >= cap:
                break
            mbid, sims = self.lb.similar_for(a, t)
            got = 0
            for s in sorted(sims, key=lambda x: -x.get("score", 0))[:per * 3]:
                if got >= per or self._st("listenbrainz_similar")["new"] >= cap:
                    break
                dz = self._deezer_lookup(s.get("artist_credit_name", ""), s.get("recording_name", ""))
                if dz:
                    got += self.offer("listenbrainz_similar", f"similar-to {mbid} ({a} - {t})", dz,
                                      mbid=s.get("recording_mbid"))

    def _deezer_lookup(self, artist: str, title: str) -> dict | None:
        if not artist or not title:
            return None
        base, _ = split_title(title)
        q = TrackRef(None, artist, title, None)
        best = None
        for c in self.dz.search(f"{artist} {base}", 10):
            m = compare(q, TrackRef(None, (c.get("artist") or {}).get("name", ""), c.get("title", ""), c.get("duration")),
                        self.tol)
            if m and m.title_sim >= 0.8 and m.artist_sim >= 0.8 and (best is None or m.title_sim + m.artist_sim > best[0]):
                best = (m.title_sim + m.artist_sim, c)
        return best[1] if best else None

    def retry_errors(self, limit: int = 100) -> dict:
        """Re-embed pool candidates whose preview failed or was interrupted earlier (fresh URL from /track/{id})."""
        c = dict(retried=0, embedded=0)
        rows = self.con.execute("SELECT id, deezer_id FROM candidates WHERE state='pool' AND (vector_status='error' OR vector_status IS NULL) "
                                "LIMIT ?", (limit,)).fetchall()
        for cid, did in rows:
            c["retried"] += 1
            try:
                t = self.dz.track(did) or {}
                if not t.get("preview"):
                    self.con.execute("UPDATE candidates SET vector_status='no_preview' WHERE id=?", (cid,)); continue
                traits.store(self.con, did, traits.analyse_preview(self.dz, t["preview"], bpm_hint=t.get("bpm")))
                self.con.execute("UPDATE candidates SET vector_status='ok' WHERE id=?", (cid,))
                c["embedded"] += 1
            except Exception as e:  # noqa: BLE001
                log.warning("retry %s failed: %r", did, e)
        return c

    def backfill_styles(self, limit: int = 200) -> dict:
        """Style (genre_discogs400) vectors for stored previews that predate the style head: liked songs first,
        then pool candidates. Needs a fresh preview URL (/track/{id}) plus the preview itself."""
        c = dict(todo=0, done=0, no_preview=0, errors=0)
        rows = self.con.execute("""
            SELECT v.deezer_id, 0 AS pri FROM deezer_vectors v JOIN liked_map m ON m.deezer_id=v.deezer_id
              WHERE v.style400 IS NULL AND m.vector_status='ok'
            UNION
            SELECT v.deezer_id, 1 AS pri FROM deezer_vectors v JOIN candidates c ON c.deezer_id=v.deezer_id
              WHERE v.style400 IS NULL AND c.state IN ('pool','offered','liked','negative') AND c.vector_status='ok'
            ORDER BY pri LIMIT ?""", (limit,)).fetchall()
        c["todo"] = len(rows)
        for did, _ in rows:
            try:
                t = self.dz.track(did) or {}
                if not t.get("preview"):
                    c["no_preview"] += 1; continue
                data = self.dz.preview_bytes(t["preview"])
                try:
                    audio = embed.decode_bytes(data)
                finally:
                    del data
                sty = embed.style_from_patches(embed.patches(embed.mel_frames(audio)))
                del audio
                vectors.put_style(self.con, did, sty)
                c["done"] += 1
            except Exception as e:  # noqa: BLE001
                c["errors"] += 1
                log.warning("style backfill %s failed: %r", did, e)
        return c

    def run(self, sources: list[str] | None = None, cap: int | None = None, retry: bool = True) -> dict:
        h = self.cfg["harvest"]
        retried = self.retry_errors() if retry else {"skipped": True}
        styles = self.backfill_styles(h["style_backfill_per_run"]) if retry else {"skipped": True}
        readings = self._readings_backfill(h.get("traits_backfill_per_run", 150)) if retry else {"skipped": True}
        cap = cap or h["max_new_per_source"]
        sources = [s for s in (sources or h["sources"]) if s not in h.get("retired_sources", [])]
        for s in sources:
            fn = getattr(self, s, None)
            if fn is None:
                log.warning("unknown source %s", s); continue
            self._st(s)
            try:
                fn(cap)
            except (DeezerError, Exception) as e:  # noqa: BLE001 - one bad source must not stop the others
                self._st(s)["errors"] += 1
                log.exception("source %s failed: %r", s, e)
            log.info("source %s: %s", s, self.stats[s])
        return {"per_source": self.stats, "retry_previous_errors": retried, "style_backfill": styles,
                "readings_backfill": readings, "deezer_requests": self.dz.requests,
                "listenbrainz_requests": self.lb.requests}

    def _readings_backfill(self, limit: int) -> dict:
        """Readings for stored previews (Discoveries and liked songs first); skipped while the models are missing."""
        if traits.missing_models():
            return {"skipped": f"models missing: {traits.missing_models()} (haystack models fetch)"}
        return traits.backfill(self.con, self.dz, limit)
