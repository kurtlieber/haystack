"""Candidate harvesting (token-free): Deezer (charts, editorial charts, new releases, artist radio, related
artists, playlists named after the listener's styles) and ListenBrainz similar-recordings; and, once configured
(ADR 0020), Last.fm's similar tracks and artists, the listener's own ListenBrainz recommendations and LB Radio, and
the newest albums of the labels they keep most. Sources only REACH candidates; they never contribute to the score.
Every candidate records its source(s), and the seed it was found from (ADR 0013). Already-in-library, already-liked
and already-known/offered are excluded. A Deezer track without a preview is looked up on iTunes
(harvest.preview_fallback)."""
from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from . import db, library, metadata, secrets, sources as source_stats, traits, vectors
from . import embed
from .deezer import Deezer, DeezerError
from .itunes import Itunes
from .lastfm import Lastfm
from .listenbrainz import ListenBrainz
from .matching import Index, TrackRef, compare
from .textnorm import key, song_key, split_title

log = logging.getLogger("haystack_music.harvest")

SEED_POLICIES = ("verdicts", "recent")
SEED_BUCKETS = ("top", "mid", "library")
SEED_WEEK_DAYS = 7          # an artist is seeded at most once a week per source while others are left
SEED_KEEP_DAYS = 90         # harvest_seeds rows older than this are dropped
SEEDED_SOURCES = ("deezer_artist_radio", "deezer_related", "listenbrainz_similar")
PLAYLISTS_CONSIDERED = 6    # deezer_style_search: search results read for their fan count (one request each)
PLAYLIST_TRACKS = 25        # deezer_style_search: new tracks taken from one playlist at most
PREVIEW_FALLBACKS = ("itunes", "off")
FALLBACK_PER_RUN = 50       # pool candidates without a preview looked up on iTunes per harvest
FALLBACK_RECHECK_DAYS = 30  # one iTunes didn't have is asked again after this long
LABEL_SEARCH = 50           # deezer_label: albums read from one label search
STAR_MASS = {5: 3, 4: 2, 3: 1}


@dataclass(frozen=True)
class Seed:
    """A track a source starts from. origin: rating | discovery | like | library | broad; None under seed_policy
    "recent", whose seeds are not tagged."""
    artist: str
    title: str = ""
    stars: int | None = None
    origin: str | None = None
    bucket: int = 0

    @property
    def tag(self) -> str | None:
        """What candidate_sources.detail records as seed=<tag>: the seed's stars, else its origin."""
        if self.origin is None:
            return None
        return str(self.stars) if self.stars else self.origin


def with_seed(detail: str, seed: Seed) -> str:
    return f"{detail} seed={seed.tag}" if seed.tag else detail


def _since(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def seed_buckets(con, cfg: dict, rng: random.Random) -> dict[str, list[Seed]]:
    """The seeds of seed_policy "verdicts" (ADR 0013), each bucket shuffled:
    top: tracks Rated 4-5 stars and discoveries kept, in the last harvest.seed_recent_days; 5 stars first, then 4,
      then kept discoveries without a Rating,
    mid: the other liked songs (Rated 3 stars or more, or Unrated),
    library: the library's files."""
    from . import verdicts
    cur = verdicts.current(con)
    since = _since(cfg["harvest"]["seed_recent_days"])
    names = {}
    for sid, name, arts in con.execute("SELECT spotify_id, name, artists_json FROM spotify_tracks"):
        arts = json.loads(arts or "[]")
        if arts and name:
            names[sid] = (arts[0], name)
    top: dict[str, Seed] = {}
    for sid, v in cur.items():
        if v.kind == "rating" and (v.stars or 0) >= 4 and (v.given_at or "")[:10] >= since[:10] and sid in names:
            top[sid] = Seed(*names[sid], stars=v.stars, origin="rating", bucket=0)
    for sid, artist, title, outcome in con.execute(
            """SELECT o.spotify_id, c.artist, c.title, o.outcome FROM offers o JOIN candidates c ON c.id=o.candidate_id
               WHERE o.outcome IN ('liked', 'rated') AND COALESCE(o.removed_at, o.added_at) >= ?""", (since,)):
        v = cur.get(sid)
        if sid not in top and artist and title and (v.keeps_heart if v else outcome == "liked"):
            top[sid] = Seed(artist, title, stars=v.stars if v and v.kind == "rating" else None, origin="discovery",
                            bucket=0)
    mid = []
    for (sid,) in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL ORDER BY spotify_id"):
        v = cur.get(sid)
        if sid not in top and sid in names and (v is None or v.keeps_heart):
            mid.append(Seed(*names[sid], stars=v.stars if v else None, origin="like", bucket=1))
    lib = []
    for arts, artist_raw, title in con.execute(
            "SELECT artists_json, artist_raw, title_raw FROM local_tracks WHERE present=1 ORDER BY id"):
        arts = json.loads(arts or "[]")
        a = arts[0] if arts else (artist_raw or "")
        if a and title:
            lib.append(Seed(a, title, origin="library", bucket=2))
    out = {"top": sorted(top.values(), key=lambda x: (x.artist, x.title)), "mid": mid, "library": lib}
    for v in out.values():
        rng.shuffle(v)
    out["top"].sort(key=lambda x: -(x.stars or 3.5))      # stable: random within each tier
    return out


def pick_seeds(buckets: dict[str, list[Seed]], n: int, quotas: dict, exclude: set[str] = frozenset()) -> list[Seed]:
    """Up to n seeds with distinct artists, best bucket first. Each bucket fills its quota (a share of n; what it
    can't fill passes to the next bucket), then the rest comes from the buckets in order. Artists in exclude
    (seeded for this source in the last week) are skipped while there are others."""
    out: list[tuple[int, int, Seed]] = []
    seen: set[str] = set()

    def take(b: int, k: int, skip) -> int:
        got = 0
        for pos, sd in enumerate(buckets.get(SEED_BUCKETS[b], [])):
            if got >= k or len(out) >= n:
                break
            a = key(sd.artist)
            if a and a not in seen and a not in skip:
                seen.add(a); out.append((b, pos, sd)); got += 1
        return got

    carry = 0
    for b, name in enumerate(SEED_BUCKETS):
        carry += round(n * quotas.get(name, 0))
        carry -= take(b, carry, exclude)
    for skip in (exclude, ()):
        for b in range(len(SEED_BUCKETS)):
            take(b, n - len(out), skip)
    return [sd for _, _, sd in sorted(out, key=lambda x: (x[0], x[1]))]


def broad_liked_seeds(con, cfg: dict, rng: random.Random) -> list[Seed]:
    """`harvest --broad`'s ListenBrainz seeds: every liked song, shuffled. Under seed_policy "verdicts" the liked
    songs Passed or Disliked since are left out and the rest go by their stars (Unrated as 3), each tagged."""
    from . import verdicts
    tagged = cfg["harvest"].get("seed_policy", "verdicts") == "verdicts"
    cur = verdicts.current(con) if tagged else {}
    out = []
    for sid, arts, name in con.execute("""SELECT s.spotify_id, s.artists_json, s.name FROM likes l
                                          JOIN spotify_tracks s ON s.spotify_id=l.spotify_id
                                          WHERE l.removed_at IS NULL ORDER BY l.spotify_id"""):
        arts = json.loads(arts or "[]")
        v = cur.get(sid)
        if arts and name and (v is None or v.keeps_heart):
            out.append(Seed(arts[0], name, stars=v.stars if v else None, origin="like" if tagged else None))
    rng.shuffle(out)
    if tagged:
        out.sort(key=lambda x: -(x.stars or 3))          # stable: random within each tier
    return out


def top_style_names(con, cfg: dict, n: int) -> list[str]:
    """The n Discogs styles with the most activation over the liked songs (evaluate.liked_genres), by their own
    name ("Jazz / Bebop" -> "Bebop")."""
    from .evaluate import liked_genres
    try:
        top = liked_genres(con, cfg, top=n).get("top_styles_mean_activation", [])
    except (OSError, ValueError, KeyError) as e:         # style vectors stored, but no style model here
        log.warning("deezer_style_search: no style names (%s)", e)
        return []
    out: list[str] = []
    for full, _ in top:
        name = full.split(" / ")[-1]
        if name not in out:
            out.append(name)
    return out[:n]


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


def label_mass(con) -> list[tuple[str, float]]:
    """[(label as Deezer writes it, kept mass)], most first: the labels of the albums (WP8's album rows) of the
    tracks the listener keeps. A Rating counts 3 for 5 stars, 2 for 4 and 1 for 3; a liked song without a verdict
    counts 1; a Pass, a Dislike or a Rating under 3 stars counts nothing."""
    from . import verdicts
    cur = verdicts.current(con)
    liked = {r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL")}
    mass: dict[str, float] = {}
    names: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    for sid, label in con.execute("""SELECT x.spotify_id, a.label FROM (
                                       SELECT spotify_id, album_id FROM candidates WHERE spotify_id IS NOT NULL
                                       UNION ALL SELECT spotify_id, album_id FROM liked_map) x
                                     JOIN deezer_albums a ON a.album_id=x.album_id WHERE a.label IS NOT NULL"""):
        k = key(label)
        if not k or sid in seen:
            continue
        seen.add(sid)
        v = cur.get(sid)
        w = (STAR_MASS.get(v.stars or 0, 0) if v.kind == "rating" else 0) if v else (1 if sid in liked else 0)
        if w:
            mass[k] = mass.get(k, 0) + w
            names.setdefault(k, {})[label] = names.setdefault(k, {}).get(label, 0) + 1
    best = {k: max(v, key=lambda n: (v[n], n)) for k, v in names.items()}
    return [(best[k], m) for k, m in sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))]


def _prompt_safe(name: str) -> str:
    """A name inside an LB Radio prompt's parentheses: no parentheses or colons of its own."""
    return " ".join(name.replace("(", " ").replace(")", " ").replace(":", " ").split())


def deezer_lookup(dz: Deezer, artist: str, title: str, tol: float) -> dict | None:
    """A song named elsewhere, on Deezer by search: the best of the first ten results whose artist and title both
    match at 0.8 or better, else None (also `haystack llm suggest --dry-run`, ADR 0019)."""
    if not artist or not title:
        return None
    base, _ = split_title(title)
    q = TrackRef(None, artist, title, None)
    best = None
    for c in dz.search(f"{artist} {base}", 10):
        m = compare(q, TrackRef(None, (c.get("artist") or {}).get("name", ""), c.get("title", ""), c.get("duration")),
                    tol)
        if m and m.title_sim >= 0.8 and m.artist_sim >= 0.8 and (best is None or m.title_sim + m.artist_sim > best[0]):
            best = (m.title_sim + m.artist_sim, c)
    return best[1] if best else None


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
        self.policy = h.get("seed_policy", "verdicts")
        if self.policy not in SEED_POLICIES:
            raise ValueError(f"harvest.seed_policy {self.policy!r} is not one of {', '.join(SEED_POLICIES)}")
        self.fallback = h.get("preview_fallback", "off")
        if self.fallback not in PREVIEW_FALLBACKS:
            raise ValueError(f"harvest.preview_fallback {self.fallback!r} is not one of {', '.join(PREVIEW_FALLBACKS)}")
        self._buckets: dict[str, list[Seed]] | None = None
        self._itunes: Itunes | None = None
        self._lastfm: Lastfm | None = None

    # ---- the optional providers (ADR 0020), made on first use ----
    def secret(self, name: str) -> str | None:
        try:
            return secrets.store(self.cfg).find(name)
        except secrets.SecretError as e:
            log.warning("%s is unreadable: %s", name, e)
            return None

    @property
    def itunes(self) -> Itunes | None:
        if self.fallback == "itunes" and self._itunes is None:
            self._itunes = Itunes(self.cfg["harvest"]["itunes_min_interval"])
        return self._itunes

    def lastfm(self) -> Lastfm | None:
        if self._lastfm is None:
            k = self.secret("lastfm-api-key")
            self._lastfm = Lastfm(k, self.cfg["harvest"]["lastfm_min_interval"]) if k else None
        return self._lastfm

    def skip(self, source: str, why: str) -> None:
        """A source that isn't configured: one line in the run's log (its stats) and nothing else."""
        self._st(source)["skipped"] = why

    # ---- seeds: by Verdicts (ADR 0013), or recent likes first, then the library ----
    def seed_tracks(self, n: int, source: str | None = None) -> list[Seed]:
        if self.policy == "verdicts":
            return self._verdict_seeds(n, source)
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
        return [Seed(a, t) for a, t in out]

    def seed_artists(self, n: int, source: str | None = None) -> list[Seed]:
        if self.seeds is not None:
            return [Seed(x["artist"], x["title"], origin="broad") for x in self.seeds]
        if self.policy == "verdicts":
            return self._verdict_seeds(n, source)
        seen, out = set(), []
        for sd in self.seed_tracks(n * 3):
            if key(sd.artist) not in seen:
                seen.add(key(sd.artist)); out.append(sd)
            if len(out) >= n:
                break
        return out

    def _verdict_seeds(self, n: int, source: str | None) -> list[Seed]:
        if self._buckets is None:
            self._buckets = seed_buckets(self.con, self.cfg, self.rng)
        return pick_seeds(self._buckets, n, self.cfg["harvest"]["seed_quotas"], self.recently_seeded(source))

    def recently_seeded(self, source: str | None) -> set[str]:
        """Artist keys this source was seeded from in the last week."""
        return {r[0] for r in self.con.execute("SELECT artist_key FROM harvest_seeds WHERE source=? AND at>=?",
                                               (source, _since(SEED_WEEK_DAYS)))}

    def seeded(self, source: str, sd: Seed) -> None:
        self.con.execute("INSERT INTO harvest_seeds(at, source, artist_key, stars, origin) VALUES(?,?,?,?,?)",
                         (db.now(), source, key(sd.artist), sd.stars, sd.origin))

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
                                                  via_itunes=0, no_preview=0, errors=0))

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
               mbid, deezer_artist_id, source, harvested_at, state, exclude_reason, album_id, deezer_rank)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (did, artist, key(artist), title, key(base), suffix, sk, dur, mbid, (t.get("artist") or {}).get("id"),
             source, db.now(), "excluded" if reason else "pool", reason, (t.get("album") or {}).get("id"),
             t.get("rank")))
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
            if self.preview_elsewhere(cid, did, artist, title, dur):
                st["embedded"] += 1; st["via_itunes"] += 1
                return True
            st["no_preview"] += 1
            self.con.execute("UPDATE candidates SET vector_status='no_preview' WHERE id=?", (cid,))
            return True
        try:
            traits.store(self.con, did, traits.analyse_preview(self.dz, url, bpm_hint=traits.deezer_bpm(self.dz, t)))
            self.con.execute("UPDATE candidates SET vector_status='ok' WHERE id=?", (cid,))
            st["embedded"] += 1
        except Exception as e:  # noqa: BLE001
            st["errors"] += 1
            log.warning("preview embed failed for deezer %s: %r", did, e)
            self.con.execute("UPDATE candidates SET vector_status='error' WHERE id=?", (cid,))
        return True

    def preview_elsewhere(self, cid: int, did: int, artist: str, title: str, duration) -> bool:
        """A Deezer track without a preview: embed Apple's preview of the same song instead, if iTunes has one
        (harvest.preview_fallback). The vectors stay under the Deezer id, with provider 'itunes'."""
        it = self.itunes
        if it is None:
            return False
        try:
            x = it.find(artist, title, duration, self.tol)
            if x:
                bpm = traits.deezer_bpm(self.dz, {"id": did})     # Deezer has the song, just not its preview
                traits.store(self.con, did, traits.analyse_preview(it, x["previewUrl"], bpm_hint=bpm),
                             provider="itunes")
                self.con.execute("UPDATE candidates SET vector_status='ok' WHERE id=?", (cid,))
        except Exception as e:  # noqa: BLE001 - the candidate stays without a preview, and is asked again later
            log.warning("itunes preview for deezer %s failed: %r", did, e)
            return False
        self.con.execute("""INSERT INTO preview_fallback(deezer_id, provider, found, checked_at) VALUES(?,?,?,?)
                            ON CONFLICT(deezer_id) DO UPDATE SET provider=excluded.provider, found=excluded.found,
                            checked_at=excluded.checked_at""", (did, "itunes", int(bool(x)), db.now()))
        return bool(x)

    def fallback_backfill(self, limit: int = FALLBACK_PER_RUN) -> dict:
        """Pool candidates without a preview, looked up on iTunes: never asked first, then those it didn't have a
        month ago."""
        if self.itunes is None:
            return {"skipped": "harvest.preview_fallback is off"}
        rows = self.con.execute("""SELECT c.id, c.deezer_id, c.artist, c.title, c.duration FROM candidates c
                                   LEFT JOIN preview_fallback f ON f.deezer_id=c.deezer_id
                                   WHERE c.state='pool' AND c.vector_status='no_preview'
                                     AND (f.deezer_id IS NULL OR (f.found=0 AND f.checked_at < ?))
                                   ORDER BY f.checked_at IS NOT NULL, f.checked_at, c.id LIMIT ?""",
                                (_since(FALLBACK_RECHECK_DAYS), limit)).fetchall()
        found = sum(self.preview_elsewhere(*r) for r in rows)
        return {"todo": len(rows), "embedded": found}

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
                t.setdefault("album", {"id": a["id"]})
            self._feed("deezer_new_releases", f"editorial/{eid}/album/{a['id']}", tracks[:1], cap, 1)

    def deezer_artist_radio(self, cap):
        src = "deezer_artist_radio"
        n = len(self.seeds) if self.seeds is not None else self.cfg["harvest"]["seed_artists"]
        per = max(2, cap // max(1, n)) if self.seeds is None else self.cfg["harvest"]["broad_radio_per_seed"]
        for sd in self.seed_artists(n, src):
            if self._st(src)["new"] >= cap:
                break
            aid = self.deezer_artist_id(sd.artist)
            self.seeded(src, sd)
            if aid:
                radio = [t for t in self.dz.artist_radio(aid, 25) if (t.get("artist") or {}).get("id") != aid]
                self._feed(src, with_seed(f"artist/{aid}/radio ({sd.artist})", sd), radio, cap, per)

    def deezer_related(self, cap):
        src = "deezer_related"
        n = len(self.seeds) if self.seeds is not None else self.cfg["harvest"]["seed_artists"]
        per = max(2, cap // max(1, n)) if self.seeds is None else self.cfg["harvest"]["broad_related_per_seed"]
        for sd in self.seed_artists(n, src):
            if self._st(src)["new"] >= cap:
                break
            aid = self.deezer_artist_id(sd.artist)
            self.seeded(src, sd)
            if not aid:
                continue
            got = 0
            rel = self.dz.artist_related(aid, 10)
            self.rng.shuffle(rel)
            for ra in rel[:4]:
                if got >= per or self._st(src)["new"] >= cap:
                    break
                got += self._feed(src, with_seed(f"artist/{aid}/related -> {ra.get('name')}", sd),
                                  self.dz.artist_top(ra["id"], 2), cap, 1)

    def deezer_style_search(self, cap):
        """Deezer playlists named after the listener's top styles (ADR 0013): for each of the
        harvest.style_search_styles styles, the harvest.style_search_playlists playlists with the most fans among
        the first search results, up to PLAYLIST_TRACKS new tracks each, at most cap / styles per style."""
        src = "deezer_style_search"
        h = self.cfg["harvest"]
        names = top_style_names(self.con, self.cfg, h["style_search_styles"])
        if not names:
            log.info("%s: no styles to search (no liked song has a style vector yet)", src)
            return
        per_style = max(1, cap // len(names))
        for name in names:
            if self._st(src)["new"] >= cap:
                break
            found = [p for p in self.dz.search_playlist(name, PLAYLISTS_CONSIDERED) if p.get("id")]
            lists = [p for p in (self.dz.playlist(x["id"]) for x in found) if p]
            lists.sort(key=lambda p: -(p.get("fans") or 0))
            got = 0
            for p in lists[:h["style_search_playlists"]]:
                if got >= per_style or self._st(src)["new"] >= cap:
                    break
                tracks = (p.get("tracks") or {}).get("data") or []
                got += self._feed(src, f"style/{name}/playlist/{p['id']}", tracks, cap,
                                  min(PLAYLIST_TRACKS, per_style - got))

    def listenbrainz_similar(self, cap):
        src = "listenbrainz_similar"
        seeds = self.liked_seeds if self.liked_seeds is not None else \
            self.seed_tracks(self.cfg["harvest"]["seed_recordings"], src)
        per = max(2, cap // max(1, len(seeds))) if self.liked_seeds is None else self.cfg["harvest"]["broad_lb_per_seed"]
        for sd in seeds:
            if self._st(src)["new"] >= cap:
                break
            mbid, sims = self.lb.similar_for(sd.artist, sd.title)
            self.seeded(src, sd)
            got = 0
            for s in sorted(sims, key=lambda x: -x.get("score", 0))[:per * 3]:
                if got >= per or self._st(src)["new"] >= cap:
                    break
                dz = self._deezer_lookup(s.get("artist_credit_name", ""), s.get("recording_name", ""))
                if dz:
                    got += self.offer(src, with_seed(f"similar-to {mbid} ({sd.artist} - {sd.title})", sd), dz,
                                      mbid=s.get("recording_mbid"))

    def lastfm_similar(self, cap):
        """Last.fm (ADR 0020): track.getSimilar for the seed tracks, then, for the seed artists, the top tracks of
        their three most similar artists; half the cap each, the names resolved on Deezer."""
        src = "lastfm_similar"
        lf = self.lastfm()
        if lf is None:
            return self.skip(src, "no lastfm-api-key secret")
        h = self.cfg["harvest"]
        half = cap - cap // 2
        seeds = self.seed_tracks(h["seed_recordings"], src)
        per = max(2, half // max(1, len(seeds)))
        for sd in seeds:
            if self._st(src)["new"] >= half:
                break
            sims = lf.similar_tracks(sd.artist, sd.title, per * 3)
            self.seeded(src, sd)
            got = 0
            for s in sims:
                if got >= per or self._st(src)["new"] >= half:
                    break
                dz = self._deezer_lookup(s["artist"], s["title"])
                if dz:
                    got += self.offer(src, with_seed(f"lastfm/track/{sd.artist} - {sd.title}", sd), dz)
        artists = self.seed_artists(h["seed_artists"], src)
        per = max(2, (cap - self._st(src)["new"]) // max(1, len(artists)))
        for sd in artists:
            if self._st(src)["new"] >= cap:
                break
            self.seeded(src, sd)
            got = 0
            for name in lf.similar_artists(sd.artist, 3):
                for x in lf.top_tracks(name, 2):
                    if got >= per or self._st(src)["new"] >= cap:
                        break
                    dz = self._deezer_lookup(x["artist"], x["title"])
                    if dz:
                        got += self.offer(src, with_seed(f"lastfm/artist/{sd.artist} -> {name}", sd), dz)

    def listenbrainz_recs(self, cap):
        """The listener's own weekly recommendations from ListenBrainz (ADR 0020), best first. A recommendation
        tried before (its MBID is in a detail) is not resolved again."""
        src = "listenbrainz_recs"
        user = (self.cfg["harvest"].get("listenbrainz_user") or "").strip()
        if not user:
            return self.skip(src, "harvest.listenbrainz_user is not set")
        done = {d.rsplit("/", 1)[-1] for (d,) in self.con.execute(
            "SELECT detail FROM candidate_sources WHERE source=?", (src,))}
        todo = [m for m in self.lb.recommendations(user, 100) if m not in done][:cap * 2]
        names = self.lb.recordings(todo)
        for m in todo:
            if self._st(src)["new"] >= cap:
                break
            r = names.get(m)
            dz = r and self._resolve(r["artist"], r["title"], r["isrc"])
            if dz:
                self.offer(src, f"lb-recs/{user}/{m}", dz, mbid=m)

    def listenbrainz_radio(self, cap):
        """LB Radio (ADR 0020): harvest.lb_radio_prompts prompts a run, a top style and a seed artist in turn
        (tag:(<style>), artist:(<name>)::nosim), each playlist's songs resolved on Deezer."""
        src = "listenbrainz_radio"
        token = self.secret("listenbrainz-token")
        if not token:
            return self.skip(src, "no listenbrainz-token secret")
        self.lb.token = token
        n = self.cfg["harvest"]["lb_radio_prompts"]
        styles = [(f"tag:({_prompt_safe(s).lower()})", None) for s in top_style_names(self.con, self.cfg, n)]
        artists = [(f"artist:({_prompt_safe(sd.artist)})::nosim", sd) for sd in self.seed_artists(n, src)]
        prompts = [p for pair in zip(styles, artists) for p in pair]
        prompts = (prompts + styles[len(prompts) // 2:] + artists[len(prompts) // 2:])[:n]
        per = max(2, cap // max(1, len(prompts)))
        for prompt, sd in prompts:
            if self._st(src)["new"] >= cap:
                break
            if sd:
                self.seeded(src, sd)
            songs = self.lb.radio(prompt, "easy")
            self.rng.shuffle(songs)
            got = 0
            for x in songs[:per * 3]:
                if got >= per or self._st(src)["new"] >= cap:
                    break
                dz = self._deezer_lookup(x["artist"], x["title"])
                if dz:
                    got += self.offer(src, with_seed(f"lb-radio {prompt}", sd) if sd else f"lb-radio {prompt}", dz,
                                      mbid=x["mbid"])

    def deezer_label(self, cap):
        """The newest albums of the labels the listener keeps most (ADR 0020): harvest.label_count labels by
        label_mass, harvest.label_albums albums each, two tracks per album (Deezer's most played first). Deezer's
        search can't sort by date, so the newest are the highest album ids; an album already in the candidates
        is skipped without a request."""
        src = "deezer_label"
        h = self.cfg["harvest"]
        labels = label_mass(self.con)[:h["label_count"]]
        if not labels:
            return self.skip(src, "no label yet (no kept track has an album row from Deezer's metadata)")
        per = max(2, -(-cap // len(labels)))
        have = {r[0] for r in self.con.execute("SELECT DISTINCT album_id FROM candidates WHERE album_id > 0")}
        for name, _ in labels:
            if self._st(src)["new"] >= cap:
                break
            found = [a for a in self.dz.search_album(f'label:"{name}"', LABEL_SEARCH)
                     if a.get("id") and a["id"] not in have]
            got = albums = 0
            for a in sorted(found, key=lambda a: -a["id"]):
                if albums >= h["label_albums"] or got >= per or self._st(src)["new"] >= cap:
                    break
                full = self.dz.album(a["id"]) or {}
                metadata.store_album(self.con, a["id"], full)
                have.add(a["id"])
                if key(full.get("label")) != key(name):        # the search matches loosely
                    continue
                albums += 1
                tracks = sorted((full.get("tracks") or {}).get("data") or [], key=lambda x: -(x.get("rank") or 0))
                for x in tracks:
                    x.setdefault("artist", full.get("artist") or {})
                    x.setdefault("album", {"id": a["id"]})
                got += self._feed(src, f"label/{name}/album/{a['id']}", tracks, cap, min(2, per - got))

    def _resolve(self, artist: str, title: str, isrc: str | None = None) -> dict | None:
        """A song named elsewhere, on Deezer: by its ISRC when it has one and the names agree, else by search."""
        if isrc:
            t = self.dz.track_by_isrc(isrc)
            if t and t.get("id") and compare(TrackRef(None, artist, title, None), TrackRef(
                    None, (t.get("artist") or {}).get("name", ""), t.get("title", ""), None), self.tol):
                return t
        return self._deezer_lookup(artist, title)

    def _deezer_lookup(self, artist: str, title: str) -> dict | None:
        return deezer_lookup(self.dz, artist, title, self.tol)

    def retry_errors(self, limit: int = 100) -> dict:
        """Re-embed pool candidates whose preview failed or was interrupted earlier (fresh URL from /track/{id})."""
        c = dict(retried=0, embedded=0)
        rows = self.con.execute("SELECT id, deezer_id, artist, title, duration FROM candidates WHERE state='pool' "
                                "AND (vector_status='error' OR vector_status IS NULL) LIMIT ?", (limit,)).fetchall()
        for cid, did, artist, title, dur in rows:
            c["retried"] += 1
            try:
                t = self.dz.track(did) or {}
                if not t.get("preview"):
                    if self.preview_elsewhere(cid, did, artist, title, dur):
                        c["embedded"] += 1
                    else:
                        self.con.execute("UPDATE candidates SET vector_status='no_preview' WHERE id=?", (cid,))
                    continue
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
        tempo = self._retempo() if retry else {"skipped": True}
        sources = [s for s in (sources or h["sources"]) if s not in h.get("retired_sources", [])]
        caps = self.caps(sources, cap)
        self.con.execute("DELETE FROM harvest_seeds WHERE at < ?", (_since(SEED_KEEP_DAYS),))
        for s in sources:
            fn = getattr(self, s, None)
            if fn is None or s.startswith("_"):
                log.warning("unknown source %s", s); continue
            self._st(s)
            try:
                fn(caps[s]["cap"])
            except (DeezerError, Exception) as e:  # noqa: BLE001 - one bad source must not stop the others
                self._st(s)["errors"] += 1
                log.exception("source %s failed: %r", s, e)
            log.info("source %s: %s", s, self.stats[s])
        meta = metadata.backfill(self.con, self.dz, h["metadata_per_run"]) if retry else {"skipped": True}
        elsewhere = self.fallback_backfill() if retry else {"skipped": True}
        return {"per_source": self.stats, "caps": caps, "retry_previous_errors": retried, "style_backfill": styles,
                "readings_backfill": readings, "tempo_fold": tempo, "metadata_backfill": meta,
                "preview_fallback": elsewhere,
                "deezer_requests": self.dz.requests, "listenbrainz_requests": self.lb.requests,
                "lastfm_requests": self._lastfm.requests if self._lastfm else 0,
                "itunes_requests": self._itunes.requests if self._itunes else 0}

    def caps(self, sources: list[str], cap: int | None = None) -> dict[str, dict]:
        """This run's cap per source: `--limit-per-source` as given; otherwise each source's own, scaled by its kept
        rate (ADR 0013), except that in broad mode the seeded sources get harvest.broad_max_new_per_source (their
        per-seed limits do the limiting)."""
        if cap:
            return {s: {"cap": cap, "why": "--limit-per-source"} for s in sources}
        out = source_stats.caps(self.cfg, sources, source_stats.rates(self.con))
        if self.seeds is not None or self.liked_seeds is not None:
            for s in set(out) & set(SEEDED_SOURCES):
                out[s] = {"cap": self.cfg["harvest"]["broad_max_new_per_source"], "why": "broad"}
        return out

    def _readings_backfill(self, limit: int) -> dict:
        """Readings for stored previews (Discoveries and liked songs first); skipped while the models are missing."""
        if traits.missing_models():
            return {"skipped": f"models missing: {traits.missing_models()} (haystack models fetch)"}
        return traits.backfill(self.con, self.dz, limit)

    def _retempo(self) -> dict:
        """Drum & bass and jungle tempos folded (traits.retempo); skipped while the models are missing."""
        if traits.missing_models():
            return {"skipped": "models missing"}
        return traits.retempo(self.con)
