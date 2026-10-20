"""The sources of ADR 0020 (Last.fm, the listener's ListenBrainz recommendations, LB Radio, the listener's labels)
and the iTunes preview fallback, over a synthetic world with fake clients; then the clients themselves over a fake
HTTP session. Nothing here touches the network or the real state and secrets folders."""
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from haystack_music import config, db, harvest, metadata, report, throttle, traits, vectors, verdicts
from haystack_music.harvest import Harvester, label_mass
from haystack_music.itunes import Itunes, ItunesError
from haystack_music.lastfm import Lastfm, LastfmError
from haystack_music.listenbrainz import ListenBrainz, ListenBrainzError

SOURCES = ["lastfm_similar", "listenbrainz_recs", "listenbrainz_radio", "deezer_label"]


def sid(n: str) -> str:
    return n.replace("_", "").ljust(22, "0")


def track(did: int, artist: str, title: str, preview=True, rank=0) -> dict:
    return {"id": did, "title": title, "duration": 200, "rank": rank, "artist": {"id": 9000 + did, "name": artist},
            "preview": f"https://example.test/{did}.mp3" if preview else ""}


class FakeDeezer:
    """A catalogue of tracks found by an exact "<artist> <title>" search, ISRCs, and label albums."""
    def __init__(self, *a, **k):
        self.requests = 0
        self.asked: list[str] = []
        self.catalog: list[dict] = []
        self.isrcs: dict[str, dict] = {}
        self.albums: dict[int, dict] = {}
        self.label_search: dict[str, list[int]] = {}

    def add(self, artist, title, preview=True) -> dict:
        t = track(100 + len(self.catalog), artist, title, preview)
        self.catalog.append(t)
        return t

    def search(self, q, limit=25):
        self.asked.append(f"search {q}")
        return [t for t in self.catalog if f"{t['artist']['name']} {t['title']}" == q][:limit]

    def track_by_isrc(self, isrc):
        self.asked.append(f"isrc {isrc}")
        return self.isrcs.get(isrc)

    def search_album(self, q, limit=50):
        self.asked.append(f"search_album {q}")
        return [{"id": i} for i in self.label_search.get(q, [])][:limit]

    def album(self, aid):
        self.asked.append(f"album/{aid}")
        return self.albums.get(aid)

    def track(self, did):
        return next((t for t in self.catalog if t["id"] == did), None)

    def chart_tracks(self, eid=0, limit=100):
        return []


class FakeLB:
    def __init__(self, *a, token=None, **k):
        self.requests = 0
        self.token = token
        self.recs: list[str] = []
        self.meta: dict[str, dict] = {}
        self.playlists: dict[str, list[dict]] = {}
        self.asked: list = []

    def recommendations(self, user, count=100):
        self.asked.append(("recs", user))
        return self.recs[:count]

    def recordings(self, mbids, per_request=25):
        self.asked.append(("recordings", tuple(mbids)))
        return {m: self.meta[m] for m in mbids if m in self.meta}

    def radio(self, prompt, mode="easy"):
        self.asked.append(("radio", prompt, self.token))
        return list(self.playlists.get(prompt, []))


class FakeLastfm:
    made: list = []
    refuse = False

    def __init__(self, key, min_interval=1.0):
        FakeLastfm.made.append(key)
        self.requests = 0
        self.tracks: dict[tuple, list[dict]] = {}
        self.artists: dict[str, list[str]] = {}

    def similar_tracks(self, artist, title, limit=20):
        if self.refuse:
            raise LastfmError("track.getSimilar: the API key was refused")
        return [{"artist": f"Sim {artist}", "title": f"Like {title} {i}", "match": 1 - i / 10} for i in range(limit)]

    def similar_artists(self, artist, limit=3):
        return [f"Near {artist} {i}" for i in range(limit)]

    def top_tracks(self, artist, limit=3):
        return [{"artist": artist, "title": f"Hit {i}"} for i in range(limit)]


class FakeItunes:
    made = 0

    def __init__(self, min_interval=3.0):
        FakeItunes.made += 1
        self.requests = 0
        self.has: set[tuple] = set()
        self.asked: list[tuple] = []

    def find(self, artist, title, duration, tol=3.0):
        self.asked.append((artist, title))
        return {"previewUrl": f"https://itunes.test/{title}.m4a"} if (artist, title) in self.has else None


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    monkeypatch.setattr(harvest, "Deezer", FakeDeezer)
    monkeypatch.setattr(harvest, "ListenBrainz", FakeLB)
    monkeypatch.setattr(harvest, "Lastfm", FakeLastfm)
    monkeypatch.setattr(harvest, "Itunes", FakeItunes)
    monkeypatch.setattr(FakeLastfm, "made", [])
    monkeypatch.setattr(FakeLastfm, "refuse", False)
    monkeypatch.setattr(FakeItunes, "made", 0)
    stored: list[tuple] = []
    monkeypatch.setattr(traits, "analyse_preview", lambda client, url, bpm_hint=None: url)
    monkeypatch.setattr(traits, "store", lambda con, did, a, provider="deezer": stored.append((did, a, provider)))
    con = db.connect(tmp_path / "t.db")
    cfg = config.load(tmp_path / "none.toml")
    cfg["secrets"]["dir"] = str(tmp_path / "secrets")
    return con, cfg, stored


def secret(cfg, name, value):
    from haystack_music import secrets
    secrets.FileStore(cfg).put(name, value)


def like(con, name: str, artist: str, stars=None, kind="rating"):
    con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(name),))
    con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json) VALUES(?,?,?)",
                (sid(name), f"Song {name}", f'["{artist}"]'))
    if stars or kind != "rating":
        verdicts.record(con, sid(name), kind, stars)


def details(con, source):
    return [r[0] for r in con.execute("SELECT detail FROM candidate_sources WHERE source=? ORDER BY rowid", (source,))]


# ---------------- gates ----------------

def test_each_new_source_skips_without_what_it_needs_and_asks_nobody(world):
    con, cfg, _ = world
    like(con, "a", "Five Star", 5)
    h = Harvester(con, cfg, rng=random.Random(1))
    out = h.run(SOURCES, cap=10, retry=False)
    why = {s: out["per_source"][s].get("skipped") for s in SOURCES}
    assert why == {"lastfm_similar": "no lastfm-api-key secret",
                   "listenbrainz_recs": "harvest.listenbrainz_user is not set",
                   "listenbrainz_radio": "no listenbrainz-token secret",
                   "deezer_label": "no label yet (no kept track has an album row from Deezer's metadata)"}
    assert not any(out["per_source"][s]["errors"] for s in SOURCES)
    assert FakeLastfm.made == [] and h.lb.asked == [] and h.dz.asked == []
    assert (out["lastfm_requests"], out["itunes_requests"]) == (0, 0)
    assert con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0


def test_the_new_sources_are_on_by_default_and_a_bad_fallback_fails(world):
    con, cfg, _ = world
    assert set(SOURCES) <= set(config.DEFAULTS["harvest"]["sources"])
    assert config.DEFAULTS["harvest"]["preview_fallback"] == "off"         # Apple is asked only once you say so
    from haystack_music.review import SOURCE_LABELS
    assert set(config.DEFAULTS["harvest"]["sources"]) <= set(SOURCE_LABELS)  # every card names its source
    cfg["harvest"]["preview_fallback"] = "spotify"
    with pytest.raises(ValueError, match="preview_fallback"):
        Harvester(con, cfg)


# ---------------- Last.fm ----------------

def test_lastfm_takes_half_from_similar_tracks_and_half_from_similar_artists(world):
    con, cfg, _ = world
    secret(cfg, "lastfm-api-key", "k3y")
    like(con, "a", "Five Star", 5)
    like(con, "b", "Four Star", 4)
    h = Harvester(con, cfg, rng=random.Random(1))
    for i in range(6):
        h.dz.add("Sim Five Star", f"Like Song a {i}")
        h.dz.add("Sim Four Star", f"Like Song b {i}")
    for who in ("Five Star", "Four Star"):
        for j in range(3):
            for i in range(2):
                h.dz.add(f"Near {who} {j}", f"Hit {i}")
    h.lastfm_similar(8)
    got = details(con, "lastfm_similar")
    tracks = [d for d in got if d.startswith("lastfm/track/")]
    artists = [d for d in got if d.startswith("lastfm/artist/")]
    assert FakeLastfm.made == ["k3y"] and len(got) == 8
    assert len(tracks) == 4 and all(d.endswith((" seed=5", " seed=4")) for d in got)
    assert "lastfm/track/Five Star - Song a seed=5" in tracks
    assert "lastfm/artist/Five Star -> Near Five Star 0 seed=5" in artists
    seeded = {r[0] for r in con.execute("SELECT artist_key FROM harvest_seeds WHERE source='lastfm_similar'")}
    assert seeded == {harvest.key("Five Star"), harvest.key("Four Star")}


def test_lastfm_names_deezer_lacks_are_passed_over_and_a_refused_key_fails_only_its_source(world):
    con, cfg, _ = world
    secret(cfg, "lastfm-api-key", "k3y")
    like(con, "a", "Five Star", 5)
    h = Harvester(con, cfg, rng=random.Random(1))
    h.lastfm_similar(6)                                  # Deezer has none of the names
    assert details(con, "lastfm_similar") == [] and any(q.startswith("search Sim Five Star") for q in h.dz.asked)
    FakeLastfm.refuse = True
    out = Harvester(con, cfg, rng=random.Random(1)).run(["lastfm_similar", "deezer_label"], cap=6, retry=False)
    assert out["per_source"]["lastfm_similar"]["errors"] == 1
    assert out["per_source"]["deezer_label"]["skipped"]


# ---------------- ListenBrainz ----------------

def test_recommendations_resolve_by_isrc_then_by_name_once_each(world):
    con, cfg, _ = world
    cfg["harvest"]["listenbrainz_user"] = "listener"
    h = Harvester(con, cfg, rng=random.Random(1))
    lb, dz = h.lb, h.dz
    lb.recs = ["m1", "m2", "m3", "m4"]
    lb.meta = {"m1": {"artist": "Isrc Artist", "title": "Isrc Song", "isrc": "XX0000000001"},
               "m2": {"artist": "Named Artist", "title": "Named Song", "isrc": "XX0000000002"},
               "m4": {"artist": "Nowhere", "title": "Not On Deezer", "isrc": None}}    # m3: unknown
    dz.isrcs["XX0000000001"] = dz.add("Isrc Artist", "Isrc Song")
    dz.isrcs["XX0000000002"] = track(999, "Someone Else", "Another Song")      # names disagree: search
    named = dz.add("Named Artist", "Named Song")
    h.listenbrainz_recs(10)
    rows = con.execute("SELECT deezer_id, mbid FROM candidates ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [(dz.catalog[0]["id"], "m1"), (named["id"], "m2")]
    assert details(con, "listenbrainz_recs") == ["lb-recs/listener/m1", "lb-recs/listener/m2"]
    assert "search Isrc Artist Isrc Song" not in dz.asked and "search Named Artist Named Song" in dz.asked
    h2 = Harvester(con, cfg, rng=random.Random(1))
    h2.lb.recs, h2.lb.meta = lb.recs, lb.meta
    h2.listenbrainz_recs(10)                             # what became a candidate is not looked up again
    assert h2.lb.asked[-1] == ("recordings", ("m3", "m4"))


def test_lb_radio_alternates_styles_and_artists_and_sends_the_token(world, monkeypatch):
    con, cfg, _ = world
    secret(cfg, "listenbrainz-token", "t0ken")
    monkeypatch.setattr(harvest, "top_style_names", lambda con, cfg, n: ["Deep House", "Dub (Roots)"])
    like(con, "a", "Five Star", 5)
    like(con, "b", "Four: Star (Live)", 4)
    cfg["harvest"]["lb_radio_prompts"] = 4
    h = Harvester(con, cfg, rng=random.Random(1))
    for p in ("tag:(deep house)", "artist:(Five Star)::nosim"):
        h.lb.playlists[p] = [{"artist": f"{p} artist", "title": f"Song {i}", "mbid": f"{p}-{i}"} for i in range(5)]
        for i in range(5):
            h.dz.add(f"{p} artist", f"Song {i}")
    h.listenbrainz_radio(8)
    asked = [x[1:] for x in h.lb.asked if x[0] == "radio"]
    assert asked == [("tag:(deep house)", "t0ken"), ("artist:(Five Star)::nosim", "t0ken"),
                     ("tag:(dub roots)", "t0ken"), ("artist:(Four Star Live)::nosim", "t0ken")]
    got = details(con, "listenbrainz_radio")
    assert got.count("lb-radio tag:(deep house)") == 2 and got.count("lb-radio artist:(Five Star)::nosim seed=5") == 2
    assert con.execute("SELECT COUNT(*) FROM candidates WHERE mbid LIKE 'tag:(deep house)-%'").fetchone()[0] == 2


# ---------------- labels ----------------

def _album(aid, label, artist="Label Artist", n=3):
    return {"id": aid, "label": label, "release_date": "2026-01-01", "record_type": "album", "genres": {"data": []},
            "artist": {"id": 7000 + aid, "name": f"{artist} {aid}"},
            "tracks": {"data": [{"id": aid * 10 + i, "title": f"Album {aid} Track {i}", "duration": 200,
                                 "rank": i, "preview": f"https://example.test/{aid}-{i}.mp3"} for i in range(n)]}}


def _kept(con, i, label, stars=None, kind="rating", liked=False):
    """A judged candidate on its own album of `label`."""
    con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, source, state, spotify_id, album_id)
                   VALUES(?,?,?,?, 'src', 'offered', ?, ?)""", (i, i, f"Kept {i}", f"Kept {i}", sid(f"k{i}"), 900 + i))
    metadata.store_album(con, 900 + i, {"label": label})
    if liked:
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(f"k{i}"),))
    if stars or kind != "rating":
        verdicts.record(con, sid(f"k{i}"), kind, stars)


def test_label_mass_counts_stars_and_likes_and_merges_spellings(world):
    con, cfg, _ = world
    _kept(con, 1, "Label A", 5)
    _kept(con, 2, "label a", 5)
    _kept(con, 3, "Label A", 4)
    _kept(con, 4, "Label B", liked=True)                 # liked, no verdict
    _kept(con, 5, "Label B", 3)
    _kept(con, 6, "Label C", 2)                          # under 3 stars: nothing
    _kept(con, 7, "Label D", kind="pass", liked=True)    # the verdict wins over the like
    assert label_mass(con) == [("Label A", 8), ("Label B", 2)]


def test_label_source_takes_the_newest_albums_it_hasnt_seen_two_tracks_each(world):
    con, cfg, _ = world
    _kept(con, 1, "Label A", 5)
    _kept(con, 2, "Label B", liked=True)
    cfg["harvest"]["label_albums"] = 2
    h = Harvester(con, cfg, rng=random.Random(1))
    dz = h.dz
    dz.label_search = {'label:"Label A"': [10, 30, 20, 40], 'label:"Label B"': [50]}
    dz.albums = {10: _album(10, "Label A"), 20: _album(20, "Label A"), 40: _album(40, "Other Label"),
                 50: _album(50, "Label B")}
    con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, source, state, album_id)
                   VALUES(3, 3, 'Pooled', 'Pooled', 'src', 'pool', 30)""")   # album 30 is a candidate's already
    h.deezer_label(10)
    assert "album/30" not in dz.asked
    assert [q for q in dz.asked if q.startswith("album/")] == ["album/40", "album/20", "album/10", "album/50"]
    got = details(con, "deezer_label")
    assert got == ["label/Label A/album/20"] * 2 + ["label/Label A/album/10"] * 2 + ["label/Label B/album/50"] * 2
    titles = [r[0] for r in con.execute("SELECT title FROM candidates WHERE source='deezer_label' ORDER BY id")]
    assert titles[:2] == ["Album 20 Track 2", "Album 20 Track 1"]        # most played first
    assert con.execute("SELECT label FROM deezer_albums WHERE album_id=40").fetchone()[0] == "Other Label"
    a = con.execute("SELECT album_id, deezer_artist_id FROM candidates WHERE title='Album 50 Track 2'").fetchone()
    assert tuple(a) == (50, 7050)


# ---------------- iTunes ----------------

def test_a_track_without_a_deezer_preview_is_embedded_from_itunes(world):
    con, cfg, stored = world
    cfg["harvest"]["preview_fallback"] = "itunes"
    h = Harvester(con, cfg, rng=random.Random(1))
    t1 = track(1, "Apple Has", "This One", preview=False)
    t2 = track(2, "Apple Lacks", "That One", preview=False)
    t3 = track(3, "Deezer Has", "A Preview")
    h.itunes.has.add(("Apple Has", "This One"))
    assert [h.offer("src", "d", t) for t in (t1, t2, t3)] == [True, True, True]
    assert stored == [(1, "https://itunes.test/This One.m4a", "itunes"), (3, "https://example.test/3.mp3", "deezer")]
    st = h.stats["src"]
    assert (st["embedded"], st["via_itunes"], st["no_preview"]) == (2, 1, 1)
    status = dict(con.execute("SELECT deezer_id, vector_status FROM candidates").fetchall())
    assert status == {1: "ok", 2: "no_preview", 3: "ok"}
    assert dict(con.execute("SELECT deezer_id, found FROM preview_fallback").fetchall()) == {1: 1, 2: 0}


def test_a_clip_from_itunes_or_deezer_gets_deezers_own_bpm(world, monkeypatch):
    con, cfg, _ = world
    cfg["harvest"]["preview_fallback"] = "itunes"
    hints = []
    monkeypatch.setattr(traits, "analyse_preview", lambda client, url, bpm_hint=None: hints.append((url, bpm_hint)))
    h = Harvester(con, cfg, rng=random.Random(1))
    t1 = track(1, "Apple Has", "This One", preview=False)
    t2 = track(2, "Deezer Has", "A Preview")
    h.dz.catalog += [{**t1, "bpm": 174.0}, {**t2, "bpm": 128.0}]       # only /track/{id} carries the bpm
    h.itunes.has.add(("Apple Has", "This One"))
    assert h.offer("src", "d", t1) and h.offer("src", "d", t2)
    assert hints == [("https://itunes.test/This One.m4a", 174.0), ("https://example.test/2.mp3", 128.0)]

def test_the_backfill_asks_never_asked_first_and_misses_again_after_a_month(world):
    con, cfg, stored = world
    cfg["harvest"]["preview_fallback"] = "itunes"
    for i in (1, 2, 3):
        con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status)
                       VALUES(?,?,?,?, 'src', 'pool', 'no_preview')""", (i, i, f"Artist {i}", f"Song {i}"))
    old = (datetime.now(timezone.utc) - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
    con.execute("INSERT INTO preview_fallback VALUES(2, 'itunes', 0, ?)", (db.now(),))     # asked today
    con.execute("INSERT INTO preview_fallback VALUES(3, 'itunes', 0, ?)", (old,))          # asked 40 days ago
    h = Harvester(con, cfg, rng=random.Random(1))
    h.itunes.has.add(("Artist 3", "Song 3"))
    assert h.fallback_backfill() == {"todo": 2, "embedded": 1}
    assert h.itunes.asked == [("Artist 1", "Song 1"), ("Artist 3", "Song 3")]
    assert [s[0] for s in stored] == [3]
    assert h.fallback_backfill() == {"todo": 0, "embedded": 0}


def test_a_retried_candidate_whose_preview_is_gone_tries_itunes_before_giving_up(world):
    con, cfg, stored = world
    cfg["harvest"]["preview_fallback"] = "itunes"
    h = Harvester(con, cfg, rng=random.Random(1))
    for did, artist in ((1, "Apple Has"), (2, "Apple Lacks")):
        h.dz.catalog.append(track(did, artist, "Gone", preview=False))
        con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status)
                       VALUES(?,?,?, 'Gone', 'src', 'pool', 'error')""", (did, did, artist))
    h.itunes.has.add(("Apple Has", "Gone"))
    assert h.retry_errors() == {"retried": 2, "embedded": 1}
    status = dict(con.execute("SELECT deezer_id, vector_status FROM candidates").fetchall())
    assert status == {1: "ok", 2: "no_preview"}
    assert stored == [(1, "https://itunes.test/Gone.m4a", "itunes")]


def test_preview_fallback_off_asks_apple_nothing(world):
    con, cfg, stored = world                             # "off" is the default
    h = Harvester(con, cfg, rng=random.Random(1))
    assert h.offer("src", "d", track(1, "Apple Has", "This One", preview=False))
    assert FakeItunes.made == 0 and stored == []
    assert h.fallback_backfill() == {"skipped": "harvest.preview_fallback is off"}


def test_vectors_remember_whose_preview_they_came_from(tmp_path):
    con = db.connect(tmp_path / "t.db")
    vectors.put_deezer_vector(con, 1, np.zeros(512), np.zeros(1280), 5, 30.0)
    vectors.put_deezer_vector(con, 2, np.zeros(512), np.zeros(1280), 5, 30.0, provider="itunes")
    assert dict(con.execute("SELECT deezer_id, provider FROM deezer_vectors").fetchall()) == {1: "deezer", 2: "itunes"}
    for i, status in ((1, "ok"), (2, "ok"), (3, "no_preview")):
        con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status)
                       VALUES(?,?,?,?, 'src', 'pool', ?)""", (i, i, f"A {i}", f"T {i}", status))
    cfg = config.load(tmp_path / "none.toml")
    p = report.pool(con, cfg)
    assert p["embedded_by_provider"] == {"deezer": 1, "itunes": 1}
    assert "2 embedded (previews: deezer 1, itunes 1)" in report.text(report.build(con, cfg, None))


# ---------------- the clients, over a fake session ----------------

class Resp:
    def __init__(self, status=200, body=None, chunks=()):
        self.status_code, self.body, self.chunks, self.headers = status, body, chunks, {}

    def json(self):
        if self.body is None:
            raise ValueError("no body")
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def iter_content(self, n):
        return iter(self.chunks)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Session:
    def __init__(self, *answers):
        self.answers, self.calls, self.headers = list(answers), [], {}

    def get(self, url, params=None, headers=None, **k):
        self.calls.append((url, params, headers))
        return self.answers.pop(0)


@pytest.fixture
def quick(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(throttle.Throttle, "wait", lambda self: None)
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_the_new_throttles_have_their_own_files_and_floors(quick):
    it, lf = Itunes(0.5), Lastfm("k", 0.1)
    assert (it.t.path.name, it.t.min_interval) == ("itunes.throttle", 3.0)
    assert (lf.t.path.name, lf.t.min_interval) == ("lastfm.throttle", 1.0)


def test_lastfm_client(quick):
    one = {"similartracks": {"track": {"name": "Only", "artist": {"name": "One"}, "match": "0.5"}}}
    s = Session(Resp(200, {"error": 29, "message": "Rate limit exceeded"}), Resp(200, one),
                Resp(200, {"error": 6, "message": "Track not found"}),
                Resp(200, {"error": 10, "message": "Invalid API key"}))
    lf = Lastfm("k", session=s)
    assert lf.similar_tracks("A", "B") == [{"artist": "One", "title": "Only", "match": 0.5}]   # after one retry
    assert s.calls[0][1]["method"] == "track.getSimilar" and s.calls[0][1]["api_key"] == "k"
    assert lf.top_tracks("Nobody") == []
    with pytest.raises(LastfmError, match="refused"):
        lf.similar_artists("A")
    assert lf.requests == 4


def test_listenbrainz_client(quick):
    recs = {"payload": {"mbids": [{"recording_mbid": "low", "score": 0.1}, {"recording_mbid": "high", "score": 0.9}]}}
    meta = {"high": {"recording": {"name": "Song", "isrcs": ["XX1"]}, "artist": {"name": "Artist"}},
            "bare": {"recording": {"name": "No Isrc"}, "artist": {"name": "Someone"}}}
    jspf = {"payload": {"jspf": {"playlist": {"track": [
        {"title": "T1", "creator": "C1", "identifier": ["https://musicbrainz.org/recording/abc"]},
        {"title": "T2", "creator": "C2", "identifier": "https://musicbrainz.org/recording/def/"},
        {"title": "", "creator": "C3"}]}}, "feedback": []}}
    s = Session(Resp(200, recs), Resp(204), Resp(200, meta), Resp(200, jspf), Resp(401, {"error": "token"}))
    lb = ListenBrainz(session=s, token="t0ken")
    assert lb.recommendations("u") == ["high", "low"]
    assert lb.recommendations("new") == []
    assert lb.recordings(["high", "bare", "gone"]) == {"high": {"artist": "Artist", "title": "Song", "isrc": "XX1"},
                                                       "bare": {"artist": "Someone", "title": "No Isrc", "isrc": None}}
    assert lb.radio("tag:(house)") == [{"artist": "C1", "title": "T1", "mbid": "abc"},
                                       {"artist": "C2", "title": "T2", "mbid": "def"}]
    assert s.calls[3][2] == {"Authorization": "Token t0ken"} and s.calls[0][2] is None
    with pytest.raises(ListenBrainzError, match="401"):
        lb.radio("tag:(house)")


def test_itunes_client(quick):
    songs = {"results": [
        {"kind": "song", "artistName": "The Artist", "trackName": "The Song", "trackTimeMillis": 260000,
         "previewUrl": "https://itunes.test/long.m4a"},                                  # a minute too long
        {"kind": "song", "artistName": "The Artist", "trackName": "The Song (Remastered)", "trackTimeMillis": 201000,
         "previewUrl": "https://itunes.test/right.m4a"},
        {"kind": "music-video", "artistName": "The Artist", "trackName": "The Song", "trackTimeMillis": 200000,
         "previewUrl": "https://itunes.test/video.m4v"}]}
    s = Session(Resp(403), Resp(200, songs), Resp(200, chunks=[b"x" * 10]), Resp(200, chunks=[b"x" * 2_000_000] * 2),
                Resp(403), Resp(403), Resp(403))
    it = Itunes(session=s)
    assert it.find("The Artist", "The Song", 200, 3.0)["previewUrl"] == "https://itunes.test/right.m4a"
    assert s.calls[1][1] == {"term": "The Artist The Song", "media": "music", "entity": "song", "limit": 10}
    assert it.preview_bytes("https://itunes.test/right.m4a") == b"x" * 10
    with pytest.raises(ItunesError, match="too large"):
        it.preview_bytes("https://itunes.test/huge.m4a")
    with pytest.raises(ItunesError, match="retries"):
        it.search("anything")


def test_a_database_from_before_0_7_0_gets_the_provider_column_and_the_report_reads_it_either_way(tmp_path):
    path = tmp_path / "old.db"
    con = db.connect(path)
    con.execute("INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status) "
                "VALUES(1, 1, 'A', 'T', 's', 'pool', 'ok')")
    vectors.put_deezer_vector(con, 1, np.zeros(512), np.zeros(1280), 5, 30.0)
    con.execute("ALTER TABLE deezer_vectors DROP COLUMN provider")      # as 0.6.0 left it
    con.commit()
    con.close()
    cfg = config.load(tmp_path / "none.toml")
    assert report.pool(db.connect_readonly(path), cfg)["embedded_by_provider"] == {"deezer": 1}
    assert db.connect(path).execute("SELECT provider FROM deezer_vectors").fetchone()[0] == "deezer"
