"""The scene feeds of ADR 0026 (Last.fm tag artists, ListenBrainz fresh releases and similar artists, the language
model's deep cuts), the editor guard on the style search, the already-known share and eligibility by live source
rows, over the synthetic world of test_more_sources with fake clients. No network."""
import json
import random

import pytest

from haystack_music import config, db, harvest, llm, scene, sources
from haystack_music.harvest import Harvester
from test_more_sources import (FakeDeezer, FakeLastfm, FakeLB, details, like, secret, track,  # noqa: F401
                               world)

NEW = ["lastfm_tag_artists", "listenbrainz_fresh", "listenbrainz_artist_similar", "llm_suggest"]
DANCE, FUNK, REGGAE, BRASIL = 2834392844, 2862678264, 753547275, 2529028862


class SceneDeezer(FakeDeezer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.playlist_search: dict[str, list[dict]] = {}
        self.playlists: dict[int, dict] = {}
        self.album_search: dict[str, list[dict]] = {}
        self.album_lists: dict[int, list[dict]] = {}
        self.charts: dict[int, list[dict]] = {}

    def search_playlist(self, q, limit=6):
        self.asked.append(f"search_playlist {q}")
        return self.playlist_search.get(q, [])[:limit]

    def playlist(self, pid):
        self.asked.append(f"playlist/{pid}")
        return self.playlists.get(pid)

    def search_album(self, q, limit=50):
        self.asked.append(f"search_album {q}")
        return self.album_search.get(q, [])[:limit]

    def album_tracks(self, aid, limit=3):
        self.asked.append(f"album_tracks/{aid}")
        return [dict(t) for t in self.album_lists.get(aid, [])][:limit]

    def chart_tracks(self, eid=0, limit=100):
        return self.charts.get(eid, [])[:limit]


class SceneLB(FakeLB):
    def __init__(self, *a, artist_algorithm=None, **k):
        super().__init__(*a, **k)
        self.artist_algorithm = artist_algorithm
        self.fresh: list[dict] = []
        self.artist_ids: dict[tuple, str] = {}
        self.sims: dict[str, list[dict]] = {}
        self.tops: dict[str, list[dict]] = {}

    def fresh_releases(self, days):
        self.asked.append(("fresh", days))
        return list(self.fresh)

    def artist_mbid(self, artist, title):
        self.asked.append(("artist_mbid", artist, title))
        return self.artist_ids.get((artist, title))

    def similar_artists(self, mbid, limit=6):
        self.asked.append(("similar_artists", mbid, limit))
        return self.sims.get(mbid, [])[:limit]

    def top_recordings(self, mbid, limit=2):
        self.asked.append(("top_recordings", mbid))
        return self.tops.get(mbid, [])[:limit]


class SceneLastfm(FakeLastfm):
    def __init__(self, key, min_interval=1.0):
        super().__init__(key, min_interval)
        self.tags: dict[str, list[str]] = {}
        self.asked: list = []

    def tag_top_artists(self, tag, limit=50):
        self.asked.append(("tag", tag))
        return self.tags.get(tag, [])[:limit]

    def top_tracks(self, artist, limit=3):
        self.asked.append(("top", artist))
        return super().top_tracks(artist, limit)


@pytest.fixture
def scene_world(world, monkeypatch):
    monkeypatch.setattr(harvest, "Deezer", SceneDeezer)
    monkeypatch.setattr(harvest, "ListenBrainz", SceneLB)
    monkeypatch.setattr(harvest, "Lastfm", SceneLastfm)
    styles: list = []
    monkeypatch.setattr(harvest.scene, "top_styles", lambda con, cfg, n: styles[:n])
    return (*world, styles)


# ---------------- the tables ----------------

def test_the_spelling_table_names_styles_as_their_listeners_tag_them():
    assert scene.lastfm_tag("Drum n Bass") == "drum and bass"
    assert scene.lastfm_tag("UK Garage") == "uk garage"
    assert scene.lastfm_tag("Bassline") == "bassline" and scene.lastfm_tag("Halftime") == "halftime"
    assert scene.lastfm_tag("Hip Hop") == "hip-hop" and scene.lastfm_tag("RnB/Swing") == "new jack swing"
    assert scene.lastfm_tag("Rhythm & Blues") == "rhythm and blues"           # not in the table: lower-cased
    assert scene.lastfm_tag("Deep House") == "deep house"
    keys = scene.tag_keys("Drum n Bass")
    from haystack_music.textnorm import key
    assert {key("Drum and Bass"), key("drum n bass"), key("DnB")} <= keys
    assert key("dubstep") not in keys


def test_a_style_splits_on_its_last_slash_and_maps_to_its_genre_editors():
    assert scene.split_style("Electronic / Bassline") == ("Electronic", "Bassline")
    assert scene.split_style("Funk / Soul / Disco") == ("Funk / Soul", "Disco")
    assert scene.editors({"Electronic"}) == {DANCE}
    assert scene.editors({"Funk / Soul"}) == {FUNK, 2928001764}
    assert scene.editors({"Non-Music"}) == set()
    assert BRASIL not in set(scene.GENRE_EDITORS.values())                   # country editors are left out


# ---------------- gates ----------------

def test_each_scene_source_skips_without_what_it_needs_and_asks_nobody(scene_world):
    con, cfg, _, styles = scene_world
    h = Harvester(con, cfg, rng=random.Random(1))
    out = h.run(NEW, cap=10, retry=False)
    per = out["per_source"]
    assert per["lastfm_tag_artists"]["skipped"] == "no lastfm-api-key secret"
    assert per["llm_suggest"]["skipped"] == "runs in the weekly `harvest --broad` only (its first shard)"
    assert per["listenbrainz_fresh"]["releases"] == 0                       # nothing of theirs to match
    assert not any(per[s]["errors"] for s in NEW)
    assert FakeLastfm.made == [] and h.dz.asked == []
    secret(cfg, "lastfm-api-key", "k3y")
    h = Harvester(con, cfg, rng=random.Random(1))
    h.run(["lastfm_tag_artists"], cap=10, retry=False)
    assert h.stats["lastfm_tag_artists"]["skipped"] == "no styles yet (no liked song has a style vector)"
    assert h.lastfm().asked == []


def test_the_scene_sources_are_in_the_defaults_with_labels_and_the_model_job_is_off():
    d = config.DEFAULTS
    assert set(NEW) <= set(d["harvest"]["sources"])
    from haystack_music.review import SOURCE_LABELS
    assert set(NEW) <= set(SOURCE_LABELS)
    assert d["llm"]["jobs"] == [] and d["llm"]["seeds_per_week"] == 100
    assert (d["harvest"]["tag_styles"], d["harvest"]["tag_artists"], d["harvest"]["fresh_days"]) == (8, 12, 7)


def test_retired_sources_stops_each_scene_source(scene_world):
    con, cfg, _, styles = scene_world
    secret(cfg, "lastfm-api-key", "k3y")
    styles.append(("Drum n Bass", {"Electronic"}))
    cfg["harvest"]["retired_sources"] = list(NEW)
    h = Harvester(con, cfg, seeds=[], lead=True, rng=random.Random(1))
    out = h.run(NEW, cap=10, retry=False)
    assert not set(NEW) & set(out["per_source"])
    assert h.lb.asked == [] and h.dz.asked == [] and FakeLastfm.made == []


# ---------------- Last.fm tag artists ----------------

def test_tag_artists_takes_the_styles_top_artists_the_listener_lacks(scene_world):
    con, cfg, _, styles = scene_world
    secret(cfg, "lastfm-api-key", "k3y")
    like(con, "a", "Five Star", 5)
    styles += [("Drum n Bass", {"Electronic"}), ("UK Garage", {"Electronic"})]
    cfg["harvest"]["tag_artists"] = 2
    h = Harvester(con, cfg, rng=random.Random(1))
    lf = h.lastfm()
    lf.tags = {"drum and bass": ["Five Star", "Alpha", "Beta", "Gamma"], "uk garage": ["Alpha", "Delta"]}
    for a in ("Alpha", "Beta", "Gamma", "Delta"):
        h.dz.add(a, "Hit 0"); h.dz.add(a, "Hit 1")
    h.run(["lastfm_tag_artists"], cap=20, retry=False)
    assert [x for x in lf.asked if x[0] == "tag"] == [("tag", "drum and bass"), ("tag", "uk garage")]
    tops = [a for k, a in lf.asked if k == "top"]
    assert tops == ["Alpha", "Beta", "Delta"]       # not their own Five Star, two per style, Alpha once
    d = details(con, "lastfm_tag_artists")
    assert d == ["lastfm/tag/Drum n Bass -> Alpha"] * 2 + ["lastfm/tag/Drum n Bass -> Beta"] * 2 + \
        ["lastfm/tag/UK Garage -> Delta"] * 2
    assert h.stats["lastfm_tag_artists"]["new"] == 6


# ---------------- ListenBrainz fresh releases ----------------

def _album(aid, artist, title):
    return {"id": aid, "title": title, "artist": {"id": 7000 + aid, "name": artist}}


def test_fresh_releases_keep_the_listeners_styles_and_artists_and_are_tried_once(scene_world):
    con, cfg, _, styles = scene_world
    like(con, "a", "Five Star", 5)
    styles.append(("Drum n Bass", {"Electronic"}))
    h = Harvester(con, cfg, rng=random.Random(1))
    h.lb.fresh = [
        {"artist": "Roller", "title": "Night Shift EP", "release_mbid": "m1", "tags": ["dnb", "jungle"], "listens": 9},
        {"artist": "Five Star feat. Friend", "title": "Collab", "release_mbid": "m2", "tags": [], "listens": 5},
        {"artist": "Pop Star", "title": "Hits", "release_mbid": "m3", "tags": ["pop"], "listens": 90}]
    h.dz.album_search = {"Roller Night Shift EP": [_album(1, "Someone Else", "Night Shift EP"),
                                                    _album(2, "Roller", "Night Shift (EP)")],
                         "Five Star feat. Friend Collab": [_album(3, "Five Star", "Collab")]}
    h.dz.album_lists = {2: [track(201, "Roller", "Low", rank=10), track(202, "Roller", "High", rank=50),
                            track(203, "Roller", "Mid", rank=30)],
                        3: [track(301, "Five Star", "Together")]}
    h.run(["listenbrainz_fresh"], cap=10, retry=False)
    assert h.stats["listenbrainz_fresh"]["releases"] == 2
    assert not any("Pop Star" in a for a in h.dz.asked)
    got = con.execute("""SELECT c.title, cs.detail FROM candidate_sources cs JOIN candidates c ON c.id=cs.candidate_id
                         WHERE cs.source='listenbrainz_fresh' ORDER BY cs.rowid""").fetchall()
    assert [tuple(r) for r in got] == [("High", "lb-fresh/m1"), ("Mid", "lb-fresh/m1"), ("Together", "lb-fresh/m2")]
    con.commit()
    h2 = Harvester(con, cfg, rng=random.Random(1))
    h2.lb.fresh = h.lb.fresh
    h2.run(["listenbrainz_fresh"], cap=10, retry=False)
    assert h2.stats["listenbrainz_fresh"]["releases"] == 0 and h2.dz.asked == []


# ---------------- ListenBrainz similar artists ----------------

def test_top_seed_artists_start_from_five_stars(scene_world):
    con, cfg, _, _ = scene_world
    like(con, "b", "Four Star", 4)
    like(con, "a", "Five Star", 5)
    h = Harvester(con, cfg, rng=random.Random(1))
    assert [s.artist for s in h.top_seed_artists(2, "listenbrainz_artist_similar")] == ["Five Star", "Four Star"]


def test_lb_artist_similar_walks_the_seeds_similar_artists_top_recordings(scene_world):
    con, cfg, _, _ = scene_world
    like(con, "a", "Five Star", 5)
    cfg["harvest"]["lb_similar_artists"] = 2
    h = Harvester(con, cfg, rng=random.Random(1))
    lb = h.lb
    lb.artist_ids = {("Five Star", "Song a"): "art5"}
    lb.sims = {"art5": [{"name": "Kin", "mbid": "k1"}, {"name": "Kith", "mbid": "k2"}, {"name": "Far", "mbid": "k3"}]}
    lb.tops = {"k1": [{"artist": "Kin", "title": "Kin One", "mbid": "r1"},
                      {"artist": "Kin", "title": "Kin Two", "mbid": "r2"}],
               "k2": [{"artist": "Kith", "title": "Kith One", "mbid": "r3"}]}
    for a, t in (("Kin", "Kin One"), ("Kin", "Kin Two"), ("Kith", "Kith One")):
        h.dz.add(a, t)
    h.run(["listenbrainz_artist_similar"], cap=10, retry=False)
    assert ("similar_artists", "art5", 2) in lb.asked and ("top_recordings", "k3") not in lb.asked
    rows = con.execute("""SELECT c.title, c.mbid, cs.detail FROM candidate_sources cs
                          JOIN candidates c ON c.id=cs.candidate_id
                          WHERE cs.source='listenbrainz_artist_similar' ORDER BY cs.rowid""").fetchall()
    assert [(t, m) for t, m, _ in rows] == [("Kin One", "r1"), ("Kin Two", "r2"), ("Kith One", "r3")]
    assert rows[0][2].startswith("lb-artist/art5 (Five Star) -> Kin")
    assert rows[2][2].startswith("lb-artist/art5 (Five Star) -> Kith")


# ---------------- the language model's deep cuts ----------------

def _llm_world(scene_world, monkeypatch, suggestions):
    con, cfg, _, _ = scene_world
    cfg["llm"].update(enabled=True, jobs=["seeds"], weekly_budget_usd=1.0, model="m-1")
    asked = []

    def ask(con_, cfg_, provider, n, classes=None):
        asked.append(n)
        return {"provider": "fake", "model": "m-1", "n": n, "tokens_in": 1000, "tokens_out": 200, "cost_usd": 0.25,
                "suggestions": [dict(s) for s in suggestions], "reason": None}

    monkeypatch.setattr(llm, "make_provider", lambda cfg_, store: object())
    monkeypatch.setattr(llm, "ask_suggestions", ask)

    def no_model():
        raise OSError("no style model")
    monkeypatch.setattr(harvest.embed, "style_classes", no_model)
    return con, cfg, asked


def test_llm_suggest_runs_once_a_week_in_the_lead_shard_and_drops_profile_and_known_songs(scene_world, monkeypatch):
    sugg = [{"artist": "Five Star", "title": "Theirs", "why": "w", "in_profile": True},
            {"artist": "Deep Act", "title": "Cut", "why": "w", "in_profile": False},
            {"artist": "Known Act", "title": "Old", "why": "w", "in_profile": False},
            {"artist": "Nobody", "title": "Imaginary", "why": "w", "in_profile": False}]
    con, cfg, asked = _llm_world(scene_world, monkeypatch, sugg)
    first = Harvester(con, cfg, rng=random.Random(1))
    first.offer("deezer_chart", "chart/0", first.dz.add("Known Act", "Old"))
    con.commit()
    for kw in ({}, {"seeds": []}):                                   # the daily run, or a later shard
        h = Harvester(con, cfg, rng=random.Random(1), **kw)
        h.llm_suggest(10)
        assert h.stats["llm_suggest"]["skipped"].startswith("runs in the weekly")
    h = Harvester(con, cfg, seeds=[], lead=True, rng=random.Random(1))
    h.dz.catalog = list(first.dz.catalog)
    for a, t in (("Five Star", "Theirs"), ("Deep Act", "Cut")):
        h.dz.add(a, t)
    h.llm_suggest(10)
    st = h.stats["llm_suggest"]
    assert asked == [100] and (st["asked"], st["returned"], st["in_profile"]) == (100, 4, 1)
    assert (st["new"], st["dupe"]) == (1, 1) and st["cost_usd"] == 0.25
    assert not any("Five Star Theirs" in a for a in h.dz.asked)       # a profile artist's song is not looked up
    wk = llm.week()
    assert set(details(con, "llm_suggest")) == {f"llm/m-1/{wk}"}
    assert [tuple(r) for r in con.execute("SELECT job, week, n, cost_usd, ok FROM llm_runs")] == [("seeds", wk, 100, 0.25, 1)]
    h = Harvester(con, cfg, seeds=[], lead=True, rng=random.Random(1))
    h.llm_suggest(10)
    assert h.stats["llm_suggest"]["skipped"] == f"ran this week ({wk})" and asked == [100]


@pytest.mark.parametrize("change, why", [
    ({"enabled": False}, "llm.enabled is false"),
    ({"jobs": []}, 'llm.jobs has no "seeds"'),
    ({"jobs": ["rank"]}, 'llm.jobs has no "seeds"'),
])
def test_llm_suggest_is_gated_by_its_switches(scene_world, monkeypatch, change, why):
    con, cfg, asked = _llm_world(scene_world, monkeypatch, [])
    cfg["llm"].update(change)
    h = Harvester(con, cfg, seeds=[], lead=True, rng=random.Random(1))
    h.llm_suggest(10)
    assert h.stats["llm_suggest"]["skipped"] == why and asked == []


def test_llm_suggest_stops_when_the_weeks_budget_is_spent(scene_world, monkeypatch):
    con, cfg, asked = _llm_world(scene_world, monkeypatch, [])
    llm.record_run(con, "rank", "m-1", 50, 10, 10, 1.0)                # another job spent it
    h = Harvester(con, cfg, seeds=[], lead=True, rng=random.Random(1))
    h.llm_suggest(10)
    assert h.stats["llm_suggest"]["skipped"] == "this week's budget is spent ($1.00 of $1.00)" and asked == []
    assert llm.spent(con) == 1.0 and not llm.ran(con, "seeds") and llm.ran(con, "rank")


def test_llm_suggest_without_its_key_is_skipped_and_records_nothing(scene_world):
    con, cfg, _, _ = scene_world
    cfg["llm"].update(enabled=True, jobs=["seeds"])
    h = Harvester(con, cfg, seeds=[], lead=True, rng=random.Random(1))
    out = h.run(["llm_suggest"], cap=10, retry=False)
    assert out["per_source"]["llm_suggest"]["skipped"] and not out["per_source"]["llm_suggest"]["errors"]
    assert con.execute("SELECT COUNT(*) FROM llm_runs").fetchone()[0] == 0


# ---------------- the style search's editor guard (#347) ----------------

def _pl(pid, title, user, tracks=(), fans=10, creator=None):
    return {"id": pid, "title": title, "user": {"id": user}}, \
        {"id": pid, "title": title, "fans": fans, "creator": {"id": creator or user},
         "tracks": {"data": list(tracks)}}


def test_the_style_search_takes_only_the_genre_editors_playlists(scene_world):
    con, cfg, _, styles = scene_world
    styles += [("Bassline", {"Electronic"}), ("Halftime", {"Electronic"}), ("Dubstep", {"Electronic"}),
               ("Trap", {"Electronic", "Hip Hop"})]
    cfg["harvest"]["style_search_styles"] = 4
    h = Harvester(con, cfg, rng=random.Random(1))
    dz = h.dz
    t = lambda i: track(i, f"Act {i}", f"Tune {i}")                         # noqa: E731
    found = {
        "Bassline": [_pl(1, "What The Funk!", FUNK, [t(11)]),                # Soul & Funk's editor: wrong genre
                     _pl(2, "Bassline House", DANCE, [t(21), t(22)]),       # Dance's editor: kept
                     _pl(6, "Bassline Mix", DANCE, [t(61)], creator=55)],   # the full playlist says otherwise
        "Halftime": [_pl(3, "Super Bowl 2022 - Playlist Officielle", 4242, [t(31)])],   # not Deezer's
        "Dubstep": [_pl(4, "Dub Essentials", REGGAE, [t(41)])],              # Reggae's editor
        "Trap": [_pl(5, "Hitou", BRASIL, [t(51)])],                          # a country editor
    }
    for q, pls in found.items():
        dz.playlist_search[q] = [s for s, _ in pls]
        dz.playlists.update({full["id"]: full for _, full in pls})
    h.run(["deezer_style_search"], cap=20, retry=False)
    assert details(con, "deezer_style_search") == ["style/Bassline/playlist/2"] * 2
    assert {a for a in dz.asked if a.startswith("playlist/")} == {"playlist/2", "playlist/6"}
    assert h.stats["deezer_style_search"]["playlists_rejected"] == 4


# ---------------- the already-known share ----------------

def test_the_known_share_of_one_run_and_of_the_last_two_weeks(tmp_path):
    assert sources.run_known_share({"a": {"new": 1, "dupe": 2, "in_library": 1, "liked": 0},
                                    "b": {"new": 0, "dupe": 0}, "x": "not counts"}) == {
        "a": {"known": 3, "intake": 4, "known_share": 0.75}, "b": {"known": 0, "intake": 0, "known_share": None}}
    con = db.connect(tmp_path / "t.db")

    def run(at, per, ok=1, job="harvest"):
        con.execute("INSERT INTO runs(job, started_at, finished_at, ok, summary) VALUES(?,?,?,?,?)",
                    (job, at, at, ok, json.dumps({"per_source": per})))

    now = db.now()
    run(now, {"deezer_editorial": {"new": 1, "dupe": 3, "in_library": 0, "liked": 0,
                                   "charts": {"116": {"new": 0, "dupe": 3}, "113": {"new": 1, "dupe": 0}}},
              "lastfm_similar": {"new": 4, "dupe": 0, "in_library": 0, "liked": 0, "skipped": None}})
    run(now, {"deezer_editorial": {"new": 1, "dupe": 1, "liked": 1, "charts": {"116": {"new": 1, "liked": 1}}}},
        job="harvest-broad")
    run(now, {"lastfm_similar": {"new": 0, "dupe": 99}}, ok=0)                # a failed run counts nothing
    run("2026-01-01T00:00:00Z", {"lastfm_similar": {"new": 0, "dupe": 99}})  # nor one older than 14 days
    run(now, {"deezer_editorial": {"new": 0, "dupe": 99}}, job="batch")
    k = sources.known_share(con)
    assert k["per_source"] == {"deezer_editorial": {"known": 5, "intake": 7, "known_share": 0.714},
                               "lastfm_similar": {"known": 0, "intake": 4, "known_share": 0.0}}
    assert k["charts"] == {"113": {"known": 0, "intake": 1, "known_share": 0.0},
                           "116": {"known": 4, "intake": 5, "known_share": 0.8}}


def test_the_editorial_source_counts_each_chart_and_the_run_reports_its_known_share(scene_world):
    con, cfg, _, _ = scene_world
    like(con, "a", "Five Star", 5)
    cfg["harvest"]["deezer_editorial_ids"] = [116, 113]
    cfg["harvest"]["chart_max_share"] = 0.99
    con.execute("""INSERT INTO candidates(id, deezer_id, artist, artist_key, title, song_key, source, state,
                   vector_status) VALUES(1, 1, 'x', 'x', 'y', 'y', 'deezer_label', 'pool', 'ok')""")
    h = Harvester(con, cfg, rng=random.Random(1))
    h.dz.charts = {116: [track(501, "Five Star", "Song a"), track(502, "Rapper", "Bars")],
                   113: [track(503, "Dancer", "Floor")]}
    out = h.run(["deezer_editorial"], cap=10, retry=False)
    ch = out["per_source"]["deezer_editorial"]["charts"]
    assert (ch["116"]["new"], ch["116"]["liked"], ch["113"]["new"]) == (1, 1, 1)
    assert out["known_share"]["deezer_editorial"] == {"known": 1, "intake": 3, "known_share": 0.333}


# ---------------- eligibility by live source rows (#343) ----------------

def test_a_pool_song_stays_eligible_while_one_live_source_holds_it(tmp_path):
    con = db.connect(tmp_path / "t.db")
    cfg = config.load(tmp_path / "none.toml")
    cfg["harvest"]["deezer_editorial_ids"] = [0, 113]
    cfg["harvest"]["retired_sources"] = ["deezer_style_search"]

    def cand(cid, source, rows=(), state="pool"):
        con.execute("""INSERT INTO candidates(id, deezer_id, artist, artist_key, title, song_key, source, state)
                       VALUES(?,?,?,?,?,?,?,?)""", (cid, cid, "a", "a", f"t{cid}", f"s{cid}", source, state))
        for src, d in rows:
            con.execute("INSERT INTO candidate_sources(candidate_id, source, detail) VALUES(?,?,?)", (cid, src, d))

    cand(1, "deezer_editorial", [("deezer_editorial", "chart/116")])                     # a dropped chart only
    cand(2, "deezer_editorial", [("deezer_editorial", "chart/116"), ("lastfm_similar", "lastfm/track/x")])
    cand(3, "deezer_editorial", [("deezer_editorial", "chart/113")])                     # a chart still there
    cand(4, "deezer_style_search")                                                       # no rows: its source
    cand(5, "deezer_new_releases", [("deezer_new_releases", "editorial/116/album/9")])
    cand(6, "deezer_style_search", [("deezer_style_search", "style/Dub/playlist/4")])
    cand(7, "deezer_chart", [("deezer_chart", "chart/0")])
    cand(8, "deezer_style_search", [("deezer_style_search", "style/x")], state="offered")  # not in the pool
    assert sources.retired_candidates(con, cfg) == {1, 4, 5, 6}
    assert sources.live_row(cfg, "deezer_editorial", "chart/113") and not sources.live_row(cfg, "deezer_editorial",
                                                                                           "chart/116")
    assert sources.live_row(cfg, "deezer_artist_radio", "radio/116")       # only the editorial ids are charts


def test_haystack_sources_shows_the_known_share_the_skip_and_the_dropped_chart(tmp_path, monkeypatch, capsys):
    from haystack_music import cli
    path = tmp_path / "t.db"
    monkeypatch.setattr(config, "DB_PATH", path)
    con = db.connect(path)
    now = db.now()
    con.execute("INSERT INTO runs(job, started_at, finished_at, ok, summary) VALUES('harvest', ?, ?, 1, ?)",
                (now, now, json.dumps({"per_source": {
                    "deezer_editorial": {"new": 1, "dupe": 3, "charts": {"116": {"new": 0, "dupe": 3},
                                                                         "113": {"new": 1, "dupe": 0}}},
                    "lastfm_tag_artists": {"new": 0, "dupe": 0, "skipped": "no lastfm-api-key secret"}}})))
    con.close()
    cfg = config.load(tmp_path / "none.toml")
    cfg["harvest"]["deezer_editorial_ids"] = [0, 113]
    assert cli.cmd_sources(cfg, cli.build_parser().parse_args(["sources", "list"])) == 0
    out = capsys.readouterr().out
    row = next(x for x in out.splitlines() if x.startswith("deezer_editorial "))
    assert " 75% " in row
    assert next(x for x in out.splitlines() if x.startswith("lastfm_tag_artists ")).endswith(
        "; skipped: no lastfm-api-key secret")
    assert "deezer_editorial by chart: 113 0% known of 1; 116 100% known of 3 (dropped)" in out
