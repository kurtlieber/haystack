"""Harvest seeds by Verdicts, the style search, adaptive caps, the chart share per batch and the weekly broad job
(ADR 0013), over a synthetic world with a fake Deezer and ListenBrainz."""
import random
from datetime import datetime

import pytest

from haystack_music import config, db, harvest, scheduler, sources, traits, verdicts
from haystack_music.harvest import Harvester, Seed, pick_seeds, seed_buckets, with_seed
from haystack_music.scoring import Pick, select_batch


def sid(n: str) -> str:
    return n.replace("_", "").ljust(22, "0")


def track(did: int, artist: str, title: str, artist_id: int | None = None) -> dict:
    return {"id": did, "title": title, "duration": 200, "preview": f"https://example.test/{did}.mp3",
            "artist": {"id": artist_id or 9000 + did, "name": artist}}


class FakeDeezer:
    """Canned answers: every artist exists, each radio and playlist has its own tracks."""
    def __init__(self, *a, **k):
        self.requests = 0
        self.playlists: dict[int, dict] = {}
        self.searches: dict[str, list[dict]] = {}
        self.radio_asked: list[int] = []

    def search_artist(self, name, limit=5):
        return [{"id": 1000 + sum(map(ord, name)), "name": name}]

    def artist_radio(self, aid, limit=25):
        self.radio_asked.append(aid)
        return [track(aid * 100 + i, f"Radio Artist {aid}-{i}", f"Radio Song {i}") for i in range(limit)]

    def artist_related(self, aid, limit=10):
        return [{"id": aid * 10 + i, "name": f"Related {aid}-{i}"} for i in range(limit)]

    def artist_top(self, aid, limit=3):
        return [track(aid * 1000 + i, f"Related Artist {aid}", f"Top Song {i}") for i in range(limit)]

    def search_playlist(self, q, limit=6):
        return self.searches.get(q, [])[:limit]

    def playlist(self, pid):
        return self.playlists.get(pid)

    def chart_tracks(self, eid=0, limit=100):
        return []


class FakeLB:
    def __init__(self, *a, **k):
        self.requests = 0
        self.asked: list[tuple[str, str]] = []

    def similar_for(self, artist, title):
        self.asked.append((artist, title))
        return None, []


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    monkeypatch.setattr(harvest, "Deezer", FakeDeezer)
    monkeypatch.setattr(harvest, "ListenBrainz", FakeLB)
    monkeypatch.setattr(traits, "analyse_preview", lambda dz, url, bpm_hint=None: {})
    monkeypatch.setattr(traits, "store", lambda con, did, r: None)
    con = db.connect(tmp_path / "t.db")
    cfg = config.load(tmp_path / "none.toml")
    return con, cfg


def like(con, name: str, artist: str, stars=None, kind="rating", given_at=None):
    con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(name),))
    con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json) VALUES(?,?,?)",
                (sid(name), f"Song {name}", f'["{artist}"]'))
    if stars or kind != "rating":
        verdicts.record(con, sid(name), kind, stars, given_at=given_at)


def library_track(con, i: int, artist: str):
    con.execute("""INSERT INTO local_tracks(id, path, size, content_hash, artist_raw, title_raw, artists_json, present)
                   VALUES(?,?,1,?,?,?,?,1)""", (i, f"/m/{i}.flac", f"h{i}", artist, f"File {i}", f'["{artist}"]'))


def discovery(con, i: int, name: str, artist: str, outcome: str, rating=None):
    con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, source, state, spotify_id)
                   VALUES(?,?,?,?, 'src_a', 'offered', ?)""", (i, i, artist, f"Found {name}", sid(name)))
    con.execute("""INSERT INTO offers(candidate_id, spotify_id, batch_id, source, added_at, outcome, removed_at)
                   VALUES(?,?,1,'src_a',?,?,?)""", (i, sid(name), db.now(), outcome, outcome and db.now()))
    if rating:
        verdicts.record(con, sid(name), "rating", rating)


def test_seed_tags():
    assert Seed("A", "x", stars=5, origin="rating").tag == "5"
    assert Seed("A", "x", origin="library").tag == "library"
    assert Seed("A", "x").tag is None
    assert with_seed("artist/1/radio (A)", Seed("A", stars=4, origin="discovery")) == "artist/1/radio (A) seed=4"
    assert with_seed("artist/1/radio (A)", Seed("A")) == "artist/1/radio (A)"


def test_buckets_put_recent_high_ratings_and_kept_discoveries_first(world):
    con, cfg = world
    like(con, "five", "Five Artist", 5)
    like(con, "four", "Four Artist", 4)
    like(con, "old_five", "Old Five", 5, given_at="2020-01-01T00:00:00Z")   # outside seed_recent_days: mid
    like(con, "three", "Three Artist", 3)
    like(con, "unrated", "Unrated Artist")
    like(con, "passed", "Passed Artist", kind="pass")                      # a Pass since: not a seed
    library_track(con, 1, "Library Artist")
    discovery(con, 50, "kept", "Kept Discovery", "liked")                  # liked, no verdict: kept
    discovery(con, 51, "rated3", "Rated Discovery", None, rating=3)        # the Rating closes it: kept
    discovery(con, 52, "turned_down", "Turned Down", "passed")
    b = seed_buckets(con, cfg, random.Random(1))
    top = [(s.artist, s.stars, s.origin) for s in b["top"]]
    assert top[:2] == [("Five Artist", 5, "rating"), ("Four Artist", 4, "rating")]
    assert set(top[2:]) == {("Kept Discovery", None, "discovery"), ("Rated Discovery", 3, "discovery")}
    assert top[-1] == ("Rated Discovery", 3, "discovery")                 # unrated kept ranks as 3.5
    mid = {(s.artist, s.stars) for s in b["mid"]}
    assert mid == {("Old Five", 5), ("Three Artist", 3), ("Unrated Artist", None)}
    assert [(s.artist, s.origin) for s in b["library"]] == [("Library Artist", "library")]


def _buckets(top=0, mid=0, lib=0):
    return {"top": [Seed(f"Top {i}", origin="rating", stars=5, bucket=0) for i in range(top)],
            "mid": [Seed(f"Mid {i}", origin="like", bucket=1) for i in range(mid)],
            "library": [Seed(f"Lib {i}", origin="library", bucket=2) for i in range(lib)]}


QUOTAS = {"top": 0.4, "mid": 0.3, "library": 0.3}


def test_quotas_and_carry_over():
    got = pick_seeds(_buckets(10, 10, 10), 10, QUOTAS)
    assert [s.origin for s in got] == ["rating"] * 4 + ["like"] * 3 + ["library"] * 3
    got = pick_seeds(_buckets(2, 10, 10), 10, QUOTAS)          # top short by 2: mid takes them
    assert [s.origin for s in got] == ["rating"] * 2 + ["like"] * 5 + ["library"] * 3
    got = pick_seeds(_buckets(0, 2, 3), 10, QUOTAS)            # not enough anywhere: everything there is
    assert len(got) == 5
    got = pick_seeds(_buckets(0, 20, 20), 10, QUOTAS)          # no Ratings yet: 70 % liked, 30 % library
    assert [s.origin for s in got] == ["like"] * 7 + ["library"] * 3


def test_distinct_artists_and_the_weekly_exclusion():
    b = _buckets(4, 4, 0)
    b["top"].append(Seed("Top 0", "another song", origin="rating", stars=5))
    got = pick_seeds(b, 8, QUOTAS)
    assert len({s.artist for s in got}) == len(got) == 8
    week = {harvest.key("Top 0"), harvest.key("Top 1")}
    got = pick_seeds(_buckets(4, 4, 0), 4, QUOTAS, exclude=week)
    assert {s.artist for s in got}.isdisjoint({"Top 0", "Top 1"})
    got = pick_seeds(_buckets(2, 0, 0), 2, QUOTAS, exclude=week)   # nothing else left: seeded again
    assert {s.artist for s in got} == {"Top 0", "Top 1"}


def test_artist_radio_records_its_seeds_and_tags_its_candidates(world):
    con, cfg = world
    for i in range(12):
        like(con, f"l{i:02d}x", f"Liked Artist {i}", 5 if i < 3 else None)
    h = Harvester(con, cfg, rng=random.Random(3))
    h.deezer_artist_radio(30)
    seeded = con.execute("SELECT artist_key, stars, origin FROM harvest_seeds WHERE source='deezer_artist_radio'"
                         ).fetchall()
    assert len(seeded) >= 3 and all(r["origin"] in ("rating", "like") for r in seeded)
    assert [r["stars"] for r in seeded[:3]] == [5, 5, 5]                  # the 5-star seeds go first
    details = [r[0] for r in con.execute("SELECT detail FROM candidate_sources")]
    assert details and all(" seed=" in d for d in details)
    assert any(d.endswith(" seed=5") for d in details)
    # a second run this week takes every artist not seeded yet, then fills up with the others
    first = {r["artist_key"] for r in seeded}
    fresh = {harvest.key(f"Liked Artist {i}") for i in range(12)} - first
    h2 = Harvester(con, cfg, rng=random.Random(3))
    nxt = {harvest.key(s.artist) for s in h2.seed_artists(cfg["harvest"]["seed_artists"], "deezer_artist_radio")}
    assert fresh and fresh <= nxt and len(nxt) == cfg["harvest"]["seed_artists"]


def test_recent_policy_leaves_seeds_untagged_and_a_bad_policy_fails(world):
    con, cfg = world
    for i in range(4):
        like(con, f"l{i:02d}x", f"Liked Artist {i}", 5)
    cfg["harvest"]["seed_policy"] = "recent"
    h = Harvester(con, cfg, rng=random.Random(3))
    h.deezer_artist_radio(10)
    details = [r[0] for r in con.execute("SELECT detail FROM candidate_sources")]
    assert details and not any("seed=" in d for d in details)
    cfg["harvest"]["seed_policy"] = "loudest"
    with pytest.raises(ValueError, match="seed_policy"):
        Harvester(con, cfg)


def test_listenbrainz_asks_the_best_rated_seeds_first(world):
    con, cfg = world
    like(con, "a", "Unrated One")
    like(con, "b", "Four Star", 4)
    like(con, "c", "Five Star", 5)
    h = Harvester(con, cfg, rng=random.Random(1))
    h.listenbrainz_similar(10)
    assert [a for a, _ in h.lb.asked] == ["Five Star", "Four Star", "Unrated One"]


def test_broad_liked_seeds_go_by_stars_and_leave_out_passes(world):
    con, cfg = world
    like(con, "a", "Unrated One")
    like(con, "b", "Two Star", 2)
    like(con, "c", "Five Star", 5)
    like(con, "d", "Passed One", kind="pass")
    got = harvest.broad_liked_seeds(con, cfg, random.Random(1))
    assert [(s.artist, s.tag) for s in got] == [("Five Star", "5"), ("Unrated One", "like")]


def test_style_search_takes_the_most_followed_playlists_within_each_styles_share(world, monkeypatch):
    con, cfg = world
    monkeypatch.setattr(harvest, "top_style_names", lambda con, cfg, n: ["Bebop", "Polka"])
    h = Harvester(con, cfg, rng=random.Random(1))
    for q, base in (("Bebop", 10), ("Polka", 20)):
        h.dz.searches[q] = [{"id": base + i} for i in range(3)]
        for i in range(3):
            pid = base + i
            h.dz.playlists[pid] = {"id": pid, "fans": i * 100,
                                   "tracks": {"data": [track(pid * 100 + j, f"Artist {pid}-{j}", f"Song {j}")
                                                       for j in range(30)]}}
    h.deezer_style_search(10)
    rows = con.execute("SELECT detail, COUNT(*) FROM candidate_sources GROUP BY detail ORDER BY detail").fetchall()
    assert dict(rows) == {"style/Bebop/playlist/12": 5, "style/Polka/playlist/22": 5}   # most fans, 10 // 2


def test_style_search_skips_without_styles(world, monkeypatch):
    con, cfg = world
    h = Harvester(con, cfg)
    h.deezer_style_search(10)                       # no liked song has a style vector
    assert con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0
    assert harvest.top_style_names(con, cfg, 8) == []


def _rates(overall, **per):
    return {"overall": {"kept_rate": overall},
            "per_source": {s: {"decided": d, "kept_rate": r} for s, (d, r) in per.items()}}


def test_adaptive_caps(world):
    _, cfg = world
    cfg["harvest"]["caps"] = {"listenbrainz_similar": 90}
    r = _rates(0.2, good=(40, 0.4), great=(40, 0.9), poor=(40, 0.0), young=(29, 0.9), listenbrainz_similar=(50, 0.1))
    c = sources.caps(cfg, ["good", "great", "poor", "young", "listenbrainz_similar", "never"], r)
    assert c["good"]["cap"] == 120 and c["good"]["multiplier"] == 2.0
    assert c["great"]["cap"] == 120                                       # clipped at 2x
    assert c["poor"]["cap"] == 15 and c["poor"]["multiplier"] == 0.25    # clipped at 0.25x
    assert c["young"]["cap"] == 60 and "29 decided" in c["young"]["why"]
    assert c["listenbrainz_similar"]["cap"] == 45 and c["listenbrainz_similar"]["base"] == 90
    assert c["never"]["cap"] == 60
    assert sources.caps(cfg, ["good"], _rates(None, good=(40, 0.0)))["good"]["cap"] == 60
    cfg["harvest"]["adaptive_caps"] = False
    assert sources.caps(cfg, ["good"], r)["good"] == {"cap": 60, "base": 60, "multiplier": 1.0,
                                                      "why": "adaptive_caps off"}


def test_run_caps_fixed_by_the_flag_and_lifted_for_seeded_sources_in_broad_mode(world):
    con, cfg = world
    h = Harvester(con, cfg)
    assert h.caps(["deezer_chart"], 5) == {"deezer_chart": {"cap": 5, "why": "--limit-per-source"}}
    b = Harvester(con, cfg, seeds=[{"artist": "A", "title": "x"}], liked_seeds=[])
    c = b.caps(["deezer_chart", "deezer_artist_radio", "listenbrainz_similar"])
    assert c["deezer_artist_radio"]["cap"] == c["listenbrainz_similar"]["cap"] == 1000
    assert c["deezer_chart"]["cap"] == 60


def test_kept_rate_by_seed(world):
    con, cfg = world
    for i, (detail, outcome) in enumerate((("artist/1/radio (A) seed=5", "liked"),
                                           ("artist/1/radio (A) seed=5", "passed"),
                                           ("artist/2/radio (B) seed=library", "passed"),
                                           ("chart/0", "liked")), start=1):
        con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, source, state, spotify_id)
                       VALUES(?,?,?,?,?, 'offered', ?)""", (i, i, f"Artist {i}", f"Title {i}",
                                                            "deezer_chart" if i == 4 else "deezer_artist_radio",
                                                            sid(f"o{i}")))
        con.execute("INSERT INTO candidate_sources(candidate_id, source, detail, seen_at) VALUES(?,?,?,?)",
                    (i, "deezer_chart" if i == 4 else "deezer_artist_radio", detail, db.now()))
        con.execute("""INSERT INTO offers(candidate_id, spotify_id, batch_id, source, added_at, outcome, removed_at)
                       VALUES(?,?,1,?,?,?,?)""", (i, sid(f"o{i}"), "deezer_chart" if i == 4 else "deezer_artist_radio",
                                                  db.now(), outcome, db.now()))
    con.execute("INSERT INTO candidate_sources(candidate_id, source, detail, seen_at) VALUES(4, 'deezer_chart', ?, ?)",
                ("chart/0 seed=4", "2999-01-01T00:00:00Z"))   # seen again later: the first sighting counts
    r = sources.rates(con)
    assert list(r["per_seed"]) == ["5★", "library", "untagged"]
    assert r["per_seed"]["5★"] == {"offered": 2, "decided": 2, "kept": 1, "kept_rate": 0.5}
    assert r["per_source"]["deezer_artist_radio"]["kept_rate"] == round(1 / 3, 3)
    assert r["overall"]["kept_rate"] == 0.5


def test_chart_picks_per_batch_are_capped():
    chart = frozenset({"deezer_chart", "deezer_editorial"})
    pool = [Pick(i, f"artist{i}", 1.0 - i / 100, "deezer_chart" if i < 10 else "deezer_related")
            for i in range(40)]
    got = select_batch(pool, 15, 0.0, max_per_artist=1, rng=random.Random(1), chart_sources=chart, max_chart=3)
    assert len(got) == 15 and sum(p.source in chart for p in got) == 3
    taken = [Pick(100, "x", 0.5, "deezer_editorial"), Pick(101, "y", 0.5, "deezer_chart")]
    got = select_batch(pool, 13, 0.0, rng=random.Random(1), taken=taken, chart_sources=chart, max_chart=3)
    assert sum(p.source in chart for p in got) == 1
    got = select_batch(pool, 15, 0.0, rng=random.Random(1))                 # no cap by default
    assert sum(p.source in chart for p in got) == 10


def test_weekly_broad_harvest_job_waits_for_its_time(tmp_path):
    cfg = config.load(tmp_path / "none.toml")
    s = scheduler.Scheduler(cfg, run=lambda job, args: 0)
    assert s.specs["harvest_broad"] == scheduler.parse("Sun 04:00")
    assert scheduler.JOBS["harvest_broad"] == ["harvest", "--broad", "150"]
    assert not scheduler.missed("harvest_broad", s.specs["harvest_broad"], None, datetime(2026, 1, 7, 12))
