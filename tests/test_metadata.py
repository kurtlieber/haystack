"""What Deezer publishes about a track (ADR 0017): the harvest keeps a new candidate's album and rank, the backfill
fetches albums and artists (judged tracks and liked songs first), and the lookups the re-ranker and the card use.
A fake Deezer; no network."""
import json
from datetime import datetime, timezone

import pytest

from haystack_music import cli, config, db, harvest, metadata, traits
from haystack_music.deezer import DeezerError
from haystack_music.harvest import Harvester


def sid(n: str) -> str:
    return n.ljust(22, "0")


class FakeDeezer:
    """Tracks 1-99 sit on album 500 + did % 3 by artist 700 + did % 2; track 13 is gone; album 502 has no label;
    album 501 is a compilation. `fail` lists ids whose next request raises."""
    def __init__(self, *a, **k):
        self.requests = 0
        self.asked: list[str] = []
        self.fail: set[str] = set()

    def _ask(self, what):
        self.requests += 1
        self.asked.append(what)
        if what in self.fail:
            self.fail.discard(what)
            raise DeezerError(f"{what}: synthetic failure")

    def track(self, did):
        self._ask(f"track/{did}")
        if did == 13:
            return None
        return {"id": did, "rank": did * 1000, "album": {"id": 500 + did % 3}, "artist": {"id": 700 + did % 2}}

    def album(self, aid):
        self._ask(f"album/{aid}")
        return {500: {"label": "Synthetic Records", "release_date": "2016-01-01", "record_type": "album",
                      "genres": {"data": [{"id": 1, "name": "Pop"}]}, "fans": 10, "nb_tracks": 12},
                501: {"label": "Various Sounds", "release_date": "2000-06-30", "record_type": "compile",
                      "genres": {"data": []}, "fans": 3, "nb_tracks": 40},
                502: {"label": "", "release_date": "0000-00-00", "record_type": "single"}}.get(aid)

    def artist(self, aid):
        self._ask(f"artist/{aid}")
        return {"id": aid, "nb_fan": 10 ** (aid - 698)}


def cand(con, cid, state="pool", at="2026-01-01T00:00:00Z", album_id=None, artist_id=None, rank=None):
    con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status, harvested_at,
                   album_id, deezer_artist_id, deezer_rank, spotify_id) VALUES(?,?,?,?, 'src', ?, 'ok', ?,?,?,?,?)""",
                (cid, cid, f"Artist {cid}", f"Title {cid}", state, at, album_id, artist_id, rank, sid(f"c{cid}")))


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "t.db")


def test_the_harvest_keeps_the_album_and_rank_it_already_has(tmp_path, monkeypatch, con):
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    monkeypatch.setattr(harvest, "Deezer", FakeDeezer)
    monkeypatch.setattr(traits, "analyse_preview", lambda dz, url, bpm_hint=None: {})
    monkeypatch.setattr(traits, "store", lambda con, did, r: None)
    h = Harvester(con, config.load(tmp_path / "none.toml"))
    t = {"id": 42, "title": "A Song", "duration": 200, "preview": "https://example.test/42.mp3", "rank": 654321,
         "artist": {"id": 9, "name": "Some Artist"}, "album": {"id": 77, "title": "Some Album"}}
    assert h.offer("src", "detail", t)
    assert h.offer("src", "detail", {**t, "id": 43, "title": "Another Song", "album": None, "rank": None})
    rows = con.execute("SELECT deezer_id, album_id, deezer_rank, deezer_artist_id FROM candidates ORDER BY deezer_id")
    assert [tuple(r) for r in rows] == [(42, 77, 654321, 9), (43, None, None, 9)]
    assert h.dz.requests == 0                                        # nothing asked: the track object had it


def test_backfill_goes_judged_liked_then_the_pool_oldest_first(con):
    cand(con, 1, at="2026-01-03T00:00:00Z")
    cand(con, 2, at="2026-01-01T00:00:00Z")
    cand(con, 3, state="offered", at="2026-01-05T00:00:00Z")
    cand(con, 4, at="2026-01-02T00:00:00Z")
    con.execute("INSERT INTO calibrations(candidate_id, spotify_id, picked_at, week, band, percentile) "
                "VALUES(4, ?, '2026-01-05', '2026-W02', '90-97', 93.0)", (sid("c4"),))
    con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid("like"),))
    con.execute("INSERT INTO liked_map(spotify_id, use_local, deezer_id, vector_status) VALUES(?, 0, 9, 'ok')",
                (sid("like"),))
    cand(con, 5, state="excluded")                                   # never asked
    order = [r[1] for r in metadata._todo(con, 10)]
    assert order == [4, 3, sid("like"), 2, 1]                        # each group oldest first
    assert [r[1] for r in metadata._todo(con, 3)] == [4, 3, sid("like")]
    assert metadata._todo(con, 0) == []


def test_backfill_fetches_each_album_and_artist_once_and_counts_what_failed(con):
    for cid in (1, 2, 3, 4, 13):
        cand(con, cid)
    cand(con, 7, album_id=501, artist_id=701, rank=7000)             # from the harvest: no track request
    dz = FakeDeezer()
    dz.fail = {"album/502"}
    out = metadata.backfill(con, dz, 10)
    assert (out["todo"], out["tracks"], out["gone"], out["errors"]) == (6, 5, 1, 1)
    assert (out["albums"], out["artists"]) == (2, 2)                 # 501 and 500; 701 and 700
    assert dz.asked == ["track/1", "album/501", "artist/701", "track/2", "album/502", "track/3", "album/500",
                        "track/4", "artist/700", "track/13"]
    rows = {r[0]: tuple(r)[1:] for r in con.execute("SELECT id, album_id, deezer_rank, deezer_artist_id "
                                                     "FROM candidates")}
    assert rows[1] == (501, 1000, 701) and rows[13] == (0, None, None)   # gone: album 0
    assert rows[7] == (501, 7000, 701)
    a = dict(con.execute("SELECT * FROM deezer_albums WHERE album_id=501").fetchone())
    assert (a["label"], a["release_date"], a["record_type"], a["nb_tracks"]) == \
        ("Various Sounds", "2000-06-30", "compile", 40)
    assert json.loads(con.execute("SELECT genres_json FROM deezer_albums WHERE album_id=500").fetchone()[0]) == ["Pop"]
    assert con.execute("SELECT nb_fan FROM deezer_artist_fans WHERE artist_id=701").fetchone()[0] == 1000
    assert out["coverage"]["pool"] == {"n": 6, "with_album": 4, "share": 0.667}   # 2 waits for album 502
    # the failed album is asked again next time; 502 has neither label nor date: known, not counted as covered
    dz2 = FakeDeezer()
    again = metadata.backfill(con, dz2, 10)
    assert dz2.asked == ["album/502"] and again["errors"] == 0 and again["albums"] == 1
    assert tuple(con.execute("SELECT label, release_date FROM deezer_albums WHERE album_id=502").fetchone()) == \
        (None, None)
    assert again["coverage"]["pool"]["with_album"] == 4
    dz3 = FakeDeezer()
    assert metadata.backfill(con, dz3, 10)["todo"] == 0 and dz3.asked == []


def test_liked_songs_get_their_album_too(con):
    for name, did in (("l1", 1), ("l2", 2)):
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(name),))
        con.execute("INSERT INTO liked_map(spotify_id, use_local, deezer_id, vector_status) VALUES(?, 0, ?, 'ok')",
                    (sid(name), did))
    con.execute("UPDATE likes SET removed_at='2026-01-02' WHERE spotify_id=?", (sid("l2"),))   # no longer liked
    out = metadata.backfill(con, FakeDeezer(), 10)
    assert out["todo"] == 1 and out["coverage"]["liked"] == {"n": 1, "with_album": 1, "share": 1.0}
    assert con.execute("SELECT album_id FROM liked_map WHERE spotify_id=?", (sid("l1"),)).fetchone()[0] == 501
    assert metadata.labels_of_tracks(con, [sid("l1"), sid("l2")]) == {sid("l1"): "various sounds"}
    assert metadata.labels_for_cards(con, [1, 2, None]) == {1: "Various Sounds"}


def test_lookups_for_the_re_ranker(con):
    cand(con, 1, album_id=501, artist_id=701, rank=1234)
    cand(con, 2, album_id=0)
    cand(con, 3)
    metadata.backfill(con, FakeDeezer(), 0)
    metadata.store_album(con, 501, FakeDeezer().album(501))
    metadata._store_artist(con, 701, {"nb_fan": 1000})
    now = datetime(2010, 6, 30, tzinfo=timezone.utc)
    md = metadata.for_candidates(con, [1, 2, 3], now=now)
    assert md[1] == {"rank": 1234, "fans": 1000, "age": pytest.approx(10.0, abs=0.01), "compilation": True,
                     "label": "various sounds"}
    assert md[2] == md[3] == {"rank": None, "fans": None, "age": None, "compilation": None, "label": None}
    assert metadata._age_years("0000-00-00") is None and metadata._age_years(None) is None
    # owners: a rated discovery, a library track mapped to a liked song, a manual seed, one with nothing known
    con.execute("INSERT INTO local_tracks(id, path, size, content_hash, spotify_id) VALUES(5, '/m/a.m4a', 1, 'h', ?)",
                (sid("lib"),))
    con.execute("INSERT INTO liked_map(spotify_id, use_local, deezer_id, album_id) VALUES(?, 1, 8, 501)",
                (sid("lib"),))
    assert metadata.owner_labels(con, [f"rated:{sid('c1')}", "local:5", "manual:1", f"like:{sid('none')}"]) == \
        ["various sounds", "various sounds", "various sounds", None]


def test_the_cli_backfills_and_prints_the_coverage(tmp_path, monkeypatch, capsys, con):
    cand(con, 1)
    con.close()
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "STATUS_PATH", tmp_path / "state" / "status.json")
    fake = FakeDeezer()
    monkeypatch.setattr("haystack_music.deezer.Deezer", lambda *a, **k: fake)
    cfg = config.load(tmp_path / "none.toml")
    a = cli.build_parser().parse_args(["metadata", "backfill", "--limit", "0"])
    assert cli.cmd_metadata(cfg, a) == 0
    assert "album known: pool 0/1, judged 0/0, liked 0/0; 0 Deezer requests" in capsys.readouterr().out
    a = cli.build_parser().parse_args(["metadata", "backfill"])
    assert a.limit == 1000 and cli.cmd_metadata(cfg, a) == 0
    assert "album known: pool 1/1, judged 0/0, liked 0/0; 3 Deezer requests, 0 errors" in capsys.readouterr().out


def test_a_database_from_before_0_6_0_knows_nothing_and_breaks_nothing(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""CREATE TABLE candidates(id INTEGER PRIMARY KEY, deezer_id INTEGER, spotify_id TEXT,
                         deezer_artist_id INTEGER);
                         CREATE TABLE liked_map(spotify_id TEXT PRIMARY KEY, local_id INTEGER, deezer_id INTEGER);
                         CREATE TABLE local_tracks(id INTEGER PRIMARY KEY, spotify_id TEXT);
                         INSERT INTO candidates VALUES(1, 1, 'x', 9);""")
    old.close()
    con = db.connect_readonly(path)                                   # `batch --plan`, `rerank show`
    assert metadata.for_candidates(con, [1]) == {}
    assert metadata.owner_labels(con, ["like:x", "local:1", "manual:1"]) == [None, None, None]
    assert metadata.labels_for_cards(con, [1]) == {}


def test_the_harvest_backfills_deezers_metadata_unless_told_not_to_retry(tmp_path, monkeypatch, con):
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    monkeypatch.setattr(harvest, "Deezer", FakeDeezer)
    calls = []
    monkeypatch.setattr(metadata, "backfill", lambda con, dz, limit: calls.append((dz, limit)) or {"albums": 0})
    cfg = config.load(tmp_path / "none.toml")
    h = Harvester(con, cfg)
    for name in ("retry_errors", "backfill_styles", "_readings_backfill"):
        monkeypatch.setattr(h, name, lambda *a: {})
    assert h.run(["no_such_source"], 1)["metadata_backfill"] == {"albums": 0}
    assert calls == [(h.dz, cfg["harvest"]["metadata_per_run"])]
    assert h.run(["no_such_source"], 1, retry=False)["metadata_backfill"] == {"skipped": True}
    assert len(calls) == 1


def test_only_the_first_shard_of_a_broad_harvest_retries_and_backfills(tmp_path, monkeypatch, con):
    con.close()
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "STATUS_PATH", tmp_path / "state" / "status.json")
    seen = []

    class FakeHarvester:
        def __init__(self, con, cfg, seeds=None, liked_seeds=None):
            self.seeds = seeds

        def run(self, sources, cap, retry=True):
            seen.append((len(self.seeds), retry))
            return {}
    monkeypatch.setattr(harvest, "Harvester", FakeHarvester)
    monkeypatch.setattr(harvest, "stratified_seeds", lambda con, n, rng: [{"bucket": "b"}] * n)
    monkeypatch.setattr(harvest, "broad_liked_seeds", lambda con, cfg, rng: [])
    cfg = config.load(tmp_path / "none.toml")
    for argv in (["--shard", "0/2"], ["--shard", "1/2"], [], ["--shard", "0/2", "--no-retry"]):
        a = cli.build_parser().parse_args(["harvest", "--broad", "4", "--no-score", *argv])
        assert cli.cmd_harvest(cfg, a) == 0
    assert seen == [(2, True), (2, False), (4, True), (2, False)]
