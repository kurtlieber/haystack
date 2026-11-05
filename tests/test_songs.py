"""The Song as a row of its own (ADR 0025): the harvest keeps a new candidate's ISRC and folds a re-release under
another Deezer id into the candidate it has, the backfill asks Deezer for older candidates' ISRCs, every row joins
a Song by its ISRC or its names, and Known tries the Song first. A fake Deezer; no network."""
import pytest

from haystack_music import cli, config, db, harvest, known, report, songs, traits
from haystack_music.deezer import DeezerError
from haystack_music.harvest import Harvester


class FakeDeezer:
    """/track/{id} answers with `isrcs[id]` (no key: no ISRC) and a bpm; `fail` lists ids that raise once."""
    def __init__(self, *a, **k):
        self.requests = 0
        self.asked: list[int] = []
        self.isrcs: dict[int, str] = {}
        self.fail: set[int] = set()

    def track(self, did):
        self.requests += 1
        self.asked.append(did)
        if did in self.fail:
            self.fail.discard(did)
            raise DeezerError(f"track/{did}: synthetic failure")
        out = {"id": did, "bpm": 120.0, "preview": f"https://example.test/{did}.mp3"}
        if did in self.isrcs:
            out["isrc"] = self.isrcs[did]
        return out


def track(did, artist, title, duration=200):
    return {"id": did, "title": title, "duration": duration, "preview": f"https://example.test/{did}.mp3",
            "artist": {"id": 9, "name": artist}}


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "t.db")


@pytest.fixture
def intake(tmp_path, monkeypatch, con):
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    monkeypatch.setattr(harvest, "Deezer", FakeDeezer)
    hints = []
    monkeypatch.setattr(traits, "analyse_preview", lambda dz, url, bpm_hint=None: hints.append(bpm_hint) or {})
    monkeypatch.setattr(traits, "store", lambda con, did, r: None)
    return lambda: Harvester(con, config.load(tmp_path / "none.toml")), hints


def cand(con, cid, artist="Some Artist", title="Some Song", isrc=None, state="pool", score=None, checked=None):
    con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, song_key, source, state, vector_status, isrc,
                   score, isrc_checked_at) VALUES(?,?,?,?,?, 'src', ?, 'ok', ?,?,?)""",
                (cid, cid, artist, title, f"song {cid}", state, isrc, score, checked))


def song_of(con, table, ref):
    k = "spotify_id" if table == "spotify_tracks" else "id"
    return con.execute(f"SELECT song_id FROM {table} WHERE {k}=?", (ref,)).fetchone()[0]


def test_the_harvest_keeps_the_isrc_and_folds_a_rerelease_into_its_candidate(intake, con):
    make, hints = intake
    h = make()
    h.dz.isrcs = {42: "usaa10000042", 43: "USAA10000042"}
    assert h.offer("src", "d", track(42, "Some Artist", "A Song"))
    # the same recording on a best-of: another Deezer id, other names, the same ISRC
    assert not h.offer("other", "d", track(43, "Some Artist & Friend", "A Song (Greatest Hits Version)"))
    assert h.offer("src", "d", track(44, "Some Artist", "Another Song"))     # Deezer has no ISRC for it
    rows = [tuple(r) for r in con.execute("SELECT deezer_id, isrc, isrc_checked_at IS NOT NULL FROM candidates "
                                          "ORDER BY deezer_id")]
    assert rows == [(42, "USAA10000042", 1), (44, None, 1)]
    assert h.dz.asked == [42, 43, 44]          # once per new track: the ISRC and the bpm from one request
    assert hints == [120.0, 120.0]
    srcs = {r[0] for r in con.execute("SELECT source FROM candidate_sources WHERE candidate_id="
                                      "(SELECT id FROM candidates WHERE deezer_id=42)")}
    assert srcs == {"src", "other"}
    assert not h.offer("src", "d", track(42, "Some Artist", "A Song"))       # a known Deezer id asks nothing
    assert h.dz.asked == [42, 43, 44]


def test_a_library_file_matches_by_the_isrc_of_its_spotify_tag(intake, con):
    con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, duration, isrc) VALUES('sp1', "
                "'Tagged Song', '[\"Tagged Artist\"]', 200.0, 'USAA10000007')")
    con.execute("""INSERT INTO local_tracks(id, path, size, content_hash, artist_raw, title_raw, artists_json,
                   duration, spotify_id, present) VALUES(7, '/m/7.m4a', 1, 'h7', 'Tagged Artist', 'Tagged Song',
                   '["Tagged Artist"]', 200.0, 'sp1', 1)""")
    make, _ = intake
    h = make()
    h.dz.isrcs = {50: "USAA10000007"}
    assert not h.offer("src", "d", track(50, "Some Compilation Artist", "Totally Different Name"))
    assert con.execute("SELECT state, exclude_reason FROM candidates WHERE deezer_id=50").fetchone()[:] == \
        ("excluded", "in_library")


def test_backfill_asks_the_pool_first_and_never_twice(con):
    cand(con, 1, state="excluded")
    cand(con, 2, state="pool", score=0.2)
    cand(con, 3, state="pool", score=0.9)
    cand(con, 4, state="liked")
    cand(con, 5, state="pool", isrc="USAA10000005")                 # has one already
    cand(con, 6, state="pool", checked="2026-01-01T00:00:00Z")      # asked before: Deezer had none
    assert songs.backfill_order(con, 10) == [(3, 3), (2, 2), (4, 4), (1, 1)]
    dz = FakeDeezer()
    dz.isrcs = {3: "USAA10000003"}
    dz.fail = {2}
    out = songs.backfill(con, dz, 3)
    assert (out["todo"], out["found"], out["none"], out["errors"]) == (3, 1, 1, 1)
    assert dz.asked == [3, 2, 4]
    assert songs.backfill_order(con, 10) == [(2, 2), (1, 1)]         # the error is asked again; 4 is not
    assert out["coverage"]["pool"] == {"n": 4, "with_isrc": 2, "isrc_share": 0.5, "with_song": 4}
    assert out["songs"]["candidates"] == 6


def test_a_song_is_found_by_isrc_then_by_names(con):
    nk = songs.name_key("AC/DC", "Hells Bells - 2009 Remaster")
    assert nk == songs.name_key("AC/DC", "Hells Bells") != songs.name_key("AC/DC", "Hells Bells - Live")
    a = songs.find_or_make(con, None, nk)
    assert songs.find_or_make(con, None, nk) == a                  # the same names
    assert songs.find_or_make(con, "USAA10000001", nk) == a        # a name-keyed Song takes the first code
    b = songs.find_or_make(con, "USAA10000002", nk)                # a remaster with a code of its own
    assert b != a
    assert songs.find_or_make(con, None, nk) == a                  # names alone: the oldest Song of them
    assert songs.find_or_make(con, "USAA10000002", nk) == b
    assert songs.norm_isrc(" usaa10000001 ") == "USAA10000001" and songs.norm_isrc("short") is None


def test_assign_joins_every_table_and_folds_a_song_when_an_isrc_joins_them(con):
    con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, isrc) VALUES('sp1', 'Song One', "
                "'[\"Band\"]', 'USAA10000001')")
    con.execute("""INSERT INTO local_tracks(id, path, size, content_hash, artist_raw, title_raw, spotify_id, present)
                   VALUES(7, '/m/7.m4a', 1, 'h7', 'Band', 'Song One (Remastered)', 'sp1', 1)""")
    cand(con, 1, "Band", "Song One")                                 # names only, for now
    cand(con, 2, "Band", "Song One - Live")                          # another recording
    cand(con, 3, "Another Band", "Song One", isrc="USAA10000003")
    out = songs.assign(con)
    assert out == {"candidates": 3, "spotify_tracks": 1, "local_tracks": 1, "merged": 0}
    s1 = song_of(con, "spotify_tracks", "sp1")
    assert song_of(con, "local_tracks", 7) == s1                    # its ISRC, by its Spotify id tag
    assert song_of(con, "candidates", 1) == s1                      # the names of a Song whose code it lacks
    assert song_of(con, "candidates", 2) not in (s1, None)
    assert songs.assign(con) == {"candidates": 0, "spotify_tracks": 0, "local_tracks": 0, "merged": 0}
    # a code that isn't one, or another spelling of the Song's, is no work the next time
    con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, isrc) VALUES('sp2', 'Song Two', "
                "'[\"Band\"]', 'not-an-isrc')")
    con.execute("UPDATE candidates SET isrc='usaa10000003 ' WHERE id=3")
    assert songs.assign(con)["spotify_tracks"] == 1
    assert songs.assign(con) == {"candidates": 0, "spotify_tracks": 0, "local_tracks": 0, "merged": 0}
    con.execute("UPDATE candidates SET isrc='USAA10000003' WHERE id=3")
    # a names-only Song whose candidate gets its code from the backfill: the code is another Song's, so they fold
    cand(con, 4, "Band", "Different Title")
    songs.assign(con)
    s4 = song_of(con, "candidates", 4)
    con.execute("UPDATE candidates SET isrc='USAA10000003' WHERE id=4")
    assert songs.assign(con)["merged"] == 1
    assert song_of(con, "candidates", 4) == song_of(con, "candidates", 3)
    assert con.execute("SELECT 1 FROM songs WHERE id=?", (s4,)).fetchone() is None
    shown = songs.show(con, "usaa10000003")
    assert {c["id"] for c in shown["candidates"]} == {3, 4}


def test_known_tries_the_song_and_still_the_names(con):
    con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, duration, isrc) VALUES('sp1', "
                "'Song One', '[\"Band\"]', 200.0, 'USAA10000001')")
    con.execute("INSERT INTO likes(spotify_id, added_at) VALUES('sp1', '2026-01-01T00:00:00Z')")
    cand(con, 1, "Band", "Song One (Extended Mix 12\")")
    con.execute("UPDATE candidates SET duration=420.0, isrc='USAA10000001' WHERE id=1")   # far longer: the code says
    cand(con, 2, "Band", "Song One - 2011 Remaster", isrc="USAA10000099")             # its own code: the names say
    con.execute("UPDATE candidates SET duration=201.0 WHERE id=2")
    cand(con, 3, "Band", "Song One - Live", isrc="USAA10000098")
    rows = lambda: {r["id"]: r for r in con.execute("SELECT * FROM candidates")}      # noqa: E731
    before = {cid: known.Known(con).why(r) for cid, r in rows().items()}             # no Songs yet: as before
    songs.assign(con)
    k = known.Known(con)
    assert k.songs["liked"] == {song_of(con, "spotify_tracks", "sp1")}
    after = {cid: k.why(r) for cid, r in rows().items()}
    assert after == before == {1: "liked", 2: "liked", 3: None}
    # a names-only Song is the listener's even when the lengths are far apart
    con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, duration) VALUES('sp2', 'Song Two', "
                "'[\"Band\"]', 200.0)")
    con.execute("INSERT INTO likes(spotify_id, added_at) VALUES('sp2', '2026-01-01T00:00:00Z')")
    cand(con, 4, "Band", "Song Two")
    con.execute("UPDATE candidates SET duration=400.0 WHERE id=4")
    assert known.Known(con).why(rows()[4]) is None
    songs.assign(con)
    assert known.Known(con).why(rows()[4]) == "liked"


def test_coverage_and_the_report_line(con, tmp_path):
    cand(con, 1, isrc="USAA10000001")
    cand(con, 2, "Other", "Other Song", isrc="USAA10000001")
    cand(con, 3, "Third", "Third Song", state="excluded")
    songs.assign(con)
    cv = songs.coverage(con)
    assert cv["pool"] == {"n": 2, "with_isrc": 2, "isrc_share": 1.0, "with_song": 2}
    assert cv["all"]["n"] == 3 and cv["songs_with_2_or_more_candidates"] == 1
    r = report.build(con, config.load(tmp_path / "none.toml"), None)
    assert "Songs (ADR 0025): ISRC known for 2 of 2 in the pool (100%), 2 of 3 in all; 1 Songs" in report.text(r)


def test_cli_songs_backfill_and_show(con, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(db, "connect", lambda *a, **k: con)
    monkeypatch.setattr(db, "connect_readonly", lambda *a, **k: con)
    import haystack_music.deezer as deezer
    fake = FakeDeezer()
    fake.isrcs = {1: "USAA10000001"}
    monkeypatch.setattr(deezer, "Deezer", lambda *a, **k: fake)
    cand(con, 1)
    assert cli.main(["songs", "backfill", "--limit", "5"]) == 0
    assert "asked 1: 1 with an ISRC" in capsys.readouterr().out
    assert cli.main(["songs", "show", "USAA10000001"]) == 0
    assert '"deezer_id": 1' in capsys.readouterr().out
    assert cli.main(["songs", "show", "USAA10000009"]) == 1
