import pytest

from haystack_music import config, db, verdicts
from haystack_music.textnorm import song_key
from haystack_music.verdicts import Verdict


def sid(name: str) -> str:
    return name.ljust(22, "0")


T1, DISC_ID, LIKE_ID = sid("trk1"), sid("disc"), sid("like")


def _con(tmp_path):
    return db.connect(tmp_path / "t.db")


def _cfg(to_rip=True):
    cfg = config.load(config.CONFIG_FILE.parent / "does-not-exist.toml")
    cfg["spotify"]["to_rip"] = to_rip
    return cfg


# ---------------- store ----------------

def test_latest_given_verdict_is_current_and_history_is_kept(tmp_path):
    con = _con(tmp_path)
    verdicts.record(con, T1, "rating", 4, given_at="2026-10-11T10:00:00Z", id="a")
    verdicts.record(con, T1, "pass", given_at="2026-10-11T10:05:00Z", id="b")
    # an outbox replay of an older verdict arrives last: it must not win
    verdicts.record(con, T1, "rating", 2, given_at="2026-10-11T09:00:00Z", id="c")
    cur = verdicts.current(con)
    assert cur[T1] == Verdict(T1, "pass", None, "2026-10-11T10:05:00Z")
    assert [h["id"] for h in verdicts.history(con, T1)] == ["c", "a", "b"]


def test_record_is_idempotent_by_id(tmp_path):
    con = _con(tmp_path)
    assert verdicts.record(con, T1, "rating", 5, id="same")["status"] == "accepted"
    assert verdicts.record(con, T1, "rating", 5, id="same")["status"] == "duplicate"
    assert len(verdicts.history(con, T1)) == 1


@pytest.mark.parametrize("kind,stars", [("rating", None), ("rating", 0), ("rating", 6), ("pass", 3),
                                         ("skip", None), ("dislike", 1)])
def test_invalid_verdicts_are_refused(tmp_path, kind, stars):
    with pytest.raises(ValueError):
        verdicts.record(_con(tmp_path), T1, kind, stars)


def test_track_id_must_look_like_a_spotify_id(tmp_path):
    with pytest.raises(ValueError):
        verdicts.record(_con(tmp_path), "not an id; drop table", "pass")


def test_feedback_rows_migrate_once(tmp_path):
    con = _con(tmp_path)
    con.executescript("""CREATE TABLE feedback (id INTEGER PRIMARY KEY, candidate_id INTEGER, spotify_id TEXT,
        artist TEXT, title TEXT, verdict TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0,
        source TEXT NOT NULL DEFAULT 'manual', note TEXT, created_at TEXT, UNIQUE (candidate_id, source));""")
    rows = [(1, "sidYes", "yes", 1.0), (2, "sidMild", "no", 0.5), (3, "sidNormal", "no", 1.0),
            (4, "sidStrong", "no", 2.0), (5, None, "no", 1.0)]
    for cid, sid, v, w in rows:
        con.execute("INSERT INTO feedback(candidate_id, spotify_id, verdict, weight, created_at) VALUES(?,?,?,?,?)",
                    (cid, sid, v, w, "2026-10-09T00:00:00Z"))
    verdicts.ensure(con)
    verdicts.ensure(con)                                    # second run: no duplicates
    cur = verdicts.current(con)
    assert cur["sidYes"].kind == "rating" and cur["sidYes"].stars == 3
    assert cur["sidMild"].kind == "pass" and cur["sidNormal"].kind == "pass"
    assert cur["sidStrong"].kind == "dislike"
    assert len(cur) == 4                                    # a row with no Spotify id cannot be keyed
    assert {h["effects"] for s in cur for h in verdicts.history(con, s)} == {"skipped"}
    assert not verdicts.pending(con)


def test_a_migrated_clear_no_can_still_be_taken_back(tmp_path):
    con = _con(tmp_path)
    con.executescript("""CREATE TABLE feedback (id INTEGER PRIMARY KEY, candidate_id INTEGER, spotify_id TEXT,
        artist TEXT, title TEXT, verdict TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0,
        source TEXT NOT NULL DEFAULT 'manual', note TEXT, created_at TEXT, UNIQUE (candidate_id, source));""")
    con.execute("INSERT INTO feedback(candidate_id, spotify_id, verdict, weight, source, created_at) "
                "VALUES(1, 'sidClear', 'no', 1.0, 'playlist', '2026-10-12T00:00:00Z')")
    con.execute("INSERT INTO feedback(candidate_id, spotify_id, verdict, weight, source, created_at) "
                "VALUES(2, 'sidManual', 'no', 1.0, 'manual', '2026-10-12T00:00:00Z')")
    verdicts.ensure(con)
    assert not verdicts.judged_elsewhere(con, "sidClear") and verdicts.judged_elsewhere(con, "sidManual")
    assert verdicts.retract_playlist_no(con, "sidClear") and not verdicts.retract_playlist_no(con, "sidManual")
    assert set(verdicts.current(con)) == {"sidManual"}


# ---------------- weights ----------------

def test_an_empty_star_bonus_is_linear_in_pool_sd_and_three_stars_are_neutral():
    cfg = _cfg()
    cfg["scoring"]["verdicts"]["star_bonus"] = []
    assert verdicts.stars_in_sd(cfg)
    b = [verdicts.positive_bonus(cfg, Verdict("t", "rating", s, "")) for s in range(1, 6)]
    assert b == pytest.approx([-0.5, -0.25, 0.0, 0.25, 0.5])


def test_the_default_star_bonus_is_fixed_offsets_and_three_stars_equal_a_discovery_like():
    cfg = _cfg()
    assert not verdicts.stars_in_sd(cfg)
    b = [verdicts.positive_bonus(cfg, Verdict("t", "rating", s, "")) for s in range(1, 6)]
    assert b == pytest.approx([0.0, 0.0025, 0.005, 0.0075, 0.01])
    assert b[2] == cfg["scoring"]["bonus_discovery_like"]


def test_unrated_like_counts_as_three_stars_and_negatives_are_not_positives():
    cfg = _cfg()
    assert verdicts.positive_bonus(cfg, None, liked=True) == pytest.approx(0.005)
    assert verdicts.positive_stars(cfg, None, liked=True) == 3
    assert verdicts.positive_bonus(cfg, None, liked=False) is None
    assert verdicts.positive_bonus(cfg, Verdict("t", "pass", None, ""), liked=True) is None
    assert verdicts.negative_strength(cfg, Verdict("t", "pass", None, "")) == 1.0
    assert verdicts.negative_strength(cfg, Verdict("t", "dislike", None, "")) == 2.0
    assert verdicts.negative_strength(cfg, Verdict("t", "rating", 1, "")) is None


def test_an_unrated_like_gains_a_star_per_doubling_of_its_recent_plays(tmp_path):
    from datetime import datetime, timedelta, timezone
    from haystack_music import baseline
    cfg = _cfg()
    assert [verdicts.positive_stars(cfg, None, liked=True, plays=n) for n in (0, 1, 2, 3, 7, 50)] == [3, 4, 4, 5, 5, 5]
    assert verdicts.positive_stars(cfg, Verdict("t", "rating", 2, ""), plays=50) == 2      # a Rating stands
    assert verdicts.positive_stars(cfg, None, liked=False, plays=50) is None
    con = _con(tmp_path)
    now = datetime.now(timezone.utc)
    for h, ctx in ((1, "needle"), (2, "spotify:playlist:x"), (3, None)):
        con.execute("INSERT INTO plays(spotify_id, played_at, context) VALUES(?,?,?)",
                    (T1, (now - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ"), ctx))
    con.execute("INSERT INTO plays(spotify_id, played_at, context) VALUES(?, '2000-01-01T00:00:00Z', 'needle')",
                (LIKE_ID,))                                         # outside unrated_like_days
    con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01'), (?, '2026-01-01')", (T1, LIKE_ID))
    assert verdicts.like_plays(con, cfg) == {T1: 3}
    assert baseline.liked_weights(con, cfg) == {T1: 5.0, LIKE_ID: 3.0}
    cfg["scoring"]["verdicts"]["unrated_like_plays"] = False
    assert verdicts.like_plays(con, cfg) == {} and baseline.liked_weights(con, cfg) == {T1: 3.0, LIKE_ID: 3.0}


def test_a_skip_is_no_play_and_a_needle_play_spotify_reports_too_counts_once(tmp_path):
    from datetime import datetime, timedelta, timezone
    cfg, con = _cfg(), _con(tmp_path)
    now = datetime.now(timezone.utc)

    def play(track, minutes_ago, ctx="needle", ms=None, done=None):
        con.execute("INSERT INTO plays(spotify_id, played_at, context, ms_played, completed) VALUES(?,?,?,?,?)",
                    (track, (now - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ"), ctx, ms, done))
    for m in (300, 200, 100):
        play(T1, m, ms=6000, done=0)                        # three skips at 6 s: soft nos, not listens
    play(LIKE_ID, 120, ms=240000, done=1)                   # a full needle play ...
    play(LIKE_ID, 116, ctx="spotify:album:x")               # ... that Spotify reports when it ends: one listen
    play(LIKE_ID, 60, ctx="spotify:album:x")                # a later Spotify play is another
    play(DISC_ID, 50, ms=6000, done=0)                      # a skip Spotify reports too: still no listen
    play(DISC_ID, 49, ctx=None)
    assert verdicts.like_plays(con, cfg) == {LIKE_ID: 2}
    con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (T1,))
    assert verdicts.positive_stars(cfg, None, liked=True, plays=verdicts.like_plays(con, cfg).get(T1, 0)) == 3


def test_weights_are_configurable(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[scoring.verdicts]\nstar_bonus = [0.0, 0.0, 0.01, 0.02, 0.04]\ndislike_strength = 3.0\n")
    cfg = config.load(p)
    assert verdicts.positive_bonus(cfg, Verdict("t", "rating", 5, "")) == 0.04
    assert verdicts.negative_strength(cfg, Verdict("t", "dislike", None, "")) == 3.0


# ---------------- effects: the ADR 0002 table ----------------

def _plan(v, **kw):
    state = dict(liked=False, in_library=False, on_to_rip=False, in_discoveries=False)
    state.update(kw)
    return set(verdicts.plan(v, **state))


@pytest.mark.parametrize("stars", [3, 4, 5])
def test_plan_three_stars_and_up(stars):
    v = Verdict("t", "rating", stars, "")
    assert _plan(v) == {"heart_add", "to_rip_add"}
    assert _plan(v, liked=True, in_library=True) == set()
    assert _plan(v, liked=True, on_to_rip=True) == set()
    assert _plan(v, in_discoveries=True, liked=True) == {"discoveries_remove", "to_rip_add"}
    # spotify.to_rip off: nothing goes onto To Rip
    assert _plan(v, to_rip=False) == {"heart_add"}
    assert _plan(v, in_discoveries=True, liked=True, to_rip=False) == {"discoveries_remove"}


@pytest.mark.parametrize("v", [Verdict("t", "rating", 1, ""), Verdict("t", "rating", 2, ""),
                               Verdict("t", "pass", None, ""), Verdict("t", "dislike", None, "")])
def test_plan_below_three_stars_pass_and_dislike(v):
    assert _plan(v) == set()
    assert _plan(v, liked=True, on_to_rip=True, in_discoveries=True) == {
        "heart_remove", "to_rip_remove", "discoveries_remove"}
    # a track the bot put on To Rip before spotify.to_rip was turned off still comes off
    assert _plan(v, liked=True, on_to_rip=True, to_rip=False) == {"heart_remove", "to_rip_remove"}


# ---------------- effects against a fake Spotify ----------------

class FakeSpotify:
    def __init__(self, liked=(), scopes_ok=True):
        self.liked = set(liked)
        self.scopes_ok = scopes_ok
        self.playlists = {"PLdisc": set(), "PLrip": set()}
        self.calls = []
        self.token_meta = {}

    def library_contains(self, ids):
        return {i: i in self.liked for i in ids}

    def library_save(self, ids):
        self.calls.append(("save", tuple(ids)))
        if not self.scopes_ok:
            from haystack_music.spotify import SpotifyError
            raise SpotifyError(403, "Insufficient client scope")
        self.liked |= set(ids)

    def library_remove(self, ids):
        self.calls.append(("remove", tuple(ids)))
        if not self.scopes_ok:
            from haystack_music.spotify import SpotifyError
            raise SpotifyError(403, "Insufficient client scope")
        self.liked -= set(ids)

    def add_items(self, pid, ids):
        self.calls.append(("add", pid, tuple(ids)))
        self.playlists[pid] |= set(ids)
        return "snap"

    def remove_items(self, pid, ids, snap=None):
        self.calls.append(("pl_remove", pid, tuple(ids)))
        self.playlists[pid] -= set(ids)
        return "snap"

    def playlist(self, pid):
        return {"id": pid, "snapshot_id": "snap"}

    def track(self, sid):
        return {"id": sid, "name": "Unknown Song", "artists": [{"name": "Nobody"}], "duration_ms": 200000,
                "album": {"name": "x", "album_type": "single", "release_date": "2020", "images": []}}


def _world(tmp_path):
    """A discovery on an active offer (in Discoveries, bot-added), plus a liked track already on To Rip."""
    con = _con(tmp_path)
    db.kv_set(con, "playlist_discoveries", "PLdisc")
    db.kv_set(con, "playlist_to_rip", "PLrip")
    for sid, name, artist in ((DISC_ID, "Flying Octopus", "Rezz"),
                              (LIKE_ID, "Some Like", "Liked Artist")):
        con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, duration) VALUES(?,?,?,?)",
                    (sid, name, f'["{artist}"]', 179.0))
    con.execute("INSERT INTO candidates(id, deezer_id, artist, title, song_key, source, state, spotify_id) "
                "VALUES(1, 11, 'Rezz', 'Flying Octopus', ?, 'deezer_artist_radio', 'offered', '" + DISC_ID + "')",
                (song_key("Rezz", "Flying Octopus"),))
    con.execute("INSERT INTO offers(candidate_id, spotify_id, batch_id, added_at) VALUES(1, '" + DISC_ID + "', 1, '2026-10-08T00:00:00Z')")
    con.execute("INSERT INTO playlist_adds(playlist, playlist_id, spotify_id, added_at) VALUES('discoveries','PLdisc','" + DISC_ID + "','x')")
    con.execute("INSERT INTO likes(spotify_id, added_at) VALUES('" + LIKE_ID + "', '2026-01-01T00:00:00Z')")
    con.execute("INSERT INTO playlist_adds(playlist, playlist_id, spotify_id, added_at) VALUES('to_rip','PLrip','" + LIKE_ID + "','x')")
    con.execute("INSERT INTO to_rip(spotify_id, added_at) VALUES('" + LIKE_ID + "', 'x')")
    sp = FakeSpotify(liked={LIKE_ID})
    sp.playlists["PLdisc"].add(DISC_ID)
    sp.playlists["PLrip"].add(LIKE_ID)
    return con, sp


def test_four_stars_on_a_discovery_hearts_it_and_takes_it_out_of_discoveries(tmp_path):
    con, sp = _world(tmp_path)
    verdicts.record(con, DISC_ID, "rating", 4, id="v1")
    out = verdicts.apply_effects(con, _cfg(), sp, library=[])
    assert out["applied"] == 1 and not verdicts.pending(con)
    assert DISC_ID in sp.liked
    assert DISC_ID not in sp.playlists["PLdisc"]
    assert DISC_ID in sp.playlists["PLrip"]             # not in the library
    o = con.execute("SELECT removed_at, outcome FROM offers").fetchone()
    assert o["removed_at"] and o["outcome"] == "rated"
    assert con.execute("SELECT state FROM candidates WHERE id=1").fetchone()[0] == "liked"
    assert con.execute("SELECT removed_at FROM likes WHERE spotify_id='" + DISC_ID + "'").fetchone()[0] is None
    # applying again changes nothing
    n = len(sp.calls)
    verdicts.apply_effects(con, _cfg(), sp, library=[])
    assert len(sp.calls) == n



def test_with_to_rip_off_nothing_goes_onto_to_rip(tmp_path):
    con, sp = _world(tmp_path)
    verdicts.record(con, DISC_ID, "rating", 4, id="v1")
    verdicts.apply_effects(con, _cfg(to_rip=False), sp, library=[])
    assert DISC_ID in sp.liked and DISC_ID not in sp.playlists["PLdisc"]
    assert DISC_ID not in sp.playlists["PLrip"]
    assert not con.execute("SELECT 1 FROM to_rip WHERE spotify_id='" + DISC_ID + "'").fetchone()
    # what the bot put on To Rip while it was on still comes off with a Pass
    verdicts.record(con, LIKE_ID, "pass", id="v2")
    verdicts.apply_effects(con, _cfg(to_rip=False), sp, library=[])
    assert LIKE_ID not in sp.playlists["PLrip"]

def test_pass_on_an_unrated_like_removes_the_heart_and_its_to_rip_entry(tmp_path):
    con, sp = _world(tmp_path)
    verdicts.record(con, LIKE_ID, "pass", id="v1")
    verdicts.apply_effects(con, _cfg(), sp, library=[])
    assert LIKE_ID not in sp.liked
    assert LIKE_ID not in sp.playlists["PLrip"]
    assert con.execute("SELECT removed_at FROM likes").fetchone()[0]
    assert con.execute("SELECT removed_at FROM to_rip").fetchone()[0]


def test_dislike_on_a_discovery_closes_the_offer_as_negative(tmp_path):
    con, sp = _world(tmp_path)
    verdicts.record(con, DISC_ID, "dislike", id="v1")
    verdicts.apply_effects(con, _cfg(), sp, library=[])
    assert con.execute("SELECT outcome FROM offers").fetchone()[0] == "disliked"
    assert con.execute("SELECT state FROM candidates").fetchone()[0] == "negative"
    assert ("save", (DISC_ID,)) not in sp.calls


def test_missing_library_scope_keeps_the_heart_pending_but_applies_the_rest(tmp_path):
    con, sp = _world(tmp_path)
    sp.scopes_ok = False
    verdicts.record(con, DISC_ID, "rating", 5, id="v1")
    out = verdicts.apply_effects(con, _cfg(), sp, library=[])
    assert out["pending"] == 1
    row = verdicts.history(con, DISC_ID)[0]
    assert row["effects"] == "pending" and "user-library-modify" in row["effects_error"]
    assert DISC_ID not in sp.playlists["PLdisc"]        # the playlist part still happened
    sp.scopes_ok = True                                                   # after re-auth
    verdicts.apply_effects(con, _cfg(), sp, library=[])
    assert DISC_ID in sp.liked and not verdicts.pending(con)


def test_only_the_current_verdict_is_applied(tmp_path):
    con, sp = _world(tmp_path)
    verdicts.record(con, DISC_ID, "rating", 5, given_at="2026-10-11T10:00:00Z", id="v1")
    verdicts.record(con, DISC_ID, "rating", 2, given_at="2026-10-11T10:01:00Z", id="v2")
    verdicts.apply_effects(con, _cfg(), sp, library=[])
    assert DISC_ID not in sp.liked                     # 2★: never hearted
    assert not any(c[0] == "save" for c in sp.calls)
    assert {h["effects"] for h in verdicts.history(con, DISC_ID)} == {"done"}


# ---------------- export / import ----------------

def _rated(tmp_path):
    from haystack_music.sync import upsert_track
    con = db.connect(tmp_path / "a.db")
    upsert_track(con, {"id": T1, "name": "Song One", "artists": [{"name": "A"}, {"name": "B"}], "duration_ms": 201500,
                       "external_ids": {"isrc": "usabc2600001"}, "album": {"name": "Album"}})
    con.execute("INSERT INTO candidates(id, deezer_id, artist, title, song_key, source, state, spotify_id) "
                "VALUES(1, 11, 'Cand Artist', 'Cand Title', 'k', 'deezer_chart', 'offered', ?)", (DISC_ID,))
    verdicts.record(con, T1, "rating", 4, given_at="2026-10-11T10:00:00Z", id="a", queue="ratings")
    verdicts.record(con, T1, "rating", 5, given_at="2026-10-12T10:00:00Z", id="b")
    verdicts.record(con, DISC_ID, "pass", given_at="2026-10-11T11:00:00Z", id="c", source="cli")
    return con


def test_export_is_every_verdict_oldest_first_with_names(tmp_path):
    doc = verdicts.export_doc(_rated(tmp_path))
    assert doc["format"] == "haystack-verdicts" and doc["version"] == 1
    assert [v["id"] for v in doc["verdicts"]] == ["a", "c", "b"]
    a, c, _ = doc["verdicts"]
    assert a == {"id": "a", "track": T1, "kind": "rating", "stars": 4, "given_at": "2026-10-11T10:00:00Z",
                 "source": "needle", "queue": "ratings", "title": "Song One", "artists": ["A", "B"],
                 "album": "Album", "isrc": "USABC2600001", "duration": 201.5}
    assert (c["title"], c["artists"], c["album"]) == ("Cand Title", ["Cand Artist"], None)   # a candidate's names


def test_import_round_trip_is_idempotent(tmp_path):
    doc = verdicts.export_doc(_rated(tmp_path))
    con = db.connect(tmp_path / "b.db")
    assert verdicts.import_doc(con, doc) == {"verdicts": 3, "added": 3, "duplicate": 0, "tracks_added": 2}
    assert verdicts.current(con) == verdicts.current(_rated(tmp_path / "again"))
    assert [(h["id"], h["source"], h["effects"]) for h in verdicts.history(con, T1)] == [
        ("a", "needle", "skipped"), ("b", "needle", "skipped")]
    assert verdicts.pending(con) == []
    t = con.execute("SELECT name, artists_json, isrc, duration FROM spotify_tracks WHERE spotify_id=?", (T1,)).fetchone()
    assert tuple(t) == ("Song One", '["A", "B"]', "USABC2600001", 201.5)
    assert verdicts.import_doc(con, doc) == {"verdicts": 3, "added": 0, "duplicate": 3, "tracks_added": 0}
    assert verdicts.export_doc(con)["verdicts"] == doc["verdicts"]


def test_import_pending_leaves_the_spotify_writes_to_the_sync(tmp_path):
    con = db.connect(tmp_path / "b.db")
    verdicts.import_doc(con, verdicts.export_doc(_rated(tmp_path)), effects="pending")
    assert verdicts.pending(con) == sorted([T1, DISC_ID])


@pytest.mark.parametrize("doc,match", [
    ([], "not a haystack-verdicts file"),
    ({"format": "other", "version": 1, "verdicts": []}, "not a haystack-verdicts file"),
    ({"format": "haystack-verdicts", "version": 2, "verdicts": []}, "version 2"),
])
def test_import_refuses_another_format(tmp_path, doc, match):
    with pytest.raises(ValueError, match=match):
        verdicts.import_doc(db.connect(tmp_path / "b.db"), doc)


def test_import_is_all_or_nothing(tmp_path):
    doc = verdicts.export_doc(_rated(tmp_path))
    doc["verdicts"][1]["stars"] = 3                       # a Pass with stars
    con = db.connect(tmp_path / "b.db")
    with pytest.raises(ValueError, match=r"verdicts\[1\]: a pass has no stars"):
        verdicts.import_doc(con, doc)
    assert verdicts.current(con) == {}
    assert con.execute("SELECT COUNT(*) FROM spotify_tracks").fetchone()[0] == 0
    del doc["verdicts"][1]["given_at"]
    with pytest.raises(ValueError, match=r"verdicts\[1\]: an id and given_at are required"):
        verdicts.import_doc(con, doc)


def test_cli_export_import(tmp_path, monkeypatch, capsys):
    import json
    import os
    import stat
    from types import SimpleNamespace

    from haystack_music import cli
    _rated(tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "a.db")
    f = tmp_path / "verdicts.json"
    f.write_text("old")
    os.chmod(f, 0o644)
    assert cli.cmd_verdict(_cfg(), SimpleNamespace(action="export", track=str(f), pending=False)) == 0
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o600
    assert len(json.loads(f.read_text())["verdicts"]) == 3
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "b.db")
    capsys.readouterr()
    assert cli.cmd_verdict(_cfg(), SimpleNamespace(action="import", track=str(f), pending=False)) == 0
    assert json.loads(capsys.readouterr().out)["added"] == 3
    f.write_text("{not json")
    assert cli.cmd_verdict(_cfg(), SimpleNamespace(action="import", track=str(f), pending=False)) == 1
    assert "verdict import:" in capsys.readouterr().err
