"""Possible covers and the Swap (ADR 0022): the signs, the stored checks and the Spotify writes of a Swap."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from haystack_music import db, originals, playlists, verdicts


def sid(name: str) -> str:
    return name.ljust(22, "0")


CFG = {"spotify": {"to_rip": True}, "matching": {"duration_tolerance": 3.0}}
LOCAL = "local file"            # a local file in a playlist: Spotify counts its place, haystack doesn't store it


def _con(tmp_path):
    return db.connect(tmp_path / "t.db")


def _track(con, name, title, artists, isrc=None):
    con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, isrc) VALUES(?,?,?,?)",
                (sid(name), title, json.dumps(artists), isrc))


def dz_track(title, artist, rank, contributors=()):
    return {"id": abs(hash((title, artist))) % 10**8 + 1, "title": title, "rank": rank, "artist": {"name": artist},
            "contributors": [{"name": c} for c in contributors]}


class FakeDeezer:
    def __init__(self, isrc=None, search=None):
        self.isrc, self.hits, self.calls = isrc or {}, search or {}, []

    def track_by_isrc(self, isrc):
        self.calls.append(("isrc", isrc))
        return self.isrc.get(isrc)

    def search(self, q, limit=25, order=None):
        self.calls.append(("search", q, order))
        return self.hits.get(q, [])


def sp_track(name, title, artists):
    return {"id": sid(name), "name": title, "artists": [{"name": a} for a in artists], "duration_ms": 200000,
            "album": {"name": "x", "album_type": "album", "release_date": "1994", "images": []}}


class FakeSpotify:
    """Playlists as ordered lists, Liked Songs as a set, and canned search results."""

    def __init__(self, lists=None, liked=(), search=None, tracks=()):
        self.lists = {k: list(v) for k, v in (lists or {}).items()}
        self.liked, self.results, self.calls, self.snap = set(liked), search or {}, [], 0
        self.tracks = {t["id"]: t for t in tracks}
        self.fail_remove = False

    def _bump(self):
        self.snap += 1
        return f"snap{self.snap}"

    def search_tracks(self, q, limit=10):
        self.calls.append(("search", q))
        return self.results.get(q, [])

    def track(self, track_id):
        return self.tracks.get(track_id) or sp_track(track_id.rstrip("0"), "Unknown", ["Nobody"])

    def playlist(self, pid):
        return {"id": pid, "snapshot_id": f"snap{self.snap}"}

    def playlist_tracks(self, pid):
        return [(i, "2026-01-01T00:00:00Z", self.track(t)) for i, t in enumerate(self.lists[pid]) if t != LOCAL]

    def add_items(self, pid, ids, position=None):
        self.calls.append(("add", pid, tuple(ids), position))
        at = len(self.lists[pid]) if position is None else position
        self.lists[pid][at:at] = ids
        return self._bump()

    def remove_items(self, pid, ids, snapshot_id=None):
        self.calls.append(("pl_remove", pid, tuple(ids)))
        if self.fail_remove:
            raise RuntimeError("Spotify said no")
        self.lists[pid] = [t for t in self.lists[pid] if t not in ids]
        return self._bump()

    def library_contains(self, ids):
        return {i: i in self.liked for i in ids}

    def library_save(self, ids):
        self.calls.append(("save", tuple(ids)))
        self.liked |= set(ids)

    def library_remove(self, ids):
        self.calls.append(("unsave", tuple(ids)))
        self.liked -= set(ids)


# ---------------- the signs ----------------

def test_a_title_that_names_the_original_artist_needs_no_lookup():
    dz = FakeDeezer()
    s = originals.signs("Zombie (Originally Performed by The Cranberries) [Karaoke Version]", ["Sing King"], "X1", dz)
    assert (s["reason"], s["title"], s["artist"]) == ("marker", "Zombie", "The Cranberries")
    assert dz.calls == []


def test_a_karaoke_track_without_a_name_takes_the_best_known_recording_as_its_original():
    dz = FakeDeezer(search={'track:"Zombie"': [dz_track("Zombie (Karaoke Version)", "Karaoke Hits", 9000),
                                                dz_track("Zombie", "The Cranberries", 950000),
                                                dz_track("Zombie", "Bad Wolves", 700000)]})
    s = originals.signs("Zombie (Karaoke Version)", ["Karaoke Hits"], None, dz)
    assert (s["reason"], s["artist"], s["other_rank"]) == ("marker", "The Cranberries", 950000)
    assert dz.calls == [("search", 'track:"Zombie"', "RANKING")]


def test_an_artist_called_tribute_or_karaoke_is_a_marker():
    s = originals.signs("Everlong", ["Foo Fighters Tribute Band"], None,
                        FakeDeezer(search={"Everlong": [dz_track("Everlong", "Foo Fighters", 800000)]}))
    assert (s["reason"], s["artist"]) == ("marker", "Foo Fighters")     # the plain-text search, after track:"…"


def test_a_live_recording_points_to_the_same_artists_studio_one():
    for title in ("Zombie - Live at Rockpalast", "Zombie [Live]", "Linger - Unplugged", "Dreams - BBC Session"):
        s = originals.signs(title, ["The Cranberries"], "X1", FakeDeezer())
        assert (s["reason"], s["artist"]) == ("live", "The Cranberries"), title
    assert originals.signs("Live Forever", ["Oasis"], None, FakeDeezer())["reason"] is None


def test_a_borrowed_isrc_names_the_original():
    dz = FakeDeezer(isrc={"USRW29600011": dz_track("Everlong", "Foo Fighters", 900000)})
    s = originals.signs("Everlong", ["Rock Covers United"], "USRW29600011", dz)
    assert (s["reason"], s["artist"], s["own_rank"]) == ("isrc", "Foo Fighters", 900000)


def test_an_isrc_under_a_featured_artist_is_the_tracks_own():
    dz = FakeDeezer(isrc={"I1": dz_track("Stay", "Rihanna", 800000, contributors=["Rihanna", "Mikky Ekko"])},
                    search={'track:"Stay"': [dz_track("Stay", "Rihanna", 800000)]})
    assert originals.signs("Stay", ["Mikky Ekko"], "I1", dz)["reason"] is None


def test_a_far_better_known_recording_by_another_artist_is_a_sign_but_a_famous_cover_is_not():
    hits = {'track:"Hallelujah"': [dz_track("Hallelujah", "Jeff Buckley", 900000),
                                   dz_track("Hallelujah", "Leonard Cohen", 700000)]}
    cover = FakeDeezer(isrc={"C1": dz_track("Hallelujah", "Pop Covers", 40000)}, search=hits)
    s = originals.signs("Hallelujah", ["Pop Covers"], "C1", cover)
    assert (s["reason"], s["artist"], s["own_rank"], s["other_rank"]) == ("rank", "Jeff Buckley", 40000, 900000)
    famous = FakeDeezer(isrc={"B1": dz_track("Hallelujah", "Jeff Buckley", 900000)}, search=hits)
    s = originals.signs("Hallelujah", ["Jeff Buckley"], "B1", famous)
    assert s["reason"] is None and s["other_rank"] == 700000        # Cohen's isn't 3x Buckley's


def test_deezers_placeholder_rank_and_an_unknown_own_rank_are_no_sign():
    hits = {'track:"Song"': [dz_track("Song", "Someone Else", originals.RANK_PLACEHOLDER)]}
    assert originals.signs("Song", ["Me"], "I1", FakeDeezer(isrc={"I1": dz_track("Song", "Me", 10)},
                                                             search=hits))["reason"] is None
    hits = {'track:"Song"': [dz_track("Song", "Someone Else", 900000)]}
    assert originals.signs("Song", ["Me"], None, FakeDeezer(search=hits))["reason"] is None


def test_a_title_with_cover_in_it_is_not_a_marker():
    dz = FakeDeezer(isrc={"I1": dz_track("Cover Me", "Bruce Springsteen", 500000)},
                    search={'track:"Cover Me"': [dz_track("Cover Me", "Bruce Springsteen", 500000)]})
    assert originals.signs("Cover Me", ["Bruce Springsteen"], "I1", dz)["reason"] is None


# ---------------- the check ----------------

ZOMBIE_RESULTS = {'track:"Zombie" artist:"The Cranberries"': [
    sp_track("cover", "Zombie (Karaoke Version)", ["Karaoke Hits"]),
    sp_track("zlive", "Zombie - Live", ["The Cranberries"]),
    sp_track("zacoustic", "Zombie - Acoustic Version", ["The Cranberries"]),
    sp_track("zombie", "Zombie", ["The Cranberries"])]}


def test_the_original_is_the_same_artists_plain_recording(tmp_path):
    con = _con(tmp_path)
    sp = FakeSpotify(search=ZOMBIE_RESULTS)
    orig = originals.find_original(sp, "Zombie", "The Cranberries", sid("cover"))
    assert orig["id"] == sid("zombie")
    assert originals.find_original(FakeSpotify(), "Zombie", "The Cranberries", sid("cover")) is None
    assert sp.calls[0] == ("search", 'track:"Zombie" artist:"The Cranberries"')
    con.close()


def test_a_check_is_stored_once_and_shows_on_the_card(tmp_path):
    con = _con(tmp_path)
    _track(con, "cover", "Zombie (Originally Performed by The Cranberries) [Karaoke Version]", ["Sing King"])
    _track(con, "plain", "Linger", ["The Cranberries"], "IE1")
    sp = FakeSpotify(search=ZOMBIE_RESULTS)
    dz = FakeDeezer(isrc={"IE1": dz_track("Linger", "The Cranberries", 800000)})
    row = originals.check(con, sid("cover"), dz, sp)
    assert (row["reason"], row["of_title"], row["of_artist"], row["of_spotify_id"]) == \
        ("marker", "Zombie", "The Cranberries", sid("zombie"))
    assert con.execute("SELECT name FROM spotify_tracks WHERE spotify_id=?", (sid("zombie"),)).fetchone()[0] == "Zombie"
    assert originals.check(con, sid("cover"), dz, sp) is None                     # checked already
    assert originals.check(con, sid("plain"), dz, sp)["reason"] is None
    f = originals.card_fields(con, [sid("cover"), sid("plain"), sid("new"), sid("newer")], {sid("newer")})
    assert f[sid("cover")] == {"original": {"title": "Zombie", "artist": "The Cranberries", "id": sid("zombie"),
                                            "reason": "marker"}, "original_check": "done"}
    assert f[sid("plain")] == {"original": None, "original_check": "done"}
    assert f[sid("new")] == {"original": None, "original_check": "unknown"}
    assert f[sid("newer")] == {"original": None, "original_check": "pending"}
    assert originals.due(con, [sid("cover"), sid("plain"), sid("new")]) == [sid("new")]


def test_a_failed_check_is_tried_again_a_day_later(tmp_path):
    con = _con(tmp_path)
    _track(con, "song", "Song", ["Me"], "I1")

    class Down(FakeDeezer):
        def track_by_isrc(self, isrc):
            raise ConnectionError("Deezer is down")

    row = originals.check(con, sid("song"), Down(), FakeSpotify())
    assert "Deezer is down" in row["error"] and row["reason"] is None
    assert originals.card_fields(con, [sid("song")])[sid("song")] == {"original": None, "original_check": "failed"}
    assert originals.card_fields(con, [sid("song")], {sid("song")})[sid("song")]["original_check"] == "pending"
    assert originals.due(con, [sid("song")]) == []
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).strftime("%Y-%m-%dT%H:%M:%SZ")
    con.execute("UPDATE original_checks SET checked_at=?", (old,))
    assert originals.due(con, [sid("song")]) == [sid("song")]
    assert originals.card_fields(con, [sid("song")])[sid("song")]["original_check"] == "unknown"


def test_a_track_not_stored_yet_is_fetched_first(tmp_path):
    con = _con(tmp_path)
    sp = FakeSpotify(tracks=[sp_track("live", "Zombie - Live", ["The Cranberries"])],
                     search={'track:"Zombie" artist:"The Cranberries"': [sp_track("zombie", "Zombie",
                                                                                  ["The Cranberries"])]})
    row = originals.check(con, sid("live"), FakeDeezer(), sp)
    assert (row["reason"], row["of_spotify_id"]) == ("live", sid("zombie"))


# ---------------- the Swap ----------------

def test_a_swap_takes_back_the_covers_verdicts_and_drops_late_ones(tmp_path):
    con = _con(tmp_path)
    verdicts.record(con, sid("cover"), "rating", 4, given_at="2026-10-23T08:00:00Z", id="v-before")
    verdicts.record(con, sid("other"), "rating", 5, given_at="2026-10-23T08:00:00Z", id="v-other")
    r = originals.record_swap(con, "swap-0001", sid("cover"), sid("orig"), "2026-10-23T09:00:00Z")
    assert r == {"id": "swap-0001", "status": "accepted", "retracted": 1}
    assert originals.record_swap(con, "swap-0001", sid("cover"), sid("orig"), "2026-10-23T09:00:00Z")["status"] == \
        "duplicate"
    assert sid("cover") not in verdicts.current(con) and sid("other") in verdicts.current(con)
    kept = json.loads(con.execute("SELECT row FROM swap_retracted WHERE verdict_id='v-before'").fetchone()[0])
    assert kept["stars"] == 4 and kept["spotify_id"] == sid("cover")
    assert originals.swallowed(con, "v-before", sid("cover"), "2026-10-23T08:00:00Z")        # sent again
    assert originals.swallowed(con, "v-late", sid("cover"), "2026-10-23T08:30:00Z")          # given before the Swap
    assert not originals.swallowed(con, "v-after", sid("cover"), "2026-10-23T10:00:00Z")     # a new mind
    assert not originals.swallowed(con, "v-orig", sid("orig"), "2026-10-23T08:30:00Z")
    assert originals.pending(con) == ["swap-0001"] and originals.swapping(con) == {sid("cover")}


def _swap_world(tmp_path, **sp_kw):
    """Rock (the listener's) holds the cover twice; Mix (the listener's) holds it and the original's other release;
    Theirs is followed. The cover is liked and haystack put it on To Rip."""
    con = _con(tmp_path)
    _track(con, "cover", "Zombie (Karaoke Version)", ["Karaoke Hits"], "KAR1")
    _track(con, "orig", "Zombie", ["The Cranberries"], "IE1")
    _track(con, "origre", "Zombie - Remastered", ["The Cranberries"], "IE1")      # the same Song
    for t in ("a", "b", "c"):
        _track(con, t, t.upper(), ["Someone"])
    for pos, (pid, mine) in enumerate((("rock", 1), ("mix", 1), ("theirs", 0))):
        con.execute("INSERT INTO playlists(id, name, owner_id, mine, total, snapshot_id, position) VALUES(?,?,?,?,?,?,?)",
                    (sid(pid), pid, "me" if mine else "them", mine, 3, "snap0", pos))
    lists = {sid("rock"): [sid("a"), sid("cover"), sid("b"), sid("cover")],
             sid("mix"): [sid("c"), sid("origre"), sid("cover")],
             sid("theirs"): [sid("cover")]}
    for pid, ids in lists.items():
        playlists.store(con, pid, "snap0", [(t, "2026-01-01T00:00:00Z") for t in ids], "web")
    con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid("cover"),))
    con.execute("INSERT INTO playlist_adds(playlist, playlist_id, spotify_id, added_at) VALUES('to_rip', ?, ?, 'x')",
                (sid("torip"), sid("cover")))
    con.execute("INSERT INTO to_rip(spotify_id, added_at) VALUES(?, 'x')", (sid("cover"),))
    lists[sid("torip")] = [sid("cover")]
    sp = FakeSpotify(lists=lists, liked={sid("cover")}, **sp_kw)
    originals.record_swap(con, "swap-0001", sid("cover"), sid("orig"), "2026-10-23T09:00:00Z")
    return con, sp


def _stored(con, pid):
    return [r[0] for r in con.execute("SELECT spotify_id FROM playlist_tracks WHERE playlist_id=? ORDER BY position",
                                      (sid(pid),))]


def test_a_swap_puts_the_original_in_the_covers_place_and_moves_the_heart(tmp_path):
    con, sp = _swap_world(tmp_path)
    out = originals.apply_swaps(con, CFG, sp, library=[])
    assert out == {"swaps": 1, "applied": 1, "pending": 0}
    assert sp.lists[sid("rock")] == [sid("a"), sid("orig"), sid("b")]                    # in the cover's first place
    assert ("add", sid("rock"), (sid("orig"),), 1) in sp.calls
    assert sp.lists[sid("mix")] == [sid("c"), sid("origre")]                            # its Song is there already
    assert sp.lists[sid("theirs")] == [sid("cover")] and sp.lists[sid("torip")] == [sid("orig")]
    assert _stored(con, "rock") == sp.lists[sid("rock")] and _stored(con, "mix") == sp.lists[sid("mix")]
    assert sp.liked == {sid("orig")}
    assert sp.calls.index(("save", (sid("orig"),))) < sp.calls.index(("unsave", (sid("cover"),)))
    assert con.execute("SELECT removed_at FROM likes WHERE spotify_id=?", (sid("orig"),)).fetchone()[0] is None
    assert con.execute("SELECT removed_at FROM likes WHERE spotify_id=?", (sid("cover"),)).fetchone()[0]
    assert con.execute("SELECT removed_at FROM to_rip WHERE spotify_id=?", (sid("cover"),)).fetchone()[0]
    assert con.execute("SELECT removed_at FROM to_rip WHERE spotify_id=?", (sid("orig"),)).fetchone()[0] is None
    assert verdicts._bot_on(con, "to_rip", sid("orig")) == sid("torip")
    res = json.loads(con.execute("SELECT result FROM swaps").fetchone()[0])
    assert res == {"playlists": [sid("rock")], "already_there": [sid("mix")], "followed": [sid("theirs")],
                   "heart": "moved", "to_rip": "moved"}
    assert originals.pending(con) == [] and originals.apply_swaps(con, CFG, sp)["swaps"] == 0


def test_a_swap_counts_the_local_files_spotify_has_before_the_cover(tmp_path):
    con, sp = _swap_world(tmp_path)
    sp.lists[sid("rock")] = [LOCAL, sid("a"), sid("cover"), LOCAL, sid("b"), sid("cover")]
    originals.apply_swaps(con, CFG, sp, library=[])
    assert sp.lists[sid("rock")] == [LOCAL, sid("a"), sid("orig"), LOCAL, sid("b")]
    assert ("add", sid("rock"), (sid("orig"),), 2) in sp.calls
    assert [tuple(r) for r in con.execute("SELECT spotify_id, position FROM playlist_tracks WHERE playlist_id=? "
                                          "ORDER BY position", (sid("rock"),))] == \
        [(sid("a"), 1), (sid("orig"), 2), (sid("b"), 4)]                      # as a fresh read would store them
    playlists.read_web(con, sp, sid("rock"), "again")
    assert [tuple(r) for r in con.execute("SELECT spotify_id, position FROM playlist_tracks WHERE playlist_id=? "
                                          "ORDER BY position", (sid("rock"),))] == \
        [(sid("a"), 1), (sid("orig"), 2), (sid("b"), 4)]


def test_a_swap_leaves_an_original_off_to_rip_that_is_in_the_library(tmp_path):
    con, sp = _swap_world(tmp_path)
    con.execute("INSERT INTO local_tracks(path, size, content_hash, spotify_id) VALUES('/m/zombie.m4a', 1, 'h', ?)",
                (sid("orig"),))
    originals.apply_swaps(con, CFG, sp, library=[])
    assert sp.lists[sid("torip")] == []
    assert json.loads(con.execute("SELECT result FROM swaps").fetchone()[0])["to_rip"] == "removed"


def test_a_swap_leaves_an_original_unhearted_that_the_listener_rated_low(tmp_path):
    con, sp = _swap_world(tmp_path)
    verdicts.record(con, sid("origre"), "rating", 2)                      # on another release of its Song
    originals.apply_swaps(con, CFG, sp, library=[])
    assert sp.liked == set() and sp.lists[sid("torip")] == []
    res = json.loads(con.execute("SELECT result FROM swaps").fetchone()[0])
    assert (res["heart"], res["to_rip"]) == ("removed", "removed")


def test_a_swap_that_failed_half_way_is_made_again_without_a_second_copy(tmp_path):
    con, sp = _swap_world(tmp_path)
    sp.fail_remove = True
    out = originals.apply_swaps(con, CFG, sp, library=[])
    assert out["pending"] == 1 and "Spotify said no" in originals.pending_writes(con)[1]
    assert sp.lists[sid("rock")] == [sid("a"), sid("orig"), sid("cover"), sid("b"), sid("cover")]
    sp.fail_remove = False                                                # the add changed the playlist: re-read
    assert originals.apply_swaps(con, CFG, sp, library=[])["applied"] == 1
    assert sp.lists[sid("rock")] == [sid("a"), sid("orig"), sid("b")] and sp.lists[sid("torip")] == [sid("orig")]
    assert _stored(con, "rock") == sp.lists[sid("rock")]
    assert originals.pending_writes(con) == (0, None)


def test_a_swap_cancels_the_covers_verdict_writes_still_to_make(tmp_path):
    con = _con(tmp_path)
    verdicts.record(con, sid("cover"), "rating", 5, given_at="2026-10-23T08:00:00Z")      # would heart it
    assert verdicts.pending(con) == [sid("cover")]
    originals.record_swap(con, "swap-0001", sid("cover"), sid("orig"), "2026-10-23T09:00:00Z")
    assert verdicts.pending(con) == []


@pytest.mark.parametrize("body,error", [
    ({"id": "s", "track": sid("a"), "original": sid("b"), "given_at": "2026-10-23T09:00:00Z"}, "id must be"),
    ({"id": "swap-0002", "track": sid("a"), "original": sid("b"), "given_at": "today"}, "given_at"),
    ({"id": "swap-0002", "track": "short", "original": sid("b"), "given_at": "2026-10-23T09:00:00Z"}, "track must"),
    ({"id": "swap-0002", "track": sid("a"), "given_at": "2026-10-23T09:00:00Z"}, "original must"),
    ({"id": "swap-0002", "track": sid("a"), "original": sid("a"), "given_at": "2026-10-23T09:00:00Z"}, "another"),
])
def test_a_malformed_swap_is_refused(body, error):
    from haystack_music import api
    with pytest.raises(ValueError, match=error):
        api.parse_swap(body)
