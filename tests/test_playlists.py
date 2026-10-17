import json

import pytest

from haystack_music import config, db, playlists, review, verdicts
from haystack_music.spotify import SpotifyError


def sid(n: str) -> str:
    return n.ljust(22, "0")


def track(name: str, artist: str = "Rezz", isrc: str | None = None) -> dict:
    return {"id": sid(name), "name": name.title(), "type": "track", "duration_ms": 180000,
            "artists": [{"name": artist}], "external_ids": {"isrc": isrc} if isrc else {},
            "album": {"name": name.title(), "album_type": "single", "release_date": "2019-02-01",
                      "images": [{"url": f"https://i.scdn.co/image/{name}-640", "width": 640},
                                 {"url": f"https://i.scdn.co/image/{name}-64", "width": 64}]}}


def listing(pid, name, owner="me", snapshot="s1", total=2, images=None):
    return {"id": sid(pid), "name": name, "owner": {"id": owner, "display_name": owner.title()},
            "snapshot_id": snapshot, "items": {"total": total},
            "images": images if images is not None else [{"url": f"https://mosaic/{pid}-640", "width": 640},
                                                         {"url": f"https://mosaic/{pid}-300", "width": 300},
                                                         {"url": f"https://mosaic/{pid}-60", "width": 60}]}


class FakeSpotify:
    """The listener's library: two playlists of their own, one they follow (dev mode: 403), and the server's
    Discoveries."""

    def __init__(self):
        self.lists = [listing("disc", "Discoveries"), listing("bass", "Wide Mouth Bass"),
                      listing("phonk", "OG Drift Phonk", owner="sxulbxrn", total=3,
                              images=[{"url": "https://image-cdn/phonk", "width": None}]),
                      listing("dub", "All the Dubstep")]
        self.items = {sid("bass"): [track("belly", "Imanbek"), track("humble", "Skrillex", "USUM71710562"),
                                    track("belly", "Imanbek")],
                      sid("dub"): [track("scary", "Skrillex"), track("humble", "Skrillex", "USUM71710562")]}
        self.reads = []

    def me(self):
        return {"id": "me"}

    def my_playlists(self):
        yield from self.lists

    def playlist_tracks(self, pid):
        self.reads.append(pid)
        if pid not in self.items:
            raise SpotifyError(403, "Forbidden")
        for t in self.items[pid]:
            yield "2026-10-01T00:00:00Z", t


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.kv_set(c, "playlist_discoveries", sid("disc"))
    db.kv_set(c, "playlist_to_rip", sid("torip"))
    return c


@pytest.fixture
def cfg(tmp_path):
    return config.load(tmp_path / "none.toml")


def test_sync_lists_playlists_and_reads_the_ones_it_can(con, cfg):
    sp = FakeSpotify()
    out = playlists.sync(con, sp, cfg)
    assert out == {"playlists": 3, "removed": 0, "read": 2, "unchanged": 0, "for_engine": 1, "errors": 0}
    rows = {r["id"]: r for r in con.execute("SELECT * FROM playlists")}
    assert sid("disc") not in rows                                           # the server's own: not offered
    assert rows[sid("bass")]["mine"] == 1 and rows[sid("phonk")]["mine"] == 0
    assert rows[sid("bass")]["image_url"] == "https://mosaic/bass-300"       # the 300 px mosaic
    assert rows[sid("phonk")]["image_url"] == "https://image-cdn/phonk"      # a playlist's own picture
    assert rows[sid("phonk")]["refused_snapshot"] == "s1" and rows[sid("phonk")]["tracks_snapshot"] is None
    # songs in order, with their metadata for the cards
    got = [r[0] for r in con.execute("SELECT spotify_id FROM playlist_tracks WHERE playlist_id=? ORDER BY position",
                                     (sid("bass"),))]
    assert got == [sid("belly"), sid("humble"), sid("belly")]
    assert con.execute("SELECT image_url FROM spotify_tracks WHERE spotify_id=?", (sid("belly"),)).fetchone()[0] \
        == "https://i.scdn.co/image/belly-640"

    sp.reads.clear()                                                         # nothing changed: no reads at all
    assert playlists.sync(con, sp, cfg)["unchanged"] == 2 and sp.reads == []

    sp.lists[1]["snapshot_id"] = "s2"                                         # one playlist changed, one is gone
    sp.items[sid("bass")] = [track("goose", "HVME")]
    del sp.lists[3]
    out = playlists.sync(con, sp, cfg)
    assert (out["read"], out["removed"]) == (1, 1) and sp.reads == [sid("bass")]
    assert [r[0] for r in con.execute("SELECT spotify_id FROM playlist_tracks WHERE playlist_id=?", (sid("bass"),))] \
        == [sid("goose")]
    assert con.execute("SELECT removed_at FROM playlists WHERE id=?", (sid("dub"),)).fetchone()[0]


def test_a_failing_playlist_does_not_stop_the_others(con, cfg):
    sp = FakeSpotify()
    real = sp.playlist_tracks

    def flaky(pid):
        if pid == sid("bass"):
            raise SpotifyError(500, "boom")
        return real(pid)
    sp.playlist_tracks = flaky
    out = playlists.sync(con, sp, cfg)
    assert (out["read"], out["errors"], out["for_engine"]) == (1, 1, 1)


def engine_track(name, **kw):
    t = {"id": sid(name), "title": name.title(), "artists": ["SXULBXRN"], "album": name.title(),
         "album_type": "single", "release_date": "2022-04-01", "duration_ms": 137042,
         "cover": f"https://i.scdn.co/image/{name}", "cover_small": f"https://i.scdn.co/image/{name}-s",
         "isrc": "BC6M22604272"}
    t.update(kw)
    return t


def test_the_engine_reads_a_followed_playlist(con, cfg):
    playlists.sync(con, FakeSpotify(), cfg)
    before = {p["id"]: p for p in playlists.summaries(con)["items"]}[sid("phonk")]
    assert (before["reader"], before["current"], before["tracks"]) == ("engine", False, None)

    out = playlists.from_engine(con, sid("phonk"), {"snapshot_id": "s1", "tracks": [engine_track("resonate"),
                                                                                    engine_track("close")]})
    assert out == {"stored": 2, "snapshot_id": "s1"}
    after = {p["id"]: p for p in playlists.summaries(con)["items"]}[sid("phonk")]
    assert after["current"] is True and after["tracks"] == [sid("resonate"), sid("close")]
    r = con.execute("SELECT name, artists_json, isrc, image_url, image_url_small, release_date FROM spotify_tracks "
                    "WHERE spotify_id=?", (sid("resonate"),)).fetchone()
    assert tuple(r) == ("Resonate", '["SXULBXRN"]', "BC6M22604272", "https://i.scdn.co/image/resonate",
                        "https://i.scdn.co/image/resonate-s", "2022-04-01")
    # the playlist changes: the server tries the Web API once for the new version, then leaves it to the engine again
    sp = FakeSpotify()
    sp.lists[2]["snapshot_id"] = "s2"
    playlists.sync(con, sp, cfg)
    again = {p["id"]: p for p in playlists.summaries(con)["items"]}[sid("phonk")]
    assert (again["reader"], again["current"], len(again["tracks"])) == ("engine", False, 2)


@pytest.mark.parametrize("bad, why", [
    ({"tracks": []}, "snapshot_id"),
    ({"snapshot_id": "s1", "tracks": "x"}, "tracks"),
    ({"snapshot_id": "s1", "tracks": [engine_track("a", id="short")]}, "track 0: id"),
    ({"snapshot_id": "s1", "tracks": [engine_track("a", artists="Rezz")]}, "artists"),
    ({"snapshot_id": "s1", "tracks": [engine_track("a", cover="http://evil/x")]}, "https"),
    ({"snapshot_id": "s1", "tracks": [engine_track("a", release_date="soon")]}, "release_date"),
    ({"snapshot_id": "s1", "tracks": [engine_track("a", duration_ms=True)]}, "duration_ms"),
    ({"snapshot_id": "s1", "tracks": [engine_track("a", title="")]}, "title"),
])
def test_engine_songs_are_validated_and_stored_all_or_nothing(con, cfg, bad, why):
    playlists.sync(con, FakeSpotify(), cfg)
    body = json.loads(json.dumps(bad))
    if isinstance(body.get("tracks"), list):
        body["tracks"].insert(0, engine_track("ok"))
        why = why.replace("track 0", "track 1")
    with pytest.raises(ValueError, match=why):
        playlists.from_engine(con, sid("phonk"), body)
    assert con.execute("SELECT COUNT(*) FROM playlist_tracks WHERE playlist_id=?", (sid("phonk"),)).fetchone()[0] == 0
    with pytest.raises(LookupError):
        playlists.from_engine(con, sid("unknown"), {"snapshot_id": "s1", "tracks": []})


def test_summaries_carry_ids_verdicts_and_top_artists(con, cfg):
    playlists.sync(con, FakeSpotify(), cfg)
    verdicts.record(con, sid("humble"), "rating", 4)
    verdicts.record(con, sid("nowhere"), "pass")                    # not on any playlist: not sent
    s = playlists.summaries(con)
    assert s["synced_at"].endswith("Z")
    bass, phonk, dub = s["items"]                                    # Spotify's order
    assert bass["name"] == "Wide Mouth Bass" and bass["mine"] and not phonk["mine"] and phonk["owner"] == "Sxulbxrn"
    assert bass["tracks"] == [sid("belly"), sid("humble")]           # Belly Dancer is listed twice, plays once
    assert bass["artists"] == ["Imanbek", "Skrillex"] and dub["artists"] == ["Skrillex"]
    assert bass["total"] == 2 and bass["current"] and bass["reader"] == "server"
    assert set(s["verdicts"]) == {sid("humble")} and s["verdicts"][sid("humble")]["stars"] == 4


def test_cards_say_where_else_a_song_is_and_whether_it_is_in_the_library(con, cfg):
    playlists.sync(con, FakeSpotify(), cfg)
    con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid("scary"),))
    con.execute("INSERT INTO local_tracks(path, size, content_hash, isrc, present) VALUES('/m/h.m4a', 1, 'h', "
                "'USUM71710562', 1)")
    out = playlists.cards(con, review.Cards(con, cfg, classes=[]), sid("dub"))
    assert out["name"] == "All the Dubstep" and [c["id"] for c in out["items"]] == [sid("scary"), sid("humble")]
    scary, humble = out["items"]
    assert scary["title"] == "Scary" and scary["cover"] == "https://i.scdn.co/image/scary-640"
    assert scary["playlists"] == [sid("dub")] and humble["playlists"] == [sid("bass"), sid("dub")]
    assert (scary["liked"], scary["in_library"]) == (True, False)
    assert (humble["liked"], humble["in_library"]) == (False, True)   # matched by ISRC
    assert playlists.cards(con, review.Cards(con, cfg, classes=[]), sid("nope")) is None


def test_cover_url_prefers_the_300px_mosaic():
    assert playlists.cover_url(None) is None
    assert playlists.cover_url([{"url": "a", "width": None}]) == "a"
    assert playlists.cover_url([{"url": "b", "width": 640}, {"url": "c", "width": 300}, {"url": "d", "width": 60}]) == "c"
