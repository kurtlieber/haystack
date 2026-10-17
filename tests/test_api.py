import http.client
import ipaddress
import json
import threading

import numpy as np
import pytest

from haystack_music import api, config, db, playlists, review, traits, vectors, verdicts

TOKEN = "t" * 40


def sid(n: str) -> str:
    return n.ljust(22, "0")


class FakeWorker:
    def __init__(self):
        self.measuring, self.effects_calls, self.measured, self.playlist_syncs = set(), 0, [], []

    def effects(self):
        self.effects_calls += 1

    def playlists(self, every=0.0):
        self.playlist_syncs.append(every)

    def measure(self, s):
        self.measured.append(s)


@pytest.fixture
def world(tmp_path):
    """A batch of two offers (one rated already), two liked songs (one rated), Readings for some."""
    path = tmp_path / "t.db"
    con = db.connect(path)
    con.execute("INSERT INTO batches(id, created_at, size) VALUES(1, '2026-10-05', 2)")
    con.execute("INSERT INTO local_tracks(id, path, size, content_hash, title_raw, artists_json) "
                "VALUES(967, '/m/x.m4a', 1, 'h', 'Cradles', '[\"Sub Urban\"]')")
    for cid, name, title, artist, explore in ((1, "rezz", "Flying Octopus", "Rezz", 1), (2, "blind", "Right Behind", "Blinders", 0)):
        con.execute("INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status, spotify_id, "
                    "nearest, raw_cos) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (cid, cid, artist, title, "deezer_artist_radio", "offered", "ok", sid(name), "local:967", 0.962))
        con.execute("INSERT INTO offers(candidate_id, spotify_id, batch_id, exploration, added_at) VALUES(?,?,1,?,'x')",
                    (cid, sid(name), explore))
        con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, album, album_type, release_date, "
                    "duration, image_url) VALUES(?,?,?,?,?,?,?,?)",
                    (sid(name), title, json.dumps([artist]), title, "single", "2018-05-04", 179.0, "https://i/c.jpg"))
        vectors.put_deezer_vector(con, cid, np.ones(512), np.zeros(1280), 10, 30.0, np.eye(400)[cid * 7])
    traits.put(con, 1, {"tempo": 102.0, "valence": 0.14})
    for name in ("like1", "like2"):
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(name),))
    verdicts.record(con, sid("blind"), "rating", 4)
    verdicts.record(con, sid("like2"), "pass")
    cfg = config.load(tmp_path / "none.toml")
    worker = FakeWorker()
    srv = api.Server(("127.0.0.1", 0), cfg, TOKEN, worker, db_path=path,
                     classes=[f"Electronic---Style{i}" for i in range(400)])
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, worker, con
    srv.shutdown()


def call(srv, method, path, body=None, token=TOKEN):
    c = http.client.HTTPConnection(*srv.server_address, timeout=5)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    c.request(method, path, body=data, headers=headers)
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"null")


def test_allowed_networks():
    nets = [ipaddress.ip_network(n) for n in ("192.168.1.0/24", "10.20.1.0/24", "10.20.0.0/24")]
    assert api.allowed("127.0.0.1", nets) and api.allowed("::1", nets) and api.allowed("::ffff:10.20.0.4", nets)
    assert api.allowed("192.168.1.108", nets) and api.allowed("10.20.0.4", nets)
    assert not api.allowed("203.0.113.9", nets) and not api.allowed("10.20.2.1", nets)
    assert not api.allowed("2001:db8::1", nets) and not api.allowed("nonsense", nets)


def test_short_tokens_are_refused(tmp_path):
    with pytest.raises(ValueError):
        api.Server(("127.0.0.1", 0), config.load(tmp_path / "none.toml"), "short")


def test_requests_need_the_token(world):
    srv, _, _ = world
    assert call(srv, "GET", "/v1/health", token=None)[0] == 401
    assert call(srv, "GET", "/v1/health", token="x" * 40)[0] == 401
    code, body = call(srv, "GET", "/v1/health")
    assert code == 200 and body["ok"] is True


def test_state_has_both_queues_with_cards(world):
    srv, worker, _ = world
    code, st = call(srv, "GET", "/v1/state")
    assert code == 200
    d = st["queues"]["discoveries"]
    assert (d["total"], d["done"]) == (2, 1) and d["next_batch_at"].endswith("Z")
    first, last = d["items"]
    assert first["id"] == sid("rezz") and first["verdict"] is None              # unreviewed first
    assert last["verdict"] == {"kind": "rating", "stars": 4, "given_at": last["verdict"]["given_at"]}
    assert first["title"] == "Flying Octopus" and first["artists"] == ["Rezz"] and first["year"] == 2018
    assert first["album_type"] == "single" and first["duration_ms"] == 179000 and first["cover"] == "https://i/c.jpg"
    assert first["why"] == {"sounds_like": {"title": "Cradles", "artist": "Sub Urban"}, "similarity": 0.96,
                            "source": "deezer_artist_radio", "source_label": "Deezer artist radio", "exploration": True}
    assert first["styles"][0] == ["Style7", 1.0] and len(first["styles"]) == 3
    assert first["readings"] == {"tempo": 102.0, "valence": 0.14} and first["measure"] == "ok"
    assert last["measure"] == "unknown"
    likes = st["queues"]["likes"]
    assert likes["total"] == 1 and [c["id"] for c in likes["items"]] == [sid("like1")]   # the Pass is reviewed
    assert set(st["baseline"]) == {"tracks", "traits"}
    assert st["spotify"]["pending_writes"] == 2
    assert sid("blind") in worker.measured and sid("like1") in worker.measured     # no Readings yet: measure


def test_verdicts_are_idempotent_and_validated(world):
    srv, worker, con = world
    good = {"id": "0b9a2c7e-1", "track": sid("rezz"), "kind": "rating", "stars": 5, "given_at": "2026-10-11T07:00:00Z",
            "queue": "discoveries"}
    bad = [{"id": "x1234567", "track": sid("rezz"), "kind": "rating", "stars": 9, "given_at": "2026-10-11T07:00:00Z"},
           {"id": "x1234568", "track": "short", "kind": "pass", "given_at": "2026-10-11T07:00:00Z"},
           {"id": "x1234569", "track": sid("rezz"), "kind": "pass", "given_at": "yesterday"},
           "not an object"]
    code, body = call(srv, "POST", "/v1/verdicts", {"verdicts": [good, *bad]})
    assert code == 200 and body["accepted"] == ["0b9a2c7e-1"] and len(body["rejected"]) == 4
    code, body = call(srv, "POST", "/v1/verdicts", {"verdicts": [good]})          # outbox replay
    assert body["accepted"] == ["0b9a2c7e-1"]
    assert len(verdicts.history(con, sid("rezz"))) == 1
    assert verdicts.current(con)[sid("rezz")].stars == 5
    assert worker.effects_calls == 2
    # the offer closed at once: Discoveries is now fully reviewed
    st = call(srv, "GET", "/v1/state")[1]["queues"]["discoveries"]
    assert (st["total"], st["done"]) == (2, 2)


def test_bad_bodies(world):
    srv, _, _ = world
    assert call(srv, "POST", "/v1/verdicts", b"{not json")[0] == 400
    assert call(srv, "POST", "/v1/verdicts", {"verdict": []})[0] == 400
    assert call(srv, "GET", "/v1/nope")[0] == 404


def test_plays_are_recorded_once(world):
    srv, _, con = world
    p = {"id": "p-1", "track": sid("rezz"), "started_at": "2026-10-11T07:00:00Z", "ms_played": 31000, "completed": False}
    for _ in range(2):
        code, body = call(srv, "POST", "/v1/plays", {"plays": [p, {"id": "p-2", "track": "bad"}]})
        assert code == 200 and body["accepted"] == ["p-1"] and len(body["rejected"]) == 1
    assert con.execute("SELECT COUNT(*), context FROM plays").fetchone()[:] == (1, "needle")


def test_unknown_track_is_measured_on_demand(world):
    srv, worker, _ = world
    code, card = call(srv, "GET", f"/v1/tracks/{sid('newone')}")
    assert code == 202 and card["measure"] == "pending" and worker.measured[-1] == sid("newone")
    code, card = call(srv, "GET", f"/v1/tracks/{sid('rezz')}")
    assert code == 200 and card["why"]["exploration"] is True


def test_next_batch_fallback_is_the_coming_monday_morning():
    from datetime import datetime, timezone
    nxt = review.next_batch_at(datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc))
    assert nxt.endswith("Z")


def test_verdict_writes_jump_the_measuring_queue(tmp_path):
    w = api.Worker(config.load(tmp_path / "none.toml"))
    for i in range(5):
        w.measure(sid(f"m{i}"))
    w.measure(sid("m0"))                                  # already queued: not twice
    w.effects()
    assert w.q.qsize() == 6
    assert w.q.get()[2] == "effects"
    assert [w.q.get()[3] for _ in range(5)] == [sid(f"m{i}") for i in range(5)]


def add_playlists(con):
    """Wide Mouth Bass (the listener's own, read) holds both discoveries; OG Drift Phonk (followed) is not read yet."""
    for pos, (pid, name, mine) in enumerate(((sid("bass"), "Wide Mouth Bass", 1), (sid("phonk"), "OG Drift Phonk", 0))):
        con.execute("INSERT INTO playlists(id, name, owner_id, mine, total, snapshot_id, position, refused_snapshot) "
                    "VALUES(?,?,?,?,?,?,?,?)", (pid, name, "me" if mine else "someone", mine, 2, "s1", pos,
                                                None if mine else "s1"))
    playlists.store(con, sid("bass"), "s1", [(sid("rezz"), None), (sid("blind"), None)], "web")


def test_state_lists_playlists_with_ids_and_verdicts(world):
    srv, worker, con = world
    add_playlists(con)
    st = call(srv, "GET", "/v1/state")[1]["playlists"]
    bass, phonk = st["items"]
    assert bass["tracks"] == [sid("rezz"), sid("blind")] and bass["current"] and bass["reader"] == "server"
    assert phonk["tracks"] is None and phonk["reader"] == "engine"
    assert st["verdicts"][sid("blind")]["stars"] == 4 and sid("rezz") not in st["verdicts"]
    assert worker.playlist_syncs == [api.PLAYLISTS_EVERY]


def test_a_playlist_comes_with_full_cards(world):
    srv, _, con = world
    add_playlists(con)
    code, pl = call(srv, "GET", f"/v1/playlists/{sid('bass')}")
    assert code == 200 and pl["name"] == "Wide Mouth Bass"
    first, second = pl["items"]
    assert first["title"] == "Flying Octopus" and first["readings"]["tempo"] == 102.0
    assert first["playlists"] == [sid("bass")] and first["liked"] is False
    assert second["verdict"]["stars"] == 4
    assert call(srv, "GET", f"/v1/playlists/{sid('nope')}")[0] == 404


def test_the_engine_sends_a_followed_playlists_songs(world):
    srv, _, con = world
    add_playlists(con)
    t = {"id": sid("resonate"), "title": "RESONATE", "artists": ["SXULBXRN"], "duration_ms": 137042,
         "cover": "https://i.scdn.co/image/ab67616d0000b273f5c3e26394e2b6cc0ec064df"}
    code, body = call(srv, "PUT", f"/v1/playlists/{sid('phonk')}/tracks", {"snapshot_id": "s1", "tracks": [t]})
    assert code == 200 and body == {"stored": 1, "snapshot_id": "s1"}
    phonk = call(srv, "GET", "/v1/state")[1]["playlists"]["items"][1]
    assert phonk["current"] and phonk["tracks"] == [sid("resonate")]
    assert call(srv, "PUT", f"/v1/playlists/{sid('phonk')}/tracks", {"snapshot_id": "s1", "tracks": [{"id": "x"}]})[0] == 400
    assert call(srv, "PUT", f"/v1/playlists/{sid('nope')}/tracks", {"snapshot_id": "s1", "tracks": []})[0] == 404
    assert call(srv, "PUT", "/v1/state", {})[0] == 404
    assert call(srv, "GET", "/v1/health")[0] == 200
    assert call(srv, "PUT", f"/v1/playlists/{sid('phonk')}/tracks", {}, token=None)[0] == 401


def test_a_verdict_from_a_playlist_review(world):
    srv, worker, con = world
    v = {"id": "pl-verdict-1", "track": sid("like1"), "kind": "rating", "stars": 3,
         "given_at": "2026-10-16T07:00:00Z", "queue": "playlists"}
    assert call(srv, "POST", "/v1/verdicts", {"verdicts": [v]})[1]["accepted"] == ["pl-verdict-1"]
    assert con.execute("SELECT queue FROM verdicts WHERE id='pl-verdict-1'").fetchone()[0] == "playlists"


def test_playlist_syncs_are_rate_limited(tmp_path):
    w = api.Worker(config.load(tmp_path / "none.toml"))
    w.playlists()
    w.playlists()                                          # already queued
    assert w.q.qsize() == 1 and w.q.get()[2] == "playlists"
    w._playlists_queued.clear()
    import time
    w._playlists_at = time.monotonic()
    w.playlists(api.PLAYLISTS_EVERY)                       # ran just now
    assert w.q.qsize() == 0
