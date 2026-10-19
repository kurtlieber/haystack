"""/v1 contract: every key and JSON type in tests/fixtures/v1-contract.json is still in the live payloads.

The fixture is the API as clients may rely on it (docs/api.md). A change may add to it, never remove from it.
The world below fills every documented field with a value at least once, so a field can't pass by being null."""
import http.client
import json
import re
import threading
from pathlib import Path

import numpy as np
import pytest

from haystack_music import __version__, api, config, db, playlists, traits, vectors, verdicts

TOKEN = "c" * 40
FIXTURE = Path(__file__).parent / "fixtures" / "v1-contract.json"
API_DOC = Path(__file__).parent.parent / "docs" / "api.md"
SCALARS = {"string": str, "boolean": bool, "null": type(None), "object": dict, "array": list}


def sid(n: str) -> str:
    return n.ljust(22, "0")


def _is(value, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, SCALARS[name])


def check(value, spec, where: str, defs: dict, seen: set) -> list[str]:
    """-> what doesn't match (empty when the value matches). `seen` gets each 'section.key' that held a non-null value."""
    if isinstance(spec, str):
        return [] if _is(value, spec) else [f"{where}: {type(value).__name__}, not {spec}"]
    if isinstance(spec, list):
        errs = []
        for s in spec:
            mine: set = set()
            e = check(value, s, where, defs, mine)
            if not e:
                seen |= mine
                return []
            errs += e
        return [f"{where}: matches none of {json.dumps(spec)} ({'; '.join(errs)})"]
    if "$ref" in spec:
        return check(value, defs[spec["$ref"]], spec["$ref"], defs, seen)
    if "$all" in spec:
        return [e for s in spec["$all"] for e in check(value, s, where, defs, seen)]
    if "$array" in spec:
        if not isinstance(value, list):
            return [f"{where}: {type(value).__name__}, not an array"]
        return [e for x in value for e in check(x, spec["$array"], f"{where}[]", defs, seen)]
    if "$map" in spec:
        if not isinstance(value, dict):
            return [f"{where}: {type(value).__name__}, not an object"]
        return [e for x in value.values() for e in check(x, spec["$map"], f"{where}{{}}", defs, seen)]
    if not isinstance(value, dict):
        return [f"{where}: {type(value).__name__}, not an object"]
    errs = []
    for k, s in spec.items():
        if k not in value:
            errs.append(f"{where}.{k}: missing")
            continue
        if value[k] is not None:
            seen.add(f"{where}.{k}")
        errs += check(value[k], s, f"{where}.{k}", defs, seen)
    return errs


def key_paths(spec, where: str, defs: dict, out: set) -> set:
    """Every 'section.key' the fixture documents (the same names check() puts in `seen`)."""
    if isinstance(spec, list):
        for s in spec:
            key_paths(s, where, defs, out)
    elif isinstance(spec, dict):
        if "$ref" in spec:
            key_paths(defs[spec["$ref"]], spec["$ref"], defs, out)
        elif "$all" in spec:
            for s in spec["$all"]:
                key_paths(s, where, defs, out)
        elif "$array" in spec:
            key_paths(spec["$array"], f"{where}[]", defs, out)
        elif "$map" in spec:
            key_paths(spec["$map"], f"{where}{{}}", defs, out)
        else:
            for k, s in spec.items():
                out.add(f"{where}.{k}")
                key_paths(s, f"{where}.{k}", defs, out)
    return out


@pytest.fixture(scope="module")
def defs():
    d = json.loads(FIXTURE.read_text())
    d.pop("_note")
    return d


class Worker:
    measuring: set = set()

    def effects(self):
        pass

    def playlists(self, every=0.0):
        pass

    def measure(self, s):
        pass


ALL_TRAITS = {"tempo": 120.0, "valence": 0.5, "energy": 0.6, "danceability": 0.7, "acousticness": 0.1,
              "instrumentalness": 0.4, "loudness": -8.0}


@pytest.fixture
def server(tmp_path):
    """Two offers (one rated, one exploration with Readings), an Unrated like, a rated like with Readings (the
    Baseline), a playlist holding both offers, and a Spotify write that failed."""
    path = tmp_path / "t.db"
    con = db.connect(path)
    con.execute("INSERT INTO batches(id, created_at, size) VALUES(1, '2026-01-05T00:00:00Z', 2)")
    con.execute("INSERT INTO local_tracks(id, path, size, content_hash, title_raw, artists_json) "
                "VALUES(5, '/m/a.m4a', 1, 'h', 'Library Title', '[\"Library Artist\"]')")
    for n, name in enumerate(("disc1", "disc2", "like1", "like2"), start=1):
        con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, album, album_type, release_date, duration, "
                    "image_url, image_url_small) VALUES(?,?,?,?,?,?,?,?,?)",
                    (sid(name), f"Track {n}", json.dumps([f"Artist {n}"]), f"Album {n}", "single", "2020-01-02",
                     200.0, "https://example.invalid/c.jpg", "https://example.invalid/s.jpg"))
    for cid, name, explore in ((1, "disc1", 1), (2, "disc2", 0)):
        con.execute("INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status, spotify_id, "
                    "nearest, raw_cos) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (cid, cid, f"Artist {cid}", f"Track {cid}", "deezer_artist_radio", "offered", "ok", sid(name),
                     "local:5", 0.91))
        con.execute("INSERT INTO offers(candidate_id, spotify_id, batch_id, exploration, added_at) VALUES(?,?,1,?,'x')",
                    (cid, sid(name), explore))
        vectors.put_deezer_vector(con, cid, np.ones(512), np.zeros(1280), 10, 30.0, np.eye(400)[cid * 3])
    traits.put(con, 1, ALL_TRAITS)
    for name in ("like1", "like2"):
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(name),))
    con.execute("INSERT INTO liked_map(spotify_id, use_local, deezer_id, vector_status) VALUES(?, 0, 10, 'ok')",
                (sid("like2"),))
    vectors.put_deezer_vector(con, 10, np.ones(512), np.zeros(1280), 10, 30.0, np.eye(400)[9])
    traits.put(con, 10, ALL_TRAITS)
    verdicts.record(con, sid("disc2"), "rating", 4)
    verdicts.record(con, sid("like2"), "rating", 4)
    con.execute("UPDATE verdicts SET effects_error='synthetic failure' WHERE spotify_id=?", (sid("disc2"),))
    con.execute("INSERT INTO playlists(id, name, owner_id, owner_name, mine, image_url, total, snapshot_id, position) "
                "VALUES(?, 'Synthetic Mix', 'me', 'Listener', 1, 'https://example.invalid/p.jpg', 2, 's1', 0)",
                (sid("mix"),))
    playlists.store(con, sid("mix"), "s1", [(sid("disc1"), None), (sid("disc2"), None)], "web")
    db.kv_set(con, "playlists_synced_at", "2026-01-05T00:00:00Z")
    srv = api.Server(("127.0.0.1", 0), config.load(tmp_path / "none.toml"), TOKEN, Worker(), db_path=path,
                     classes=[f"Parent---Style{i}" for i in range(400)])
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def call(srv, method, path, body=None):
    c = http.client.HTTPConnection(*srv.server_address, timeout=5)
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    c.request(method, path, body=None if body is None else json.dumps(body).encode(), headers=headers)
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"null")


def payloads(srv) -> dict:
    """One live payload per fixture section that is a whole response."""
    out = {}
    for name, method, path, body in (
            ("health", "GET", "/v1/health", None),
            ("state", "GET", "/v1/state", None),
            ("card", "GET", f"/v1/tracks/{sid('disc1')}", None),
            ("playlist", "GET", f"/v1/playlists/{sid('mix')}", None),
            ("verdicts_response", "POST", "/v1/verdicts",
             {"verdicts": [{"id": "contract-1", "track": sid("like1"), "kind": "pass", "given_at": "2026-01-06T00:00:00Z"},
                           {"id": "contract-2", "track": "short", "kind": "pass"}]}),
            ("plays_response", "POST", "/v1/plays",
             {"plays": [{"id": "p1", "track": sid("disc1"), "started_at": "2026-01-06T00:00:00Z"}, {"id": "p2"}]}),
            ("playlist_put_response", "PUT", f"/v1/playlists/{sid('mix')}/tracks",
             {"snapshot_id": "s2", "tracks": [{"id": sid("disc1"), "title": "Track 1", "artists": ["Artist 1"]}]})):
        code, out[name] = call(srv, method, path, body)
        assert code == 200, (name, out[name])
    return out


def test_live_payloads_keep_every_documented_key_and_type(server, defs):
    seen: set = set()
    errs = []
    for name, value in payloads(server).items():
        errs += check(value, defs[name], name, defs, seen)
    assert errs == []
    documented = set()
    for name in ("health", "state", "card", "playlist", "verdicts_response", "plays_response", "playlist_put_response"):
        key_paths(defs[name], name, defs, documented)
    assert documented - seen == set(), "give these a non-null value in the test world"


def test_every_fixture_section_is_checked(defs):
    used: set = set()

    def refs(spec):
        if isinstance(spec, list):
            for s in spec:
                refs(s)
        elif isinstance(spec, dict):
            for k, s in spec.items():
                if k == "$ref":
                    used.add(s)
                else:
                    refs(s)
    roots = {"health", "state", "card", "playlist", "verdicts_response", "plays_response", "playlist_put_response"}
    for name in roots:
        refs(defs[name])
    assert set(defs) == roots | used


def test_the_check_catches_a_removed_or_retyped_field(defs):
    good = {"ok": True, "version": "1", "time": "2026-01-01T00:00:00Z",
            "server": {"version": "0.1.0", "api": "1", "features": []}}
    assert check(good, defs["health"], "health", defs, set()) == []
    assert check({**good, "extra": 1}, defs["health"], "health", defs, set()) == []          # additions are fine
    assert check({k: v for k, v in good.items() if k != "time"}, defs["health"], "health", defs, set()) == \
        ["health.time: missing"]
    assert check({**good, "ok": "yes"}, defs["health"], "health", defs, set()) == ["health.ok: str, not boolean"]
    assert check({**good, "server": {**good["server"], "features": [1]}}, defs["health"], "health", defs, set())
    assert check(True, "integer", "x", defs, set())                                          # a bool is no integer


def test_server_object_and_features_are_documented(server):
    _, h = call(server, "GET", "/v1/health")
    assert h["server"] == {"version": __version__, "api": api.VERSION, "features": list(api.FEATURES)}
    assert h["version"] == "1"                     # the old field keeps its meaning: the API version
    doc = API_DOC.read_text()
    for f in api.FEATURES:
        assert re.search(rf"`{re.escape(f)}`", doc), f"feature {f} is not in docs/api.md"
