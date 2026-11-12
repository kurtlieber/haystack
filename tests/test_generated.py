"""The generated-song store (ADR 0029): Ideas, Takes and Verdicts on takes, for a client such as spindle.

Off (the default) it leaves no trace: no table, no route, no feature. On, it keeps verdicts as ADR 0005 does, and
nothing that recommends music reads it."""
import ast
import http.client
import json
import random
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest

from haystack_music import api, baseline, batch, config, db, generated, review, traits, vectors, verdicts

TOKEN = "g" * 40
SRC = Path(__file__).resolve().parents[1] / "src" / "haystack_music"
TABLES = ("generated_ideas", "generated_takes", "generated_verdicts")


def sid(n: str) -> str:
    return n.ljust(22, "0")


class FakeWorker:
    measuring: set = set()
    checking: set = set()

    def effects(self):
        pass

    def playlists(self, every=0.0):
        pass

    def measure(self, s):
        pass

    def original(self, s, soon=False):
        pass


def _music(con):
    """A batch of one offer with Readings, two liked songs (one rated): enough for the Review and the Baseline."""
    con.execute("INSERT INTO batches(id, created_at, size) VALUES(1, '2026-10-05T00:00:00Z', 1)")
    con.execute("INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status, spotify_id) "
                "VALUES(1, 1, 'Rezz', 'Flying Octopus', 'deezer_artist_radio', 'offered', 'ok', ?)", (sid("rezz"),))
    con.execute("INSERT INTO offers(candidate_id, spotify_id, batch_id, exploration, added_at) VALUES(1, ?, 1, 0, 'x')",
                (sid("rezz"),))
    con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json) VALUES(?, 'Flying Octopus', '[\"Rezz\"]')",
                (sid("rezz"),))
    vectors.put_deezer_vector(con, 1, np.ones(512), np.zeros(1280), 10, 30.0, np.eye(400)[7])
    traits.put(con, 1, {"tempo": 102.0, "valence": 0.14})
    for n, name in enumerate(("like1", "like2"), start=10):
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(name),))
        con.execute("INSERT INTO liked_map(spotify_id, use_local, deezer_id, vector_status) VALUES(?, 0, ?, 'ok')",
                    (sid(name), n))
        traits.put(con, n, {"tempo": 120.0 + n, "valence": 0.5})
    verdicts.record(con, sid("like2"), "rating", 4, given_at="2026-10-06T00:00:00Z")


def _serve(path, enabled: bool):
    cfg = config.load(None)
    cfg["generated"]["enabled"] = enabled
    srv = api.Server(("127.0.0.1", 0), cfg, TOKEN, FakeWorker(), db_path=path,
                     classes=[f"Electronic---Style{i}" for i in range(400)])
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def off(tmp_path):
    path = tmp_path / "t.db"
    _music(db.connect(path))
    srv = _serve(path, False)
    yield srv, path
    srv.shutdown()


@pytest.fixture
def on(tmp_path):
    path = tmp_path / "t.db"
    _music(db.connect(path))
    srv = _serve(path, True)
    yield srv, db.connect(path)
    srv.shutdown()


def call(srv, method, path, body=None):
    c = http.client.HTTPConnection(*srv.server_address, timeout=5)
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    c.request(method, path, body=None if body is None else json.dumps(body).encode(), headers=headers)
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"null")


IDEA = {"id": "idea-0001", "created_at": "2026-11-12T10:00:00Z", "service": "acme",
        "idea": {"title": "Night Bus", "style": "dark synthwave, 104 bpm", "lyrics_by": "model"},
        "reason": "synthwave is 5★ in 6 of 9 new songs", "request": {"prompt": "dark synthwave", "model": "m5"},
        "takes": [{"id": "take-a", "title": "Night Bus", "model": "m5"}, {"id": "take-b", "title": "Night Bus"}]}


def verdict(vid="verdict-001", take="take-a", kind="rating", stars=4, note="the hook lands", given="2026-11-12T11:00:00Z",
            service="acme"):
    return {"id": vid, "service": service, "take": take, "kind": kind, "stars": stars, "note": note, "given_at": given}


def tables(con) -> dict:
    return {t: [tuple(r) for r in con.execute(f"SELECT * FROM {t} ORDER BY rowid")] for t in TABLES}


# ---- off: the default ----

def test_off_is_the_default():
    assert config.load(None)["generated"] == {"enabled": False}
    assert not generated.enabled(config.load(None)) and not generated.enabled(None)


def test_off_creates_no_table_lists_no_feature_and_answers_404_as_an_unknown_path(off):
    srv, path = off
    con = db.connect(path)
    assert con.execute("SELECT name FROM sqlite_master WHERE name LIKE 'generated%'").fetchall() == []
    _, h = call(srv, "GET", "/v1/health")
    assert h["server"]["features"] == list(api.FEATURES)
    _, st = call(srv, "GET", "/v1/state")
    assert "generated" not in st["server"]["features"]
    unknown = call(srv, "GET", "/v1/nothing-here")
    assert unknown == (404, {"error": "no such endpoint"})
    for path_ in ("/v1/generated/takes", "/v1/generated/notes", "/v1/generated/notes?limit=5"):
        assert call(srv, "GET", path_) == unknown
    for path_, key in (("/v1/generated/ideas", "ideas"), ("/v1/generated/verdicts", "verdicts"),
                       ("/v1/generated/kept", "kept")):
        assert call(srv, "POST", path_, {key: []}) == call(srv, "POST", "/v1/nothing-here", {key: []}) == unknown
    assert con.execute("SELECT name FROM sqlite_master WHERE name LIKE 'generated%'").fetchall() == []


# ---- on ----

def test_on_creates_the_tables_and_lists_the_feature(on):
    srv, con = on
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(TABLES) <= names
    _, h = call(srv, "GET", "/v1/health")
    assert h["server"]["features"] == list(api.FEATURES) + ["generated"]
    assert call(srv, "GET", "/v1/generated/takes") == (200, {"takes": [], "ideas": {}})
    assert call(srv, "GET", "/v1/generated/notes") == (200, {"notes": []})


def test_a_verdict_with_a_note_goes_in_comes_back_and_a_resend_changes_nothing(on):
    srv, con = on
    assert call(srv, "POST", "/v1/generated/ideas", {"ideas": [IDEA]}) == (200, {"accepted": ["idea-0001"],
                                                                                  "rejected": []})
    assert call(srv, "POST", "/v1/generated/verdicts", {"verdicts": [verdict()]}) == \
        (200, {"accepted": ["verdict-001"], "rejected": []})
    _, got = call(srv, "GET", "/v1/generated/takes")
    a = next(t for t in got["takes"] if t["id"] == "take-a")
    assert a == {"service": "acme", "id": "take-a", "idea": "idea-0001", "title": "Night Bus", "model": "m5",
                 "kept_at": None,
                 "verdict": {"kind": "rating", "stars": 4, "note": "the hook lands", "given_at": "2026-11-12T11:00:00Z"}}
    assert next(t for t in got["takes"] if t["id"] == "take-b")["verdict"] is None
    assert got["ideas"] == {"idea-0001": {"created_at": "2026-11-12T10:00:00Z", "service": "acme",
                                          "idea": IDEA["idea"], "reason": IDEA["reason"], "request": IDEA["request"]}}
    _, notes = call(srv, "GET", "/v1/generated/notes")
    assert notes == {"notes": [{"service": "acme", "take": "take-a", "idea": "idea-0001", "title": "Night Bus",
                                "kind": "rating", "stars": 4, "note": "the hook lands",
                                "given_at": "2026-11-12T11:00:00Z", "current": True}]}
    before, changes = tables(con), con.total_changes
    # the outbox sends again: same ids, and a resent idea whose content differs keeps the first
    assert call(srv, "POST", "/v1/generated/verdicts", {"verdicts": [verdict(stars=1, note="changed")]})[1][
        "accepted"] == ["verdict-001"]
    assert call(srv, "POST", "/v1/generated/ideas", {"ideas": [{**IDEA, "reason": "other"}]})[1]["accepted"] == \
        ["idea-0001"]
    assert tables(con) == before
    assert call(srv, "GET", "/v1/generated/takes")[1] == got
    assert con.total_changes == changes


def test_the_latest_verdict_by_given_at_counts_and_older_notes_stay_in_the_log(on):
    srv, con = on
    call(srv, "POST", "/v1/generated/ideas", {"ideas": [IDEA]})
    call(srv, "POST", "/v1/generated/verdicts", {"verdicts": [
        verdict("verdict-002", given="2026-11-12T12:00:00Z", kind="dislike", stars=None, note="too busy"),
        verdict("verdict-001", given="2026-11-12T11:00:00Z"),          # arrives later, given earlier
        verdict("verdict-003", take="take-b", stars=5, note=None, given="2026-11-12T13:00:00Z")]})
    takes = {t["id"]: t for t in call(srv, "GET", "/v1/generated/takes")[1]["takes"]}
    assert takes["take-a"]["verdict"] == {"kind": "dislike", "stars": None, "note": "too busy",
                                          "given_at": "2026-11-12T12:00:00Z"}
    assert takes["take-b"]["verdict"]["stars"] == 5 and takes["take-b"]["verdict"]["note"] is None
    notes = call(srv, "GET", "/v1/generated/notes")[1]["notes"]
    assert [(n["note"], n["current"]) for n in notes] == [("too busy", True), ("the hook lands", False)]
    assert [n["note"] for n in call(srv, "GET", "/v1/generated/notes?limit=1")[1]["notes"]] == ["too busy"]
    for bad in ("0", "501", "x", "-1"):
        code, body = call(srv, "GET", f"/v1/generated/notes?limit={bad}")
        assert code == 400 and "limit" in body["error"]


def test_a_verdict_or_a_keep_on_a_take_haystack_has_not_seen_records_a_take_with_no_idea(on):
    srv, con = on
    assert call(srv, "POST", "/v1/generated/verdicts", {"verdicts": [verdict(take="old-take")]})[1]["accepted"] == \
        ["verdict-001"]
    assert call(srv, "POST", "/v1/generated/kept", {"kept": [
        {"id": "kept-0001", "service": "acme", "take": "older-take", "kept_at": "2026-11-12T11:30:00Z"}]}) == \
        (200, {"accepted": ["kept-0001"], "rejected": []})
    takes = {t["id"]: t for t in call(srv, "GET", "/v1/generated/takes")[1]["takes"]}
    assert takes["old-take"]["idea"] is None and takes["old-take"]["title"] is None
    assert takes["old-take"]["verdict"]["note"] == "the hook lands"
    assert takes["older-take"]["kept_at"] == "2026-11-12T11:30:00Z" and takes["older-take"]["verdict"] is None
    # an idea recorded later claims the take and fills what is missing, never what is there
    call(srv, "POST", "/v1/generated/ideas", {"ideas": [{**IDEA, "takes": [{"id": "old-take", "title": "Found"}]}]})
    call(srv, "POST", "/v1/generated/ideas", {"ideas": [{**IDEA, "id": "idea-0002",
                                                         "takes": [{"id": "old-take", "title": "Other"}]}]})
    t = {t["id"]: t for t in call(srv, "GET", "/v1/generated/takes")[1]["takes"]}["old-take"]
    assert (t["idea"], t["title"], t["model"]) == ("idea-0001", "Found", None)


def test_the_first_keep_wins_and_a_resend_is_a_no_op(on):
    srv, con = on
    call(srv, "POST", "/v1/generated/ideas", {"ideas": [IDEA]})
    k = {"id": "kept-0001", "service": "acme", "take": "take-a", "kept_at": "2026-11-12T11:30:00Z"}
    call(srv, "POST", "/v1/generated/kept", {"kept": [k]})
    changes = con.total_changes
    assert call(srv, "POST", "/v1/generated/kept", {"kept": [k, {**k, "id": "kept-0002",
                                                                 "kept_at": "2026-11-13T00:00:00Z"}]})[0] == 200
    assert con.total_changes == changes
    takes = {t["id"]: t for t in call(srv, "GET", "/v1/generated/takes")[1]["takes"]}
    assert takes["take-a"]["kept_at"] == "2026-11-12T11:30:00Z" and takes["take-b"]["kept_at"] is None


def test_an_idea_resent_with_its_takes_adds_them(on):
    srv, con = on
    call(srv, "POST", "/v1/generated/ideas", {"ideas": [{**IDEA, "takes": []}]})     # sent, nothing back yet
    assert call(srv, "GET", "/v1/generated/takes")[1]["takes"] == []
    assert "idea-0001" in call(srv, "GET", "/v1/generated/takes")[1]["ideas"]
    call(srv, "POST", "/v1/generated/ideas", {"ideas": [IDEA]})
    assert [t["id"] for t in call(srv, "GET", "/v1/generated/takes")[1]["takes"]] == ["take-a", "take-b"]


def test_takes_are_newest_first(on):
    srv, con = on
    call(srv, "POST", "/v1/generated/ideas", {"ideas": [
        IDEA, {**IDEA, "id": "idea-0002", "created_at": "2026-11-12T12:00:00Z", "takes": [{"id": "take-c"}]},
        {**IDEA, "id": "idea-0003", "created_at": "2026-11-11T12:00:00Z", "takes": [{"id": "take-d"}],
         "request": None, "reason": None}]})
    got = call(srv, "GET", "/v1/generated/takes")[1]
    assert [t["id"] for t in got["takes"]] == ["take-c", "take-a", "take-b", "take-d"]
    assert got["ideas"]["idea-0003"]["request"] is None and got["ideas"]["idea-0003"]["reason"] is None


@pytest.mark.parametrize("path, key, item, error", [
    ("ideas", "ideas", {**IDEA, "id": "short"}, "id"),
    ("ideas", "ideas", {**IDEA, "created_at": "yesterday"}, "created_at"),
    ("ideas", "ideas", {**IDEA, "service": "Acme Music"}, "service"),
    ("ideas", "ideas", {**IDEA, "idea": "a song"}, "idea"),
    ("ideas", "ideas", {**IDEA, "idea": {"lyrics": "x" * 70000}}, "idea"),
    ("ideas", "ideas", {**IDEA, "request": ["prompt"]}, "request"),
    ("ideas", "ideas", {**IDEA, "takes": [{"id": "t"}] * 21}, "takes"),
    ("ideas", "ideas", {**IDEA, "takes": [{"title": "no id"}]}, "take"),
    ("ideas", "ideas", {**IDEA, "takes": [{"id": "bad\nid"}]}, "take"),
    ("ideas", "ideas", {**IDEA, "reason": 7}, "reason"),
    ("verdicts", "verdicts", verdict(stars=6), "stars"),
    ("verdicts", "verdicts", verdict(kind="pass", stars=None), "kind"),
    ("verdicts", "verdicts", verdict(kind="dislike", stars=2), "stars"),
    ("verdicts", "verdicts", verdict(kind="rating", stars=True), "stars"),
    ("verdicts", "verdicts", verdict(note="x" * 4001), "note"),
    ("verdicts", "verdicts", verdict(take=""), "take"),
    ("verdicts", "verdicts", verdict(given="2026-11-12 11:00"), "given_at"),
    ("verdicts", "verdicts", verdict(service=None), "service"),
    ("kept", "kept", {"id": "kept-0001", "service": "acme", "take": "take-a"}, "kept_at"),
])
def test_a_malformed_item_is_rejected_with_a_reason_and_the_rest_are_stored(on, path, key, item, error):
    srv, con = on
    good = {"ideas": IDEA, "verdicts": verdict("verdict-ok1"), "kept": {"id": "kept-ok01", "service": "acme",
                                                                         "take": "take-a",
                                                                         "kept_at": "2026-11-12T11:30:00Z"}}[key]
    code, body = call(srv, "POST", f"/v1/generated/{path}", {key: [item, good]})
    assert code == 200 and body["accepted"] == [good["id"]]
    assert len(body["rejected"]) == 1 and error in body["rejected"][0]["error"]
    assert body["rejected"][0]["id"] == (item["id"] if isinstance(item.get("id"), str) else None)


def test_a_malformed_body_is_a_400(on):
    srv, con = on
    for path, key, most in (("ideas", "ideas", 100), ("verdicts", "verdicts", 500), ("kept", "kept", 500)):
        assert call(srv, "POST", f"/v1/generated/{path}", {"other": []})[0] == 400
        assert call(srv, "POST", f"/v1/generated/{path}", {key: [{}] * (most + 1)})[0] == 400
    code, body = call(srv, "POST", "/v1/generated/verdicts", {"verdicts": ["not an object", 7]})
    assert code == 200 and [r["id"] for r in body["rejected"]] == [None, None]
    assert tables(con) == {t: [] for t in TABLES}


# ---- apart from the music haystack recommends ----

def _imports_generated(tree) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in ("generated", "haystack_music.generated"):
                return True
            if node.module in (None, "haystack_music") and any(a.name == "generated" for a in node.names):
                return True
        if isinstance(node, ast.Import) and any(a.name == "haystack_music.generated" for a in node.names):
            return True
    return False


def test_only_the_store_names_its_tables_and_only_the_api_imports_it():
    """Harvest, scoring, the re-ranker, the Baseline, the Review and every other module: none can read a table it
    never names, and none but the API can call the store."""
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "generated.py":
            continue
        text = path.read_text()
        assert not any(t in text for t in TABLES), f"{path.name} names a generated-song table"
        assert path.name == "api.py" or not _imports_generated(ast.parse(text)), f"{path.name} imports the store"


def test_recommending_reads_no_generated_table(tmp_path, monkeypatch):
    """With the store on and full: what scoring, the re-ranker and selection (`batch --plan`), the Baseline and the
    Review read, table by table."""
    from test_batch_plan import _no_network
    from test_rerank import _world
    con, cfg, path = _world(tmp_path, monkeypatch)
    cfg["generated"]["enabled"] = True
    generated.ensure(con)
    _fill(con)
    read: set = set()

    def note(action, a1, a2, dbname, source):
        if action == sqlite3.SQLITE_READ:
            read.add(a1)
        return sqlite3.SQLITE_OK

    _no_network(monkeypatch)
    con.set_authorizer(note)
    out = batch.plan(con, cfg, rng=random.Random(1))
    assert out["picked"] and out["rerank"]
    baseline.compute(con, cfg)
    review.state(con, cfg)
    con.set_authorizer(None)
    assert {"verdicts", "candidates", "readings"} <= read           # it watched the reads
    assert not read & set(TABLES)


def _fill(con):
    generated.receive(con, "ideas", [IDEA])
    generated.receive(con, "verdicts", [verdict(), verdict("verdict-002", take="take-b", kind="dislike", stars=None),
                                        verdict("verdict-003", take="elsewhere", stars=5)])
    generated.receive(con, "kept", [{"id": "kept-0001", "service": "acme", "take": "take-a",
                                     "kept_at": "2026-11-12T11:30:00Z"}])


def test_state_and_the_baseline_are_the_same_with_and_without_generated_rows(on):
    srv, con = on
    cfg = srv.cfg
    _, before = call(srv, "GET", "/v1/state")
    base_before = baseline.compute(con, cfg)
    assert before["queues"]["discoveries"]["items"] and before["baseline"]["tracks"]
    _fill(con)
    assert con.execute("SELECT COUNT(*) FROM generated_verdicts").fetchone()[0] == 3
    _, after = call(srv, "GET", "/v1/state")
    for st in (before, after):
        st.pop("generated_at")
    assert after == before
    assert baseline.compute(con, cfg) == base_before
