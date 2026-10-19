"""Discoveries clear: mocked Spotify only. Never talks to the live API."""
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from haystack_music import config, db, feedback, vectors
from haystack_music.clear import clear_discoveries
from haystack_music.cli import build_parser
from haystack_music.matching import Index
from haystack_music.textnorm import song_key

PID = "pid_discoveries"
NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
OLD = "2026-10-01T12:00:00Z"       # 11 days
EXACT = "2026-10-11T12:00:00Z"     # exactly 24h
RECENT = "2026-10-12T11:00:00Z"    # 1h
SECRET_SID = "DoNotLogThisSpotifyIdx"


class FakeSpotify:
    def __init__(self, items, liked=()):
        self.items = [dict(x) for x in items]
        self.liked = set(liked)
        self.removed = []
        self.remove_calls = 0
        self.snapshot_ids = []
        self.fail = None  # "before" | "after" | None
        self.on_remove = None

    def playlist_items(self, pid):
        assert pid == PID
        return [dict(x) for x in self.items]

    def library_contains(self, ids):
        return {i: i in self.liked for i in ids}

    def playlist(self, pid):
        return {"id": pid, "snapshot_id": "snap"}

    def remove_items(self, pid, ids, snapshot_id=None):
        if self.fail == "before":
            raise RuntimeError("spotify down")
        if self.on_remove:
            self.on_remove(list(ids))
        self.remove_calls += 1
        self.snapshot_ids.append(snapshot_id)
        self.removed.extend(ids)
        gone = set(ids)
        self.items = [it for it in self.items if it["spotify_id"] not in gone]
        if self.fail == "after":
            raise RuntimeError("lost the db update")
        return "snap2"


def _db(tmp_path, name="t.db"):
    return db.connect(tmp_path / name)


def _cfg(tmp_path, text=""):
    p = tmp_path / "c.toml"
    if text:
        p.write_text(text)
    return config.load(p)


def _add(con, cid, sid, artist, title, added_at, *, offer=True, playlist_add=True):
    con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, song_key, source, state, spotify_id, vector_status)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (cid, cid, artist, title, song_key(artist, title), "deezer_chart", "offered", sid, "ok"))
    if offer:
        con.execute("INSERT INTO offers(candidate_id, spotify_id, source, added_at) VALUES(?,?,?,?)",
                    (cid, sid, "deezer_chart", added_at))
    if playlist_add:
        con.execute("""INSERT INTO playlist_adds(playlist, playlist_id, spotify_id, added_at)
                       VALUES('discoveries',?,?,?)""", (PID, sid, added_at))


def _item(sid, added_at, name=None, artists=None):
    return {"spotify_id": sid, "added_at": added_at, "added_by": "user", "name": name, "artists": artists or []}


def _clear(con, cfg, sp, **kw):
    kw.setdefault("playlist_id", PID)
    kw.setdefault("resolve_playlist", False)
    kw.setdefault("now", NOW)
    return clear_discoveries(con, cfg, sp, **kw)


def _feedback(con):
    return [dict(r) for r in feedback.judged(con)]


def _offer(con, sid):
    return con.execute("SELECT outcome, removed_at FROM offers WHERE spotify_id=?", (sid,)).fetchone()


def test_liked_removed_without_verdict_unliked_old_gets_no(tmp_path):
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "likedtrack00000000001", "Liked", "Song", RECENT)
    _add(con, 2, SECRET_SID, "Old", "Track", OLD)
    sp = FakeSpotify([
        _item("likedtrack00000000001", RECENT, "Song", ["Liked"]),
        _item(SECRET_SID, OLD, "Track", ["Old"]),
    ], liked={"likedtrack00000000001"})
    out = _clear(con, cfg, sp)
    assert out["liked"] == 1 and out["no"] == 1 and out["removed"] == 2 and out["feedback"] == 1
    assert out["recent_removed"] == 0 and out["dry_run"] is False
    assert sp.removed == ["likedtrack00000000001", SECRET_SID]
    assert sp.snapshot_ids == ["snap"]
    rows = _feedback(con)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "no" and rows[0]["weight"] == 1.0 and rows[0]["kind"] == "pass"
    assert [r[0] for r in con.execute("SELECT source FROM verdicts")] == ["playlist"]
    assert rows[0]["artist"] == "Old" and rows[0]["spotify_id"] == SECRET_SID
    assert _offer(con, "likedtrack00000000001")["outcome"] == "liked"
    assert _offer(con, SECRET_SID)["outcome"] == "negative"
    states = dict(con.execute("SELECT spotify_id, state FROM candidates"))
    assert states["likedtrack00000000001"] == "liked" and states[SECRET_SID] == "negative"
    reasons = dict(con.execute("SELECT spotify_id, removed_reason FROM playlist_adds"))
    assert reasons == {"likedtrack00000000001": "liked", SECRET_SID: "clear_no"}
    assert con.execute("SELECT COUNT(*) FROM clear_pending").fetchone()[0] == 0
    # local likes table is not the signal: the liked track was not inserted there
    assert con.execute("SELECT COUNT(*) FROM likes").fetchone()[0] == 0


def test_stale_local_like_does_not_suppress_a_no(tmp_path):
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "stalelike0000000000001", "Stale", "Like", OLD)
    con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?,?)", ("stalelike0000000000001", OLD))
    sp = FakeSpotify([_item("stalelike0000000000001", OLD)], liked=())
    out = _clear(con, cfg, sp)
    assert out["no"] == 1 and out["liked"] == 0
    assert _feedback(con)[0]["verdict"] == "no"


def test_recent_remove_records_nothing_keep_leaves_them(tmp_path):
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "oldtrack00000000000001", "Old", "One", OLD)
    _add(con, 2, "newtrack00000000000001", "New", "One", RECENT)
    _add(con, 3, "likednew00000000000001", "Liked", "New", RECENT)
    items = [
        _item("oldtrack00000000000001", OLD),
        _item("newtrack00000000000001", RECENT),
        _item("likednew00000000000001", RECENT),
    ]
    sp = FakeSpotify(items, liked={"likednew00000000000001"})
    out = _clear(con, cfg, sp, recent="remove")
    assert out["no"] == 1 and out["recent_removed"] == 1 and out["liked"] == 1 and out["recent_kept"] == 0
    assert set(sp.removed) == {"oldtrack00000000000001", "newtrack00000000000001", "likednew00000000000001"}
    assert _feedback(con)[0]["spotify_id"] == "oldtrack00000000000001"
    new = _offer(con, "newtrack00000000000001")
    assert new["outcome"] == "unplayed" and new["removed_at"] is not None
    pooled = con.execute("SELECT state, unplayed_expiries FROM candidates WHERE id=2").fetchone()
    assert pooled["state"] == "pool" and pooled["unplayed_expiries"] == 0
    assert con.execute("SELECT removed_reason FROM playlist_adds WHERE spotify_id='newtrack00000000000001'").fetchone()[0] == "unplayed"

    con2, cfg2 = _db(tmp_path, "keep.db"), _cfg(tmp_path)
    # fresh db: keep mode leaves the recent unliked track and still removes the liked one
    _add(con2, 1, "oldtrack00000000000001", "Old", "One", OLD)
    _add(con2, 2, "newtrack00000000000001", "New", "One", RECENT)
    _add(con2, 3, "likednew00000000000001", "Liked", "New", RECENT)
    sp2 = FakeSpotify(items, liked={"likednew00000000000001"})
    out2 = _clear(con2, cfg2, sp2, recent="keep")
    assert out2["recent_kept"] == 1 and out2["recent_removed"] == 0 and out2["liked"] == 1 and out2["no"] == 1
    assert set(sp2.removed) == {"oldtrack00000000000001", "likednew00000000000001"}
    assert sp2.items == [_item("newtrack00000000000001", RECENT)]
    assert _offer(con2, "newtrack00000000000001")["removed_at"] is None
    assert _feedback(con2)[0]["spotify_id"] == "oldtrack00000000000001"


def test_exactly_24h_is_judged_younger_is_not(tmp_path):
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "exact00000000000000001", "Exact", "Day", EXACT)
    _add(con, 2, "young00000000000000001", "Young", "Hour", "2026-10-11T12:00:01Z")
    sp = FakeSpotify([
        _item("exact00000000000000001", EXACT),
        _item("young00000000000000001", "2026-10-11T12:00:01Z"),
    ])
    out = _clear(con, cfg, sp, recent="keep")
    assert out["no"] == 1 and out["recent_kept"] == 1
    assert _feedback(con)[0]["spotify_id"] == "exact00000000000000001"
    assert sp.items[0]["spotify_id"] == "young00000000000000001"


def test_playlist_added_at_wins_over_older_offer(tmp_path):
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "readded000000000000001", "Re", "Added", OLD)
    sp = FakeSpotify([_item("readded000000000000001", RECENT)])
    out = _clear(con, cfg, sp, recent="keep")
    assert out["recent_kept"] == 1 and out["no"] == 0 and out["feedback"] == 0
    assert _feedback(con) == []


def test_missing_playlist_timestamp_falls_back_then_unknown_is_a_no(tmp_path):
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "fallback00000000000001", "Fall", "Back", OLD)
    _add(con, 2, "unknown000000000000001", "No", "Time", OLD, offer=False, playlist_add=False)
    sp = FakeSpotify([
        _item("fallback00000000000001", None),
        _item("unknown000000000000001", None),
    ])
    out = _clear(con, cfg, sp)
    assert out["no"] == 2 and out["age_unknown"] == 1 and out["feedback"] == 2
    sids = {r["spotify_id"] for r in _feedback(con)}
    assert sids == {"fallback00000000000001", "unknown000000000000001"}


def test_duplicate_rows_use_the_earliest_add_and_one_verdict(tmp_path):
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "dupe000000000000000001", "Dupe", "Song", OLD)
    sp = FakeSpotify([
        _item("dupe000000000000000001", RECENT),
        _item("dupe000000000000000001", OLD),
    ])
    out = _clear(con, cfg, sp)
    assert out["items"] == 2 and out["tracks"] == 1 and out["no"] == 1 and out["feedback"] == 1
    assert sp.removed == ["dupe000000000000000001"]
    assert len(_feedback(con)) == 1


def test_unlinked_track_is_removed_with_no_feedback_row(tmp_path):
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    sp = FakeSpotify([_item("nocandidate0000000001", OLD, "Ghost", ["Nobody"])])
    out = _clear(con, cfg, sp)
    assert out["no"] == 1 and out["unlinked"] == 1 and out["feedback"] == 0 and out["removed"] == 1
    assert _feedback(con) == []
    assert sp.items == []


def test_dry_run_writes_nothing(tmp_path, caplog):
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, SECRET_SID, "Old", "Track", OLD)
    sp = FakeSpotify([_item(SECRET_SID, OLD)])
    caplog.set_level(logging.INFO)
    out = _clear(con, cfg, sp, dry_run=True)
    assert out["dry_run"] is True and out["would_remove"] == 1 and out["removed"] == 0 and out["feedback"] == 0
    assert sp.remove_calls == 0 and sp.items
    assert _feedback(con) == []
    assert _offer(con, SECRET_SID)["removed_at"] is None
    assert con.execute("SELECT COUNT(*) FROM clear_pending").fetchone()[0] == 0
    assert SECRET_SID not in caplog.text
    assert "refresh" not in caplog.text.lower() and "token" not in caplog.text.lower()
    assert "no=1" in caplog.text and "dry_run=True" in caplog.text


def test_second_run_is_a_no_op(tmp_path):
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "oldtrack00000000000001", "Old", "Track", OLD)
    sp = FakeSpotify([_item("oldtrack00000000000001", OLD)])
    first = _clear(con, cfg, sp)
    second = _clear(con, cfg, sp)
    assert first["feedback"] == 1 and first["removed"] == 1
    assert second["items"] == 0 and second["removed"] == 0 and second["feedback"] == 0 and second["no"] == 0
    assert sp.remove_calls == 1
    assert len(_feedback(con)) == 1


def test_retry_after_delete_keeps_the_verdict_sync_already_saw(tmp_path):
    from haystack_music.sync import process_offers
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "oldtrack00000000000001", "Old", "Track", OLD)
    sp = FakeSpotify([_item("oldtrack00000000000001", OLD)])
    sp.fail = "after"
    with pytest.raises(RuntimeError, match="lost the db update"):
        _clear(con, cfg, sp)
    # The no was committed before the delete, so a sync that now sees an empty playlist
    # cannot turn it into manual_removed.
    assert sp.items == []
    assert len(_feedback(con)) == 1 and _feedback(con)[0]["verdict"] == "no"
    assert _offer(con, "oldtrack00000000000001")["outcome"] == "negative"
    sync = process_offers(con, sp, cfg, {"discoveries": PID, "to_rip": "torip"}, Index([]))
    assert sync["manual_removed"] == 0
    assert _offer(con, "oldtrack00000000000001")["outcome"] == "negative"
    sp.fail = None
    out = _clear(con, cfg, sp)
    assert out["removed"] == 0 and out["reconciled"] == 0
    assert len(_feedback(con)) == 1


def test_retry_before_delete_already_closed_the_offer(tmp_path):
    from haystack_music.sync import process_offers
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "oldtrack00000000000001", "Old", "Track", OLD)
    sp = FakeSpotify([_item("oldtrack00000000000001", OLD)])
    sp.fail = "before"
    with pytest.raises(RuntimeError, match="spotify down"):
        _clear(con, cfg, sp)
    assert sp.remove_calls == 0 and sp.items
    assert len(_feedback(con)) == 1
    assert _offer(con, "oldtrack00000000000001")["outcome"] == "negative"
    assert _offer(con, "oldtrack00000000000001")["removed_at"] is not None
    sync = process_offers(con, sp, cfg, {"discoveries": PID, "to_rip": "torip"}, Index([]))
    assert sync["manual_removed"] == 0
    sp.fail = None
    out = _clear(con, cfg, sp)
    assert out["removed"] == 1
    assert len(_feedback(con)) == 1
    assert sp.items == []


def test_no_penalty_and_recent_unplayed_are_not_penalized(tmp_path, monkeypatch):
    """Sync must not mark these manual_removed, and vectors must not treat them as negatives."""
    from haystack_music.sync import process_offers
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "liked0000000000000001", "Liked", "Song", OLD)
    _add(con, 2, "oldtrack00000000000001", "Old", "Track", OLD)
    _add(con, 3, "newtrack00000000000001", "New", "Track", RECENT)
    for i in (1, 2, 3):
        vectors.put_deezer_vector(con, i, np.ones(512, np.float32) * i, np.zeros(1280), 1, 30.0)
    sp = FakeSpotify([
        _item("liked0000000000000001", OLD),
        _item("oldtrack00000000000001", OLD),
        _item("newtrack00000000000001", RECENT),
    ], liked={"liked0000000000000001"})

    def on_remove(ids):
        assert set(ids) == {"liked0000000000000001", "oldtrack00000000000001", "newtrack00000000000001"}
        assert _offer(con, "liked0000000000000001")["outcome"] == "liked"
        assert _offer(con, "oldtrack00000000000001")["outcome"] == "unplayed"
        assert _offer(con, "newtrack00000000000001")["outcome"] == "unplayed"
        for sid in ids:
            assert _offer(con, sid)["removed_at"] is not None
        assert con.execute("SELECT COUNT(*) FROM offers WHERE removed_at IS NULL").fetchone()[0] == 0

    sp.on_remove = on_remove
    out = _clear(con, cfg, sp, no_penalty=True)
    assert out["no_penalty"] is True and out["no"] == 0 and out["feedback"] == 0
    assert out["liked"] == 1 and out["unplayed"] == 2 and out["recent_removed"] == 1
    assert _feedback(con) == []
    states = dict(con.execute("SELECT id, state FROM candidates"))
    assert states == {1: "liked", 2: "pool", 3: "pool"}
    assert con.execute("SELECT unplayed_expiries FROM candidates WHERE id=2").fetchone()[0] == 0
    sync = process_offers(con, sp, cfg, {"discoveries": PID, "to_rip": "torip"}, Index([]))
    assert sync["manual_removed"] == 0
    assert _offer(con, "liked0000000000000001")["outcome"] == "liked"
    assert _offer(con, "oldtrack00000000000001")["outcome"] == "unplayed"
    assert vectors.negatives(con, cfg) == []
    _, owners, _ = vectors.positives(con, cfg)
    assert "discovery:liked0000000000000001" in owners


def test_recent_remove_without_no_penalty_does_not_penalize_the_recent_track(tmp_path, monkeypatch):
    from haystack_music.sync import process_offers
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "oldtrack00000000000001", "Old", "Track", OLD)
    _add(con, 2, "newtrack00000000000001", "New", "Track", RECENT)
    vectors.put_deezer_vector(con, 1, np.ones(512, np.float32), np.zeros(1280), 1, 30.0)
    vectors.put_deezer_vector(con, 2, np.full(512, 0.2, np.float32), np.zeros(1280), 1, 30.0)
    sp = FakeSpotify([
        _item("oldtrack00000000000001", OLD),
        _item("newtrack00000000000001", RECENT),
    ])

    def on_remove(_ids):
        assert _offer(con, "oldtrack00000000000001")["outcome"] == "negative"
        assert _offer(con, "newtrack00000000000001")["outcome"] == "unplayed"
        assert _offer(con, "newtrack00000000000001")["removed_at"] is not None

    sp.on_remove = on_remove
    out = _clear(con, cfg, sp, recent="remove")
    assert out["no"] == 1 and out["unplayed"] == 1 and out["recent_removed"] == 1 and out["no_penalty"] is False
    sync = process_offers(con, sp, cfg, {"discoveries": PID, "to_rip": "torip"}, Index([]))
    assert sync["manual_removed"] == 0
    assert _offer(con, "newtrack00000000000001")["outcome"] == "unplayed"
    assert con.execute("SELECT state FROM candidates WHERE id=2").fetchone()[0] == "pool"
    negs = vectors.negatives(con, cfg)
    assert [n["candidate_id"] for n in negs] == [1]
    assert negs[0]["kind"] == "pass"


def test_sync_still_penalizes_a_hand_removal(tmp_path, monkeypatch):
    from haystack_music.sync import process_offers
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "hand000000000000000001", "Hand", "Removed", OLD)
    vectors.put_deezer_vector(con, 1, np.ones(512, np.float32), np.zeros(1280), 1, 30.0)
    sp = FakeSpotify([])  # already gone, offer still active: taken off by hand
    out = process_offers(con, sp, cfg, {"discoveries": PID, "to_rip": "torip"}, Index([]))
    assert out["manual_removed"] == 1
    assert _offer(con, "hand000000000000000001")["outcome"] == "manual_removed"
    assert con.execute("SELECT state FROM candidates WHERE id=1").fetchone()[0] == "negative"
    negs = vectors.negatives(con, cfg)
    assert len(negs) == 1 and negs[0]["candidate_id"] == 1 and negs[0]["kind"] == "removed"


def test_playlist_no_is_a_negative_signal(tmp_path):
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "oldtrack00000000000001", "Old", "Track", OLD)
    vec = np.ones(512, np.float32)
    vectors.put_deezer_vector(con, 1, vec, np.zeros(1280), 1, 30.0)
    sp = FakeSpotify([_item("oldtrack00000000000001", OLD)])
    _clear(con, cfg, sp)
    negs = vectors.negatives(con, cfg)
    assert len(negs) == 1
    assert negs[0]["candidate_id"] == 1 and negs[0]["strength"] == 1.0 and negs[0]["kind"] == "pass"


def test_empty_playlist_and_bad_config(tmp_path):
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    sp = FakeSpotify([])
    out = _clear(con, cfg, sp)
    assert out["items"] == 0 and out["removed"] == 0 and sp.remove_calls == 0
    cfg["batch"]["clear_recent"] = "drop"
    boom = FakeSpotify([_item("x", OLD)])

    def _no(*a, **k):
        raise AssertionError("spotify should not be called")

    boom.playlist_items = _no
    with pytest.raises(ValueError, match="clear_recent"):
        _clear(con, cfg, boom)


def test_config_override_keep(tmp_path):
    con = _db(tmp_path)
    cfg = _cfg(tmp_path, '[batch]\nclear_recent = "keep"\nclear_min_age_hours = 48\n')
    assert cfg["batch"]["size"] == 15
    _add(con, 1, "thirtyhours0000000001", "Mid", "Age", "2026-10-11T06:00:00Z")  # 30h: recent at 48, old at 24
    sp = FakeSpotify([_item("thirtyhours0000000001", "2026-10-11T06:00:00Z")])
    out = _clear(con, cfg, sp)
    assert out["recent"] == "keep" and out["min_age_hours"] == 48 and out["recent_kept"] == 1
    assert sp.remove_calls == 0 and _feedback(con) == []


def test_config_no_penalty_closes_old_unliked_as_unplayed(tmp_path):
    con = _db(tmp_path)
    cfg = _cfg(tmp_path, "[batch]\nclear_no_penalty = true\n")
    _add(con, 1, "oldtrack00000000000001", "Old", "Track", OLD)
    sp = FakeSpotify([_item("oldtrack00000000000001", OLD)])
    out = _clear(con, cfg, sp)
    assert out["no_penalty"] is True and out["no"] == 0 and out["unplayed"] == 1 and out["feedback"] == 0
    assert _offer(con, "oldtrack00000000000001")["outcome"] == "unplayed"
    assert con.execute("SELECT state FROM candidates WHERE id=1").fetchone()[0] == "pool"
    assert _feedback(con) == []


def test_defaults_are_24h_and_remove(tmp_path):
    cfg = config.load(tmp_path / "missing.toml")
    assert cfg["batch"]["clear_min_age_hours"] == 24
    assert cfg["batch"]["clear_recent"] == "remove"
    assert cfg["batch"]["clear_no_penalty"] is False


def test_cli_flags_and_units():
    p = build_parser()
    a = p.parse_args(["batch", "--dry-run", "--clear-recent", "keep", "--clear-min-age-hours", "48"])
    assert a.clear_recent == "keep" and a.clear_min_age_hours == 48.0 and a.dry_run
    a = p.parse_args(["clear-discoveries", "--dry-run"])
    assert a.dry_run and a.clear_recent is None and a.clear_min_age_hours is None and a.no_penalty is None
    a = p.parse_args(["clear-discoveries", "--no-penalty"])
    assert a.no_penalty is True
    a = p.parse_args(["first-batch", "--clear-recent", "remove", "--clear-min-age-hours", "24", "--no-penalty"])
    assert a.clear_recent == "remove" and a.clear_min_age_hours == 24.0 and a.no_penalty is True
    root = Path(__file__).resolve().parents[1] / "systemd"
    batch = (root / "haystack-batch.service").read_text()
    first = (root / "haystack-first-batch.service").read_text()
    for text, cmd in ((batch, "haystack batch"), (first, "haystack first-batch")):
        start = next(line for line in text.splitlines() if line.startswith("ExecStart="))
        assert cmd in start
        assert "--clear-recent remove" in start
        assert "--clear-min-age-hours 24" in start
    assert sum(1 for line in batch.splitlines() if line.startswith("ExecStart=")) == 1


def test_skipped_batch_does_not_clear(tmp_path, monkeypatch):
    import haystack_music.batch as batch
    monkeypatch.setattr(batch, "clear_discoveries", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cleared")))
    con = db.connect(tmp_path / "skip.db")
    out = batch.run(con, _cfg(tmp_path, f'[library]\nroot = "{tmp_path}"\n'), sp=object(), dry_run=False, force=False)
    assert out["skipped"] is True and "clear" not in out


def test_batch_clears_before_scoring(tmp_path, monkeypatch):
    import haystack_music.batch as batch
    order = []

    def fake_ensure(*a, **k):
        order.append("ensure")
        return {"discoveries": PID, "to_rip": "tr"}

    def fake_clear(con, cfg, sp, **kw):
        order.append("clear")
        assert kw["playlist_id"] == PID and kw["resolve_playlist"] is False
        assert kw["dry_run"] is True and kw["recent"] == "keep" and kw["min_age_hours"] == 3
        assert kw["no_penalty"] is True
        return {"removed": 0, "dry_run": True, "liked": 0, "no": 0}

    monkeypatch.setattr(batch, "ensure_playlists", fake_ensure)
    monkeypatch.setattr(batch, "clear_discoveries", fake_clear)
    monkeypatch.setattr(batch, "Deezer", lambda *a, **k: object())

    def fake_score(*a, **k):
        order.append("score")
        return {"scored": 0}

    monkeypatch.setattr(batch.vectors, "score_candidates", fake_score)
    con = db.connect(tmp_path / "batch.db")
    out = batch.run(con, _cfg(tmp_path), sp=object(), dry_run=True, clear_recent="keep",
                    clear_min_age_hours=3, no_penalty=True)
    assert order == ["ensure", "clear", "score"]
    assert out["clear"]["dry_run"] is True and out["dry_run"] is True
    assert out["variant_choice"]["used"] == out["variant"] == "raw"            # the weekly eval ran, and wrote nothing
    assert con.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] == 0 and db.kv_get(con, "scoring_variant") is None


def test_a_verdict_the_listener_gave_survives_the_clear(tmp_path):
    """A 4-star Rating whose heart is still pending (no user-library-modify yet) is not liked on Spotify, so the
    clear sees an unliked old track. It must not bury the Rating under a Pass."""
    from haystack_music import verdicts
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "oldtrack00000000000001", "Old", "Track", OLD)
    verdicts.record(con, "oldtrack00000000000001", "rating", 4, source="needle", queue="discoveries")
    sp = FakeSpotify([_item("oldtrack00000000000001", OLD)])
    out = _clear(con, cfg, sp)
    assert out["removed"] == 1 and out["feedback"] == 0
    cur = verdicts.current(con)["oldtrack00000000000001"]
    assert cur.kind == "rating" and cur.stars == 4
    assert con.execute("SELECT state FROM candidates WHERE id=1").fetchone()[0] == "liked"
    assert vectors.negatives(con, cfg) == []


def test_clear_pass_skips_its_own_spotify_writes(tmp_path):
    from haystack_music import verdicts
    con, cfg = _db(tmp_path), _cfg(tmp_path)
    _add(con, 1, "oldtrack00000000000001", "Old", "Track", OLD)
    _clear(con, cfg, FakeSpotify([_item("oldtrack00000000000001", OLD)]))
    assert verdicts.pending(con) == []
