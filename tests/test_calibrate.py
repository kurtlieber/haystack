"""The Calibrate queue (ADR 0015): its bands, its order, the picks it makes from the pool and the hourly top-up.
Synthetic data; no Spotify, no Deezer."""
import hashlib
import json
import random
import sqlite3
from collections import Counter

import pytest

from haystack_music import batch, calibrate, config, db, verdicts
from test_rerank import _world, sid

BANDS = calibrate.bands(config.load(None))


def _rows(per_band=40, start=0, **kw):
    """per_band rows in each default band, one artist each, no chance and no disagreement unless given."""
    out, i = [], start
    for name, lo, hi, _ in BANDS:
        for j in range(per_band):
            out.append({"id": i, "artist_key": f"a{i}", "source": "src_a",
                        "percentile": lo + (hi - lo) * j / per_band, "rerank_p": None, "disagreement": None, **kw})
            i += 1
    return out


def _digest(path) -> str:
    con = sqlite3.connect(path)
    try:
        return hashlib.sha256("\n".join(con.iterdump()).encode()).hexdigest()
    finally:
        con.close()


# ---------------- bands, order and reasons ----------------

def test_each_percentile_falls_in_one_band():
    assert [b[0] for b in BANDS] == ["50-75", "75-90", "90-97", "97-100"]
    assert sum(b[3] for b in BANDS) == pytest.approx(1.0)
    got = {p: calibrate.band_of(p, BANDS) for p in (49.99, 50, 74.99, 75, 89.9, 90, 96.99, 97, 99.5, 100)}
    assert got == {49.99: None, 50: "50-75", 74.99: "50-75", 75: "75-90", 89.9: "75-90", 90: "90-97",
                   96.99: "90-97", 97: "97-100", 99.5: "97-100", 100: "97-100"}
    assert calibrate.band_of(None, BANDS) is None and calibrate.band_of(float("nan"), BANDS) is None


def test_order_puts_even_odds_first_then_disagreement():
    rows = [{"id": 1, "rerank_p": 0.9, "disagreement": 30.0}, {"id": 2, "rerank_p": None, "disagreement": 5.0},
            {"id": 3, "rerank_p": 0.52, "disagreement": None}, {"id": 4, "rerank_p": None, "disagreement": None},
            {"id": 5, "rerank_p": None, "disagreement": 20.0}, {"id": 6, "rerank_p": 0.4, "disagreement": 0.0}]
    for seed in range(5):
        assert [r["id"] for r in calibrate.order(rows, random.Random(seed))] == [3, 6, 1, 5, 2, 4]


def test_the_reason_says_why_haystack_is_unsure():
    R = calibrate.REASONS
    assert calibrate.reason({"rerank_p": 0.62, "disagreement": 40.0}, False) == R["odds"]
    assert calibrate.reason({"rerank_p": 0.9, "disagreement": 12.0}, False) == R["split"]
    assert calibrate.reason({"rerank_p": None, "disagreement": 10.0}, True) == R["split"]
    assert calibrate.reason({"rerank_p": 0.9, "disagreement": 2.0}, True) == R["close"]
    assert calibrate.reason({"rerank_p": None, "disagreement": None}, False) == R["far"]


# ---------------- select ----------------

def test_select_fills_the_bands_by_their_share_of_the_week():
    picks = calibrate.select(_rows(), 30, BANDS, random.Random(1))
    assert Counter(p["band"] for p in picks) == {"50-75": 6, "75-90": 9, "90-97": 9, "97-100": 6}
    assert all(calibrate.band_of(p["percentile"], BANDS) == p["band"] for p in picks)
    # six already picked this week in the lowest band: the rest of the week goes to the others
    picks = calibrate.select(_rows(), 24, BANDS, random.Random(1), have={"50-75": 6})
    assert Counter(p["band"] for p in picks) == {"75-90": 9, "90-97": 9, "97-100": 6}
    # a band with nothing left is skipped and the picks go to the others
    rows = [r for r in _rows() if r["percentile"] < 97]
    picks = calibrate.select(rows, 30, BANDS, random.Random(1))
    assert len(picks) == 30 and "97-100" not in {p["band"] for p in picks}
    # below the lowest band: never asked about
    assert calibrate.select([{**_rows()[0], "percentile": 20.0}], 5, BANDS, random.Random(1)) == []


def test_select_takes_the_most_uncertain_first_in_each_band():
    rows = _rows(per_band=10)
    for r in rows:
        r["rerank_p"] = 0.95 if r["id"] % 10 else 0.5         # one even-odds row per band
    picks = calibrate.select(rows, 4, BANDS, random.Random(2))
    assert sorted(p["id"] for p in picks) == [0, 10, 20, 30]
    assert {p["reason"] for p in picks} == {calibrate.REASONS["odds"]}


def test_select_takes_one_per_artist_and_caps_the_charts():
    rows = _rows(per_band=10)
    for r in rows:
        r["artist_key"] = f"a{r['id'] % 5}"
    picks = calibrate.select(rows, 30, BANDS, random.Random(3), artists={"a0"})
    assert sorted(p["artist_key"] for p in picks) == ["a1", "a2", "a3", "a4"]
    chart = _rows(per_band=10, source="deezer_chart")
    picks = calibrate.select(chart + _rows(per_band=10, start=100), 30, BANDS, random.Random(3),
                             chart_sources={"deezer_chart"}, max_chart=6, chart_have=2)
    assert len(picks) == 30 and sum(p["source"] == "deezer_chart" for p in picks) <= 4


# ---------------- plan: the pool ----------------

def test_plan_skips_disliked_artists_liked_owned_and_judged_tracks(tmp_path, monkeypatch):
    con, cfg, path = _world(tmp_path, monkeypatch)
    con.execute("UPDATE candidates SET artist_key='artist lab3' WHERE id=1000")        # lab3 has a Dislike
    con.execute("UPDATE candidates SET spotify_id=? WHERE id=1001", (sid("like0"),))    # already liked
    con.execute("INSERT INTO local_tracks(path, size, content_hash, present, spotify_id) VALUES('x.flac', 1, 'h', 1, ?)",
                (sid("pool2"),))                                                     # in the library
    verdicts.record(con, sid("pool3"), "pass", None)                                    # judged already
    con.execute("UPDATE candidates SET source='src_old' WHERE id=1004")
    cfg["harvest"]["retired_sources"] = ["src_old"]
    con.commit()
    before = _digest(path)
    out = calibrate.plan(db.connect_readonly(path), cfg, random.Random(0), n=150)
    assert _digest(path) == before                                                      # no writes
    assert out["enabled"] and out["scored"] == 149 and out["eligible"] == 145      # the Pass moved pool3 on
    assert out["fitted"] and out["rerank"]["n_labels"] == 61 and out["need"] == 30       # and is a label
    ids = [p["id"] for p in out["picks"]]
    assert not set(ids) & {1000, 1001, 1002, 1003, 1004}
    assert len({p["artist_key"] for p in out["picks"]}) == len(ids)
    assert all(p["percentile"] >= 50 and p["rerank_p"] is not None for p in out["picks"])
    assert {p["reason"] for p in out["picks"]} <= set(calibrate.REASONS.values())
    assert sum(out["by_band"].values()) == len(ids)
    # by default it shows what the queue takes now
    assert len(calibrate.plan(db.connect_readonly(path), cfg, random.Random(0))["picks"]) == 30


def test_plan_on_a_fresh_install_picks_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    con = db.connect(tmp_path / "t.db")
    out = calibrate.plan(con, config.load(tmp_path / "none.toml"), random.Random(0))
    assert out["picks"] == [] and out["eligible"] == 0 and not out["fitted"] and out["outstanding"] == 0


def test_state_tolerates_a_database_from_before_the_queue(tmp_path):
    con = sqlite3.connect(tmp_path / "old.db")
    con.row_factory = sqlite3.Row
    st = calibrate.state(con, config.load(tmp_path / "none.toml"))
    assert st["outstanding"] == 0 and st["need"] == 30 and calibrate.picked(con) == {}


# ---------------- top-up ----------------

class FakeSpotify:
    def __init__(self, liked=()):
        self.liked = set(liked)
        self.searches = []

    def library_contains(self, ids):
        return {i: i in self.liked for i in ids}

    def search_tracks(self, q, limit=10):
        self.searches.append(q)
        return []

    def track(self, s):
        return {"id": s, "name": "Some Song", "artists": [{"name": "Someone"}], "duration_ms": 200000,
                "album": {"name": "x", "album_type": "single", "release_date": "2020",
                          "images": [{"url": "https://i.example/a.jpg", "width": 640}]}}


class FakeDeezer:
    def track(self, did):
        return {}


def _picks(con):
    return con.execute("SELECT * FROM calibrations ORDER BY id").fetchall()


def test_top_up_fills_the_queue_and_then_waits_for_the_week(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    out = calibrate.top_up(con, cfg, FakeSpotify(), FakeDeezer(), random.Random(0))
    assert out["picked"] == 30 and out["outstanding"] == 30 and sum(out["by_band"].values()) == 30
    rows = _picks(con)
    assert len(rows) == 30 and {r["week"] for r in rows} == {calibrate.week()}
    assert all(r["judged_at"] is None and r["band"] and r["reason"] in calibrate.REASONS.values() for r in rows)
    assert all(isinstance(json.loads(r["rerank_reason"]), list) for r in rows)        # the model was fitted
    cands = {r["id"]: r for r in con.execute("SELECT id, state, artist_key FROM candidates")}
    assert all(cands[r["candidate_id"]]["state"] == "pool" for r in rows)              # a pick stays in the pool
    assert len({cands[r["candidate_id"]]["artist_key"] for r in rows}) == 30
    assert con.execute("SELECT COUNT(*) FROM spotify_tracks WHERE image_url IS NOT NULL").fetchone()[0] == 30
    # full: the next hour picks nothing
    assert calibrate.top_up(con, cfg, FakeSpotify(), FakeDeezer(), random.Random(0))["picked"] == 0
    # ten judged (a Verdict closes its pick): ten open places, but the week already had its 30
    for r in rows[:10]:
        verdicts.record(con, r["spotify_id"], "pass", None)
    assert con.execute("SELECT COUNT(*) FROM calibrations WHERE judged_at IS NOT NULL").fetchone()[0] == 10
    st = calibrate.state(con, cfg)
    assert st["outstanding"] == 20 and st["this_week"] == 30 and st["need"] == 0
    assert calibrate.top_up(con, cfg, FakeSpotify(), FakeDeezer(), random.Random(0))["picked"] == 0
    # a new week: ten more
    cfg["calibrate"]["per_week"] = 30
    con.execute("UPDATE calibrations SET week='2000-W01'")
    out = calibrate.top_up(con, cfg, FakeSpotify(), FakeDeezer(), random.Random(0))
    assert out["picked"] == 10 and out["outstanding"] == 30
    # and never the same track twice
    assert len({r["spotify_id"] for r in _picks(con)}) == 40


def test_top_up_skips_unmapped_and_known_tracks(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    cfg["calibrate"].update(outstanding=10, per_week=10)
    con.execute("UPDATE candidates SET spotify_id=NULL WHERE state='pool' AND id % 2 = 0")
    liked = {sid(f"pool{j}") for j in range(1, 150, 4)}
    sp = FakeSpotify(liked)
    out = calibrate.top_up(con, cfg, sp, FakeDeezer(), random.Random(0))
    assert out["picked"] == 10 and out["unmapped"] > 0 and out["known"] > 0 and sp.searches
    rows = _picks(con)
    assert all(r["candidate_id"] % 2 for r in rows) and not {r["spotify_id"] for r in rows} & liked
    st = dict(con.execute("SELECT spotify_status, COUNT(*) FROM candidates WHERE state='pool' AND id % 2 = 0 "
                          "AND spotify_status IS NOT NULL GROUP BY 1").fetchall())
    assert st == {"not_found": out["unmapped"]}
    assert con.execute("SELECT COUNT(*) FROM candidates WHERE exclude_reason='known on spotify'"
                       ).fetchone()[0] == out["known"]


def test_top_up_does_nothing_when_off_or_on_a_fresh_install(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch, n_lab=6, n_pool=20)
    cfg["calibrate"]["enabled"] = False
    assert calibrate.top_up(con, cfg, FakeSpotify(), FakeDeezer())["picked"] == 0 and _picks(con) == []
    fresh = db.connect(tmp_path / "fresh.db")
    out = calibrate.top_up(fresh, config.load(tmp_path / "none.toml"), FakeSpotify(), FakeDeezer())
    assert out["picked"] == 0 and _picks(fresh) == []


def test_the_batch_never_offers_a_calibrate_pick(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    allowed = {f"Artist pool{j}" for j in range(20)}
    for j in range(20, 150):
        con.execute("INSERT INTO calibrations(candidate_id, spotify_id, picked_at, week) VALUES(?,?,?,?)",
                    (1000 + j, sid(f"pool{j}"), db.now(), calibrate.week()))
    out = batch.plan(con, cfg, rng=random.Random(1))
    assert out["picked"] and {t["artist"] for t in out["tracks"]} <= allowed
