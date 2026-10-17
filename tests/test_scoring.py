import random

import numpy as np

from haystack_music.scoring import Pick, can_retire, like_rates, nearest_positive, select_batch
from haystack_music.windows import window_starts


def test_nearest_positive_best_window_wins():
    pos = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    cands = np.array([[0.1, 0.9, 0.0], [0.0, 0.0, 2.0]], np.float32)
    s, raw, idx = nearest_positive(cands, pos)
    assert list(idx) == [1, 2]
    assert abs(raw[1] - 1.0) < 1e-6


def test_bonus_is_additive():
    pos = np.array([[1, 0], [0.9, 0.1]], np.float32)
    s, raw, idx = nearest_positive(np.array([[1, 0]], np.float32), pos, np.array([0, 0.1], np.float32))
    assert idx[0] == 1 and s[0] > raw[0]


def _pool(n=100, artists=100):
    return [Pick(i, f"a{i % artists}", 1 - i / n, "src") for i in range(n)]


def test_select_batch_size_explore_and_artist_cap():
    picks = select_batch(_pool(100, 30), 15, 0.2, (0.4, 0.75), 1, rng=random.Random(1))
    assert len(picks) == 15
    assert len({p.artist_key for p in picks}) == 15
    explore = [p for p in picks if p.exploration]
    assert len(explore) == 3
    # exploration picks come from the mid band, not the top
    assert all(0.25 <= 1 - p.score <= 0.6 for p in explore)


def test_select_batch_respects_excluded_artists():
    picks = select_batch(_pool(10, 10), 5, 0.0, exclude_artists={"a0", "a1"})
    assert "a0" not in {p.artist_key for p in picks} and "a1" not in {p.artist_key for p in picks}


def test_like_rates_and_retirement_threshold():
    offers = [{"source": "x", "outcome": "liked", "played": 1}] * 3 + [{"source": "x", "outcome": "negative", "played": 1}] * 7 \
        + [{"source": "y", "outcome": None, "played": 0}]
    r = like_rates(offers)
    assert r["per_source"]["x"]["like_rate"] == 0.3
    assert r["overall"]["offered"] == 11
    assert r["per_source"]["y"]["like_rate"] is None
    assert not can_retire(r["per_source"]["x"], 30)
    assert can_retire({"offered": 30}, 30)


def test_window_starts():
    w = window_starts(300.0)
    assert len(w) == 9 and w[0] == 15.0 and w[-1] <= 270.0
    assert window_starts(20.0) == [0.0]
    assert len(window_starts(40.0)) < 9


def test_embedding_guard(tmp_path):
    from haystack_music import config, db
    from haystack_music.batch import embedding_progress
    con = db.connect(tmp_path / "t.db")
    cfg = config.load(tmp_path / "none.toml")
    cfg["library"]["root"] = "/m"
    for i in range(100):
        con.execute("INSERT INTO local_tracks(path, size, content_hash) VALUES(?, 1, ?)", (f"/m/{i}", f"h{i}"))
    for i in range(97):
        con.execute("INSERT INTO local_embeddings(content_hash, status, attempts) VALUES(?, 'ok', 1)", (f"h{i}",))
    ok, msg, prog = embedding_progress(con, cfg)
    assert not ok and "97/100" in msg
    con.execute("INSERT INTO local_embeddings(content_hash, status, attempts) VALUES('h97', 'ok', 1)")
    ok, msg, prog = embedding_progress(con, cfg)
    assert ok and prog["embedded"] == 98
