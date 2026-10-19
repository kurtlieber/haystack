import random
from collections import Counter
from dataclasses import replace

import numpy as np

from haystack_music.scoring import (Pick, adapt_exploration, can_retire, like_rates, nearest_positive,
                                   select_batch)
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


def _old_select_batch(pool, size, explore_fraction=0.2, band=(0.4, 0.75), max_per_artist=1, exclude_artists=None,
                      rng=None):
    """select_batch before ADR 0010, kept here to pin down that diversity 0 with no caps changes nothing."""
    rng = rng or random.Random()
    counts = {a: max_per_artist for a in (exclude_artists or ())}
    ranked = sorted(pool, key=lambda p: p.score, reverse=True)
    n = len(ranked)
    n_explore = int(round(size * explore_fraction))
    chosen, chosen_ids = [], set()

    def ok(p):
        return p.cand_id not in chosen_ids and counts.get(p.artist_key, 0) < max_per_artist

    def take(p, explore):
        chosen.append(replace(p, exploration=explore))
        chosen_ids.add(p.cand_id)
        counts[p.artist_key] = counts.get(p.artist_key, 0) + 1

    if n_explore and n:
        lo, hi = int(n * (1 - band[1])), int(n * (1 - band[0]))
        mid = ranked[lo:max(hi, lo + 1)]
        rng.shuffle(mid)
        for p in mid:
            if len([c for c in chosen if c.exploration]) >= n_explore:
                break
            if ok(p):
                take(p, True)
    for p in ranked:
        if len(chosen) >= size:
            break
        if ok(p):
            take(p, False)
    chosen.sort(key=lambda p: (p.exploration, -p.score))
    return chosen


def test_diversity_zero_and_no_caps_is_the_old_selection():
    pool = [replace(p, style=p.cand_id % 2, nearest=f"like:{p.cand_id % 3}") for p in _pool(200, 60)]
    for seed in range(5):
        old = _old_select_batch(pool, 15, 0.2, (0.4, 0.75), 1, {"a3"}, random.Random(seed))
        new = select_batch(pool, 15, 0.2, (0.4, 0.75), 1, {"a3"}, random.Random(seed))
        assert new == old


def _unit(*xs):
    v = np.zeros(4, np.float32)
    v[: len(xs)] = xs
    return v / np.linalg.norm(v)


def test_mmr_picks_a_less_similar_second_track():
    """Two near-copies at the top and a slightly lower track pointing elsewhere: by score alone the copies both go
    in; with diversity the second pick is the different one."""
    pool = [Pick(1, "a1", 0.95, "s"), Pick(2, "a2", 0.94, "s"), Pick(3, "a3", 0.90, "s")] + \
        [Pick(i, f"a{i}", 0.5 - i / 100, "s") for i in range(4, 20)]
    vecs = {1: _unit(1, 0), 2: _unit(0.99, 0.14), 3: _unit(0, 1)} | {i: _unit(0, 0, 1) for i in range(4, 20)}
    plain = select_batch(pool, 2, 0.0, vectors=vecs, diversity=0.0)
    assert [p.cand_id for p in plain] == [1, 2]
    spread = select_batch(pool, 2, 0.0, vectors=vecs, diversity=0.5)
    assert [p.cand_id for p in spread] == [1, 3]


def test_mmr_never_marks_down_a_dissimilar_or_vectorless_track():
    pool = [Pick(1, "a1", 0.9, "s"), Pick(2, "a2", 0.8, "s"), Pick(3, "a3", 0.7, "s")]
    vecs = {1: _unit(1, 0), 2: _unit(-1, 0)}                        # 2 points away (cosine -1: no bonus either)
    assert [p.cand_id for p in select_batch(pool, 3, 0.0, vectors=vecs, diversity=5.0)] == [1, 2, 3]


def test_style_share_caps_the_top_picks_and_leaves_exploration_alone():
    pool = [replace(p, style=0 if p.cand_id < 60 else p.cand_id % 7 + 1) for p in _pool(100, 100)]
    picks = select_batch(pool, 10, 0.0, max_style_share=0.4)
    assert len(picks) == 10 and sum(p.style == 0 for p in picks) == 4     # the top 60 are all style 0
    explore = select_batch(pool, 10, 0.5, band=(0.45, 0.99), max_style_share=0.4, rng=random.Random(2))
    assert sum(p.style == 0 for p in explore if not p.exploration) <= 2   # 40 % of the 5 top picks
    no_style = [replace(p, style=None) for p in pool]
    assert [p.cand_id for p in select_batch(no_style, 10, 0.0, max_style_share=0.4)] == list(range(10))


def test_at_most_max_per_nearest_picks_share_a_liked_song():
    pool = [replace(p, nearest="like:hub" if p.cand_id < 50 else f"like:{p.cand_id}") for p in _pool(100, 100)]
    picks = select_batch(pool, 15, 0.2, max_per_nearest=2, rng=random.Random(4))
    assert len(picks) == 15 and sum(p.nearest == "like:hub" for p in picks) == 2
    assert sum(p.nearest == "like:hub" for p in select_batch(pool, 15, 0.0)) == 15     # 0: no cap


def test_caps_give_way_when_the_pool_is_too_narrow():
    pool = [replace(p, style=0, nearest="like:hub") for p in _pool(20, 20)]
    picks = select_batch(pool, 10, 0.0, max_style_share=0.4, max_per_nearest=2)
    assert [p.cand_id for p in picks] == list(range(10))
    # one style but many liked songs: the style cap gives way first, and the nearest cap still holds
    pool = [replace(p, style=0, nearest=f"like:{p.cand_id // 3}") for p in _pool(30, 30)]
    picks = select_batch(pool, 10, 0.0, max_style_share=0.4, max_per_nearest=2)
    assert len(picks) == 10 and max(Counter(p.nearest for p in picks).values()) == 2


def test_earlier_picks_count_toward_the_caps():
    pool = [replace(p, style=0, nearest="like:hub") for p in _pool(40, 40)] + \
        [Pick(i, f"b{i}", 0.1, "s", style=i % 5 + 1, nearest=f"like:{i}") for i in range(100, 110)]
    taken = [Pick(500, "z1", 1.0, "s", style=0, nearest="like:hub"), Pick(501, "z2", 1.0, "s", style=0,
                                                                         nearest="like:hub")]
    picks = select_batch(pool, 8, 0.0, taken=taken, max_style_share=0.4, max_per_nearest=2)
    assert len(picks) == 8 and not [p for p in picks if p.nearest == "like:hub"]    # the hub already has two
    picks = select_batch(pool, 8, 0.0, taken=taken[1:], max_style_share=0.4)
    assert sum(p.style == 0 for p in picks) == 2      # 3 of 9 top picks (40 %, rounded down), and 1 already is


def test_exploration_share_adapts_and_clips():
    prior = adapt_exploration(0.2, (3, 7), (1, 30))
    assert prior["fraction"] == 0.2 and prior["why"].startswith("prior")     # too few judged exploration offers
    up = adapt_exploration(0.2, (4, 10), (3, 30))                           # 0.4 vs 0.1: + 0.5 * 0.3
    assert up["fraction"] == 0.35 and up["explore_rate"] == 0.4 and up["exploit_judged"] == 30
    assert adapt_exploration(0.2, (10, 10), (0, 30))["fraction"] == 0.4    # clipped
    assert adapt_exploration(0.2, (0, 10), (20, 30))["fraction"] == 0.1
    assert adapt_exploration(0.2, (5, 10), (15, 30))["fraction"] == 0.2    # the same rate: the prior
