import numpy as np
import pytest

from haystack_music import config, db, vectors, verdicts
from haystack_music.scoring import l2norm


def sid(n: str) -> str:
    return n.ljust(22, "0")


def _db(tmp_path, monkeypatch):
    """Three liked songs with preview vectors (one unrated, one rated 5 stars, one passed), a 2-star discovery
    that is not in Liked Songs, and a liked discovery."""
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    con = db.connect(tmp_path / "t.db")
    rng = np.random.default_rng(0)
    for i, name in enumerate(("unrated", "five", "passed"), start=1):
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(name),))
        con.execute("INSERT INTO liked_map(spotify_id, use_local, deezer_id, vector_status) VALUES(?, 0, ?, 'ok')",
                    (sid(name), i))
        vectors.put_deezer_vector(con, i, l2norm(rng.normal(size=512)), np.zeros(1280), 10, 30.0,
                                  np.abs(rng.normal(size=400)))
    for cid, name, state in ((10, "two", "offered"), (11, "discliked", "liked")):
        con.execute("INSERT INTO candidates(id, deezer_id, artist, title, song_key, source, state, vector_status, "
                    "spotify_id) VALUES(?,?,?,?,?,?,?,?,?)", (cid, cid, "A", name, name, "x", state, "ok", sid(name)))
        vectors.put_deezer_vector(con, cid, l2norm(rng.normal(size=512)), np.zeros(1280), 10, 30.0,
                                  np.abs(rng.normal(size=400)))
    con.execute("INSERT INTO offers(candidate_id, spotify_id, added_at, outcome) VALUES(11, ?, 'x', 'liked')",
                (sid("discliked"),))
    verdicts.record(con, sid("five"), "rating", 5)
    verdicts.record(con, sid("passed"), "pass")
    verdicts.record(con, sid("two"), "rating", 2)
    return con


def test_positive_bonuses_follow_the_verdicts(tmp_path, monkeypatch):
    con = _db(tmp_path, monkeypatch)
    cfg = config.load(tmp_path / "none.toml")
    pos, owners, bonus, stars = vectors.positives_with_stars(con, cfg)
    by = dict(zip(owners, bonus))
    assert by[f"like:{sid('unrated')}"] == pytest.approx(0.005)            # Unrated like = 3 stars
    assert by[f"like:{sid('five')}"] == pytest.approx(0.01)
    assert f"like:{sid('passed')}" not in by                                # a Pass is never a positive
    assert by["manual:10"] == pytest.approx(0.0025)                         # 2 stars, not in Liked Songs
    assert by[f"discovery:{sid('discliked')}"] == pytest.approx(0.005)
    assert len(owners) == len(set(owners)) == 4
    assert dict(zip(owners, stars))[f"like:{sid('five')}"] == 5


def test_an_empty_star_bonus_counts_stars_in_pool_sd(tmp_path, monkeypatch):
    con = _db(tmp_path, monkeypatch)
    p = tmp_path / "c.toml"
    p.write_text("[scoring.verdicts]\nstar_bonus = []\n")
    pos, owners, bonus = vectors.positives(con, config.load(p))
    by = dict(zip(owners, bonus))
    assert by[f"like:{sid('unrated')}"] == pytest.approx(0.0)              # Unrated like = 3 stars = neutral
    assert by[f"like:{sid('five')}"] == pytest.approx(0.5)                 # 2 stars above 3, in pool-score sd
    assert by["manual:10"] == pytest.approx(-0.25)


def test_genre_positives_carry_the_same_bonus(tmp_path, monkeypatch):
    con = _db(tmp_path, monkeypatch)
    labels, st, bonus = vectors.liked_styles(con, config.load(tmp_path / "none.toml"))
    by = dict(zip(labels, bonus))
    assert set(by) == {sid("unrated"), sid("five"), sid("two"), sid("discliked")}
    assert by[sid("five")] == pytest.approx(0.01) and by[sid("two")] == pytest.approx(0.0025)
    assert st.shape == (4, 400)


def test_a_star_is_worth_the_same_share_of_any_pools_spread():
    """star_unit is the sd of the pool's nearest-positive similarity, so the sd bonus scales with the spread."""
    cfg = config.load(config.CONFIG_FILE.parent / "does-not-exist.toml")
    assert vectors.star_unit(cfg, np.ones((3, 4)), np.ones((2, 4))) == 1.0      # fixed offsets: no scaling
    cfg["scoring"]["verdicts"]["star_bonus"] = []
    rng = np.random.default_rng(1)
    pos = l2norm(rng.normal(size=(20, 16)))
    cands = l2norm(pos[rng.integers(0, 20, 200)] + rng.normal(scale=0.5, size=(200, 16)))
    unit = vectors.star_unit(cfg, cands, pos)
    assert unit == pytest.approx(float(np.std(vectors.nearest_positive(cands, pos)[0])))
    assert 0 < unit < 1


def test_a_pass_is_a_negative_at_normal_strength(tmp_path, monkeypatch):
    con = _db(tmp_path, monkeypatch)
    negs = vectors.negatives(con, config.load(tmp_path / "none.toml"))
    assert [(n["kind"], n["strength"]) for n in negs] == [("pass", 1.0)]
