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
    pos, owners, bonus = vectors.positives(con, cfg)
    by = dict(zip(owners, bonus))
    assert by[f"like:{sid('unrated')}"] == pytest.approx(0.005)            # Unrated like = 3 stars
    assert by[f"like:{sid('five')}"] == pytest.approx(0.01)
    assert f"like:{sid('passed')}" not in by                                # a Pass is never a positive
    assert by["manual:10"] == pytest.approx(0.0025)                         # 2 stars, not in Liked Songs
    assert by[f"discovery:{sid('discliked')}"] == pytest.approx(0.005)
    assert len(owners) == len(set(owners)) == 4


def test_genre_positives_carry_the_same_bonus(tmp_path, monkeypatch):
    con = _db(tmp_path, monkeypatch)
    labels, st, bonus = vectors.liked_styles(con, config.load(tmp_path / "none.toml"))
    by = dict(zip(labels, bonus))
    assert set(by) == {sid("unrated"), sid("five"), sid("two"), sid("discliked")}
    assert by[sid("five")] == pytest.approx(0.01) and by[sid("two")] == pytest.approx(0.0025)
    assert st.shape == (4, 400)


def test_a_pass_is_a_negative_at_normal_strength(tmp_path, monkeypatch):
    con = _db(tmp_path, monkeypatch)
    negs = vectors.negatives(con, config.load(tmp_path / "none.toml"))
    assert [(n["kind"], n["strength"]) for n in negs] == [("pass", 1.0)]
