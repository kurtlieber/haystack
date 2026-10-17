import numpy as np
import pytest

from haystack_music import config, db, feedback, vectors
from haystack_music.scoring import l2norm, negative_excess, negative_penalty
from haystack_music.textnorm import song_key


def test_excess_zero_below_margin_one_for_copy():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.2, 0.8, size=(1000, 1)).astype(np.float32)
    S = np.array([[0.3], [0.5], [np.quantile(ref, 0.95)], [1.0]], np.float32)
    e = negative_excess(S, ref, 0.95)[:, 0]
    assert e[0] == 0 and e[1] == 0 and e[2] == pytest.approx(0, abs=1e-5)
    assert e[3] == pytest.approx(1.0)
    m = np.quantile(ref, 0.95)
    mid = negative_excess(np.array([[(m + 1) / 2]], np.float32), ref, 0.95)[0, 0]
    assert mid == pytest.approx(0.5, abs=1e-4)


def test_penalty_off_strength_and_self_mask():
    E = np.array([[0.4, 0.3], [0.0, 0.0], [1.0, 0.0]], np.float32)
    p, j = negative_penalty(E, [0.5, 2.0], scale=2.0, w=0)
    assert not p.any() and (j == -1).all()                       # w = 0: off
    p, j = negative_penalty(E, [0.5, 2.0], scale=2.0, w=0.75)
    # row 0: the strong no (0.3 * 2 = 0.6) beats the nearer mild one (0.4 * 0.5 = 0.2)
    assert j[0] == 1 and p[0] == pytest.approx(0.75 * 2.0 * 0.6)
    assert p[1] == 0 and j[1] == -1                               # typical: untouched
    assert p[2] == pytest.approx(0.75 * 2.0 * 0.5)
    mask = np.array([[False, False], [False, False], [True, False]])
    p, j = negative_penalty(E, [0.5, 2.0], scale=2.0, w=0.75, self_mask=mask)
    assert p[2] == 0 and j[2] == -1                               # leave-one-out


def _db(tmp_path, monkeypatch, rng, dim=512):
    """Pool of 200 random previews; a liked direction L; one manual 'yes' (positive) near L and a strong 'no'
    W. Candidate 'copy' is a near-copy of W that also sits closest to L (top without the penalty)."""
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    con = db.connect(tmp_path / "t.db")
    L = l2norm(rng.normal(size=dim))
    W = l2norm(L + 0.9 * l2norm(rng.normal(size=dim)))          # the rejected track: liked-adjacent
    sL = np.abs(rng.normal(size=400)).astype(np.float32)
    sW = (sL + 0.6 * np.abs(rng.normal(size=400))).astype(np.float32)   # liked-adjacent style

    def add(i, artist, title, vec, style, state="pool"):
        con.execute("INSERT INTO candidates(id, deezer_id, artist, title, song_key, source, state, vector_status, "
                    "spotify_id) VALUES(?,?,?,?,?,?,?,?,?)", (i, i, artist, title, song_key(artist, title),
                                                               "deezer_artist_radio", state, "ok", f"sp{i}".ljust(22, "0")))
        vectors.put_deezer_vector(con, i, vec, np.zeros(1280), 10, 30.0, style)

    add(1, "Yes Artist", "Yes Song", L, sL)
    add(2, "wavve", "Afterglow (Spa)", W, sW)
    feedback.record(con, "yes", "Yes Artist", "Yes Song")
    feedback.record(con, "no", "wavve", "Afterglow (Spa)", "strong")
    # 'copy': a near-copy of W (embedding and style) -> similar to the like too
    add(10, "Copycat", "Copy", l2norm(W + 0.02 * rng.normal(size=dim)), sW + 0.01)
    # 'liked': close to L and not to W
    add(11, "Fan", "Close To Like", l2norm(L + 0.6 * l2norm(rng.normal(size=dim))), sL + 0.05)
    for i in range(12, 212):
        a = rng.uniform(0, 1.0)                                  # spread of similarity to the like
        add(i, f"A{i}", f"T{i}", l2norm(l2norm(rng.normal(size=dim)) + a * L),
            np.abs(rng.normal(size=400)) * 0.6 + a * sL)
    con.commit()
    return con


@pytest.mark.parametrize("variant", ["raw", "topk_mean", "centred", "genre", "genre_blend"])
def test_score_candidates_penalises_copy_of_strong_no(tmp_path, monkeypatch, variant):
    con = _db(tmp_path, monkeypatch, np.random.default_rng(3))
    cfg = config.load(tmp_path / "none.toml")

    def ranks(w):
        cfg["scoring"]["negative_penalty"]["w"] = w
        out = vectors.score_candidates(con, cfg, variant=variant)
        sc = dict(con.execute("SELECT id, score FROM candidates WHERE state='pool'").fetchall())
        order = sorted(sc, key=lambda k: -sc[k])
        return out, sc, {cid: order.index(cid) + 1 for cid in order}

    off, s_off, r_off = ranks(0)
    assert off["negative_penalty"]["penalised"] == 0
    assert con.execute("SELECT COUNT(*) FROM candidates WHERE neg_penalty IS NOT NULL").fetchone()[0] == 0
    on, s_on, r_on = ranks(0.75)
    assert on["negative_penalty"]["negatives"] == 1
    assert r_off[10] <= 5 and r_on[10] > 15                      # the near-copy of a strong no drops out
    assert r_off[11] == 1 and r_on[11] == 1                      # a strong match to the likes still wins
    assert r_on[10] > r_off[10]
    row = con.execute("SELECT neg_penalty, nearest_negative FROM candidates WHERE id=10").fetchone()
    assert row[0] > 0 and row[1] == "wavve - Afterglow (Spa)"
    moved = [abs(r_on[c] - r_off[c]) for c in range(12, 212)]
    assert np.median(moved) <= 2                                 # typical candidates barely move
    # judged tracks are never in the pool
    assert {1, 2}.isdisjoint(s_on)


def test_negatives_include_played_expiries(tmp_path, monkeypatch):
    con = _db(tmp_path, monkeypatch, np.random.default_rng(2))
    con.execute("INSERT INTO offers(candidate_id, spotify_id, added_at, outcome) VALUES(11, 'sp11', '2026-10-01', "
                "'negative')")
    con.execute("UPDATE candidates SET state='negative' WHERE id=11")
    cfg = config.load(tmp_path / "none.toml")
    cfg["scoring"]["negative_penalty"]["weight_expired"] = 0.7
    negs = {n["candidate_id"]: (n["strength"], n["kind"]) for n in vectors.negatives(con, cfg)}
    assert negs == {2: (2.0, "dislike"), 11: (0.7, "expired_played")}
