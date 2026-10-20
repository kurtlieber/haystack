"""The second stage (ADR 0011): labels, the logistic re-ranker, its gate, its reasons and the order it gives the
top of a batch. Synthetic data; no Spotify, no Deezer."""
import json
import random
import sqlite3

import numpy as np
import pytest

from haystack_music import baseline, batch, cli, config, db, rerank, traits, vectors, verdicts
from haystack_music.scoring import VARIANTS, Pick, l2norm, select_batch


def sid(n: str) -> str:
    return n.ljust(22, "z")              # "z", not "0": like1 and like10 stay apart


def _world(tmp_path, monkeypatch, n_lab=60, n_pool=150, seed=0):
    """20 liked songs near one direction, with a mid valence (the Baseline's usual range is about 0.48-0.53).
    n_lab judged offers whose verdicts follow valence alone (bright: a yes, dark: a no) while their similarity to
    the likes is random, so stage 1 can't tell them apart and the model can. A pool of n_pool with random valence."""
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    path = tmp_path / "t.db"
    con = db.connect(path)
    rng = np.random.default_rng(seed)
    L = l2norm(rng.normal(size=512))
    sL = np.abs(rng.normal(size=400))
    did = iter(range(1000, 100000))

    def vec(a, valence):
        d = next(did)
        vectors.put_deezer_vector(con, d, l2norm(a * L + l2norm(rng.normal(size=512))), np.zeros(1280), 10, 30.0,
                                  np.abs(rng.normal(size=400)) * 0.6 + a * sL)
        traits.put(con, d, {"valence": valence})
        return d

    def cand(cid, name, state, a, valence, source="src_a"):
        con.execute("""INSERT INTO candidates(id, deezer_id, artist, artist_key, title, song_key, source, state,
                       vector_status, spotify_id) VALUES(?,?,?,?,?,?,?,?, 'ok', ?)""",
                    (cid, vec(a, valence), f"Artist {name}", f"artist {name}", f"Title {name}", f"song {name}",
                     source, state, sid(name)))

    for i in range(20):
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(f"like{i}"),))
        con.execute("INSERT INTO liked_map(spotify_id, use_local, deezer_id, vector_status) VALUES(?, 0, ?, 'ok')",
                    (sid(f"like{i}"), vec(1.5, 0.475 + 0.05 * rng.random())))
    con.execute("INSERT INTO batches(id, created_at, size) VALUES(1, '2026-01-05T00:00:00Z', ?)", (n_lab,))
    no_kinds = (("pass", None), ("dislike", None), ("rating", 2))
    for i in range(n_lab):
        name = f"lab{i}"
        yes = i % 2 == 0
        cand(i + 1, name, "offered", rng.uniform(0.3, 1.5), (0.8 if yes else 0.1) + 0.1 * rng.random(),
             source="src_a" if i % 3 else "src_b")
        con.execute("INSERT INTO offers(candidate_id, spotify_id, batch_id, added_at, outcome) "
                    "VALUES(?,?,1,'2026-01-05','unplayed')", (i + 1, sid(name)))
        kind, stars = ("rating", 3 + (i // 2) % 2) if yes else no_kinds[(i // 2) % 3]
        verdicts.record(con, sid(name), kind, stars, given_at="2026-01-06T00:00:00Z")
    for j in range(n_pool):
        cand(1000 + j, f"pool{j}", "pool", rng.uniform(0.0, 1.5), rng.random(), source="src_a" if j % 3 else "src_b")
    con.commit()
    cfg = config.load(tmp_path / "none.toml")
    cfg["scoring"]["variant"] = "raw"
    return con, cfg, path


def _valence(con, cids):
    rows = con.execute(f"SELECT c.id, r.valence FROM candidates c JOIN readings r ON r.deezer_id=c.deezer_id "
                       f"WHERE c.id IN ({','.join('?' * len(cids))})", list(cids)).fetchall()
    return np.mean([r[1] for r in rows])


def _run(con, cfg, variant="raw"):
    return rerank.run(con, cfg, variant, vectors.compute_scores(con, cfg, variant))


# ---------------- the model ----------------

def test_wauc_counts_ties_half_and_weights_both_sides():
    assert rerank.wauc([2, 1], [1, 0], [1, 1]) == 1.0
    assert rerank.wauc([1, 2], [1, 0], [1, 1]) == 0.0
    assert rerank.wauc([1, 1], [1, 0], [1, 1]) == 0.5
    # two no's, the heavier one outscores the yes: 1 * 1 / (1 * (1 + 3))
    assert rerank.wauc([2, 1, 3], [1, 0, 0], [1, 1, 3]) == pytest.approx(0.25)
    assert rerank.wauc([1, 2], [1, 1], [1, 1]) is None


def test_fit_recovers_the_sign_and_its_penalty_is_per_unit_of_label_weight():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(200, 3))
    y = (Z[:, 0] - 0.5 * Z[:, 1] + rng.normal(scale=0.8, size=200) > 0).astype(np.int8)
    w = np.ones(200)
    b = rerank.fit(Z, y, w, 0.01)
    assert b[1] > 0 and b[2] < 0 and abs(b[3]) < 0.5 * abs(b[1])
    # each label counted twice, or each weighing double: the same model (the penalty scales with the weight)
    assert np.allclose(rerank.fit(np.vstack([Z, Z]), np.r_[y, y], np.r_[w, w], 0.01), b, atol=1e-6)
    assert np.allclose(rerank.fit(Z, y, 2 * w, 0.01), b, atol=1e-6)
    assert np.abs(rerank.fit(Z, y, w, 10.0)[1:]).max() < np.abs(b[1:]).max()      # more penalty, smaller weights
    loo = rerank.loo_logits(Z[:40], y[:40], w[:40], 0.01)
    assert loo.shape == (40,) and rerank.wauc(loo, y[:40], w[:40]) > 0.7


def test_direction_and_explain_skip_a_weight_that_only_corrects_another():
    Z = np.array([[1.0, 1.0], [-1.0, -1.0]])
    assert list(rerank.direction(Z, np.array([1, 0]), np.ones(2))) == [1, 1]
    beta = np.array([0.0, 2.0, -0.5])             # feature 1 points against its own direction: no reason
    names = ["knn_stars", "topk"]
    agree = np.array([True, False])
    side = np.zeros(len(traits.TRAITS), np.int8)
    assert rerank.explain(beta, np.array([1.0, -1.0]), names, side, agree) == ["close to your 4-5★ tracks"]
    assert rerank.explain(beta, np.array([-1.0, 1.0]), names, side, agree, sign=-1) == \
        ["close to tracks you rated low"]
    assert rerank.explain(beta, np.array([1.0, -1.0]), names, side, agree, skip=["close to your 4-5★ tracks"]) == []
    assert rerank.explain(beta, np.array([0.04, -1.0]), names, side, agree) == []    # 0.08: below REASON_MIN


def test_trait_phrases_follow_the_usual_range():
    assert rerank._phrase("trait:valence", True, 1) == "brighter than you usually like"
    assert rerank._phrase("trait:tempo", False, -1) == "slower than you usually like"
    assert rerank._phrase("trait:valence", False, -1) == "darker than you usually like"
    assert rerank._phrase("trait:valence", True, -1) is None              # less dark than the pool, still dark
    assert rerank._phrase("trait:valence", True, 0) == "mood in your usual range"     # and the pool is not
    assert rerank._phrase("trait:tempo", False, 0) == "tempo in your usual range"
    assert rerank._phrase("trait:tempo", True, 0, up=False) is None       # never held back for being usual
    assert rerank._phrase("trait:tempo", True, rerank.NO_READING) is None
    assert rerank._phrase("pca:3", True, None) is None
    assert rerank._phrase("source:src_x", True, None) == "found via src_x"
    assert rerank._phrase("source:src_x", False, None) is None


# ---------------- labels ----------------

def test_labels_sort_verdicts_and_outcomes_into_yes_and_no(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch, n_lab=6, n_pool=5)
    got = {x["sid"]: (x["y"], x["w"], x["kind"]) for x in rerank.labels(con)}
    assert got[sid("lab0")] == (1, 1.0, "3★") and got[sid("lab2")] == (1, 1.25, "4★")
    assert got[sid("lab1")] == (0, 1.0, "pass") and got[sid("lab3")] == (0, 1.5, "dislike")
    assert got[sid("lab5")] == (0, 0.75, "2★")                           # 1-2 stars: a weak no
    # outcomes count only without a verdict; unplayed is no label; the Rate your likes queue is not a label
    con.execute("UPDATE offers SET outcome='liked' WHERE candidate_id=1001")
    con.execute("UPDATE offers SET outcome='negative' WHERE candidate_id=1002")
    for cid, name in ((1001, "pool1"), (1002, "pool2"), (1003, "pool3")):
        con.execute("INSERT INTO offers(candidate_id, spotify_id, batch_id, added_at, outcome) "
                    "VALUES(?,?,1,'2026-01-05',?)", (cid, sid(name), {1001: "liked", 1002: "negative"}.get(cid,
                                                                                                    "unplayed")))
    verdicts.record(con, sid("pool4"), "rating", 5, queue="likes")
    got = {x["sid"]: (x["y"], x["kind"]) for x in rerank.labels(con)}
    assert got[sid("pool1")] == (1, "liked") and got[sid("pool2")] == (0, "negative")
    assert sid("pool3") not in got and sid("pool4") not in got
    verdicts.record(con, sid("pool1"), "pass", None, given_at="2099-01-01T00:00:00Z")
    assert {x["sid"]: x["y"] for x in rerank.labels(con)}[sid("pool1")] == 0    # the verdict wins over the outcome


def _play(con, name, at, ms=None, completed=None, context="needle"):
    con.execute("INSERT INTO plays(spotify_id, played_at, context, ms_played, completed) VALUES(?,?,?,?,?)",
                (sid(name), at, context, ms, completed))


def test_a_short_unfinished_needle_play_is_a_soft_no(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch, n_lab=6, n_pool=10)
    for j in range(1, 7):
        con.execute("INSERT INTO offers(candidate_id, spotify_id, batch_id, added_at, outcome) "
                    "VALUES(?,?,1,'2026-01-05','unplayed')", (1000 + j, sid(f"pool{j}")))
    con.execute("INSERT INTO calibrations(candidate_id, spotify_id, picked_at, week) VALUES(1007, ?, 'x', 'w')",
                (sid("pool7"),))
    _play(con, "pool1", "2026-01-06T10:00:00Z", 12000, 0)                   # short, unfinished: a soft no
    _play(con, "pool2", "2026-01-06T10:00:00Z", 12000, 1)                   # finished (a short track)
    _play(con, "pool3", "2026-01-06T10:00:00Z", 45000, 0)                   # long enough
    _play(con, "pool4", "2026-01-06T10:00:00Z", context="spotify:playlist:x")   # Spotify carries no length
    _play(con, "pool5", "2026-01-06T10:00:00Z", 5000, 0)
    _play(con, "pool5", "2026-01-07T10:00:00Z", 200000, 1)                  # a later full play wins
    _play(con, "pool6", "2026-01-06T10:00:00Z")                             # an older needle: no length
    _play(con, "pool7", "2026-01-06T11:00:00Z", 3000, 0)                    # a Calibrate pick counts too
    _play(con, "pool8", "2026-01-06T11:00:00Z", 3000, 0)                    # never offered or picked
    soft = {x["sid"]: x for x in rerank.labels(con, cfg) if x["soft"]}
    assert set(soft) == {sid("pool1"), sid("pool7")}
    assert (soft[sid("pool1")]["y"], soft[sid("pool1")]["w"], soft[sid("pool1")]["kind"],
            soft[sid("pool1")]["at"]) == (0, 0.5, "soft_pass", "2026-01-06T10:00:00Z")
    assert not any(x["soft"] for x in rerank.labels(con))                   # without cfg: hard labels only
    cfg["rerank"]["soft_labels"] = False
    assert not any(x["soft"] for x in rerank.labels(con, cfg))
    cfg["rerank"]["soft_labels"] = True
    con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-07')", (sid("pool7"),))
    assert sid("pool7") not in {x["sid"] for x in rerank.labels(con, cfg) if x["soft"]}   # liked on Spotify since
    verdicts.record(con, sid("pool1"), "rating", 4)                         # a Verdict always wins
    got = {x["sid"]: x for x in rerank.labels(con, cfg)}
    assert (got[sid("pool1")]["y"], got[sid("pool1")]["soft"]) == (1, False)


def test_soft_labels_train_the_model_but_not_its_gate(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    for j in range(10):
        con.execute("INSERT INTO offers(candidate_id, spotify_id, batch_id, added_at, outcome) "
                    "VALUES(?,?,1,'2026-01-05','unplayed')", (1000 + j, sid(f"pool{j}")))
        _play(con, f"pool{j}", "2026-01-06T10:00:00Z", 8000, 0)
    r = _run(con, cfg)
    assert (r["n_labels"], r["n_pos"], r["n_neg"], r["n_soft"]) == (60, 30, 30, 10)
    assert rerank.summary(r)["n_soft"] == 10
    cfg["rerank"]["min_labels"] = 61                                        # 70 labels, 60 of them hard
    assert not _run(con, cfg)["used"]
    rerank.record(con, r)
    assert rerank.runs(con)[0]["n_soft"] == 10


def test_a_calibrate_verdict_is_a_label_and_an_eval_query(tmp_path, monkeypatch):
    from haystack_music import evaluate
    con, cfg, _ = _world(tmp_path, monkeypatch, n_lab=6, n_pool=10)
    con.execute("INSERT INTO calibrations(candidate_id, spotify_id, picked_at, week) VALUES(1008, ?, 'x', 'w')",
                (sid("pool8"),))
    verdicts.record(con, sid("pool8"), "dislike", None, queue="calibrate")
    verdicts.record(con, sid("pool9"), "pass", None)                        # neither offered nor picked
    assert {x["sid"]: (x["y"], x["kind"]) for x in rerank.labels(con, cfg)}[sid("pool8")] == (0, "dislike")
    q, _ = evaluate._verdict_queries(con, cfg, [], set(), set())
    down = {x["sid"] for x in q if x["kind"] == "turned down"}
    assert sid("pool8") in down and sid("pool9") not in down


# ---------------- stage 1 and the second stage ----------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_stage1_is_compute_scores_for_every_pool_row(tmp_path, monkeypatch, variant):
    con, cfg, _ = _world(tmp_path, monkeypatch, n_lab=12, n_pool=60)
    res = vectors.compute_scores(con, cfg, variant)
    ctx = rerank._prepare(con, cfg, variant, res, rerank.labels(con))
    s1, base, pen = rerank._stage1(ctx)
    P = ctx["n_pool"]
    assert P == len(res["rows"]) and len(s1) == P + 12
    assert np.allclose(s1[:P], res["score"], atol=1e-4, equal_nan=True)
    assert np.allclose(base[:P], res["base"], atol=1e-4, equal_nan=True)
    assert np.isfinite(s1[P:]).all() and (pen >= 0).all()


def test_the_model_learns_what_the_verdicts_follow_and_orders_the_top(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    r = _run(con, cfg)
    assert r["used"], r["reason"]
    assert (r["n_labels"], r["n_pos"], r["n_neg"]) == (60, 30, 30)
    assert r["loo_auc_model"] > 0.9 and r["loo_auc_stage1"] < 0.75
    m = r["model"]
    strongest = max(zip(m["features"], m["beta"][1:]), key=lambda x: abs(x[1]))
    assert strongest[0] == "trait:valence" and strongest[1] > 0
    assert m["sources"] == ["src_a", "src_b"] and m["pca"] is None        # PCA waits for min_labels_embed
    pool = [c for (c,) in con.execute("SELECT id FROM candidates WHERE state='pool'")]
    assert set(r["ranks"]) == set(pool) and set(r["top"]) == set(pool)  # pool_top 400 covers the whole pool
    by_rank = sorted(pool, key=lambda c: -r["ranks"][c])[:15]
    res = vectors.compute_scores(con, cfg, "raw")
    by_score = [int(res["rows"][i][0]) for i in np.argsort(-res["score"])[:15]]
    assert _valence(con, by_rank) > _valence(con, by_score) + 0.2
    phrases = {p for pair in rerank.PHRASES.values() for p in pair if p}
    for c in by_rank:
        t = r["top"][c]
        assert 0 < t["p"] < 1 and len(t["reasons"]) <= 2 and len(t["held_back"]) <= 2
        assert len(set(t["reasons"])) == len(t["reasons"]) and not set(t["reasons"]) & set(t["held_back"])
        for p in t["reasons"] + t["held_back"]:
            assert p in phrases or p.endswith(("than you usually like", "in your usual range")) or \
                p.startswith("found via ")
    # the bright ones say so; the rest are in the usual range (the rated discoveries widen it) and the pool isn't
    lo, hi = (baseline.compute(con, cfg)["traits"]["valence"][k] for k in ("lo", "hi"))
    first = {c: (r["top"][c]["reasons"] or [None])[0] for c in by_rank}
    assert all(first[c] == "brighter than you usually like" for c in by_rank if _valence(con, [c]) > hi)
    assert all(first[c] == "mood in your usual range" for c in by_rank if lo <= _valence(con, [c]) <= hi)
    assert sum(f is not None for f in first.values()) >= 12
    # the turned-down tracks happen to sit nearer the yes here; that weight orders, but is never quoted
    assert dict(zip(m["features"], m["beta"][1:]))["neg_penalty"] > 0
    assert not any("sounds like a track you turned down" in t["reasons"] for t in r["top"].values())


def test_below_pool_top_the_pool_keeps_its_stage1_order_under_the_top(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    cfg["rerank"]["pool_top"] = 40
    res = vectors.compute_scores(con, cfg, "raw")
    r = rerank.run(con, cfg, "raw", res)
    assert r["used"] and len(r["top"]) == 40 and len(r["ranks"]) == len(res["rows"])
    s1 = {int(row[0]): float(s) for row, s in zip(res["rows"], res["score"])}
    rest = [c for c in r["ranks"] if c not in r["top"]]
    assert min(r["ranks"][c] for c in r["top"]) > max(r["ranks"][c] for c in rest)
    assert sorted(rest, key=lambda c: -r["ranks"][c]) == sorted(rest, key=lambda c: -s1[c])
    assert min(s1[c] for c in r["top"]) >= max(s1[c] for c in rest)       # the top is stage 1's top 40


def test_the_gate_keeps_stage1_until_the_model_earns_it(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    cfg["rerank"]["min_labels"] = 100
    r = _run(con, cfg)
    assert not r["used"] and r["reason"] == "60 labels (needs 100)"
    assert r["ranks"] is None and r["top"] is None
    assert r["loo_auc_model"] is not None and r["model"]["beta"]          # still fitted and reported
    cfg["rerank"]["min_labels"] = 40
    cfg["rerank"]["min_gain"] = 0.99
    r = _run(con, cfg)
    assert not r["used"] and "(needs +0.99)" in r["reason"]
    cfg["rerank"]["enabled"] = False
    assert _run(con, cfg)["reason"] == "rerank.enabled is false"


def test_too_few_labels_of_one_kind_fit_nothing(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch, n_lab=3, n_pool=20)       # 2 yes, 1 no
    r = _run(con, cfg)
    assert not r["used"] and r["reason"] == "2 yes and 1 no labels: too few to fit" and r["model"] is None


def test_record_write_pool_and_the_cli(tmp_path, monkeypatch, capsys):
    con, cfg, path = _world(tmp_path, monkeypatch)
    r = _run(con, cfg)
    rerank.record(con, r)
    rerank.write_pool(con, r)
    assert rerank.last(con)["used"] is True and rerank.last(con)["features"] == r["model"]["features"]
    assert rerank.last(con)["n_features"] == len(r["model"]["features"]) == r["n_features"]
    assert [x["used"] for x in rerank.runs(con)] == [1]
    n = con.execute("SELECT COUNT(*) FROM candidates WHERE state='pool' AND rerank_p IS NOT NULL").fetchone()[0]
    assert n == 150
    reason = con.execute("SELECT rerank_reason FROM candidates WHERE rerank_p IS NOT NULL LIMIT 1").fetchone()[0]
    assert isinstance(json.loads(reason), list)
    assert con.execute("SELECT COUNT(*) FROM candidates WHERE rerank_score IS NOT NULL").fetchone()[0] == 150
    off = {**r, "used": False, "top": None}
    rerank.write_pool(con, off)              # fitted, not used: the chance stays (Calibrate), the order is cleared
    assert con.execute("SELECT COUNT(*) FROM candidates WHERE rerank_p IS NOT NULL").fetchone()[0] == 150
    assert con.execute("SELECT COUNT(*) FROM candidates WHERE rerank_score IS NOT NULL").fetchone()[0] == 0
    rerank.write_pool(con, {**off, "chance": None})                        # no model: the columns are cleared
    assert con.execute("SELECT COUNT(*) FROM candidates WHERE rerank_p IS NOT NULL").fetchone()[0] == 0
    con.close()
    monkeypatch.setattr(config, "DB_PATH", path)
    a = cli.build_parser().parse_args(["rerank", "show"])
    assert cli.cmd_rerank(cfg, a) == 0
    out = capsys.readouterr().out
    assert "re-ranker (raw): used, 60 labels (30 yes, 30 no); leave-one-out AUC" in out
    assert "weights (per pool sd, largest first): trait:valence +" in out
    # a database from before the re-ranker: no table, no crash
    old = tmp_path / "old.db"
    sqlite3.connect(old).execute("CREATE TABLE kv(k TEXT PRIMARY KEY, v TEXT)").connection.close()
    monkeypatch.setattr(config, "DB_PATH", old)
    assert cli.cmd_rerank(cfg, a) == 0
    assert "no re-ranker fit yet" in capsys.readouterr().out


# ---------------- the batch ----------------

def test_select_batch_takes_the_top_by_rank_and_the_band_by_score():
    pool = [Pick(i, f"a{i}", 1 - i / 100, "src", rank=i / 100) for i in range(100)]
    picks = select_batch(pool, 5, 0.0, rng=random.Random(1))
    assert sorted(p.cand_id for p in picks) == [95, 96, 97, 98, 99]       # the re-ranker's top, not the score's
    picks = select_batch(pool, 10, 0.2, (0.4, 0.75), rng=random.Random(1))
    band = [p.cand_id for p in picks if p.exploration]
    assert len(band) == 2 and all(25 <= c < 60 for c in band)             # the band is by score percentile
    plain = [Pick(i, f"a{i}", 1 - i / 100, "src") for i in range(100)]
    assert sorted(p.cand_id for p in select_batch(plain, 5, 0.0)) == [0, 1, 2, 3, 4]


def test_plan_shows_the_reranked_top_and_stage1_on_request(tmp_path, monkeypatch):
    con, cfg, path = _world(tmp_path, monkeypatch)
    con.close()
    ro = db.connect_readonly(path)
    out = batch.plan(ro, cfg, variant="raw", rng=random.Random(1))
    assert out["rerank"]["used"] is True and out["rerank"]["n_labels"] == 60
    top = [t for t in out["tracks"] if not t["exploration"]]
    assert all(0 < t["confidence"] < 1 for t in top) and sum(bool(t["reasons"]) for t in top) >= len(top) // 2
    assert all(t["confidence"] is None and t["reasons"] == [] for t in out["tracks"] if t["exploration"])
    s1 = batch.plan(ro, cfg, variant="raw", rng=random.Random(1), stage1=True)
    assert s1["rerank"] == {"used": False, "reason": "--stage1", **{k: None for k in s1["rerank"]
                                                                    if k not in ("used", "reason")}}
    assert {t["artist"] for t in top} != {t["artist"] for t in s1["tracks"] if not t["exploration"]}
    ro.close()
