"""The eval's objective (ADR 0009): kept vs turned-down offers, AUC, the weekly re-choice with hysteresis and eval_runs.
Synthetic data; no Spotify, no Deezer."""
import json

import numpy as np
import pytest

from haystack_music import batch, config, db, evaluate, firstbatch, report, status, vectors, verdicts
from haystack_music.scoring import l2norm


def sid(n: str) -> str:
    return n.ljust(22, "0")


def test_auc_counts_ties_half_and_weights_the_yes_side():
    assert evaluate.auc([3, 4], [1, 1], [1, 2]) == 1.0
    assert evaluate.auc([1], [1], [2]) == 0.0
    assert evaluate.auc([2], [1], [2]) == 0.5                            # a tie counts half
    assert evaluate.auc([3, 0], [3, 1], [1]) == pytest.approx(0.75)      # the 3-star yes weighs three times
    assert evaluate.auc([], [], [1]) is None and evaluate.auc([1], [1], []) is None
    assert evaluate.auc([1], [0], [0]) is None


def _verdict_world(tmp_path, monkeypatch):
    """Liked songs, offers and verdicts of every kind the eval sorts into kept, turned down or neither."""
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    con = db.connect(tmp_path / "t.db")
    con.execute("INSERT INTO batches(id, created_at, size) VALUES(1, '2026-01-05T00:00:00Z', 9)")
    rng = np.random.default_rng(0)
    did = iter(range(100, 200))

    def vec():
        d = next(did)
        vectors.put_deezer_vector(con, d, l2norm(rng.normal(size=512)), np.zeros(1280), 10, 30.0, None)
        return d

    def track(name):
        con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json) VALUES(?,?,?)",
                    (sid(name), f"Title {name}", f'["Artist {name}"]'))

    def liked(name):
        track(name)
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(name),))
        con.execute("INSERT INTO liked_map(spotify_id, use_local, deezer_id, vector_status) VALUES(?, 0, ?, 'ok')",
                    (sid(name), vec()))

    def offered(cid, name, outcome, with_vector=True):
        con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status, spotify_id)
                       VALUES(?,?,?,?, 'src_a', 'offered', ?, ?)""",
                    (cid, vec() if with_vector else 999, f"Artist {name}", f"Title {name}",
                     "ok" if with_vector else None, sid(name)))
        con.execute("""INSERT INTO offers(candidate_id, spotify_id, batch_id, added_at, outcome, removed_at)
                       VALUES(?,?,1,'2026-01-05',?,?)""", (cid, sid(name), outcome, outcome and "2026-01-12"))

    for name in ("plainlike", "likedfive", "likedfour", "likeddisliked", "likeddisc"):
        liked(name)
    verdicts.record(con, sid("likedfive"), "rating", 5)
    verdicts.record(con, sid("likedfour"), "rating", 4)
    verdicts.record(con, sid("likeddisliked"), "dislike", None)
    offered(1, "likedfive", "liked")                       # a liked discovery, rated: kept, and a liked query too
    offered(2, "likeddisc", "liked")                       # an Unrated like that was offered: kept
    offered(3, "rated4", "unplayed")
    verdicts.record(con, sid("rated4"), "rating", 4)
    offered(4, "passed", "unplayed")
    verdicts.record(con, sid("passed"), "pass", None)
    offered(5, "disliked", "negative")
    verdicts.record(con, sid("disliked"), "dislike", None)
    offered(6, "expired", "negative")                      # played and expired, no verdict: turned down
    offered(7, "removed", "manual_removed")                # removed by hand, no verdict: turned down
    offered(8, "removedrated", "manual_removed")           # removed, then rated 3: kept
    verdicts.record(con, sid("removedrated"), "rating", 3)
    offered(9, "rated2", "unplayed")                       # 2 stars: neither
    verdicts.record(con, sid("rated2"), "rating", 2)
    offered(10, "inlibrary", "unplayed")                   # passed, but the library has the same song: left out
    verdicts.record(con, sid("inlibrary"), "pass", None)
    con.execute("INSERT INTO local_tracks(id, path, size, content_hash, artist_raw, title_raw, present) "
                "VALUES(1, '/m/a.m4a', 1, 'h', 'Artist inlibrary', 'Title inlibrary', 1)")
    offered(11, "novector", "unplayed", with_vector=False)  # passed, no preview vector: left out
    verdicts.record(con, sid("novector"), "pass", None)
    for name, kind, stars in (("ratedelsewhere", "rating", 4), ("passedelsewhere", "pass", None)):
        track(name)                                        # never offered (a playlist track), found on Deezer on demand:
        con.execute("INSERT INTO spotify_deezer(spotify_id, deezer_id, status) VALUES(?, ?, 'ok')",
                    (sid(name), vec()))
        verdicts.record(con, sid(name), kind, stars)       # left out of both sides
    return con, config.load(tmp_path / "none.toml")


def test_offers_are_sorted_into_kept_and_turned_down(tmp_path, monkeypatch):
    con, cfg = _verdict_world(tmp_path, monkeypatch)
    d = evaluate.load(con, cfg)
    by = {}
    for x in d["queries"]:
        by.setdefault(x["kind"], {})[x["sid"].rstrip("0")] = x
    assert set(by["preview"]) == {"plainlike", "likedfive", "likedfour", "likeddisliked", "likeddisc"}
    assert {k: x["weight"] for k, x in by["preview"].items()} == \
        {"plainlike": 3.0, "likedfive": 5.0, "likedfour": 4.0, "likeddisliked": 0.0, "likeddisc": 3.0}
    assert {k: x["weight"] for k, x in by["kept"].items()} == \
        {"likedfive": 5.0, "likeddisc": 3.0, "rated4": 4.0, "removedrated": 3.0}
    assert set(by["turned down"]) == {"passed", "disliked", "expired", "removed"}
    own = by["kept"]["likedfive"]["owners"]
    assert {"manual:1", f"like:{sid('likedfive')}", f"discovery:{sid('likedfive')}", f"rated:{sid('likedfive')}"} <= own
    assert len(d["q"]) == len(d["queries"]) == len(d["q_style"])


def _ev(chosen="csls", objective="auc", aucs=None, degenerate=(), **kw):
    aucs = aucs or {"raw": 0.60, "csls": 0.70}
    return {"chosen": chosen, "objective_used": objective, "n_pos": 30, "n_neg": 20,
            "variants": {v: {"auc": a, "degenerate": v in degenerate} for v, a in aucs.items()}, **kw}


def test_hysteresis_keeps_on_a_small_gain_and_switches_on_a_large_one():
    h = evaluate._hysteresis
    assert h(_ev(), "csls", 0.03) == ("csls", "kept: still the best")
    assert h(_ev(aucs={"raw": 0.68, "csls": 0.70}), "raw", 0.03)[0] == "raw"             # +0.02: kept
    v, why = h(_ev(aucs={"raw": 0.60, "csls": 0.70}), "raw", 0.03)
    assert v == "csls" and why.startswith("switched: AUC +0.100")
    v, why = h(_ev(aucs={"raw": 0.69, "csls": 0.70}, degenerate=("raw",)), "raw", 0.03)
    assert v == "csls" and "degenerate" in why                                          # degenerate: always
    v, why = h(_ev(), "genre", 0.03)
    assert v == "csls" and "not evaluated" in why                                       # e.g. genre skipped
    v, why = h(_ev(objective="mean_pct", objective_note="auc needs 15"), "raw", 0.03)
    assert v == "raw" and "fallback" in why                                             # no switch on mean pct


@pytest.fixture
def con(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "STATUS_PATH", tmp_path / "state" / "status.json")
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def _canned(monkeypatch, ev):
    monkeypatch.setattr(evaluate, "run", lambda con, cfg, data=None: json.loads(json.dumps(ev)))


def _runs(con):
    return [dict(r) for r in con.execute("SELECT * FROM eval_runs ORDER BY id")]


def test_set_stores_the_choice_and_records_the_run(con, monkeypatch):
    cfg = config.load(path=None)
    _canned(monkeypatch, _ev())
    ev = evaluate.choose(con, cfg, "set")
    assert (ev["used"], ev["previous"], ev["switched"], ev["decision"]) == ("csls", "raw", True, "set by the eval")
    assert db.kv_get(con, "scoring_variant") == "csls" and vectors.active_variant(con, cfg) == "csls"
    (r,) = _runs(con)
    assert (r["trigger"], r["objective"], r["chosen"], r["used"], r["n_pos"], r["n_neg"], r["switched"]) == \
        ("set", "auc", "csls", "csls", 30, 20, 1)
    assert json.loads(r["auc_json"]) == {"raw": 0.6, "csls": 0.7}
    assert status.read()["scoring_eval"]["used"] == "csls"


def test_weekly_switches_only_past_the_margin(con, monkeypatch):
    cfg = config.load(path=None)
    db.kv_set(con, "scoring_variant", "raw")
    _canned(monkeypatch, _ev(aucs={"raw": 0.68, "csls": 0.70}))
    ev = evaluate.choose(con, cfg, "weekly")
    assert ev["used"] == "raw" and not ev["switched"] and db.kv_get(con, "scoring_variant") == "raw"
    _canned(monkeypatch, _ev(aucs={"raw": 0.60, "csls": 0.70}))
    ev = evaluate.choose(con, cfg, "weekly")
    assert ev["used"] == "csls" and ev["switched"] and db.kv_get(con, "scoring_variant") == "csls"
    assert [(r["trigger"], r["used"], r["switched"]) for r in _runs(con)] == [("weekly", "raw", 0), ("weekly", "csls", 1)]


def test_report_and_pinned_never_change_the_variant(con, monkeypatch):
    cfg = config.load(path=None)
    db.kv_set(con, "scoring_variant", "raw")
    _canned(monkeypatch, _ev())
    ev = evaluate.choose(con, cfg, "report")
    assert ev["used"] == "raw" and ev["decision"] == "kept: report only"
    cfg["scoring"]["variant"] = "centred"
    ev = evaluate.choose(con, cfg, "set")
    assert (ev["used"], ev["pinned"], ev["decision"]) == ("centred", "centred", "pinned in config")
    assert db.kv_get(con, "scoring_variant") == "raw"
    assert [r["trigger"] for r in _runs(con)] == ["report", "set"]
    with pytest.raises(ValueError):
        evaluate.choose(con, cfg, "monthly")


def test_without_write_nothing_is_stored(con, monkeypatch):
    cfg = config.load(path=None)
    _canned(monkeypatch, _ev())
    ev = evaluate.choose(con, cfg, "weekly", write=False)
    assert ev["used"] == "csls"                                      # what the batch would use
    assert db.kv_get(con, "scoring_variant") is None and _runs(con) == []
    assert not config.STATUS_PATH.exists()


def test_first_batch_sets_and_the_batch_rechecks(con, monkeypatch):
    cfg = config.load(path=None)
    _canned(monkeypatch, _ev())
    assert firstbatch.choose(con, cfg)["used"] == "csls"
    assert db.kv_get(con, "scoring_variant") == "csls"
    _canned(monkeypatch, _ev(chosen="raw", aucs={"raw": 0.80, "csls": 0.70}))
    s = batch._rechoose(con, cfg, write=True)
    assert s == {"objective_used": "auc", "chosen": "raw", "previous": "csls", "used": "raw", "switched": True,
                 "decision": "switched: AUC +0.100 over csls (more than 0.03)", "n_pos": 30, "n_neg": 20,
                 "auc_used": 0.8}
    assert db.kv_get(con, "scoring_variant") == "raw"
    monkeypatch.setattr(evaluate, "run", lambda *a, **k: 1 / 0)
    assert batch._rechoose(con, cfg, write=True)["error"].startswith("ZeroDivisionError")
    cfg["scoring"]["variant"] = "raw"
    assert batch._rechoose(con, cfg, write=True) is None             # pinned: no eval at all


def test_a_fresh_install_keeps_the_default(con):
    cfg = config.load(path=None)
    ev = evaluate.choose(con, cfg, "weekly")
    assert ev["chosen"] is None and ev["used"] == "raw" and not ev["switched"]
    assert ev["decision"].startswith("kept: not enough data")
    assert db.kv_get(con, "scoring_variant") is None
    (r,) = _runs(con)
    assert r["chosen"] is None and r["used"] == "raw" and json.loads(r["auc_json"]) == {}


def test_markdown_and_report_show_the_objective(tmp_path):
    variants = {v: {"auc": a, "gap": 10.0, "pct_yes": 70.0, "pct_no": 60.0, "top50_band": 1.2, "degenerate": d,
                    "mean_pct": 60.0, "seed_sd": 1.0, "recall_at_15": 0.1, "recall_at_50": 0.3,
                    "chart_share_top15": 0.2, "top15_score_band": [0.1, 0.2]}
                for v, a, d in (("raw", 0.6, False), ("genre", 0.9, True), ("genre_blend", 0.7, False))}
    r = {"at": "x", "liked_with_vectors": 10, "holdout": 0.2, "seeds": [1], "pool": 100, "pool_total": 120,
         "positive_tracks": 50, "pool_chart_share": 0.3, "recall_k": 15, "tie_tolerance": 0.5, "auc_tie": 0.02,
         "min_verdicts": 15, "degenerate_sd": 0.25, "objective": "auc", "objective_used": "auc",
         "n_pos": 30, "n_neg": 20, "verdict_queries": {"kept": 30, "turned down": 20},
         "chosen": "genre_blend", "used": "genre_blend",
         "previous": "raw", "switched": True, "decision": "set by the eval", "pinned": None,
         "queries_by_kind": {"preview": 10}, "held_out_per_seed": {"preview": 2}, "min_queries": 5,
         "variants": variants}
    md = evaluate.markdown(r)
    assert "AUC, kept above turned down" in md and "raw -> **genre_blend** (set by the eval)" in md
    e = report.eval_table({"scoring_eval": r})
    assert (e["objective"], e["n_pos"], e["n_neg"]) == ("auc", 30, 20)
    assert e["variants"]["genre"]["auc"] == 0.9 and e["variants"]["genre"]["degenerate"] is True
    md_fb = evaluate.markdown({**r, "objective_used": "mean_pct", "objective_note": "auc needs 15 kept and 15 "
                               "turned-down offers with a preview vector, there are 30 and 4"})
    assert "mean-percentile fallback" in md_fb and "there are 30 and 4" in md_fb
    db.connect(tmp_path / "fresh.db").close()
    txt = report.text(report.build(db.connect_readonly(tmp_path / "fresh.db"), config.load(tmp_path / "none.toml"),
                                   {"scoring_eval": r}))
    assert "chosen by auc; offers: 30 kept, 20 turned down" in txt and "0.900 deg" in txt
