import numpy as np
import pytest

from haystack_music import evaluate
from haystack_music.scoring import Space, choose_variant, hubness, l2norm, score_variant


def _space(vecs, k=2):
    v = l2norm(np.asarray(vecs, np.float32))
    return Space(mu=v.mean(0), ref=v, ref_owner=[f"r{i}" for i in range(len(v))], k=k)


def test_centring_removes_shared_direction():
    common = np.array([5, 0, 0], np.float32)
    a, b = common + [0, 1, 0], common + [0, -1, 0]
    sp = _space([a, b])
    raw = l2norm(np.array([a]))[0] @ l2norm(np.array([b]))[0]
    cen = sp.prep(np.array([a]), True)[0] @ sp.prep(np.array([b]), True)[0]
    assert raw > 0.9 and cen < -0.9


def test_hubness_excludes_self_and_identical_rows():
    ref = l2norm(np.array([[1, 0], [1, 0.1], [0, 1], [1, 0]], np.float32))
    h = hubness(ref[:1], ref, k=1, self_idx=np.array([0]))
    assert h[0] == pytest.approx(float(ref[0] @ ref[1]), abs=1e-6)   # self (and its identical copy) skipped
    h2 = hubness(ref[:1], ref, k=1)                                   # identical vector excluded by threshold
    assert h2[0] < 0.99999


def test_csls_penalises_a_hub():
    rng = np.random.default_rng(0)
    pos = l2norm(rng.normal(size=(1, 16)))
    hub = l2norm(rng.normal(size=(1, 16)))
    ref = l2norm(hub + 0.05 * rng.normal(size=(40, 16)))               # the hub is close to everything in ref
    spec = l2norm(pos + 0.9 * rng.normal(size=(1, 16)))
    hubc = l2norm(0.55 * pos + hub)
    cands = np.vstack([hubc, spec])
    sp = Space(mu=np.zeros(16, np.float32), ref=ref, ref_owner=list(range(40)), k=10)
    raw, _, _ = score_variant("raw", cands, pos, None, sp)
    csls, _, _ = score_variant("csls", cands, pos, None, sp)
    assert (csls[0] - csls[1]) < (raw[0] - raw[1])


def test_variant_validation():
    with pytest.raises(ValueError):
        score_variant("bogus", np.ones((1, 2)), np.ones((1, 2)), None)
    with pytest.raises(ValueError):
        score_variant("csls", np.ones((1, 2)), np.ones((1, 2)), None, Space())


def test_choose_variant_ties_go_to_simpler():
    assert choose_variant({"raw": 70.0, "centred": 70.3, "csls": 75.0, "csls_centred": 75.2}, 0.5) == "csls"
    assert choose_variant({"raw": 70.0, "centred": 70.4, "csls": 60.0, "csls_centred": 61.0}, 0.5) == "raw"
    assert choose_variant({"raw": 60.0, "centred": 80.0, "csls": 60.0, "csls_centred": 79.0}, 0.5) == "centred"


def test_owner_max_reduceat():
    p = np.eye(4, dtype=np.float32)
    starts = np.array([0, 2])                                           # owner A = rows 0,1; owner B = rows 2,3
    out = evaluate._owner_max(np.array([[0, 1, 0, 0], [0, 0, 0, 1]], np.float32), p, np.zeros(4, np.float32), starts)
    assert out.tolist() == [[1, 0], [0, 1]]


def test_markdown_table():
    r = {"at": "x", "liked_with_vectors": 10, "holdout": 0.2, "seeds": [1], "pool": 100,
         "pool_total": 120, "positive_tracks": 50, "pool_chart_share": 0.3, "recall_k": 15, "tie_tolerance": 0.5,
         "chosen": "csls", "used": "csls", "pinned": None, "queries_by_kind": {"preview": 10}, "held_out_per_seed": {"preview": 2}, "min_queries": 5,
         "variants": {v: {"mean_pct": 60.0, "seed_sd": 1.0, "recall_at_15": 0.1, "chart_share_top15": 0.2,
                          "top15_score_band": [0.1, 0.2]} for v in ("raw", "csls")}}
    md = evaluate.markdown(r)
    assert "| csls **(chosen)** | 60.00 |" in md and "used for the batch: **csls**" in md


def test_topk_and_style_and_blend():
    from haystack_music.scoring import style_topk, topk_mean, zblend
    S = np.array([[1, 5, 3, 2], [0, 0, 0, 9]], np.float32)
    assert topk_mean(S, 2).tolist() == [4.0, 4.5]
    st = style_topk(np.array([[1, 0, 0]], np.float32), np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32), 2)
    assert st[0] == pytest.approx((1 + 2 ** -0.5) / 2, abs=1e-6)
    z = zblend(np.array([1.0, 2.0, 3.0]), np.array([30.0, 20.0, 10.0]))
    assert np.allclose(z, 0.0)


def _synthetic(rng, n_lib=60, n_q=40, n_pool=300, dim=16):
    centers = l2norm(rng.normal(size=(4, dim)))          # 4 "genres"; the listener likes genres 0 and 1
    def around(c, n, s=0.35):
        return l2norm(centers[c] + s * rng.normal(size=(n, dim)))
    lib = [around(rng.integers(2), 2) for _ in range(n_lib)]
    pos = np.vstack(lib).astype(np.float32)
    owners = [f"local:{i}" for i in range(n_lib) for _ in range(2)]
    q = np.vstack([around(rng.integers(2), 1) for _ in range(n_q)]).astype(np.float32)
    pool = np.vstack([around(rng.integers(4), 1) for _ in range(n_pool)]).astype(np.float32)
    style_c = np.eye(4, dtype=np.float32)
    qs = np.vstack([style_c[0] + 0.1] * n_q)
    ps = np.vstack([style_c[i % 4] + 0.1 for i in range(n_pool)])
    allv = np.vstack([pos, pool])
    sp = Space(mu=allv.mean(0), ref=allv[::3], ref_owner=[f"r{i}" for i in range(len(allv[::3]))], k=5)
    starts = np.array([i for i in range(len(owners)) if i == 0 or owners[i] != owners[i - 1]])
    queries = [dict(sid=f"s{i}", owner=f"like:s{i}", kind="preview", keys={f"k{i}"}, did=i, label="") for i in range(n_q)]
    rows = [dict(source="deezer_chart" if i % 5 == 0 else "deezer_artist_radio", artist_key=f"a{i}") for i in range(n_pool)]
    judged = [dict(label="yes1", verdict="yes", weight=1.0, owner="manual:1", vec=q[0], style=qs[0]),
              dict(label="no1", verdict="no", weight=2.0, owner="manual:2", vec=pool[3], style=ps[3])]
    return dict(pos=pos, owners=owners, bonus=np.zeros(len(pos), np.float32), starts=starts, uo=[owners[i] for i in starts],
                okeys={}, space=sp, judged=judged, queries=queries, q=q, q_style=qs, pool_style=ps, pool_ids=list(range(n_pool)),
                pool_m=pool, pool_rows=rows, pool_total=n_pool, centers=centers)


def test_eval_end_to_end_synthetic():
    from haystack_music import config
    cfg = config.load(path=None)
    d = _synthetic(np.random.default_rng(1))
    r = evaluate.run(None, cfg, data=d)
    assert set(r["variants"]) == {"raw", "topk_mean", "centred", "csls", "csls_centred", "genre", "genre_blend"}
    for v, x in r["variants"].items():
        assert x["mean_pct"] > 55, (v, x)                 # the liked genres are findable by every variant
    assert r["chosen"] in r["variants"]
    assert r["variants"]["raw"]["paired_vs_raw"][0] == 0
    md = evaluate.markdown(r)
    assert "genre_blend" in md
    assert set(r["judged_check"]["variants"]) == set(r["variants"]) and "Sanity check" in md


def test_judged_check_with_negative_penalty_leave_one_out():
    from haystack_music import config
    cfg = config.load(path=None)
    cfg["scoring"]["negative_penalty"].update(w=0.75, margin_quantile=0.95)
    d = _synthetic(np.random.default_rng(1))
    d["judged"][0]["candidate_id"], d["judged"][1]["candidate_id"] = 1, 2
    # the only negative is the judged 'no' itself: it must not be penalised by its own verdict, while the pool
    # around it is -> its penalised percentile can only go up
    d["negs"] = [dict(candidate_id=2, strength=2.0, label="no1", vec=d["judged"][1]["vec"], style=d["judged"][1]["style"])]
    r = evaluate.run(None, cfg, data=d)
    jc = r["judged_check"]
    assert jc["penalty"]["negatives"] == 1
    for v, x in jc["variants"].items():
        assert "pct_pen" in x, v
        assert x["pct_pen"][1] >= x["pct"][1], (v, x)
    assert "With the negative penalty" in evaluate.markdown(r)
    cfg["scoring"]["negative_penalty"]["w"] = 0
    r = evaluate.run(None, cfg, data=d)
    assert all("pct_pen" not in x for x in r["judged_check"]["variants"].values())


def _with_verdicts(d, rng, n_yes=20, n_no=20, *, emb_sep=False, style_sep=False):
    """Add kept (3-5 stars) and turned-down offers. emb_sep: the turned-down ones come from the genres the
    listener doesn't like, in embedding space; style_sep: they carry another genre's style vector."""
    c, st = d["centers"], np.eye(4, dtype=np.float32)

    def one(genre):
        return l2norm(c[genre] + 0.35 * rng.normal(size=(1, c.shape[1]))).astype(np.float32)
    yv = np.vstack([one(rng.integers(2)) for _ in range(n_yes)])
    nv = np.vstack([one(rng.integers(2, 4) if emb_sep else rng.integers(2)) for _ in range(n_no)])
    qs = [dict(sid=f"y{i}", owner=f"cand:{1000 + i}", owners={f"rated:y{i}"}, kind="kept", keys={f"yk{i}"},
               did=1000 + i, label="", weight=float(3 + i % 3)) for i in range(n_yes)]
    qs += [dict(sid=f"n{i}", owner=f"cand:{2000 + i}", owners=set(), kind="turned down", keys={f"nk{i}"},
                did=2000 + i, label="", weight=1.0) for i in range(n_no)]
    d["queries"] = d["queries"] + qs
    d["q"] = np.vstack([d["q"], yv, nv])
    d["q_style"] = np.vstack([d["q_style"], np.vstack([st[0] + 0.1] * n_yes),
                              np.vstack([(st[2] if style_sep else st[0]) + 0.1] * n_no)])
    return d


def test_auc_picks_the_variant_that_separates_the_verdicts():
    """The turned-down tracks differ in sound, not in genre: AUC picks an embedding variant, while the held-out
    liked songs alone (mean percentile) reward the genre blend that cannot tell yes from no."""
    from haystack_music import config
    cfg = config.load(path=None)
    rng = np.random.default_rng(1)
    d = _with_verdicts(_synthetic(rng), rng, emb_sep=True)
    r = evaluate.run(None, cfg, data=d)
    assert (r["objective_used"], r["n_pos"], r["n_neg"]) == ("auc", 20, 20)
    assert r["chosen"] not in ("genre", "genre_blend")
    assert r["variants"][r["chosen"]]["auc"] > 0.85 and r["variants"]["genre"]["auc"] == 0.5
    assert r["variants"][r["chosen"]]["gap"] > 20
    cfg["eval"]["objective"] = "mean_pct"
    r2 = evaluate.run(None, cfg, data=d)
    assert r2["objective_used"] == "mean_pct" and r2["chosen"] == "genre_blend"
    assert r2["objective_note"] == "eval.objective is mean_pct"


def test_a_degenerate_variant_is_never_chosen():
    """The style vectors separate yes from no perfectly, but the genre variant ties its whole top 50."""
    from haystack_music import config
    cfg = config.load(path=None)
    rng = np.random.default_rng(1)
    r = evaluate.run(None, cfg, data=_with_verdicts(_synthetic(rng), rng, style_sep=True))
    g = r["variants"]["genre"]
    assert g["auc"] == 1.0 and g["degenerate"] and g["top50_band"] < 0.25
    assert r["chosen"] == "genre_blend" and not r["variants"]["genre_blend"]["degenerate"]
    assert "degenerate" in evaluate.markdown(r)


def test_too_few_turned_down_tracks_fall_back_to_mean_percentile():
    from haystack_music import config
    cfg = config.load(path=None)
    rng = np.random.default_rng(1)
    r = evaluate.run(None, cfg, data=_with_verdicts(_synthetic(rng), rng, n_no=4, emb_sep=True))
    assert r["objective_used"] == "mean_pct" and r["n_neg"] == 4
    assert r["objective_note"] == "auc needs 15 kept and 15 turned-down offers with a preview vector, there are 20 and 4"
    assert r["chosen"] == choose_variant({v: x["mean_pct"] for v, x in r["variants"].items()}, 0.5)
    assert "mean-percentile fallback" in evaluate.markdown(r)
    assert evaluate._hysteresis(r, "raw", 0.03)[0] == "raw"         # the weekly rule never switches on it


def test_kept_offers_leave_themselves_out():
    """Every kept offer is also a positive (ADR 0002). Scored against itself it would always win; left out, the
    kept offers are no closer to the positives than the turned-down ones (same genres)."""
    from haystack_music import config
    cfg = config.load(path=None)
    rng = np.random.default_rng(2)
    d = _with_verdicts(_synthetic(rng), rng)
    yes = [i for i, x in enumerate(d["queries"]) if x["kind"] == "kept"]
    d["pos"] = np.vstack([d["pos"], d["q"][yes]])
    d["owners"] = d["owners"] + [f"rated:y{k}" for k in range(len(yes))]
    d["bonus"] = np.zeros(len(d["pos"]), np.float32)
    d["starts"] = np.arange(len(d["owners"]))[[i == 0 or d["owners"][i] != d["owners"][i - 1]
                                               for i in range(len(d["owners"]))]]
    d["uo"] = [d["owners"][i] for i in d["starts"]]
    r = evaluate.run(None, cfg, data=d, variants=("raw", "centred"))
    for v, x in r["variants"].items():
        assert x["auc"] < 0.75, (v, x["auc"])
    for x in d["queries"]:
        x["owners"] = set()                                            # without the leave-one-out ...
    r = evaluate.run(None, cfg, data=d, variants=("raw", "centred"))
    assert all(x["auc"] > 0.9 for x in r["variants"].values())        # ... each kept offer finds itself
