"""haystack report: read-only counts and shares over a synthetic world."""
import json

import numpy as np
import pytest

from haystack_music import cli, config, db, report, vectors, verdicts
from haystack_music.scoring import l2norm


def sid(n: str) -> str:
    return n.replace("_", "").ljust(22, "0")


# (name, batch, exploration, outcome, played, current verdict)
OFFERS = [
    ("kept_liked", 1, 0, "liked", 1, None),                 # liked, no verdict: kept
    ("liked_then_disliked", 1, 0, "liked", 1, ("dislike", None)),   # current verdict wins: not kept
    ("rated4", 1, 1, "rated", 1, ("rating", 4)),             # kept, exploration
    ("rated2", 2, 0, "rated", 1, ("rating", 2)),             # judged, not kept
    ("passed", 2, 0, "passed", 0, ("pass", None)),
    ("unplayed", 2, 0, "unplayed", 0, None),                 # decided, not judged
    ("unplayed_rated", 2, 1, "unplayed", 0, ("rating", 3)),  # rated later: judged and kept
    ("active", 3, 0, None, 0, None),                         # still in Discoveries
]


def _world(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    path = tmp_path / "t.db"
    con = db.connect(path)
    for b, at in ((1, "2026-01-05T00:00:00Z"), (2, "2026-01-12T00:00:00Z"), (3, "2026-01-19T00:00:00Z")):
        con.execute("INSERT INTO batches(id, created_at, size, n_explore) VALUES(?,?,?,?)", (b, at, 3, 1))
    for i, (name, b, ex, outcome, played, v) in enumerate(OFFERS, start=1):
        con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status, spotify_id)
                       VALUES(?,?,?,?,?,?,?,?)""", (100 + i, 100 + i, f"Offered Artist {i}", f"Offered Title {i}",
                                                     "src_a" if i % 2 else "src_b", "offered", "ok", sid(name)))
        con.execute("""INSERT INTO offers(candidate_id, spotify_id, batch_id, source, exploration, added_at, played,
                       outcome, removed_at) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (100 + i, sid(name), b, "src_a" if i % 2 else "src_b", ex, "2026-01-01T00:00:00Z", played,
                     outcome, outcome and "2026-01-02T00:00:00Z"))
        if v:
            verdicts.record(con, sid(name), v[0], v[1])
    # Liked Songs: two, one rated
    for n in ("ls1", "ls2"):
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(n),))
        con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json) VALUES(?,?,?)",
                    (sid(n), f"Liked Title {n}", f'["Liked Artist {n}"]'))
    verdicts.record(con, sid("ls1"), "rating", 5)
    # pool: 4 scored candidates, nearest a library window, a like, a manual rating and a rated track
    rng = np.random.default_rng(1)
    style = np.zeros(400); style[7] = 1.0
    for j, (owner, score) in enumerate((("local:1", 0.9), (f"like:{sid('ls1')}", 0.8), ("manual:101", 0.7),
                                        (f"rated:{sid('x')}", 0.6)), start=1):
        con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status, score,
                       nearest, scored_at) VALUES(?,?,?,?,?, 'pool', 'ok', ?, ?, '2026-01-19T00:00:00Z')""",
                    (j, j, f"Pool Artist {j}", f"Pool Title {j}", "src_a", score, owner))
        vectors.put_deezer_vector(con, j, l2norm(rng.normal(size=512)), np.zeros(1280), 10, 30.0,
                                  style if j < 4 else None)
    con.execute("INSERT INTO candidates(id, deezer_id, source, state, vector_status) VALUES(50, 50, 'src_b', 'pool', NULL)")
    db.kv_set(con, "scored_variant", "raw")
    con.close()
    cfg = config.load(tmp_path / "none.toml")
    return path, cfg


STATUS = {"scoring_eval": {
    "at": "2026-01-19 00:00 UTC", "chosen": "raw", "used": "raw", "pinned": None, "pool": 4, "recall_k": 15,
    "queries_by_kind": {"preview": 30, "local file": 2},
    "variants": {"genre": {"mean_pct": 61.5, "recall_at_15": 0.02}, "raw": {"mean_pct": 55.25, "recall_at_15": 0.031}},
    "judged_check": {"tracks": [{"label": "Secret Artist - Secret Title", "verdict": "yes", "weight": 1},
                                {"label": "Other Secret - Name", "verdict": "no", "weight": 1}],
                     "variants": {"raw": {"mean_yes": 70.1, "mean_no": 80.2, "mean_yes_pen": 69.0,
                                          "mean_no_pen": 75.0}}}}}


def test_offer_rates(tmp_path, monkeypatch):
    path, cfg = _world(tmp_path, monkeypatch)
    o = report.offers(db.connect_readonly(path))
    a = o["all"]
    assert (a["offered"], a["decided"], a["judged"], a["kept"]) == (8, 7, 6, 3)
    assert a["kept_of_decided"] == pytest.approx(3 / 7, abs=1e-3)
    assert a["kept_of_judged"] == pytest.approx(0.5)
    assert a["liked_of_decided"] == pytest.approx(2 / 7, abs=1e-3)       # the `haystack sources` rate
    assert (o["exploit"]["decided"], o["exploit"]["kept"]) == (5, 1)
    assert (o["exploration"]["decided"], o["exploration"]["kept"]) == (2, 2)
    b = {x["batch"]: x for x in o["batches"]}
    assert b[1]["at"].startswith("2026-01-05") and b[1]["exploit"]["kept_of_decided"] == 0.5
    assert b[2]["all"]["judged"] == 3 and b[3]["all"]["decided"] == 0 and b[3]["all"]["kept_of_decided"] is None
    assert sum(s["offered"] for s in o["sources"].values()) == 8


def test_verdicts_pool_and_head(tmp_path, monkeypatch):
    path, cfg = _world(tmp_path, monkeypatch)
    con = db.connect_readonly(path)
    v = report.verdict_counts(con)
    assert v["ratings"] == {"1": 0, "2": 1, "3": 1, "4": 1, "5": 1} and v["pass"] == 1 and v["dislike"] == 1
    assert (v["liked_songs"], v["liked_songs_rated"]) == (2, 1)
    p = report.pool(con, cfg)
    assert (p["pool"], p["embedded"], p["scored"]) == (5, 4, 4)
    assert p["active_variant"] == "raw" and p["scored_variant"] == "raw"
    assert (p["top"], p["top_library"], p["top_preview"]) == (4, 1, 3)
    assert p["top_preview_by_kind"] == {"like": 1, "manual": 1, "rated": 1}
    names = ["Parent---Style %d" % i for i in range(400)]
    h = report.head(con, names)
    assert h["n"] == 4 and h["score_max"] == 0.9 and h["score_min"] == 0.6
    assert h["top1_styles"] == {"Style 7": 3, "(no style vector)": 1}
    assert "tracks" not in h
    assert report.head(con, [])["top1_styles"]["style #7"] == 3          # no model metadata: numbered
    t = report.head(con, names, tracks=True)["tracks"]
    assert t[1]["sounds_like"] == {"title": "Liked Title ls1", "artist": "Liked Artist ls1"}


def test_eval_table_counts_judged_tracks_without_naming_them():
    e = report.eval_table(STATUS)
    assert list(e["variants"]) == ["raw", "genre"]                        # scoring.VARIANTS order
    assert e["variants"]["raw"]["recall"] == 0.031 and e["variants"]["genre"]["judged_yes"] is None
    assert (e["judged_yes"], e["judged_no"], e["queries"]) == (1, 1, 32)
    assert "Secret" not in json.dumps(e)
    assert report.eval_table({}) is None


def test_text_has_no_names_unless_asked(tmp_path, monkeypatch):
    path, cfg = _world(tmp_path, monkeypatch)
    con = db.connect_readonly(path)
    txt = report.text(report.build(con, cfg, STATUS))
    for word in ("Artist", "Title", "Secret"):
        assert word not in txt
    assert "kept/dec" in txt and "batch 1 (2026-01-05)" in txt and "nearest positive of the top 4" in txt
    named = report.text(report.build(con, cfg, STATUS, tracks=True))
    assert "Pool Artist 1 - Pool Title 1" in named and "sounds like Liked Artist ls1" in named
    assert "Secret" not in named


def test_plays_per_week_and_the_gates(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from haystack_music import rerank, sources
    path, cfg = _world(tmp_path, monkeypatch)
    con = db.connect(path)
    now = datetime.now(timezone.utc)
    at = [(now - timedelta(minutes=m)).strftime("%Y-%m-%dT%H:%M:%SZ") for m in range(5)]
    for t, ctx, ms in ((at[0], "needle", 10000), (at[1], "needle", 60000), (at[2], "needle", None),
                       (at[3], "spotify:playlist:x", None), ("2001-01-01T00:00:00Z", "needle", 1000)):
        con.execute("INSERT INTO plays(spotify_id, played_at, context, ms_played) VALUES(?,?,?,?)",
                    (sid("kept_liked"), t, ctx, ms))
    con.commit()
    pl = report.plays(con)
    assert len(pl["weeks"]) == report.WEEKS and list(pl["weeks"])[-1] == report._weeks()[-1]
    assert pl["weeks"][report._weeks()[-1]] == {"needle": 3, "spotify": 1, "needle_with_length": 2, "needle_short": 1}
    assert (pl["needle_with_length"], pl["needle_short_share"]) == (2, 0.5)
    # nothing has evidence enough yet: every gate closed, none opened before
    g = report.gates(con, cfg)
    assert set(g) == {"auc_objective", "rerank", "exploration", "adaptive_caps"}
    assert not any(x["open"] or x["opened_at"] for x in g.values())
    assert g["auc_objective"]["value"] is None and g["rerank"]["gain"] is None
    assert g["rerank"]["value"] == len(rerank.labels(con, cfg)) and g["rerank"]["soft"] == 0
    # an AUC eval with enough of both kinds, a re-ranker run that beat stage 1, a busy source
    con.execute("INSERT INTO eval_runs(at, trigger, objective, n_pos, n_neg) VALUES('2026-01-12T00:00:00Z', 'weekly', "
                "'auc', 20, 16)")
    con.execute("INSERT INTO rerank_runs(at, trigger, variant, n_labels, loo_auc_model, loo_auc_stage1, used) "
                "VALUES('2026-01-19T00:00:00Z', 'batch', 'raw', 6, 0.81, 0.7, 1)")
    cfg["rerank"]["min_labels"] = 1
    cfg["harvest"]["sources"] = ["src_a", "src_b"]
    busiest = max(sources.rates(con)["per_source"][s]["decided"] for s in ("src_a", "src_b"))
    cfg["sources"]["min_offers_before_retire"] = busiest
    g = report.gates(con, cfg)
    assert g["auc_objective"] == {**g["auc_objective"], "value": 16, "threshold": 15, "open": True,
                                  "opened_at": "2026-01-12T00:00:00Z"}
    assert (g["rerank"]["open"], g["rerank"]["gain"], g["rerank"]["opened_at"]) == (True, 0.11, "2026-01-19T00:00:00Z")
    assert g["adaptive_caps"]["open"] and g["adaptive_caps"]["value"] == busiest
    cfg["eval"]["objective"] = "mean_pct"                                    # pinned to the old rule: closed
    assert not report.gates(con, cfg)["auc_objective"]["open"]
    cfg["eval"]["objective"] = "auc"
    txt = report.text(report.build(con, cfg, STATUS))
    assert "Plays per week" in txt and "Gates" in txt and "auc_objective  open" in txt and "1/2" in txt


def test_report_on_an_empty_install(tmp_path):
    path = tmp_path / "fresh.db"
    db.connect(path).close()
    r = report.build(db.connect_readonly(path), config.load(tmp_path / "none.toml"), {})
    assert r["offers"]["all"]["offered"] == 0 and r["head"] == {"n": 0} and r["eval"] is None
    txt = report.text(r)
    assert "none yet" in txt


def test_cli_report_is_read_only(tmp_path, monkeypatch, capsys):
    path, cfg = _world(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(config, "STATUS_PATH", tmp_path / "status.json")
    (tmp_path / "status.json").write_text(json.dumps(STATUS))
    monkeypatch.setattr(cli, "job", lambda *a, **k: pytest.fail("report must not record a run"))
    before = path.read_bytes()
    p = cli.build_parser()
    assert cli.cmd_report(cfg, p.parse_args(["report"])) == 0
    out = capsys.readouterr().out
    assert out.startswith("haystack report") and "Latest eval" in out and "Secret" not in out
    assert cli.cmd_report(cfg, p.parse_args(["report", "--json"])) == 0
    assert json.loads(capsys.readouterr().out)["pool"]["scored"] == 4
    assert path.read_bytes() == before
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "missing.db")
    assert cli.cmd_report(cfg, p.parse_args(["report"])) == 1
