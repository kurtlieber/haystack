"""batch --plan and the selection it shares with the live batch. Synthetic data; no Spotify, no Deezer."""
import hashlib
import random
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from haystack_music import batch, cli, config, db, vectors
from haystack_music.scoring import l2norm


def sid(n: str) -> str:
    return n.ljust(22, "0")


def _world(tmp_path, monkeypatch):
    """Three liked songs (preview vectors) and a pool of 40 candidates near them, half already mapped to Spotify.
    Plus one candidate per filter: its track is already liked, its source is retired, Spotify has no match."""
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    path = tmp_path / "t.db"
    con = db.connect(path)
    rng = np.random.default_rng(7)
    base = [l2norm(rng.normal(size=512)) for _ in range(3)]
    for i, b in enumerate(base, start=1):
        s = sid(f"like{i}")
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (s,))
        con.execute("INSERT INTO liked_map(spotify_id, use_local, deezer_id, vector_status) VALUES(?, 0, ?, 'ok')",
                    (s, 1000 + i))
        con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json) VALUES(?, ?, ?)",
                    (s, f"Liked Title {i}", f'["Liked Artist {i}"]'))
        vectors.put_deezer_vector(con, 1000 + i, b, np.zeros(1280), 10, 30.0, np.abs(rng.normal(size=400)))

    def cand(cid, *, source="src_a", spotify_id=None, spotify_status=None, near=0, noise=0.5):
        con.execute("""INSERT INTO candidates(id, deezer_id, artist, artist_key, title, song_key, source, state,
                       vector_status, spotify_id, spotify_status) VALUES(?,?,?,?,?,?,?, 'pool', 'ok', ?, ?)""",
                    (cid, cid, f"Artist {cid}", f"artist {cid}", f"Title {cid}", f"song {cid}", source,
                     spotify_id, spotify_status))
        v = base[near] + noise * rng.normal(size=512)
        vectors.put_deezer_vector(con, cid, l2norm(v), np.zeros(1280), 10, 30.0, np.abs(rng.normal(size=400)))

    for cid in range(1, 41):
        cand(cid, source="src_a" if cid % 2 else "src_b", spotify_id=sid(f"c{cid}") if cid % 2 else None,
             near=cid % 3, noise=0.02 * cid)
    cand(41, spotify_id=sid("like1"), noise=0.0)                   # the same track as a liked song
    cand(42, source="src_retired", noise=0.0)
    cand(43, spotify_status="not_found", noise=0.0)
    cfg = config.load(tmp_path / "none.toml")
    cfg["scoring"]["variant"] = "raw"
    cfg["harvest"]["retired_sources"] = ["src_retired"]
    return con, cfg, path


def _digest(path) -> str:
    con = sqlite3.connect(path)
    try:
        return hashlib.sha256("\n".join(con.iterdump()).encode()).hexdigest()
    finally:
        con.close()


def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("batch --plan must not reach Spotify or Deezer")
    monkeypatch.setattr(batch, "Spotify", boom)
    monkeypatch.setattr(batch, "Deezer", boom)
    monkeypatch.setattr(batch, "map_to_spotify", boom)


def test_plan_scores_in_memory_and_writes_nothing(tmp_path, monkeypatch):
    con, cfg, path = _world(tmp_path, monkeypatch)
    _no_network(monkeypatch)
    con.close()
    before = _digest(path)
    ro = db.connect_readonly(path)
    out = batch.plan(ro, cfg, rng=random.Random(1))
    assert out["plan"] is True and out["variant"] == "raw"
    assert out["picked"] == cfg["batch"]["size"] == 15
    assert out["exploration"] == round(15 * cfg["batch"]["exploration_fraction"])
    ids = {t["artist"] for t in out["tracks"]}
    assert not ids & {"Artist 41", "Artist 42", "Artist 43"}      # known, retired, not found
    assert out["unmapped"] == sum(1 for t in out["tracks"] if t["spotify_id"] is None) > 0
    for t in out["tracks"]:
        assert t["nearest"].startswith("like:")
        assert t["sounds_like"]["artist"].startswith("Liked Artist")
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("UPDATE candidates SET score=1")
    ro.close()
    assert _digest(path) == before
    con = db.connect(path)
    assert con.execute("SELECT COUNT(*) FROM candidates WHERE score IS NOT NULL").fetchone()[0] == 0



def test_plan_rechecks_an_auto_variant_without_writing(tmp_path, monkeypatch):
    """scoring.variant auto: the plan runs the weekly eval in memory and says which variant the batch would use."""
    con, cfg, path = _world(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "STATUS_PATH", tmp_path / "state" / "status.json")
    _no_network(monkeypatch)
    cfg["scoring"]["variant"] = "auto"
    con.close()
    before = _digest(path)
    out = batch.plan(db.connect_readonly(path), cfg, rng=random.Random(1))
    c = out["variant_choice"]
    assert out["variant"] == c["used"] == "raw" and c["switched"] is False
    assert c["decision"].startswith("kept: not enough data")             # three liked songs: too few for an eval
    assert _digest(path) == before and not config.STATUS_PATH.exists()

def test_plan_picks_what_the_live_selection_picks(tmp_path, monkeypatch):
    """Same pool, same seed: the in-memory scores (--plan) and the stored scores (the live batch) select the same
    candidates with the same exploration flags."""
    con, cfg, path = _world(tmp_path, monkeypatch)
    planned = batch.plan(db.connect_readonly(path), cfg, rng=random.Random(3))
    vectors.score_candidates(con, cfg, variant="raw")
    live = batch.select_for_batch(con, cfg, random.Random(3), resolve=lambda c: c["spotify_id"] or sid(f"m{c['id']}"))
    assert [(f"Artist {p.cand_id}", p.exploration) for p in live] == \
        [(t["artist"], t["exploration"]) for t in planned["tracks"]]
    assert all(p.spotify_id for p in live) and all(p.nearest.startswith("like:") for p in live)


def test_selection_callbacks(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    vectors.score_candidates(con, cfg, variant="raw")
    known, tried = [], []

    def resolve(c):
        tried.append(c["id"])
        return None if c["id"] == 1 else (c["spotify_id"] or sid(f"m{c['id']}"))

    picks = batch.select_for_batch(con, cfg, random.Random(0), size=40, resolve=resolve,
                                   on_known=lambda c: known.append(c["id"]))
    assert 1 in tried and 1 not in {p.cand_id for p in picks}      # no Spotify match: dropped
    assert known == [41]                                            # already liked: reported, skipped
    assert not {42, 43} & set(tried)                                # filtered before selection


def test_cli_plan_is_token_free(tmp_path, monkeypatch, capsys):
    con, cfg, path = _world(tmp_path, monkeypatch)
    con.close()
    _no_network(monkeypatch)
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(cli, "job", lambda *a, **k: pytest.fail("--plan must not record a run"))
    p = cli.build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["batch", "--plan", "--dry-run"])
    a = p.parse_args(["batch", "--plan", "--size", "5"])
    assert cli.cmd_batch(cfg, a) == 0
    out = capsys.readouterr().out
    assert "plan: variant raw" in out and "5 picks" in out and "(nearest: Liked Artist" in out
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "missing.db")
    assert cli.cmd_batch(cfg, a) == 1
    assert not (tmp_path / "missing.db").exists()


def test_connect_readonly_needs_a_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        db.connect_readonly(tmp_path / "none.db")
    assert not (tmp_path / "none.db").exists()


def test_cmd_batch_without_plan_still_runs_the_job(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "job", lambda name, fn, needs_spotify=False: seen.append((name, needs_spotify)) or 0)
    a = SimpleNamespace(plan=False, size=None, dry_run=True, force=False, variant=None, clear_min_age_hours=None,
                        clear_recent=None, no_penalty=None)
    assert cli.cmd_batch({}, a) == 0 and seen == [("batch", True)]
