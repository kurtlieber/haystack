"""The preview alignment (ADR 0012): fit, store, use, refit, and the eval's check. Synthetic data; no Spotify, no
Deezer."""
import json

import numpy as np
import pytest

from haystack_music import align, batch, config, db, evaluate, vectors
from haystack_music.scoring import l2norm


def sid(n) -> str:
    return f"s{n:05}".ljust(22, "x")


def _domains(k=12, seed=0):
    """Songs live in a k-dimensional subspace. A library file is the song plus the library's common offset; its
    preview is the song rotated within the subspace plus another offset (the two domains)."""
    rng = np.random.default_rng(seed)
    B = np.linalg.qr(rng.normal(size=(512, k)))[0]                     # orthonormal basis of the subspace
    R = np.linalg.qr(rng.normal(size=(k, k)))[0]                       # the rotation between the domains
    c_lib, c_prev = 1.5 * l2norm(rng.normal(size=512)), 1.5 * l2norm(rng.normal(size=512))

    def song():
        return B @ rng.normal(size=k)

    def lib(z, n=3):
        return l2norm(z + c_lib + 0.05 * rng.normal(size=(n, 512))).astype(np.float32)

    def prev(z):
        return l2norm(B @ (R @ (B.T @ z)) + c_prev + 0.05 * rng.normal(size=512)).astype(np.float32)

    return song, lib, prev


def _world(tmp_path, monkeypatch, pairs=60, like_only=20, lib_only=40, pool=30):
    """pairs liked songs with a local file and a preview, like_only liked songs with a preview alone, lib_only
    library files nobody liked, and a pool of candidates (previews of new songs)."""
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "STATUS_PATH", tmp_path / "state" / "status.json")
    config.EMB_DIR.mkdir(parents=True)
    con = db.connect(tmp_path / "t.db")
    song, lib, prev = _domains()
    did = iter(range(1000, 9000))
    lid = iter(range(1, 9000))

    def local(z, name):
        i = next(lid)
        f = config.EMB_DIR / f"h{i}.npz"
        np.savez(f, win512=lib(z))
        con.execute("INSERT INTO local_tracks(id, path, size, content_hash, artist_raw, title_raw, present) "
                    "VALUES(?, ?, 1, ?, ?, ?, 1)", (i, f"/m/{i}.m4a", f"h{i}", f"Artist {name}", f"Title {name}"))
        con.execute("INSERT INTO local_embeddings(content_hash, status, attempts, n_windows, file) "
                    "VALUES(?, 'ok', 1, 3, ?)", (f"h{i}", str(f)))
        return i

    def preview(z):
        d = next(did)
        vectors.put_deezer_vector(con, d, prev(z), np.zeros(1280), 10, 30.0, None)
        return d

    for n in range(pairs):
        z = song()
        i = local(z, n)
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(n),))
        con.execute("INSERT INTO liked_map(spotify_id, use_local, local_id, deezer_id, vector_status) "
                    "VALUES(?, 1, ?, ?, 'ok')", (sid(n), i, preview(z)))
    for n in range(pairs, pairs + like_only):
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(n),))
        con.execute("INSERT INTO liked_map(spotify_id, use_local, deezer_id, vector_status) VALUES(?, 0, ?, 'ok')",
                    (sid(n), preview(song())))
    for n in range(lib_only):
        local(song(), f"lib{n}")
    for n in range(pool):
        con.execute("""INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status)
                       VALUES(?, ?, ?, ?, 'src_a', 'pool', 'ok')""", (n + 1, preview(song()), f"A{n}", f"T{n}"))
    cfg = config.load(tmp_path / "none.toml")
    cfg["library"]["root"] = "/m"
    cfg["scoring"]["align_min_pairs"] = 40
    return con, cfg, (song, lib, prev)


def test_a_rotation_between_two_domains_is_recovered_and_improves_self_retrieval(tmp_path, monkeypatch):
    con, cfg, (song, lib, prev) = _world(tmp_path, monkeypatch)
    r = align.fit(con, cfg)
    assert r["status"] == "fitted" and r["n_pairs"] == 60 and r["improved"]
    assert r["unaligned"]["top10"] < 0.5                    # previews find each other, not their own files
    assert r["cv"]["top1"] > 0.9 and r["cv"]["top10"] == 1.0
    assert align.path().exists() and db.kv_get(con, "preview_align_version") == r["version"]
    m = align.active(con, cfg)
    assert m is not None and m.meta["method"] == r["method"]
    z = [song() for _ in range(20)]                          # new songs: the map was not fitted on them
    got = l2norm(m(np.vstack([prev(x) for x in z])))
    want = l2norm(np.vstack([lib(x, 1)[0] for x in z]))
    ceiling = np.mean(np.sum(want * np.vstack([lib(x, 1)[0] for x in z]), 1))     # two files of the same song
    assert np.mean(np.sum(got * want, 1)) > ceiling - 0.05
    assert np.mean(np.sum(l2norm(np.vstack([prev(x) for x in z])) * want, 1)) < 0.5


def test_off_scores_exactly_as_without_a_map(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    before = vectors.compute_scores(con, cfg, "raw")
    align.fit(con, cfg)
    cfg_off = align.with_mode(cfg, "off")
    off = vectors.compute_scores(con, cfg_off, "raw")
    for k in ("score", "base", "raw", "idx"):
        assert np.array_equal(np.asarray(before[k]), np.asarray(off[k])), k
    assert before["owners"] == off["owners"]
    on = vectors.compute_scores(con, cfg, "raw")
    assert not np.array_equal(np.asarray(before["score"]), np.asarray(on["score"]))
    assert np.array_equal(vectors.preview_vectors(con, cfg_off, [1000])[1000],
                          vectors.get_deezer_vectors(con, [1000])[1000])


def test_alignment_moves_the_nearest_positive_from_previews_to_files(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    off = vectors.compute_scores(con, align.with_mode(cfg, "off"), "raw")
    align.fit(con, cfg)
    on = vectors.compute_scores(con, cfg, "raw")

    def local_share(r):
        return np.mean([r["owners"][int(i)].startswith("local:") for i in r["idx"]])
    assert local_share(on) > local_share(off)


def test_too_few_pairs_means_no_map_and_removes_a_stored_one(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    assert align.fit(con, cfg)["status"] == "fitted" and align.path().exists()
    cfg["scoring"]["align_min_pairs"] = 100
    r = align.fit(con, cfg)
    assert r["status"] == "too few pairs" and "need 100" in r["reason"]
    assert not align.path().exists() and db.kv_get(con, "preview_align_version") is None
    assert align.active(con, cfg) is None


def test_no_library_means_no_map_and_no_domain_numbers(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    align.fit(con, cfg)
    cfg["library"]["root"] = None
    assert align.active(con, cfg) is None and align.candidate(con, cfg) is None
    assert evaluate.domain(con, cfg, {}) == {"status": "no library"}
    assert "no library" in "\n".join(evaluate._align_md({"domain": {"status": "no library"}}))
    r = align.fit(con, cfg)
    assert r["status"] == "no library" and not align.path().exists()


def test_the_library_job_refits_when_the_pairs_move_by_more_than_a_tenth(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    cfg["scoring"]["align_min_pairs"] = 100
    assert align.refit(con, cfg)["skipped"]                                  # 60 pairs, no map: nothing to do
    cfg["scoring"]["align_min_pairs"] = 40
    assert align.refit(con, cfg)["status"] == "fitted"                       # no map yet: fit
    v = align.load().meta["version"]
    con.execute(f"UPDATE likes SET removed_at='2026-02-01' WHERE spotify_id IN ({','.join('?' * 6)})",
                [sid(n) for n in range(6)])
    assert align.refit(con, cfg)["skipped"]                                  # 54 of 60: within a tenth
    con.execute("UPDATE likes SET removed_at='2026-02-01' WHERE spotify_id=?", (sid(6),))
    r = align.refit(con, cfg)                                                # 53 of 60: refit
    assert r["status"] == "fitted" and r["n_pairs"] == 53 and r["version"] >= v
    assert align.refit(con, align.with_mode(cfg, "off"))["skipped"]


def test_an_unknown_mode_is_refused(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch, pairs=0, like_only=1, lib_only=1, pool=1)
    with pytest.raises(ValueError, match="scoring.align"):
        align.active(con, align.with_mode(cfg, "on"))


def _ev(objective="mean_pct", **variants):
    return {"objective_used": objective, "recall_k": 15, "held_out_per_seed": {"preview": 10}, "chosen": "raw",
            "variants": {v: {"auc": a, "mean_pct": p, "recall_at_15": r} for v, (a, p, r) in variants.items()}}


def test_the_gate_turns_alignment_off_only_when_it_is_worse_for_the_variant_used():
    cfg = config.load(path=None)
    tie = cfg["eval"]["tie_tolerance"]
    g = evaluate.align_gate(_ev(raw=(None, 60.0, 0.1)), _ev(raw=(None, 60.0 + tie / 2, 0.1)), "raw", cfg)
    assert g["ok"] and g["objective"] == "mean_pct"                          # within the tie
    g = evaluate.align_gate(_ev(raw=(None, 58.0, 0.1)), _ev(raw=(None, 60.0, 0.1)), "raw", cfg)
    assert not g["ok"] and g["why"].startswith("off: mean_pct")
    g = evaluate.align_gate(_ev(raw=(None, 60.0, 0.1), genre=(None, 50, 0)),
                            _ev(raw=(None, 70.0, 0.1), genre=(None, 50, 0)), "genre", cfg)
    assert g["ok"]                                                           # worse only for another variant
    g = evaluate.align_gate(_ev(raw=(None, 60.0, 0.05)), _ev(raw=(None, 60.0, 0.2)), "raw", cfg)
    assert not g["ok"] and "recall_at_15" in g["why"]                        # recall lower by > one song a seed
    g = evaluate.align_gate(_ev(raw=(None, 60.0, 0.15)), _ev(raw=(None, 60.0, 0.2)), "raw", cfg)
    assert g["ok"]                                                           # one song a seed or less
    g = evaluate.align_gate(_ev("auc", raw=(0.60, 50, 0.1)), _ev("auc", raw=(0.66, 70, 0.1)), "raw", cfg)
    assert not g["ok"] and g["objective"] == "auc"                           # by AUC when both used it
    g = evaluate.align_gate(_ev("auc", raw=(0.66, 50, 0.1)), _ev("auc", raw=(0.66, 70, 0.1)), "raw", cfg)
    assert g["ok"]
    assert evaluate.align_gate({"skipped": True}, {"skipped": True}, "raw", cfg)["ok"]


@pytest.mark.parametrize("worse", [False, True])
def test_the_eval_runs_both_ways_and_records_its_verdict(tmp_path, monkeypatch, worse):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    align.fit(con, cfg)
    db.kv_set(con, "scoring_variant", "raw")
    seen = []

    def run(con, cfg, data=None):
        mode = cfg["scoring"]["align"]
        seen.append(mode)
        p = 61.0 if mode == "candidate" else 60.0
        if worse:
            p = 50.0 if mode == "candidate" else 60.0
        return {**_ev(raw=(None, p, 0.1)), "domain": {"status": "ok", "aligned": mode == "candidate"}}
    monkeypatch.setattr(evaluate, "run", run)
    ev = evaluate.choose(con, cfg, "report")
    assert seen == ["candidate", "off"]
    g = json.loads(db.kv_get(con, "preview_align_gate"))
    assert g["ok"] is (not worse) and g["map_version"] == align.load().meta["version"]
    assert ev["align"]["gate"]["ok"] is (not worse)
    assert ev["variants"]["raw"]["mean_pct"] == (60.0 if worse else 61.0)   # the run kept is the one used
    assert (align.active(con, cfg) is None) is worse
    assert align.candidate(con, cfg) is not None                             # the next eval tries it again
    assert "Preview alignment" in "\n".join(evaluate._align_md(ev))
    seen.clear()
    evaluate.choose(con, align.with_mode(cfg, "off"), "report")
    assert seen == ["off"]                                                   # off: one run, no verdict


def test_a_pinned_variant_still_checks_the_alignment(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    cfg["scoring"]["variant"] = "raw"
    assert batch._rechoose(con, cfg, write=False) is None                   # no map: nothing to check
    align.fit(con, cfg)
    monkeypatch.setattr(evaluate, "run", lambda con, cfg, data=None: _ev(raw=(None, 60.0, 0.1)))
    r = batch._rechoose(con, cfg, write=False)
    assert r["used"] == "raw" and r["decision"] == "pinned in config"
