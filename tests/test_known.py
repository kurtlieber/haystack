"""Songs the listener already has, on any release (ADR 0021, note of 2026-10-23): the batch and Calibrate skip
them and the hourly sync takes them out of the pool. Synthetic data; no Spotify, no Deezer."""
import random
from types import SimpleNamespace

import numpy as np

from haystack_music import batch, calibrate, db, known, sync, vectors, verdicts
from haystack_music.scoring import l2norm
from test_batch_plan import _world, sid


def _more(con):
    """On top of _world's pool: two other releases of liked song 1 (a remaster, a remix), the same song as a
    library file, a new song by the liked artist, the liked song's ISRC under other names, and a judged song."""
    con.execute("""INSERT INTO local_tracks(id, path, size, content_hash, artist_raw, title_raw, artists_json,
                   duration, present) VALUES(7, '/m/7.m4a', 1, 'h7', 'File Artist', 'File Title',
                   '["File Artist"]', 200.0, 1)""")
    con.execute("UPDATE spotify_tracks SET duration=210.0, isrc='USAA10000001' WHERE spotify_id=?", (sid("like1"),))
    con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, duration) VALUES(?, 'Judged Title', "
                "'[\"Judged Artist\"]', 180.0)", (sid("judged"),))
    verdicts.record(con, sid("judged"), "pass", None)
    rng = np.random.default_rng(3)
    base = vectors.get_deezer_vectors(con, [1001])[1001]

    def cand(cid, artist, title, duration=None, isrc=None):
        con.execute("""INSERT INTO candidates(id, deezer_id, artist, artist_key, title, song_key, duration, isrc,
                       source, state, vector_status) VALUES(?,?,?,?,?,?,?,?,'src_a','pool','ok')""",
                    (cid, cid, artist, artist.lower(), title, f"song {cid}", duration, isrc))
        vectors.put_deezer_vector(con, cid, l2norm(base + 0.01 * rng.normal(size=512)), np.zeros(1280), 10, 30.0,
                                  np.abs(rng.normal(size=400)))
    cand(44, "Liked Artist 1", "Liked Title 1 (2015 Remaster)", 211.0)     # the liked song, another release
    cand(45, "Liked Artist 1", "Liked Title 1 (Somebody Remix)", 211.0)    # a remix: a different recording
    cand(46, "File Artist", "File Title", 199.0)                           # a song in the library
    cand(47, "Liked Artist 1", "Another Song", 150.0)                      # new to the listener
    cand(48, "Someone Else", "Different Name", 300.0, isrc="USAA10000001")  # the liked song's ISRC
    cand(49, "Judged Artist", "Judged Title (Remastered)", 181.0)          # a Passed song, another release
    cand(50, "Liked Artist 1", 'Liked Title 1 - From "Some Film" Soundtrack', 215.0)   # decorated, 5 s off
    cand(51, "Liked Artist 1", "Liked Title 1 (Extended)", 260.0)         # a longer cut of the liked song
    cand(52, "Liked Artist 1", "Liked Title 1 (Acoustic)", 212.0)         # an acoustic version: a different recording
    cand(53, "Liked Artist 1", "Liked Title 1 2", 212.0)                   # a sequel
    cand(54, "Liked Artist 1", "Liked Title 10", 210.0)                    # another song with a title alike
    cand(55, "Liked Artist 1", "Liked Title 1 (Take 1)", 210.0)            # another take
    cand(56, "Liked Artist 1", "Liked Title 1", 30.0)                      # Deezer's length is the preview's
    return {cid: dict(con.execute("SELECT * FROM candidates WHERE id=?", (cid,)).fetchone()) for cid in range(44, 57)}


def test_known_tells_other_releases_from_other_recordings(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    rows = _more(con)
    k = known.Known(con)
    assert {cid: k.why(r) for cid, r in rows.items()} == {44: "liked", 45: None, 46: "in_library", 47: None,
                                                        48: "liked", 49: "judged", 50: "liked", 51: "liked",
                                                        52: None, 53: None, 54: None, 55: None, 56: "liked"}


def test_the_batch_and_the_plan_skip_songs_the_listener_has(tmp_path, monkeypatch):
    con, cfg, path = _world(tmp_path, monkeypatch)
    _more(con)
    vectors.score_candidates(con, cfg, variant="raw")
    stats = {}
    picks = batch.select_for_batch(con, cfg, random.Random(0), size=60, stats=stats,
                                   resolve=lambda c: c["spotify_id"] or sid(f"m{c['id']}"))
    ids = {p.cand_id for p in picks}
    assert not ids & {44, 46, 48, 49, 50, 51, 56} and ids & {45, 47, 52, 53, 54, 55}   # one: one song per artist
    assert stats["known_songs"] == 7
    plan = batch.plan(db.connect_readonly(path), cfg, rng=random.Random(0), size=60)
    assert plan["known_songs"] == 7
    assert "Liked Title 1 (2015 Remaster)" not in {t["title"] for t in plan["tracks"]}


def test_calibrate_skips_songs_the_listener_has(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    _more(con)
    ids = {r["id"] for r in calibrate.candidates(con, cfg)["rows"]}
    assert not ids & {44, 46, 48, 49, 50, 51, 56} and {45, 47, 52, 53, 54, 55} <= ids


def test_the_sync_takes_liked_and_owned_songs_out_of_the_pool(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    _more(con)
    assert known.exclude_known(con, cfg) == {"checked": 56, "in_library": 1, "liked": 5}
    state = dict(con.execute("SELECT id, state || ':' || COALESCE(exclude_reason, '') FROM candidates "
                             "WHERE id >= 44").fetchall())
    assert state == {44: "excluded:liked", 45: "pool:", 46: "excluded:in_library", 47: "pool:",
                     48: "excluded:liked", 49: "pool:",      # a judged song's other release is skipped, not excluded
                     50: "excluded:liked", 51: "excluded:liked", 52: "pool:", 53: "pool:", 54: "pool:", 55: "pool:",
                     56: "excluded:liked"}
    assert known.exclude_known(con, cfg) == {"checked": 50, "in_library": 0, "liked": 0}
    # the hourly sync runs it after the likes, before the Calibrate top-up
    for name in ("ensure_playlists", "poll_plays", "poll_likes", "map_likes", "liked_preview_vectors",
                 "process_offers", "torip_check", "token_check"):
        monkeypatch.setattr(sync, name, lambda *a, **k: {})
    monkeypatch.setattr(sync.calibrate, "top_up", lambda *a, **k: {})
    monkeypatch.setattr(sync.verdicts, "apply_effects", lambda *a, **k: {})
    monkeypatch.setattr(sync.playlists, "sync", lambda *a, **k: {})
    out = sync.run(con, cfg, sp=SimpleNamespace(token_meta={}))
    assert out["known_songs"] == {"checked": 50, "in_library": 0, "liked": 0}
