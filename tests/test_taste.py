"""The Taste brief (ADR 0028): what the listener's Verdicts say about styles, tempo and instrumental level."""
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from haystack_music import baseline, config, db, taste, traits, vectors, verdicts

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)
CLASSES = [f"Genre{i // 100}---Style{i}" for i in range(400)]


def sid(n: str) -> str:
    return n.ljust(22, "0")


def add(con, name: str, deezer_id: int | None, style: int | None, readings: dict | None, isrc: str | None = None,
        title: str | None = None, artist: str = "Artist"):
    con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, isrc) VALUES(?,?,?,?)",
                (sid(name), title or name, json.dumps([artist]), isrc))
    if deezer_id is None:
        return
    con.execute("INSERT INTO candidates(deezer_id, artist, title, source, state, vector_status, spotify_id) "
                "VALUES(?, ?, ?, 'deezer_related', 'pool', 'ok', ?)", (deezer_id, artist, title or name, sid(name)))
    if style is not None:
        vec = np.zeros(400)
        vec[style] = 0.6
        vec[style + 1] = 0.05                    # below the 0.10 a style needs to count
        vectors.put_deezer_vector(con, deezer_id, np.ones(512), np.zeros(1280), 10, 30.0, vec)
    if readings:
        traits.put(con, deezer_id, readings)


def rate(con, name: str, kind: str, stars=None, given="2026-11-01T00:00:00Z", queue=None):
    verdicts.record(con, sid(name), kind, stars, given_at=given, queue=queue)


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.db")
    # new to the listener: 5 stars, fast, vocal, style 10
    add(c, "a", 1, 10, {"tempo": 170.0, "instrumentalness": 0.02, "valence": 0.4}, isrc="XXA0000001", title="Alpha")
    rate(c, "a", "rating", 5, "2026-11-10T00:00:00Z", "discoveries")
    # new: 4 stars, style 10, mid tempo, instrumental
    add(c, "b", 2, 10, {"tempo": 120.0, "instrumentalness": 0.9})
    rate(c, "b", "rating", 4, "2026-11-09T00:00:00Z", "calibrate")
    # new: a Dislike, style 20, recent
    add(c, "c", 3, 20, {"tempo": 100.0, "instrumentalness": 0.4})
    rate(c, "c", "dislike", None, "2026-11-08T00:00:00Z", "discoveries")
    # a liked song (not new), 5 stars, style 20, played three times; an old Verdict
    add(c, "d", 4, 20, {"tempo": 140.0, "instrumentalness": 0.1}, title="Delta")
    rate(c, "d", "rating", 5, "2026-01-01T00:00:00Z", "likes")
    for t in ("2026-11-01T00:00:00Z", "2026-11-02T00:00:00Z", "2026-11-03T00:00:00Z"):
        c.execute("INSERT INTO plays(spotify_id, played_at, context) VALUES(?, ?, 'needle')", (sid("d"), t))
    # unmeasured: counts among all songs only
    add(c, "e", None, None, None)
    rate(c, "e", "pass", None, "2026-11-05T00:00:00Z", "discoveries")
    # two releases of one Song: the newer Verdict (a Pass on the other release) is the Song's
    add(c, "a2", 5, 10, {"tempo": 170.0}, isrc="XXA0000001", title="Alpha (Remastered)")
    rate(c, "a2", "pass", None, "2026-11-11T00:00:00Z", "playlists")
    # an old 5-star discovery, outside the 45 days
    add(c, "f", 6, 30, {"tempo": 90.0, "instrumentalness": 0.5})
    rate(c, "f", "rating", 5, "2026-08-01T00:00:00Z", "discoveries")
    return c


def brief(con, **kw):
    return taste.brief(con, config.load(None), CLASSES, now=NOW, **kw)


def test_one_song_per_isrc_and_its_newest_verdict(con):
    b = brief(con)
    assert b["counts"] == {"songs": 6, "measured": 5, "new": 3, "recent_five_stars": 0, "recent_dislikes": 1}
    # the Song a/a2 counts once, as the Pass given last, on the a2 release (not new to the listener)
    assert b["totals"]["all"] == {"n": 6, "five": 2, "four_five": 3, "down": 3}
    assert b["totals"]["new"] == {"n": 3, "five": 1, "four_five": 2, "down": 1}


def test_style_tables_keep_groups_of_min_n_and_put_the_best_five_star_rate_first(con):
    b = brief(con, min_n=1)
    rows = b["styles"]["all"]
    assert [(r["style"], r["genre"]) for r in rows] == [("Style30", "Genre0"), ("Style20", "Genre0"),
                                                        ("Style10", "Genre0")]
    assert rows[1] == {"style": "Style20", "genre": "Genre0", "n": 2, "five": 1, "four_five": 1, "down": 1}
    assert all(r["style"] != "Style11" for r in rows)          # 0.05 is under the activation a style needs
    assert [r["style"] for r in brief(con, min_n=2)["styles"]["all"]] == ["Style20", "Style10"]


def test_bands_cover_every_range_even_empty_ones(con):
    b = brief(con)
    tempo = b["tempo"]["all"]
    assert [(t["lo"], t["hi"]) for t in tempo] == [(0, 95), (95, 110), (110, 132), (132, 145), (145, 165),
                                                   (165, 200), (200, None)]
    assert tempo[5] == {"lo": 165, "hi": 200, "n": 1, "five": 0, "four_five": 0, "down": 1}   # a2's Pass
    inst = b["instrumentalness"]["new"]
    assert [(i["lo"], i["hi"]) for i in inst] == [(0, 0.05), (0.05, 0.3), (0.3, 0.8), (0.8, 1)]
    assert inst[3]["n"] == 1 and inst[3]["four_five"] == 1
    assert inst[0]["n"] == 0                                    # a2 has no instrumentalness: in no band


def test_recent_lists_and_most_played(con):
    b = brief(con)
    assert [s["title"] for s in b["recent"]["five_stars"]] == []        # a's Song is a2's Pass now; f is too old
    assert [s["title"] for s in b["recent"]["dislikes"]] == ["c"]
    d = b["recent"]["dislikes"][0]
    assert d["id"] == sid("c") and d["artist"] == "Artist" and d["kind"] == "dislike" and d["stars"] is None
    assert d["readings"] == {"tempo": 100.0, "instrumentalness": 0.4}
    assert d["styles"][:2] == [["Style20", 0.6], ["Style21", 0.05]] and len(d["styles"]) == 3
    assert [(s["title"], s["plays"]) for s in b["most_played"]] == [("Delta", 3)]
    assert brief(con, days=200)["recent"]["five_stars"][0]["title"] == "f"


def test_the_recent_lists_are_capped_but_their_counts_are_not(con, monkeypatch):
    monkeypatch.setattr(taste, "RECENT_DISLIKES", 0)
    b = brief(con)
    assert b["recent"]["dislikes"] == [] and b["counts"]["recent_dislikes"] == 1


def test_the_trait_ranges_are_the_baselines(con):
    assert brief(con)["traits"] == baseline.compute(con, config.load(None))["traits"]


def test_no_verdicts_and_no_style_names(tmp_path):
    c = db.connect(tmp_path / "empty.db")
    b = taste.brief(c, config.load(None), [], now=NOW)
    assert b["counts"] == {"songs": 0, "measured": 0, "new": 0, "recent_five_stars": 0, "recent_dislikes": 0}
    assert b["styles"] == {"all": [], "new": []}
    assert b["totals"]["all"] == {"n": 0, "five": 0, "four_five": 0, "down": 0}
    assert b["generated_at"] == "2026-11-12T12:00:00Z" and b["days"] == taste.DAYS


def test_the_brief_never_writes(con, tmp_path):
    before = con.total_changes
    brief(con)
    assert con.total_changes == before


def test_the_brief_loads_no_language_model_review_or_rerank_code():
    import os
    import subprocess
    import sys
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src"
    r = subprocess.run([sys.executable, "-B", "-c", "import sys, haystack_music.taste; "
                        "print(' '.join(m for m in sys.modules if m.startswith('haystack_music.')))"],
                       env={**os.environ, "PYTHONPATH": str(src)}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    loaded = set(r.stdout.split())
    assert "haystack_music.taste" in loaded
    assert not loaded & {"haystack_music.llm", "haystack_music.review", "haystack_music.rerank"}, loaded
