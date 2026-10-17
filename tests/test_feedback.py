import pytest

from haystack_music import db, feedback, verdicts
from haystack_music.textnorm import song_key


def _con(tmp_path):
    con = db.connect(tmp_path / "t.db")
    for i, (a, t) in enumerate([("wavve", "Afterglow (Spa)"), ("Tom Odell", "Another Love"), ("Never", "Offered")]):
        con.execute("INSERT INTO candidates(deezer_id, artist, title, song_key, source, state, vector_status, spotify_id) "
                    "VALUES(?,?,?,?,?,?,?,?)", (i + 1, a, t, song_key(a, t), "deezer_chart", "pool", "ok",
                                                f"sp{i}".ljust(22, "0") if i < 2 else None))
    return con


def test_feedback_is_an_alias_of_verdicts(tmp_path):
    con = _con(tmp_path)
    r = feedback.record(con, "no", "wavve", "Afterglow (Spa)", "strong", "hard no")
    assert (r["kind"], r["stars"]) == ("dislike", None)
    feedback.record(con, "yes", "Tom Odell", "Another Love")
    states = dict(con.execute("SELECT artist, state FROM candidates WHERE spotify_id IS NOT NULL").fetchall())
    assert states == {"wavve": "negative", "Tom Odell": "liked"}            # judged: never offered again
    cur = verdicts.current(con)
    assert cur["sp1".ljust(22, "0")].stars == 3
    j = {x["artist"]: (x["verdict"], x["weight"]) for x in feedback.judged(con)}
    assert j == {"wavve": ("no", 2.0), "Tom Odell": ("yes", 1.0)}


def test_mild_and_normal_no_are_a_pass(tmp_path):
    assert feedback.as_verdict("no", "mild") == ("pass", None)
    assert feedback.as_verdict("no", "normal") == ("pass", None)


def test_a_candidate_never_mapped_to_spotify_cannot_be_judged(tmp_path):
    with pytest.raises(LookupError):
        feedback.record(_con(tmp_path), "yes", "Never", "Offered")
