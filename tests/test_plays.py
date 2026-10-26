"""The 15-minute plays job and play gaps (ADR 0023): a full page of recently-played is recorded as a gap."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from haystack_music import cli, db, sync
from haystack_music.spotify import RECENTLY_PLAYED_LIMIT

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def _ms(t: datetime) -> int:
    return int(t.timestamp() * 1000)


def _item(i: int, t: datetime) -> dict:
    return {"played_at": t.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "context": {"uri": "spotify:playlist:x"},
            "track": {"id": f"track{i:017d}", "name": f"Song {i}", "artists": [{"name": "Band"}], "album": {},
                      "duration_ms": 200000}}


class FakeSpotify:
    def __init__(self, items):
        self.items, self.calls = items, []

    def recently_played(self, after_ms=None):
        self.calls.append(after_ms)
        return [dict(x) for x in self.items]


def _page(n: int, start: datetime) -> list[dict]:
    """n plays, newest first, three minutes apart, the oldest at start."""
    return [_item(i, start + timedelta(minutes=3 * i)) for i in reversed(range(n))]


def _gaps(con):
    return [tuple(r) for r in con.execute("SELECT from_ms, to_ms FROM play_gaps ORDER BY to_ms")]


def test_a_short_page_records_no_gap_and_moves_the_cursor(tmp_path):
    con = db.connect(tmp_path / "t.db")
    db.kv_set(con, "recently_played_after_ms", _ms(T0))
    items = _page(RECENTLY_PLAYED_LIMIT - 1, T0 + timedelta(minutes=1))
    sp = FakeSpotify(items)
    assert sync.poll_plays(con, sp) == RECENTLY_PLAYED_LIMIT - 1
    assert sp.calls == [_ms(T0)]
    assert _gaps(con) == []
    newest = T0 + timedelta(minutes=1 + 3 * (RECENTLY_PLAYED_LIMIT - 2))
    assert int(db.kv_get(con, "recently_played_after_ms")) == _ms(newest)


def test_a_full_page_records_the_span_it_may_have_dropped(tmp_path, caplog):
    con = db.connect(tmp_path / "t.db")
    db.kv_set(con, "recently_played_after_ms", _ms(T0))
    oldest = T0 + timedelta(hours=2)
    sp = FakeSpotify(_page(RECENTLY_PLAYED_LIMIT, oldest))
    assert sync.poll_plays(con, sp) == RECENTLY_PLAYED_LIMIT
    assert _gaps(con) == [(_ms(T0), _ms(oldest))]
    assert "may be missing" in caplog.text
    # the same page again (two polls at once): no new plays, and still one gap
    db.kv_set(con, "recently_played_after_ms", _ms(T0))
    assert sync.poll_plays(con, sp) == 0
    assert _gaps(con) == [(_ms(T0), _ms(oldest))]
    assert con.execute("SELECT found_at FROM play_gaps").fetchone()[0].endswith("Z")


def test_a_first_poll_with_a_full_page_starts_the_gap_at_zero(tmp_path):
    con = db.connect(tmp_path / "t.db")
    sp = FakeSpotify(_page(RECENTLY_PLAYED_LIMIT, T0))
    sync.poll_plays(con, sp)
    assert sp.calls == [None]
    assert _gaps(con) == [(0, _ms(T0))]


def test_an_empty_poll_leaves_the_cursor(tmp_path):
    con = db.connect(tmp_path / "t.db")
    db.kv_set(con, "recently_played_after_ms", _ms(T0))
    assert sync.poll_plays(con, FakeSpotify([])) == 0
    assert int(db.kv_get(con, "recently_played_after_ms")) == _ms(T0) and _gaps(con) == []


def test_a_new_database_has_no_gap_but_an_old_cursor_is_unknown_history(tmp_path):
    con = db.connect(tmp_path / "new.db")
    assert _gaps(con) == [] and db.kv_get(con, "play_gaps_since")
    # A database polled before gap tracking: its plays up to the cursor were never checked for a full page.
    con = db.connect(tmp_path / "old.db")
    db.kv_set(con, "recently_played_after_ms", _ms(T0))
    con.execute("DELETE FROM kv WHERE k='play_gaps_since'")
    con.close()
    con = db.connect(tmp_path / "old.db")
    assert _gaps(con) == [(0, _ms(T0))]
    since = db.kv_get(con, "play_gaps_since")
    con.close()
    con = db.connect(tmp_path / "old.db")                  # once only
    assert _gaps(con) == [(0, _ms(T0))] and db.kv_get(con, "play_gaps_since") == since


def test_status_shows_the_count_and_the_latest_gap(tmp_path):
    con = db.connect(tmp_path / "t.db")
    assert cli._play_gaps(con)["count"] == 0 and cli._play_gaps(con)["latest"] is None
    con.execute("INSERT INTO play_gaps VALUES(0, ?, '2026-10-20T12:00:00Z')", (_ms(T0),))
    con.execute("INSERT INTO play_gaps VALUES(?, ?, '2026-10-21T12:00:00Z')",
                (_ms(T0 + timedelta(days=1)), _ms(T0 + timedelta(days=1, hours=3))))
    g = cli._play_gaps(con)
    assert g["count"] == 2
    assert g["latest"] == {"from": "2026-10-21T12:00:00Z", "to": "2026-10-21T15:00:00Z",
                           "found_at": "2026-10-21T12:00:00Z"}
    assert g["tracked_since"] == db.kv_get(con, "play_gaps_since")
    con.execute("DELETE FROM play_gaps WHERE from_ms>0")
    assert cli._play_gaps(con)["latest"]["from"] == "the start"


def test_the_plays_job(tmp_path):
    con = db.connect(tmp_path / "t.db")
    assert sync.plays(con, {}, FakeSpotify(_page(3, T0))) == {"plays_new": 3}
    assert cli.build_parser().parse_args(["plays"]).cmd == "plays"


def test_the_plays_timer_runs_every_15_minutes():
    timer = (ROOT / "systemd" / "haystack-plays.timer").read_text()
    service = (ROOT / "systemd" / "haystack-plays.service").read_text()
    assert "OnCalendar=*:0/15" in timer and "Persistent=true" in timer
    assert "/bin/haystack plays\n" in service and "OnFailure=" in service
    assert "haystack-plays.timer" in (ROOT / "scripts" / "install-units.sh").read_text()
