import threading
from datetime import datetime, timezone

import pytest

from haystack_music import alert, config, db, review, scheduler
from haystack_music.scheduler import last_due, missed, next_after, parse

MON = datetime(2026, 10, 19, 8, 0)          # a Monday


def test_parse():
    assert parse("hourly").hourly
    assert parse("off") is None and parse("") is None
    d = parse("daily 03:15")
    assert (d.hour, d.minute, len(d.days)) == (3, 15, 7)
    w = parse("Mon,Thursday  08:00")
    assert w.days == {0, 3} and w.hour == 8
    for bad in ("weekly", "Mon 25:00", "Funday 08:00", "daily 8"):
        with pytest.raises(ValueError):
            parse(bad)


def test_next_after_is_strictly_later():
    w = parse("Mon 08:00")
    assert next_after(w, MON) == datetime(2026, 10, 26, 8, 0)
    assert next_after(w, datetime(2026, 10, 19, 7, 59)) == MON
    assert next_after(parse("daily 03:15"), datetime(2026, 10, 19, 3, 15)) == datetime(2026, 10, 20, 3, 15)
    assert next_after(parse("hourly"), datetime(2026, 10, 19, 23, 30)) == datetime(2026, 10, 20, 0, 0)


def test_last_due():
    w = parse("Mon 08:00")
    assert last_due(w, MON) == MON
    assert last_due(w, datetime(2026, 10, 19, 7, 59)) == datetime(2026, 10, 12, 8, 0)
    assert last_due(parse("hourly"), datetime(2026, 10, 19, 9, 41)) == datetime(2026, 10, 19, 9, 0)


def test_missed_runs_are_caught_up_but_a_new_batch_waits():
    w = parse("Mon 08:00")
    wed = datetime(2026, 10, 21, 12, 0)
    assert missed("batch", w, datetime(2026, 10, 12, 8, 0), wed)       # down over Monday 08:00
    assert not missed("batch", w, datetime(2026, 10, 19, 8, 0), wed)   # ran on time
    assert not missed("batch", w, None, wed)                          # never ran: waits for its time
    assert missed("library", parse("daily 03:15"), None, wed)         # never ran: starts now


def test_last_start_reads_the_runs_table(tmp_path):
    con = db.connect(tmp_path / "t.db")
    assert scheduler.last_start(con, "sync") is None
    con.execute("INSERT INTO runs(job, started_at, ok) VALUES('sync', '2026-10-19T08:00:00Z', 1)")
    con.execute("INSERT INTO runs(job, started_at, ok) VALUES('sync', '2026-10-19T09:00:00Z', 0)")
    got = scheduler.last_start(con, "sync")
    assert got == datetime(2026, 10, 19, 9, 0, tzinfo=timezone.utc).astimezone().replace(tzinfo=None)


def _sched(codes, release=None):
    ran = []

    def run(job, args):
        ran.append((job, args))
        if release:
            release.wait(5)
        return codes.get(job, 0)
    return scheduler.Scheduler(config.load(), run=run), ran


def _join(s):
    for th in list(s.running.values()):
        th.join(5)


def test_a_running_job_is_not_started_twice():
    release = threading.Event()
    s, ran = _sched({}, release)
    s.launch("sync")
    s.launch("sync")
    release.set()
    _join(s)
    assert [j for j, _ in ran] == ["sync"]


def test_first_good_library_or_harvest_run_starts_the_first_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    s, ran = _sched({})
    s.launch("library")
    _join(s); _join(s)
    assert [j for j, _ in ran] == ["library", "first_batch"]
    assert ran[1][1] == scheduler.FIRST_BATCH

    s, ran = _sched({})
    s.launch("harvest")
    _join(s); _join(s)
    assert [j for j, _ in ran] == ["harvest", "first_batch"]

    s, ran = _sched({})
    s.launch("sync")
    _join(s)
    assert [j for j, _ in ran] == ["sync"]

    (tmp_path / "first-batch.done").write_text("done\n")
    s, ran = _sched({})
    s.launch("library")
    _join(s)
    assert [j for j, _ in ran] == ["library"]


def test_failures_alert_except_needs_auth(monkeypatch):
    sent = []
    monkeypatch.setattr(alert, "send", lambda cfg, job, msg, *a, **kw: sent.append((job, msg)) or True)
    s, _ = _sched({"harvest": 1, "sync": 3})
    s.launch("harvest"); s.launch("sync")
    _join(s)
    assert [j for j, _ in sent] == ["harvest"]
    assert "harvest failed" in sent[0][1] and "haystack serve" in sent[0][1]


def test_commands_run_at_nice_19(monkeypatch):
    monkeypatch.setattr(scheduler.shutil, "which", lambda name: "/usr/bin/nice")
    assert scheduler.command(["sync"])[:3] == ["/usr/bin/nice", "-n", "19"]
    assert scheduler.command(["sync"])[-3:] == ["-m", "haystack_music.cli", "sync"]
    monkeypatch.setattr(scheduler.shutil, "which", lambda name: None)
    assert scheduler.command(["sync"])[1:] == ["-m", "haystack_music.cli", "sync"]


def test_next_batch_without_systemd_follows_the_schedule(monkeypatch):
    def no_systemctl(*a, **kw):
        raise FileNotFoundError("systemctl")
    monkeypatch.setattr(review.subprocess, "run", no_systemctl)
    sun = datetime(2026, 10, 18, 12, 0).astimezone()

    def utc(local):
        return local.astimezone().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert review.next_batch_at(sun, "Mon 08:00") == utc(datetime(2026, 10, 19, 8, 0))
    assert review.next_batch_at(sun, "Thu 09:30") == utc(datetime(2026, 10, 22, 9, 30))
    assert review.next_batch_at(sun, "off") is None
