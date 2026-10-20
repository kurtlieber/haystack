import json
import re
import sqlite3
import stat
from datetime import datetime, timezone

from haystack_music import alert, db

ON = {"alert": {"enabled": True}}


class FakeStore:
    def __init__(self, **values):
        self.values = {"alert-webhook-url": "https://example.invalid/hook", "alert-webhook-key": "k", **values}

    def find(self, name):
        return self.values.get(name)

    def read(self, name):
        assert self.values.get(name), name
        return self.values[name]


class Resp:
    def __init__(self, code): self.status_code = code


def test_payload_prefix_and_job():
    p = alert.payload("harvest", "fail", "boom", 1, host="server1",
                      now=datetime(2026, 10, 11, tzinfo=timezone.utc))
    assert p["message"] == "haystack: boom"
    assert p["job"] == "haystack-harvest" and p["exit_code"] == "1" and p["host"] == "server1"
    assert alert.payload("haystack-sync", "fail", "haystack: x")["message"] == "haystack: x"


def test_send_posts_bearer(monkeypatch):
    monkeypatch.delenv("HAYSTACK_ALERT_DRY_RUN", raising=False)
    seen = {}
    def post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers); return Resp(200)
    assert alert.send(ON, "auth", "re-auth soon", "warn", store=FakeStore(), post=post)
    assert seen["url"] == "https://example.invalid/hook"
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert seen["json"]["message"].startswith("haystack: ")


def test_send_without_a_key_sends_no_authorization(monkeypatch):
    monkeypatch.delenv("HAYSTACK_ALERT_DRY_RUN", raising=False)
    seen = {}
    def post(url, json, headers, timeout):
        seen.update(headers=headers); return Resp(204)
    assert alert.send(ON, "auth", "x", store=FakeStore(**{"alert-webhook-key": None}), post=post)
    assert seen["headers"] == {}


def test_send_is_off_by_default(monkeypatch):
    monkeypatch.delenv("HAYSTACK_ALERT_DRY_RUN", raising=False)
    calls = []
    post = lambda *a, **k: calls.append(k) or Resp(200)
    assert alert.send({}, "auth", "x", store=FakeStore(), post=post) is False
    assert alert.send({"alert": {"enabled": False}}, "auth", "x", store=FakeStore(), post=post) is False
    assert not calls


def test_send_never_raises(monkeypatch):
    monkeypatch.delenv("HAYSTACK_ALERT_DRY_RUN", raising=False)
    def post(*a, **k): raise OSError("down")
    assert alert.send(ON, "auth", "x", store=FakeStore(), post=post) is False
    assert alert.send(ON, "auth", "x", store=FakeStore(), post=lambda *a, **k: Resp(500)) is False
    # no webhook URL set: logged, not raised
    assert alert.send(ON, "auth", "x", store=FakeStore(**{"alert-webhook-url": None}), post=post) is False


def test_unit_failure_skips_needs_auth(monkeypatch):
    monkeypatch.delenv("HAYSTACK_ALERT_DRY_RUN", raising=False)
    calls = []
    post = lambda *a, **k: calls.append(k) or Resp(200)
    assert alert.unit_failure(ON, "haystack-sync.service", "3", store=FakeStore(), post=post) is False
    assert not calls
    assert alert.unit_failure(ON, "haystack-harvest.service", "1", store=FakeStore(), post=post)
    assert "harvest.service failed" in calls[0]["json"]["message"]


def test_once_a_day(tmp_path, monkeypatch):
    con = sqlite3.connect(tmp_path / "t.db")
    store = {}
    monkeypatch.setattr(db, "kv_get", lambda c, k, d=None: store.get(k, d))
    monkeypatch.setattr(db, "kv_set", lambda c, k, v: store.__setitem__(k, v))
    assert alert.once_a_day(con, "k", "2026-10-11")
    assert not alert.once_a_day(con, "k", "2026-10-11")
    assert alert.once_a_day(con, "k", "2026-10-12")


def test_job_failure_names_the_log(monkeypatch):
    monkeypatch.delenv("HAYSTACK_ALERT_DRY_RUN", raising=False)
    seen = {}
    def post(url, json, headers, timeout):
        seen.update(json); return Resp(200)
    assert alert.job_failure(ON, "harvest", "harvest", 1, "see the log of `haystack serve`",
                             store=FakeStore(), post=post)
    assert seen["job"] == "haystack-harvest" and "see the log of `haystack serve`" in seen["message"]
    assert alert.job_failure(ON, "sync", "sync", 3, "x", store=FakeStore(), post=post) is False


class NoStore:
    """The outbox needs no secret: any read fails the test."""
    def find(self, name): raise AssertionError(name)
    read = find


def no_post(*a, **k): raise AssertionError("POST with an outbox set")


def outbox(folder) -> dict:
    return {"alert": {"enabled": True, "outbox": str(folder)}}


def test_outbox_writes_one_file_and_no_post(tmp_path, monkeypatch):
    monkeypatch.delenv("HAYSTACK_ALERT_DRY_RUN", raising=False)
    box = tmp_path / "outbox"
    assert alert.unit_failure(outbox(box), "haystack-harvest-broad.service", "1", store=NoStore(), post=no_post)
    files = list(box.iterdir())
    assert len(files) == 1
    # The name dotfiles' backup-alert.sh and failure-alerts accept.
    assert re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[A-Za-z0-9_-]+-[0-9]+\.json", files[0].name), files[0].name
    assert "-haystack-harvest-broad-" in files[0].name
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
    assert stat.S_IMODE(box.stat().st_mode) == 0o700
    body = json.loads(files[0].read_text())
    assert set(body) == {"job", "host", "status", "exit_code", "message", "ts"}
    assert body["job"] == "haystack-harvest-broad" and body["status"] == "fail" and body["exit_code"] == "1"
    assert body["message"].startswith("haystack: haystack-harvest-broad.service failed")


def test_outbox_expands_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HAYSTACK_ALERT_DRY_RUN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert alert.send(outbox("~/box"), "auth", "re-auth soon", "warn", store=NoStore(), post=no_post)
    assert len(list((tmp_path / "box").glob("*-haystack-auth-*.json"))) == 1


def test_a_full_outbox_takes_no_more(tmp_path, monkeypatch):
    monkeypatch.delenv("HAYSTACK_ALERT_DRY_RUN", raising=False)
    monkeypatch.setattr(alert, "OUTBOX_MAX", 1)
    box = tmp_path / "outbox"
    assert alert.send(outbox(box), "auth", "one", store=NoStore(), post=no_post)
    assert alert.send(outbox(box), "auth", "two", store=NoStore(), post=no_post) is False
    assert len(list(box.iterdir())) == 1


def test_an_unwritable_outbox_never_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("HAYSTACK_ALERT_DRY_RUN", raising=False)
    (tmp_path / "file").write_text("")
    assert alert.send(outbox(tmp_path / "file" / "box"), "auth", "x", store=NoStore(), post=no_post) is False
