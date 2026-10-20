"""Alerts: a JSON POST to a webhook, or a file in an outbox, when a job fails or the Spotify login needs renewing.

Off unless [alert] enabled. With [alert] outbox set, each alert is one file in that folder for a collector to
pick up (write_outbox); nothing goes over the network and no secret is needed. Otherwise the URL is the
alert-webhook-url secret, and the alert-webhook-key secret, if set, goes along as `Authorization: Bearer <key>`.
The body is {job, host, status, exit_code, message, ts}, and every message starts with "haystack:" so it can be
told apart from other alerts on the same hook. Alerting never raises: a missing secret, a failed POST or an
outbox it cannot write is logged and ignored. HAYSTACK_ALERT_DRY_RUN=1 logs the payload instead of sending it."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

PREFIX = "haystack: "
EXIT_NEEDS_AUTH = 3
# A full outbox means its collector is not running; stop adding to it.
OUTBOX_MAX = 500


def payload(job: str, status: str, message: str, exit_code: str | int | None = None,
            host: str | None = None, now: datetime | None = None) -> dict:
    msg = message if message.startswith(PREFIX) else PREFIX + message
    return {"job": f"haystack-{job}" if not job.startswith("haystack") else job,
            "host": host or socket.gethostname().split(".")[0], "status": status,
            "exit_code": "" if exit_code is None else str(exit_code), "message": msg,
            "ts": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")}


def send(cfg: dict, job: str, message: str, status: str = "fail", exit_code=None, store=None, post=None) -> bool:
    if not cfg.get("alert", {}).get("enabled"):
        return False
    body = payload(job, status, message, exit_code)
    if os.environ.get("HAYSTACK_ALERT_DRY_RUN") == "1":
        log.info("alert (dry run): %s", json.dumps(body))
        return True
    if cfg["alert"].get("outbox"):
        return write_outbox(Path(cfg["alert"]["outbox"]).expanduser(), body)
    try:
        if store is None:
            from . import secrets
            store = secrets.store(cfg)
        url, key = store.read("alert-webhook-url"), store.find("alert-webhook-key")
        if post is None:
            import requests
            post = requests.post
        r = post(url, json=body, headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=20)
        ok = 200 <= getattr(r, "status_code", 0) < 300
        if not ok:
            log.warning("alert POST returned HTTP %s", getattr(r, "status_code", "?"))
        return ok
    except Exception as e:  # noqa: BLE001
        log.warning("alert not sent: %s", type(e).__name__)
        return False


def write_outbox(folder: Path, body: dict) -> bool:
    """One alert as one file, YYYYMMDDTHHMMSSZ-JOB-N.json, mode 0600, written under a dot name and then renamed so a
    reader never sees half of it. A collector posts the files and deletes them (in dotfiles, failure-alerts reads
    the backup-alert outbox). Never raises."""
    try:
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        if sum(1 for _ in folder.glob("*.json")) >= OUTBOX_MAX:
            log.warning("alert not written: the outbox holds %s alerts (is its collector running?)", OUTBOX_MAX)
            return False
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job = re.sub(r"[^A-Za-z0-9_-]", "-", str(body.get("job") or "haystack"))
        name = f"{stamp}-{job}-{os.getpid()}{random.randrange(10**6):06d}.json"
        tmp = folder / f".{name}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(body) + "\n")
        os.replace(tmp, folder / name)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("alert not written to the outbox: %s", type(e).__name__)
        return False


def once_a_day(con, key: str, today: str | None = None) -> bool:
    """True the first time `key` is seen on a local calendar day (then records it)."""
    from . import db
    today = today or datetime.now().strftime("%Y-%m-%d")
    if db.kv_get(con, key) == today:
        return False
    db.kv_set(con, key, today)
    return True


def unit_exit_code(unit: str) -> str:
    try:
        return subprocess.run(["systemctl", "--user", "show", "-p", "ExecMainStatus", "--value", unit],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def job_failure(cfg: dict, job: str, name: str, exit_code, where: str, **kw) -> bool:
    """A failed job, from a unit's OnFailure= hook or `haystack serve`. Exit 3 (needs auth) is skipped: the daily
    auth alert covers it, not every hourly sync."""
    if str(exit_code) == str(EXIT_NEEDS_AUTH):
        log.info("%s exited %s (needs auth): covered by the daily auth alert", name, exit_code)
        return False
    host = socket.gethostname().split(".")[0]
    return send(cfg, job, f"{name} failed on {host} (exit {exit_code or '?'}); {where}", "fail", exit_code, **kw)


def unit_failure(cfg: dict, unit: str, exit_code: str | None = None, **kw) -> bool:
    """OnFailure= hook."""
    code = exit_code if exit_code is not None else unit_exit_code(unit)
    return job_failure(cfg, unit.removesuffix(".service"), unit, code, f"see journalctl --user -u {unit}", **kw)
