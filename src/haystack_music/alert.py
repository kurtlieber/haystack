"""Webhook alerts: a JSON POST when a job fails or the Spotify login needs renewing.

Off unless [alert] enabled. The URL is the alert-webhook-url secret; the alert-webhook-key secret, if set, goes
along as `Authorization: Bearer <key>`. The body is {job, host, status, exit_code, message, ts}, and every
message starts with "haystack:" so it can be told apart from other alerts on the same hook. Alerting never
raises: a missing secret or a failed POST is logged and ignored. HAYSTACK_ALERT_DRY_RUN=1 logs the payload instead
of sending it."""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
from datetime import datetime, timezone

log = logging.getLogger(__name__)

PREFIX = "haystack: "
EXIT_NEEDS_AUTH = 3


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


def unit_failure(cfg: dict, unit: str, exit_code: str | None = None, **kw) -> bool:
    """OnFailure= hook. Exit 3 (needs auth) is skipped: the daily auth alert covers it, not every hourly sync."""
    code = exit_code if exit_code is not None else unit_exit_code(unit)
    if str(code) == str(EXIT_NEEDS_AUTH):
        log.info("%s exited %s (needs auth): covered by the daily auth alert", unit, code)
        return False
    host = socket.gethostname().split(".")[0]
    return send(cfg, unit.removesuffix(".service"),
                f"{unit} failed on {host} (exit {code or '?'}); see journalctl --user -u {unit}", "fail", code, **kw)
