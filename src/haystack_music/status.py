"""Short machine-readable status at ~/.local/state/haystack/status.json (for monitoring)."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from . import config


@contextmanager
def _locked():
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.STATE_DIR / ".status.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        yield


def read() -> dict:
    try:
        return json.loads(config.STATUS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def update(section: str | None = None, data: dict | None = None, **top) -> dict:
    """Merge `data` into status[section] (or top-level keys from **top) and write atomically."""
    with _locked():
        st = read()
        if section:
            cur = st.get(section) if isinstance(st.get(section), dict) else {}
            cur.update(data or {})
            st[section] = cur
        st.update(top)
        st["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        tmp = config.STATUS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1, sort_keys=True, default=str))
        os.replace(tmp, config.STATUS_PATH)
        return st


def job(name: str, **data) -> None:
    jobs = read().get("jobs", {})
    cur = jobs.get(name, {})
    cur.update(data)
    jobs[name] = cur
    update(None, jobs=jobs)


def token_expiry_info(issued_at: str | None, warn_days: int, lifetime_days: int, now: datetime | None = None) -> dict:
    """Pure: token age / expiry estimate / warning flag from the refresh token's issue date."""
    if not issued_at:
        return {"issued_at": None, "expires_estimate": None, "expiry_warning": False}
    now = now or datetime.now(timezone.utc)
    issued = datetime.fromisoformat(issued_at.replace("Z", "+00:00"))
    exp = issued + timedelta(days=lifetime_days)
    return {"issued_at": issued_at, "expires_estimate": exp.strftime("%Y-%m-%d"),
            "age_days": (now - issued).days,
            "expiry_warning": now >= issued + timedelta(days=warn_days)}


def notify(summary: str, body: str, urgency: str = "normal") -> bool:
    exe = shutil.which("notify-send") or "/usr/bin/notify-send"
    try:
        subprocess.run([exe, "-u", urgency, "-a", "haystack", summary, body], timeout=10, check=True,
                       capture_output=True)
        return True
    except Exception:
        return False
