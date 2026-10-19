"""`haystack serve`'s job schedule: the systemd timers' work, for hosts without systemd (the container).

The times come from [schedule]: "hourly", "daily HH:MM", "Mon HH:MM" (days Mon..Sun, comma-separated:
"Mon,Thu 08:00") or "off", in the server's local time. Each job runs as its own `haystack <job>` child at nice 19,
as the units do, so a crash or a TensorFlow memory spike in one never takes the API down. A job still running is
not started again; different jobs may overlap (the first library embedding takes hours, and the hourly sync must
not wait for it).

A run missed while haystack was down is caught up on start, like the timers' Persistent=true. The last start of
each job comes from the runs table, which the units fill too, so moving from the units to the container doesn't
replay a batch. A job that has never run starts at once, except the batch, which waits for its time. Until the
first batch is done, it follows every good library or harvest run (the units: OnSuccess= of the library unit). The
harvest fills the pool, so without a library a new install gets its first batch the same day; with one, the
first batch skips until the library is embedded. A failure (any exit but 3, needs auth) sends an alert."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import alert, config, db

log = logging.getLogger(__name__)

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
CLEAR = ["--clear-recent", "remove", "--clear-min-age-hours", "24"]   # what the batch units pin
JOBS = {
    "sync": ["sync"],
    "library": ["library", "update"],
    "harvest": ["harvest"],
    "batch": ["batch", *CLEAR],
}
FIRST_BATCH = ["first-batch", *CLEAR]
WAIT_IF_NEVER_RUN = {"batch"}
TICK = 30            # seconds between checks: a job starts at most this late


@dataclass(frozen=True)
class Spec:
    hourly: bool
    days: frozenset
    hour: int = 0
    minute: int = 0


def parse(spec: str) -> Spec | None:
    """A [schedule] value; None for "off". ValueError for anything else it can't read."""
    s = " ".join(str(spec).lower().split())
    if s in ("", "off"):
        return None
    if s == "hourly":
        return Spec(True, frozenset(range(7)))
    try:
        when, hm = s.rsplit(" ", 1)
        hour, minute = (int(x) for x in hm.split(":"))
        days = range(7) if when == "daily" else [DAYS.index(d.strip()[:3]) for d in when.split(",")]
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError
    except ValueError:
        raise ValueError(f"schedule: can't read {spec!r}; use hourly, daily HH:MM, Mon HH:MM or off") from None
    return Spec(False, frozenset(days), hour, minute)


def next_after(spec: Spec, t: datetime) -> datetime:
    """The first run time strictly after t (naive local time)."""
    if spec.hourly:
        return t.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    for d in range(8):
        c = (t + timedelta(days=d)).replace(hour=spec.hour, minute=spec.minute, second=0, microsecond=0)
        if c > t and c.weekday() in spec.days:
            return c
    raise AssertionError("unreachable: every spec has a day")


def last_due(spec: Spec, t: datetime) -> datetime:
    """The latest run time at or before t (naive local time)."""
    if spec.hourly:
        return t.replace(minute=0, second=0, microsecond=0)
    for d in range(8):
        c = (t - timedelta(days=d)).replace(hour=spec.hour, minute=spec.minute, second=0, microsecond=0)
        if c <= t and c.weekday() in spec.days:
            return c
    raise AssertionError("unreachable: every spec has a day")


def last_start(con, job: str) -> datetime | None:
    """When the job last started (naive local time), from the runs table."""
    v = con.execute("SELECT MAX(started_at) FROM runs WHERE job=?", (job,)).fetchone()[0]
    if not v:
        return None
    return datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)


def missed(job: str, spec: Spec, last: datetime | None, now: datetime) -> bool:
    """Should the job run on start? Yes if a run time passed since it last started."""
    if last is None:
        return job not in WAIT_IF_NEVER_RUN
    return last < last_due(spec, now)


def command(args: list[str]) -> list[str]:
    cmd = [sys.executable, "-m", "haystack_music.cli", *args]
    nice = shutil.which("nice")
    return [nice, "-n", "19", *cmd] if nice else cmd


class Scheduler:
    def __init__(self, cfg: dict, clock=datetime.now, run=None):
        self.cfg = cfg
        self.clock = clock
        self.run = run or self._run_child
        self.specs = {}
        for job in JOBS:
            spec = parse(cfg["schedule"][job])
            if spec:
                self.specs[job] = spec
            else:
                log.info("schedule: %s is off", job)
        self.running: dict[str, threading.Thread] = {}
        self.children: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self.stopping = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self.loop, name="scheduler", daemon=True).start()

    def loop(self) -> None:
        now = self.clock()
        con = db.connect()
        nxt = {}
        for job, spec in self.specs.items():
            if missed(job, spec, last_start(con, job), now):
                log.info("schedule: %s missed a run (or never ran): starting it now", job)
                self.launch(job)
            nxt[job] = next_after(spec, now)
            log.info("schedule: %s next at %s", job, nxt[job].strftime("%a %Y-%m-%d %H:%M"))
        con.close()
        while not self.stopping.wait(TICK):
            now = self.clock()
            for job, t in nxt.items():
                if now >= t:
                    self.launch(job)
                    nxt[job] = next_after(self.specs[job], now)

    def launch(self, job: str, args: list[str] | None = None) -> None:
        with self.lock:
            th = self.running.get(job)
            if th and th.is_alive():
                log.info("schedule: %s is still running; this run is skipped", job)
                return
            th = threading.Thread(target=self._job, args=(job, args or JOBS[job]), name=job, daemon=True)
            self.running[job] = th
            th.start()

    def _job(self, job: str, args: list[str]) -> None:
        code = self.run(job, args)
        if self.stopping.is_set():
            return
        if code == 0 and job in ("library", "harvest"):
            from .firstbatch import DONE
            if not (config.STATE_DIR / DONE).exists():
                self.launch("first_batch", FIRST_BATCH)
        if code != 0:
            alert.job_failure(self.cfg, job, job, code, "see the log of `haystack serve`")

    def _run_child(self, job: str, args: list[str]) -> int:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "3"}
        log.info("schedule: starting %s", job)
        p = subprocess.Popen(command(args), env=env, stdin=subprocess.DEVNULL)
        with self.lock:
            self.children[job] = p
        code = p.wait()
        with self.lock:
            self.children.pop(job, None)
        log.info("schedule: %s exited %s", job, code)
        return code

    def stop(self, grace: float = 20) -> None:
        """Stop scheduling and end the running jobs (SIGTERM, then SIGKILL after `grace` seconds)."""
        self.stopping.set()
        with self.lock:
            children = list(self.children.items())
        for job, p in children:
            log.info("schedule: stopping %s", job)
            p.terminate()
        for _, p in children:
            try:
                p.wait(grace)
            except subprocess.TimeoutExpired:
                p.kill()
