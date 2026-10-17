"""Cross-process minimum-interval throttle (a lock file + timestamp), so concurrent units share one rate."""
from __future__ import annotations

import fcntl
import os
import time

from . import config


class Throttle:
    def __init__(self, name: str, min_interval: float):
        self.min_interval = float(min_interval)
        config.STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = config.STATE_DIR / f"{name}.throttle"

    def wait(self) -> None:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            raw = os.pread(fd, 32, 0).decode().strip()
            last = float(raw) if raw else 0.0
            d = time.time() - last
            if d < self.min_interval:
                time.sleep(self.min_interval - d)
            os.ftruncate(fd, 0)
            os.pwrite(fd, f"{time.time():.3f}".encode(), 0)
        finally:
            os.close(fd)
