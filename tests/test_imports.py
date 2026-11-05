"""Every module imports on its own, in a fresh interpreter, as a service's entry point does.

pytest imports modules in one process, so an import cycle that only breaks for one first module (haystack-api
starts at api -> review -> calibrate -> sources -> report -> review) passes the suite and fails in production."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
MODULES = sorted(p.stem for p in (SRC / "haystack_music").glob("*.py") if p.stem != "__init__")


@pytest.mark.parametrize("name", MODULES)
def test_module_imports_alone(name):
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    r = subprocess.run([sys.executable, "-B", "-c", f"import haystack_music.{name}"], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
