"""No test may read or write a real installation (kurtlieber/dotfiles#292).

config.py takes its directories from HAYSTACK_DATA, HAYSTACK_STATE, HAYSTACK_CONFIG_DIR and HAYSTACK_CONFIG once,
at import, and its defaults (the secrets dir, the model path) are built from them. So this file points the four
at an empty temporary directory before any test module imports haystack_music (removed when the run ends), and
gives every test its own empty directories and defaults on top. A test that wants files there creates them, or
patches the paths itself as before."""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(tempfile.mkdtemp(prefix="haystack-tests-"))
atexit.register(shutil.rmtree, _ROOT, ignore_errors=True)
os.environ["HAYSTACK_DATA"] = str(_ROOT / "data")
os.environ["HAYSTACK_STATE"] = str(_ROOT / "state")
os.environ["HAYSTACK_CONFIG_DIR"] = str(_ROOT / "config")
os.environ["HAYSTACK_CONFIG"] = str(_ROOT / "config" / "config.toml")

from haystack_music import config  # noqa: E402


@pytest.fixture(autouse=True)
def _empty_installation(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("installation")
    data, state, conf = root / "data", root / "state", root / "config"
    for name, value in (("DATA_DIR", data), ("STATE_DIR", state), ("CONFIG_DIR", conf),
                        ("CONFIG_FILE", conf / "config.toml"), ("DB_PATH", data / "discovery.db"),
                        ("EMB_DIR", data / "embeddings"), ("STATUS_PATH", state / "status.json")):
        monkeypatch.setattr(config, name, value)
    monkeypatch.setitem(config.DEFAULTS["secrets"], "dir", str(conf / "secrets"))
    monkeypatch.setitem(config.DEFAULTS["embedding"], "model",
                        str(data / "models" / Path(config.DEFAULTS["embedding"]["model"]).name))
