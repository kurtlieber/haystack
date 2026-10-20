"""conftest.py keeps every test away from a real installation (kurtlieber/dotfiles#292)."""
import os
from pathlib import Path

import conftest
from haystack_music import config


def test_every_path_is_a_fresh_temporary_directory(tmp_path_factory):
    tmp = tmp_path_factory.getbasetemp().resolve()
    for p in (config.DATA_DIR, config.STATE_DIR, config.CONFIG_DIR, config.CONFIG_FILE, config.DB_PATH,
              config.EMB_DIR, config.STATUS_PATH, config.DEFAULTS["secrets"]["dir"],
              config.DEFAULTS["embedding"]["model"]):
        assert tmp in Path(p).resolve().parents, p
    assert not config.DB_PATH.exists() and not config.EMB_DIR.exists()


def test_the_directories_seen_at_import_are_temporary_too():
    root = conftest._ROOT.resolve()
    for name in ("HAYSTACK_DATA", "HAYSTACK_STATE", "HAYSTACK_CONFIG_DIR", "HAYSTACK_CONFIG"):
        assert root in Path(os.environ[name]).resolve().parents, name
