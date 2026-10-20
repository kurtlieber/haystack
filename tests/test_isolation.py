"""conftest.py keeps every test away from a real installation (kurtlieber/dotfiles#292)."""
import tempfile
from pathlib import Path

from haystack_music import config


def test_every_path_is_a_fresh_temporary_directory():
    tmp = Path(tempfile.gettempdir()).resolve()
    for p in (config.DATA_DIR, config.STATE_DIR, config.CONFIG_DIR, config.CONFIG_FILE, config.DB_PATH,
              config.EMB_DIR, config.STATUS_PATH):
        assert tmp in Path(p).resolve().parents, p
    assert not config.DB_PATH.exists() and not config.EMB_DIR.exists()


def test_the_defaults_built_at_import_are_temporary_too():
    tmp = Path(tempfile.gettempdir()).resolve()
    assert tmp in Path(config.DEFAULTS["secrets"]["dir"]).resolve().parents
    assert tmp in Path(config.DEFAULTS["embedding"]["model"]).resolve().parents
