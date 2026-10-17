import json
import os
import stat
from types import SimpleNamespace

import pytest

from haystack_music import cli, config, secrets


def _cfg(tmp_path, **over):
    cfg = config.load(tmp_path / "none.toml")
    cfg["secrets"]["dir"] = str(tmp_path / "secrets")
    for k, v in over.items():
        cfg[k].update(v)
    return cfg


def _mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch):
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)


# ---------------- files ----------------

def test_files_is_the_default_backend(tmp_path):
    assert isinstance(secrets.store(config.load(tmp_path / "none.toml")), secrets.FileStore)


def test_file_store_put_get_is_private(tmp_path):
    st = secrets.store(_cfg(tmp_path))
    assert st.get("api-token") is None
    st.put("api-token", "tok\n")
    assert st.get("api-token") == "tok"
    assert _mode(tmp_path / "secrets") == 0o700
    assert _mode(tmp_path / "secrets" / "api-token") == 0o600
    assert not list((tmp_path / "secrets").glob(".*.tmp"))


def test_file_store_refuses_a_readable_secret(tmp_path):
    st = secrets.store(_cfg(tmp_path))
    st.put("api-token", "tok")
    os.chmod(tmp_path / "secrets" / "api-token", 0o644)
    with pytest.raises(secrets.SecretError, match="0600"):
        st.get("api-token")


def test_read_names_a_missing_secret(tmp_path):
    with pytest.raises(secrets.SecretError, match="spotify-client-id is not set"):
        secrets.store(_cfg(tmp_path)).read("spotify-client-id")


def test_a_systemd_credential_comes_first(tmp_path, monkeypatch):
    st = secrets.store(_cfg(tmp_path))
    st.put("api-token", "from-file")
    creds = tmp_path / "creds"
    creds.mkdir()
    (creds / "api-token").write_text("from-credential\n")
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(creds))
    assert st.find("api-token") == "from-credential"
    assert st.find("alert-webhook-url") is None


def test_refresh_token_round_trip(tmp_path):
    s = secrets.SpotifySecrets(_cfg(tmp_path))
    assert s.refresh_token() == (None, {})
    s.store_refresh_token("rt1", issued=True, scopes="a b")
    tok, meta = s.refresh_token()
    assert tok == "rt1" and meta["scopes"] == "a b" and meta["issued at"] == meta["rotated at"]
    s.store_refresh_token("rt2", issued=False)
    tok, meta2 = s.refresh_token()
    assert tok == "rt2" and meta2["issued at"] == meta["issued at"] and meta2["scopes"] == "a b"
    s.write_check()
    assert s.refresh_token()[1]["last write check"]
    p = tmp_path / "secrets" / secrets.TOKEN_FILE
    assert _mode(p) == 0o600
    assert json.loads(p.read_text())["refresh_token"] == "rt2"


def test_unknown_backend(tmp_path):
    with pytest.raises(secrets.SecretError, match="unknown secrets.backend"):
        secrets.store(_cfg(tmp_path, secrets={"backend": "vault"}))


# ---------------- 1password ----------------

class FakeOp:
    def __init__(self):
        self.items = {"client": {"client id": "cid", "client secret": "csec"},
                      "tok": {"refresh token": "rt", "issued at": "2026-01-01T00:00:00Z", "notes": "x"}}
        self.edits = []

    def read(self, item, field):
        return self.items[item].get(field, "")

    def item(self, item):
        return {"fields": [{"label": k, "value": v} for k, v in self.items[item].items()]}

    def set_fields(self, item, values):
        self.edits.append((item, values))
        self.items.setdefault(item, {}).update({k: v for k, (_, v) in values.items()})


def _op_cfg(tmp_path):
    return _cfg(tmp_path, secrets={"backend": "1password"},
                onepassword={"vault": "v", "client_item": "client", "token_item": "tok", "api_token_item": "api"})


def test_one_password_store(tmp_path):
    op = FakeOp()
    st = secrets.OnePasswordStore(_op_cfg(tmp_path), op=op)
    assert secrets.SpotifySecrets(_op_cfg(tmp_path), st).client() == ("cid", "csec")
    assert st.find("alert-webhook-url") is None          # no alert_item: not set, and op is not asked
    st.put("api-token", "t")
    assert op.edits[-1] == ("api", {"password": ("CONCEALED", "t")})
    tok, meta = secrets.SpotifySecrets(_op_cfg(tmp_path), st).refresh_token()
    assert tok == "rt" and meta == {"issued at": "2026-01-01T00:00:00Z"}
    secrets.SpotifySecrets(_op_cfg(tmp_path), st).store_refresh_token("rt2", issued=False)
    item, values = op.edits[-1]
    assert item == "tok" and values["refresh token"] == ("CONCEALED", "rt2")
    assert values["rotated at"][0] == "STRING"


def test_op_token_is_never_taken_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "from-env")
    with pytest.raises(secrets.SecretError, match="service-account token"):
        secrets._op_token(_op_cfg(tmp_path))
    p = tmp_path / "op-token"
    p.write_text("from-file\n")
    os.chmod(p, 0o600)
    assert secrets._op_token(_cfg(tmp_path, onepassword={"token_file": str(p)})) == "from-file"


# ---------------- haystack token / check-secrets ----------------

def test_token_new_show_force(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    assert cli.cmd_token(cfg, SimpleNamespace(action="show", force=False)) == 1
    assert cli.cmd_token(cfg, SimpleNamespace(action="new", force=False)) == 0
    out = capsys.readouterr()
    assert out.out == ""                                 # the new token is not echoed
    first = secrets.store(cfg).get("api-token")
    assert len(first) >= 40
    assert cli.cmd_token(cfg, SimpleNamespace(action="new", force=False)) == 2
    assert secrets.store(cfg).get("api-token") == first
    assert cli.cmd_token(cfg, SimpleNamespace(action="show", force=False)) == 0
    assert capsys.readouterr().out == first + "\n"
    assert cli.cmd_token(cfg, SimpleNamespace(action="new", force=True)) == 0
    assert secrets.store(cfg).get("api-token") != first


def test_check_secrets_shows_lengths_not_values(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    secrets.store(cfg).put("spotify-client-id", "abcdef")
    assert cli.cmd_check(cfg, SimpleNamespace(write=False)) == 0
    out = capsys.readouterr().out
    assert "spotify-client-id: set (6 characters)" in out and "abcdef" not in out
    assert "api-token: not set" in out
    os.chmod(tmp_path / "secrets" / "spotify-client-id", 0o644)
    assert cli.cmd_check(cfg, SimpleNamespace(write=False)) == 1
