"""Secrets: the Spotify app's client id and secret, the Spotify refresh token, needle's API token and the alert
webhook.

Two backends, `[secrets] backend`:
  files      (the default) one 0600 file per secret in `[secrets] dir` (0700), named as in NAMES. The refresh
             token and its non-secret metadata share spotify-token.json there, rewritten whole on each rotation.
  1password  the `op` CLI with a service-account token; the items and fields are set in `[onepassword]`.

A systemd credential comes first for every secret but the refresh token (the jobs rewrite that one):
$CREDENTIALS_DIRECTORY/<name>, from LoadCredential=<name>:<file> or LoadCredentialEncrypted=.

The 1Password service-account token is never taken from the shell's environment. Order:
  1. $CREDENTIALS_DIRECTORY/op-token   (LoadCredential=op-token:<0600 file>)
  2. [onepassword].token_file          (a 0600 file)
It is passed only to the `op` child process. Secret values are never logged; `haystack token show` prints the
API token, on request, to stdout."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

NAMES = ("spotify-client-id", "spotify-client-secret", "api-token", "alert-webhook-url", "alert-webhook-key")
TOKEN_FILE = "spotify-token.json"
REFRESH = "refresh token"
META = ("issued at", "rotated at", "scopes", "last write check")


class SecretError(RuntimeError):
    pass


def _private(p: Path) -> Path:
    st = p.stat()
    if st.st_mode & 0o077:
        raise SecretError(f"{p} must be mode 0600 (is {oct(st.st_mode & 0o777)})")
    return p


def _credential(name: str) -> str | None:
    d = os.environ.get("CREDENTIALS_DIRECTORY")
    if not d:
        return None
    try:
        return (Path(d) / name).read_text().strip() or None
    except FileNotFoundError:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    """One backend. get() is None for a secret that isn't set; token_fields() is the refresh token's record,
    {"refresh token": ..., "issued at": ..., ...}, empty before the first `haystack auth`."""
    where = ""

    def get(self, name: str) -> str | None:
        raise NotImplementedError

    def put(self, name: str, value: str) -> None:
        raise NotImplementedError

    def token_fields(self) -> dict:
        raise NotImplementedError

    def set_token_fields(self, values: dict[str, str]) -> None:
        raise NotImplementedError

    def find(self, name: str) -> str | None:
        """The systemd credential, else the backend's value, else None."""
        return _credential(name) or self.get(name)

    def read(self, name: str) -> str:
        v = self.find(name)
        if not v:
            raise SecretError(f"the secret {name} is not set ({self.where})")
        return v


class FileStore(Store):
    def __init__(self, cfg: dict):
        self.dir = Path(os.path.expanduser(cfg["secrets"]["dir"]))
        self.where = str(self.dir)

    def get(self, name: str) -> str | None:
        p = self.dir / name
        if not p.exists():
            return None
        return _private(p).read_text().strip() or None

    def put(self, name: str, value: str) -> None:
        self._write(name, value.strip() + "\n")

    def token_fields(self) -> dict:
        p = self.dir / TOKEN_FILE
        if not p.exists():
            return {}
        return {k.replace("_", " "): v for k, v in json.loads(_private(p).read_text()).items()}

    def set_token_fields(self, values: dict[str, str]) -> None:
        fields = {**self.token_fields(), **values}
        self._write(TOKEN_FILE, json.dumps({k.replace(" ", "_"): v for k, v in fields.items()}, indent=1) + "\n")

    def _write(self, name: str, text: str) -> None:
        """A new 0600 file renamed over the old one, so a crash leaves one or the other, never half of either."""
        self.dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        tmp = self.dir / f".{name}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.dir / name)


def _op_token(cfg: dict) -> str:
    cred = _credential("op-token")
    if cred:
        return cred
    path = cfg["onepassword"]["token_file"]
    if path:
        p = Path(os.path.expanduser(path))
        if p.exists():
            tok = _private(p).read_text().strip()
            if tok:
                return tok
    raise SecretError("no 1Password service-account token (LoadCredential=op-token or onepassword.token_file)")


class OnePassword:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.op = cfg["onepassword"]["op_binary"]
        self.vault = cfg["onepassword"]["vault"]

    def _run(self, args: list[str], stdin: str | None = None) -> str:
        env = {k: v for k, v in os.environ.items() if not k.startswith("OP_")}
        env["OP_SERVICE_ACCOUNT_TOKEN"] = _op_token(self.cfg)
        try:
            r = subprocess.run([self.op, *args], input=stdin, capture_output=True, text=True, env=env, timeout=90)
        except subprocess.TimeoutExpired as e:
            raise SecretError(f"op {args[0]} timed out") from e
        if r.returncode != 0:
            # op error messages do not contain secret values; keep them short
            raise SecretError(f"op {' '.join(args[:2])} failed: {r.stderr.strip()[:300]}")
        return r.stdout

    def read(self, item: str, field: str) -> str:
        return self._run(["read", "--no-newline", f"op://{self.vault}/{item}/{field}"])

    def item(self, item: str) -> dict:
        return json.loads(self._run(["item", "get", item, "--vault", self.vault, "--format", "json"]))

    def set_fields(self, item: str, values: dict[str, tuple[str, str]]) -> None:
        """values: label -> (type, value). Edits via a JSON template on stdin (never on the command line)."""
        it = self.item(item)
        fields = it.setdefault("fields", [])
        for label, (ftype, val) in values.items():
            for f in fields:
                if f.get("label") == label:
                    f["value"] = val
                    break
            else:
                fields.append({"id": label.replace(" ", "_"), "label": label, "type": ftype, "value": val})
        self._run(["item", "edit", item, "--vault", self.vault, "--format", "json"], stdin=json.dumps(it))


class OnePasswordStore(Store):
    """Items (by id, or by a name `op://` accepts) in [onepassword].vault. An empty item means that secret isn't
    set."""

    def __init__(self, cfg: dict, op: OnePassword | None = None):
        self.o = cfg["onepassword"]
        self.op = op or OnePassword(cfg)
        self.where = f"1Password vault {self.o['vault'] or '(onepassword.vault is empty)'}"

    def _ref(self, name: str) -> tuple[str, str]:
        o = self.o
        return {"spotify-client-id": (o["client_item"], o["client_id_field"]),
                "spotify-client-secret": (o["client_item"], o["client_secret_field"]),
                "api-token": (o["api_token_item"], o["api_token_field"]),
                "alert-webhook-url": (o["alert_item"], o["alert_url_field"]),
                "alert-webhook-key": (o["alert_item"], o["alert_key_field"])}[name]

    def get(self, name: str) -> str | None:
        item, field = self._ref(name)
        return (self.op.read(item, field).strip() or None) if item else None

    def put(self, name: str, value: str) -> None:
        item, field = self._ref(name)
        if not item:
            raise SecretError(f"no 1Password item for {name}: set it in [onepassword]")
        self.op.set_fields(item, {field: ("CONCEALED", value)})

    def token_fields(self) -> dict:
        if not self.o["token_item"]:
            raise SecretError("no 1Password item for the refresh token: set onepassword.token_item")
        out = {}
        for f in self.op.item(self.o["token_item"]).get("fields", []):
            label = REFRESH if f.get("label") == self.o["token_field"] else f.get("label")
            if label in (REFRESH, *META):
                out[label] = f.get("value") or None
        return out

    def set_token_fields(self, values: dict[str, str]) -> None:
        self.op.set_fields(self.o["token_item"], {
            (self.o["token_field"] if k == REFRESH else k): ("CONCEALED" if k == REFRESH else "STRING", v)
            for k, v in values.items()})


def store(cfg: dict) -> Store:
    backend = cfg["secrets"]["backend"]
    if backend == "files":
        return FileStore(cfg)
    if backend == "1password":
        return OnePasswordStore(cfg)
    raise SecretError(f"unknown secrets.backend {backend!r} (files or 1password)")


class SpotifySecrets:
    """Client credentials + refresh token, read from the store at run time."""

    def __init__(self, cfg: dict, store_: Store | None = None):
        self.cfg = cfg
        self.store = store_ or store(cfg)

    def client(self) -> tuple[str, str]:
        return self.store.read("spotify-client-id"), self.store.read("spotify-client-secret")

    def refresh_token(self) -> tuple[str | None, dict]:
        """-> (refresh token or None, non-secret metadata {issued at, rotated at, scopes, last write check})."""
        f = self.store.token_fields()
        return (f.get(REFRESH) or "").strip() or None, {k: f[k] or None for k in META if k in f}

    def store_refresh_token(self, token: str, *, issued: bool, scopes: str | None = None) -> None:
        ts = _now()
        vals = {REFRESH: token, "rotated at": ts}
        if issued:
            vals["issued at"] = ts
        if scopes is not None:
            vals["scopes"] = scopes
        self.store.set_token_fields(vals)

    def write_check(self) -> str:
        ts = _now()
        self.store.set_token_fields({"last write check": ts})
        return ts
