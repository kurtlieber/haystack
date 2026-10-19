"""One-time Authorization Code login. `haystack auth` prints Spotify's authorize URL; after you approve, Spotify
sends the browser to the redirect URI (a 127.0.0.1 address) with a one-time code. Two ways back:

* the listener on the redirect URI's loopback port picks it up, when the browser runs on this computer or comes
  through an SSH tunnel (tunnel());
* from any other browser (in a container, on a phone) that page can't load, but its address holds the code: paste
  it at the prompt. Needs a terminal (stdin a tty).

Whichever arrives first wins; both check the same state value.

Only non-secret output: the authorize URL (contains the public client id + a one-time state), the Spotify
display name and product. The code, access token and refresh token are never printed or logged."""
from __future__ import annotations

import logging
import secrets as pysecrets
import socket
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import db, status
from .secrets import SpotifySecrets
from .spotify import Spotify, authorize_url, exchange_code

log = logging.getLogger("haystack_music.auth")

READY_TEMPLATE = """# Spotify login for haystack: ready ({created})

1. Open this URL in a browser and approve access:

   {url}

   {hint}

2. Redirect URI used: `{redirect}`
   - It must be listed in the Spotify app's Redirect URIs.
   - If Spotify shows **"INVALID_CLIENT: Invalid redirect URI"**: open https://developer.spotify.com/dashboard,
     select the app, go to Settings > Redirect URIs, add exactly `{redirect}`, save, then reload the URL above
     (it stays valid while the listener runs).

3. The listener runs as the transient user unit `haystack-auth` until {deadline} (15 min).
   Result (success or failure) shows up in:
   - `{log_file}`
   - `journalctl --user -u haystack-auth`
"""


def tunnel(cfg: dict) -> str:
    """The SSH command that brings this host's auth listener to a browser on another computer."""
    port = urlparse(cfg["spotify"]["redirect_uri"]).port or 80
    return f"ssh -L {port}:127.0.0.1:{port} {socket.gethostname().split('.')[0]}"


def reauth_hint(cfg: dict) -> str:
    """How to renew the login: `haystack auth` here, then the URL in any browser."""
    return (f"Run `haystack auth` on {socket.gethostname().split('.')[0]}, open the URL it prints in any browser, "
            "and paste back the address the browser lands on.")


STALE = "state mismatch (stale URL?)"
NOT_CALLBACK = "that isn't the address Spotify sent the browser to"


def callback(query: str, state: str) -> tuple[str | None, str | None]:
    """(code, error) from the redirect's query string."""
    p = parse_qs(query)
    if p.get("state", [None])[0] != state:
        return None, STALE
    if "error" in p:
        return None, p["error"][0]
    code = p.get("code", [None])[0]
    return (code, None) if code else (None, "no code in the address")


def pasted(text: str, redirect: str, state: str) -> tuple[str | None, str | None]:
    """(code, error) from an address pasted from the browser's address bar."""
    q = urlparse(text.strip())
    if q.path != urlparse(redirect).path or not q.query:
        return None, NOT_CALLBACK
    return callback(q.query, state)


def _read_pasted(redirect: str, state: str, finish) -> None:
    """The paste prompt, until an address with a code (or Spotify's error) arrives, or the listener got one."""
    while True:
        try:
            line = input("Paste the address here: ")
        except (EOFError, OSError):
            return
        if not line.strip():
            continue
        code, err = pasted(line, redirect, state)
        if err in (STALE, NOT_CALLBACK):
            print(f"  {err}: copy the whole address, starting {redirect}?code=", flush=True)
            continue
        finish(code, err)
        return


def run(cfg: dict, timeout: int = 900, ready_file: str | None = None, log_file: str | None = None,
        paste: bool | None = None) -> int:
    sp_cfg = cfg["spotify"]
    redirect = sp_cfg["redirect_uri"]
    u = urlparse(redirect)
    if u.hostname != "127.0.0.1":
        log.error("redirect_uri must be http://127.0.0.1:<port>/... (Spotify rejects 'localhost')")
        return 2
    paste = sys.stdin.isatty() if paste is None else paste
    sec = SpotifySecrets(cfg)
    cid, csecret = sec.client()
    state = pysecrets.token_urlsafe(24)
    result: dict = {}
    done = threading.Event()
    lock = threading.Lock()

    def finish(code: str | None, err: str | None) -> bool:
        """The first code (or Spotify error) wins; a stale state is noted but keeps waiting."""
        with lock:
            if done.is_set():
                return False
            if code:
                result.pop("error", None)
                result["code"] = code
            else:
                result["error"] = err
            if code or err != STALE:
                done.set()
            return True

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            q = urlparse(self.path)
            if q.path != u.path:
                self.send_response(404); self.end_headers(); return
            code, err = callback(q.query, state)
            finish(code, err)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(("<h2>haystack: " + ("authorised, finishing up. You can close this tab."
                              if code else "authorisation failed: " + (err or "?")) + "</h2>").encode())

    srv = None
    try:
        srv = HTTPServer(("127.0.0.1", u.port or 80), H)
    except OSError as e:
        if not paste:
            log.error("can't listen on %s: %s", redirect, e)
            return 1
        log.warning("can't listen on %s (%s): paste the address instead", redirect, e)
    if srv:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = authorize_url(cid, redirect, sp_cfg["scopes"], state)
    deadline = datetime.fromtimestamp(datetime.now().timestamp() + timeout).strftime("%Y-%m-%d %H:%M %Z").strip()
    if ready_file:
        Path(ready_file).write_text(READY_TEMPLATE.format(
            created=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"), url=url, redirect=redirect,
            deadline=deadline, log_file=log_file or "(journal only)",
            hint=f"From another computer, open the tunnel first: `{tunnel(cfg)}`"))
        log.info("wrote %s", ready_file)
    log.info("waiting for the redirect until %s", deadline)
    if paste:
        print(f"Open this URL in any browser (a phone works too) and approve access:\n\n  {url}\n\n"
              f"Spotify then sends the browser to {redirect}?code=...\n"
              "That page will most likely say it can't connect. That's expected: copy the whole address from the\n"
              "address bar and paste it below, within a few minutes. (On this machine's own browser, the login\n"
              "may finish by itself.)\n", flush=True)
        threading.Thread(target=_read_pasted, args=(redirect, state, finish), daemon=True).start()
    else:
        print(f"Open this URL in a browser and approve access:\n\n  {url}\n\nFrom another computer, open the "
              f"tunnel first: {tunnel(cfg)}\n", flush=True)
    got = done.wait(timeout)
    if srv:
        srv.shutdown()
    if not got or not result.get("code"):
        log.error("AUTH FAILED: no authorisation received (%s)", result.get("error", "timeout after 15 min"))
        status.update("auth", {"ok": False, "message": f"auth failed: {result.get('error', 'timeout')}"})
        return 1
    try:
        tok = exchange_code(cid, csecret, result["code"], redirect)
    except Exception as e:  # noqa: BLE001
        log.error("AUTH FAILED: code exchange: %s", e)
        status.update("auth", {"ok": False, "message": "code exchange failed"})
        return 1
    if not tok.get("refresh_token"):
        log.error("AUTH FAILED: Spotify returned no refresh token"); return 1
    sec.store_refresh_token(tok["refresh_token"], issued=True, scopes=tok.get("scope", ""))
    log.info("refresh token + issue date stored (%s)", sec.store.where)
    sp = Spotify(cfg, sec)
    me = sp.me()   # uses the stored refresh token end to end, so this also proves the secret store's round trip
    product = me.get("product") or "not returned by /v1/me (field removed for dev-mode apps in 2026)"
    log.info("AUTH OK: GET /v1/me -> display name: %s | product: %s", me.get("display_name") or "(none)", product)
    _, meta = sec.refresh_token()
    info = status.token_expiry_info(meta.get("issued at"), sp_cfg["token_warn_days"], sp_cfg["token_lifetime_days"])
    status.update("auth", {"ok": True, "message": "authorised", "display_name": me.get("display_name"),
                           "product": me.get("product"), **info})
    con = db.connect()
    db.kv_set(con, "spotify_user", me.get("id"))
    log.info("token issued at %s, expected expiry ~%s", info.get("issued_at"), info.get("expires_estimate"))
    return 0
