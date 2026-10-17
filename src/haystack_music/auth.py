"""One-time Authorization Code login. `haystack auth` listens on the redirect URI's loopback port; open the
printed URL in a browser on this computer, or on another one through an SSH tunnel (tunnel()).

Only non-secret output: the authorize URL (contains the public client id + a one-time state), the Spotify
display name and product. The code, access token and refresh token are never printed or logged."""
from __future__ import annotations

import logging
import secrets as pysecrets
import socket
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
    """How to renew the login: `haystack auth` here, through an SSH tunnel from a computer with a browser."""
    return f"Run `haystack auth` on {socket.gethostname().split('.')[0]} (from another computer: {tunnel(cfg)})."


def run(cfg: dict, timeout: int = 900, ready_file: str | None = None, log_file: str | None = None) -> int:
    sp_cfg = cfg["spotify"]
    redirect = sp_cfg["redirect_uri"]
    u = urlparse(redirect)
    if u.hostname != "127.0.0.1":
        log.error("redirect_uri must be http://127.0.0.1:<port>/... (Spotify rejects 'localhost')")
        return 2
    sec = SpotifySecrets(cfg)
    cid, csecret = sec.client()
    state = pysecrets.token_urlsafe(24)
    result: dict = {}
    done = threading.Event()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            q = urlparse(self.path)
            if q.path != u.path:
                self.send_response(404); self.end_headers(); return
            p = parse_qs(q.query)
            if p.get("state", [None])[0] != state:
                result["error"] = "state mismatch (stale URL?)"
            elif "error" in p:
                result["error"] = p["error"][0]
            else:
                result["code"] = p.get("code", [None])[0]
            ok = bool(result.get("code"))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(("<h2>haystack: " + ("authorised, finishing up. You can close this tab."
                              if ok else "authorisation failed: " + result.get("error", "?")) + "</h2>").encode())
            if ok or result.get("error") not in ("state mismatch (stale URL?)",):
                done.set()

    srv = HTTPServer(("127.0.0.1", u.port or 80), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    url = authorize_url(cid, redirect, sp_cfg["scopes"], state)
    deadline = datetime.fromtimestamp(datetime.now().timestamp() + timeout).strftime("%Y-%m-%d %H:%M %Z").strip()
    if ready_file:
        Path(ready_file).write_text(READY_TEMPLATE.format(
            created=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"), url=url, redirect=redirect,
            deadline=deadline, log_file=log_file or "(journal only)",
            hint=f"From another computer, open the tunnel first: `{tunnel(cfg)}`"))
        log.info("wrote %s", ready_file)
    print(f"Open this URL in a browser and approve access:\n\n  {url}\n\nFrom another computer, open the tunnel "
          f"first: {tunnel(cfg)}\n", flush=True)
    log.info("listening on %s until %s", redirect, deadline)
    got = done.wait(timeout)
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
