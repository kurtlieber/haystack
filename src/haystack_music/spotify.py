"""The ONLY module that talks to Spotify (Web API, 2026 Development-Mode endpoint set).

Not used (unavailable in dev mode): audio-features, recommendations, browse/new-releases, artist top-tracks,
batch GET /tracks. Search is capped at limit=10. Playlist endpoints use the /items paths and snapshot_id."""
from __future__ import annotations

import base64
import logging
import os
import time
from urllib.parse import urlencode

import requests

from . import config
from .secrets import SpotifySecrets

log = logging.getLogger("haystack_music.spotify")
API = "https://api.spotify.com/v1"
ACCOUNTS = "https://accounts.spotify.com"
PENDING_TOKEN = "refresh-token.pending"   # emergency 0600 copy if a rotated token cannot be stored


class AuthMissing(RuntimeError):
    """No usable refresh token: run `haystack auth`."""


class SpotifyError(RuntimeError):
    def __init__(self, status: int, msg: str):
        super().__init__(f"Spotify HTTP {status}: {msg}")
        self.status = status


def authorize_url(client_id: str, redirect_uri: str, scopes: list[str], state: str) -> str:
    return ACCOUNTS + "/authorize?" + urlencode({"client_id": client_id, "response_type": "code",
                                                  "redirect_uri": redirect_uri, "scope": " ".join(scopes),
                                                  "state": state, "show_dialog": "true"})


def _basic(cid: str, secret: str) -> str:
    return "Basic " + base64.b64encode(f"{cid}:{secret}".encode()).decode()


def exchange_code(cid: str, secret: str, code: str, redirect_uri: str) -> dict:
    r = requests.post(ACCOUNTS + "/api/token", data={"grant_type": "authorization_code", "code": code,
                                                    "redirect_uri": redirect_uri},
                      headers={"Authorization": _basic(cid, secret)}, timeout=30)
    if r.status_code != 200:
        raise SpotifyError(r.status_code, r.text[:200])
    return r.json()


class Spotify:
    def __init__(self, cfg: dict, secrets: SpotifySecrets | None = None, session: requests.Session | None = None):
        self.cfg = cfg
        self.secrets = secrets or SpotifySecrets(cfg)
        self.s = session or requests.Session()
        self._access = None
        self._exp = 0.0
        self._refresh = None
        self.token_meta: dict = {}
        self.calls = 0

    # ---- auth ----
    def _load_refresh(self) -> str:
        if self._refresh is None:
            self._flush_pending()
            tok, meta = self.secrets.refresh_token()
            if not tok:
                raise AuthMissing("no Spotify refresh token stored - run `haystack auth`")
            self._refresh, self.token_meta = tok, meta
        return self._refresh

    def _flush_pending(self) -> None:
        p = config.STATE_DIR / PENDING_TOKEN
        if p.exists():
            tok = p.read_text().strip()
            if tok:
                self.secrets.store_refresh_token(tok, issued=False)
            p.unlink()
            log.warning("flushed a pending rotated refresh token to the secret store")

    def _refresh_access(self) -> None:
        rt = self._load_refresh()
        cid, secret = self.secrets.client()
        r = self.s.post(ACCOUNTS + "/api/token", data={"grant_type": "refresh_token", "refresh_token": rt},
                        headers={"Authorization": _basic(cid, secret)}, timeout=30)
        if r.status_code in (400, 401):
            err = (r.json() if r.headers.get("content-type", "").startswith("application/json") else {}).get("error")
            raise AuthMissing(f"refresh token rejected ({err or r.status_code}) - run `haystack auth`")
        if r.status_code != 200:
            raise SpotifyError(r.status_code, "token refresh failed")
        j = r.json()
        self._access = j["access_token"]
        self._exp = time.time() + int(j.get("expires_in", 3600)) - 60
        new_rt = j.get("refresh_token")
        if new_rt and new_rt != rt:
            # rotation: persist immediately, before anything else can fail
            try:
                self.secrets.store_refresh_token(new_rt, issued=False)
                log.info("refresh token rotated; new token stored")
            except Exception:
                p = config.STATE_DIR / PENDING_TOKEN
                fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as f:
                    f.write(new_rt)
                log.error("could not store the rotated refresh token; kept a 0600 copy for the next run")
                raise
            self._refresh = new_rt

    def _token(self) -> str:
        if not self._access or time.time() >= self._exp:
            self._refresh_access()
        return self._access

    # ---- HTTP ----
    def request(self, method: str, path: str, params: dict | None = None, json: dict | None = None,
                tries: int = 5):
        url = path if path.startswith("http") else API + path
        for i in range(tries):
            self.calls += 1
            r = self.s.request(method, url, params=params, json=json, timeout=30,
                               headers={"Authorization": f"Bearer {self._token()}"})
            if r.status_code == 401 and i == 0:
                self._access = None; continue
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "5"))
                if wait > 900:
                    raise SpotifyError(429, f"rate limited for {wait}s; giving up this run")
                log.warning("429 from Spotify; sleeping %ss (Retry-After)", wait)
                time.sleep(wait + 1); continue
            if r.status_code >= 500:
                time.sleep(2 ** i); continue
            if r.status_code >= 400:
                raise SpotifyError(r.status_code, r.text[:300])
            return r.json() if r.content else {}
        raise SpotifyError(-1, f"{method} {path}: retries exhausted")

    def _pages(self, path: str, params: dict | None = None):
        j = self.request("GET", path, params)
        while True:
            yield from j.get("items", [])
            if not j.get("next"):
                break
            j = self.request("GET", j["next"])

    # ---- endpoints ----
    def me(self) -> dict:
        return self.request("GET", "/me")

    def saved_tracks(self):
        """Liked Songs, newest first: yields (added_at, track)."""
        for it in self._pages("/me/tracks", {"limit": 50}):
            t = it.get("track") or it.get("item")
            if t and t.get("id"):
                yield it.get("added_at"), t

    def recently_played(self, after_ms: int | None = None) -> list[dict]:
        p = {"limit": 50}
        if after_ms:
            p["after"] = int(after_ms)
        return self.request("GET", "/me/player/recently-played", p).get("items", [])

    def my_playlists(self):
        yield from self._pages("/me/playlists", {"limit": 50})

    def create_playlist(self, name: str, description: str, public: bool = False) -> dict:
        return self.request("POST", "/me/playlists", json={"name": name, "description": description, "public": public})

    def playlist(self, playlist_id: str) -> dict:
        return self.request("GET", f"/playlists/{playlist_id}", {"fields": "id,name,snapshot_id,public,owner(id)"})

    def playlist_items(self, playlist_id: str) -> list[dict]:
        out = []
        for it in self._pages(f"/playlists/{playlist_id}/items", {"limit": 100}):
            t = it.get("item") or it.get("track")
            if t and t.get("id"):
                out.append({"spotify_id": t["id"], "added_at": it.get("added_at"),
                            "added_by": (it.get("added_by") or {}).get("id")})
        return out

    def playlist_tracks(self, playlist_id: str):
        """A playlist's songs in order, with their full track objects: yields (added_at, track). Local files and
        podcast episodes are left out. Dev mode answers 403 for a playlist the listener follows but doesn't own."""
        for it in self._pages(f"/playlists/{playlist_id}/items", {"limit": 100}):
            t = it.get("item") or it.get("track")
            if t and t.get("id") and t.get("type", "track") == "track" and not t.get("is_local"):
                yield it.get("added_at"), t

    def add_items(self, playlist_id: str, track_ids: list[str], position: int | None = None) -> str | None:
        """Add tracks at the end, or from `position` (0-based) on."""
        snap = None
        for i in range(0, len(track_ids), 100):
            body = {"uris": [f"spotify:track:{t}" for t in track_ids[i:i + 100]]}
            if position is not None:
                body["position"] = position + i
            j = self.request("POST", f"/playlists/{playlist_id}/items", json=body)
            snap = j.get("snapshot_id", snap)
        return snap

    def remove_items(self, playlist_id: str, track_ids: list[str], snapshot_id: str | None = None) -> str | None:
        snap = snapshot_id
        for i in range(0, len(track_ids), 100):
            body = {"items": [{"uri": f"spotify:track:{t}"} for t in track_ids[i:i + 100]]}
            if snap:
                body["snapshot_id"] = snap
            j = self.request("DELETE", f"/playlists/{playlist_id}/items", json=body)
            snap = j.get("snapshot_id", snap)
        return snap

    def library_contains(self, track_ids: list[str]) -> dict[str, bool]:
        out = {}
        for i in range(0, len(track_ids), 40):
            chunk = track_ids[i:i + 40]
            j = self.request("GET", "/me/library/contains", {"uris": ",".join(f"spotify:track:{t}" for t in chunk)})
            out.update(dict(zip(chunk, j if isinstance(j, list) else [])))
        return out

    def library_save(self, track_ids: list[str]) -> None:
        """Add the heart (2026: PUT /me/library with URIs, at most 40; needs user-library-modify)."""
        for i in range(0, len(track_ids), 40):
            self.request("PUT", "/me/library", {"uris": ",".join(f"spotify:track:{t}" for t in track_ids[i:i + 40])})

    def library_remove(self, track_ids: list[str]) -> None:
        for i in range(0, len(track_ids), 40):
            self.request("DELETE", "/me/library", {"uris": ",".join(f"spotify:track:{t}" for t in track_ids[i:i + 40])})

    def search_tracks(self, q: str, limit: int = 10) -> list[dict]:
        return self.request("GET", "/search", {"q": q, "type": "track", "limit": min(10, limit)}) \
            .get("tracks", {}).get("items", [])

    def track(self, track_id: str) -> dict:
        return self.request("GET", f"/tracks/{track_id}")
