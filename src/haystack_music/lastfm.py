"""Last.fm's public API (an API key, no sign-in): similar tracks, similar artists and an artist's top tracks
(ADR 0020). Read-only methods only, at most one request per harvest.lastfm_min_interval across all processes.
Collaborative filtering over a listener base many times ListenBrainz's; the answers are names, which the harvest
resolves on Deezer like listenbrainz_similar's."""
from __future__ import annotations

import logging
import time

import requests

from . import USER_AGENT
from .throttle import Throttle

log = logging.getLogger("haystack_music.lastfm")
API = "https://ws.audioscrobbler.com/2.0/"
RETRY_CODES = {8, 11, 16, 29}      # operation failed, service offline, temporary error, rate limit exceeded
KEY_CODES = {10, 26}               # invalid API key, suspended API key


class LastfmError(RuntimeError):
    pass


def _list(x) -> list[dict]:
    """Last.fm gives a single result as an object instead of a list of one."""
    if isinstance(x, dict):
        return [x]
    return [y for y in (x or []) if isinstance(y, dict)]


def _name(x) -> str:
    return (x.get("name") or "") if isinstance(x, dict) else (x or "")


class Lastfm:
    def __init__(self, api_key: str, min_interval: float = 1.0, session: requests.Session | None = None):
        self.key = api_key
        self.t = Throttle("lastfm", max(1.0, min_interval))
        self.s = session or requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.requests = 0

    def _get(self, method: str, tries: int = 3, **params) -> dict | None:
        """None when Last.fm doesn't know the track or artist; LastfmError when the key is refused."""
        for i in range(tries):
            self.t.wait(); self.requests += 1
            try:
                r = self.s.get(API, params={"method": method, "api_key": self.key, "format": "json",
                                            "autocorrect": 1, **params}, timeout=20)
            except requests.RequestException as e:
                log.warning("lastfm %s: %r", method, e); time.sleep(3 * (i + 1)); continue
            try:
                j = r.json()
            except ValueError:
                j = {}
            code = j.get("error") if isinstance(j, dict) else None
            if r.status_code == 429 or r.status_code >= 500 or code in RETRY_CODES:
                time.sleep(min(60, 5 * (i + 1))); continue
            if code in KEY_CODES:
                raise LastfmError(f"{method}: the API key was refused ({j.get('message')})")
            if code or r.status_code != 200:
                return None
            return j
        raise LastfmError(f"{method}: retries exhausted")

    def similar_tracks(self, artist: str, title: str, limit: int = 20) -> list[dict]:
        """[{artist, title, match}], most similar first."""
        j = self._get("track.getSimilar", artist=artist, track=title, limit=limit) or {}
        out = []
        for t in _list((j.get("similartracks") or {}).get("track")):
            a, n = _name(t.get("artist")), t.get("name") or ""
            if a and n:
                out.append({"artist": a, "title": n, "match": float(t.get("match") or 0)})
        return sorted(out, key=lambda x: -x["match"])

    def similar_artists(self, artist: str, limit: int = 3) -> list[str]:
        j = self._get("artist.getSimilar", artist=artist, limit=limit) or {}
        return [_name(a) for a in _list((j.get("similarartists") or {}).get("artist")) if _name(a)][:limit]

    def top_tracks(self, artist: str, limit: int = 3) -> list[dict]:
        """[{artist, title}], the artist's most played first."""
        j = self._get("artist.getTopTracks", artist=artist, limit=limit) or {}
        return [{"artist": _name(t.get("artist")) or artist, "title": t["name"]}
                for t in _list((j.get("toptracks") or {}).get("track")) if t.get("name")][:limit]
