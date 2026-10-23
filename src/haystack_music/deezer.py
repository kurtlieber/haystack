"""Deezer public API (no key). Used to reach candidate tracks and their 30 s previews, and the facts Deezer publishes
about a track (its album's label and date, its rank and its artist's fans: ADR 0017). Never a score on its own.

Safeguards against the terms risk (README, Deezer notice):
  * at most one HTTP request per 1.1 s across ALL processes (API and preview CDN share one throttle),
  * preview audio is held in memory only, decoded through an ffmpeg pipe, embedded and dropped
    (it is never written to disk, not even to a temp file),
  * only the resulting vectors and those facts are kept, locally, never shared."""
from __future__ import annotations

import logging
import time

import requests

from . import USER_AGENT
from .throttle import Throttle

log = logging.getLogger("haystack_music.deezer")
API = "https://api.deezer.com/"
MAX_PREVIEW_BYTES = 3_000_000


class DeezerError(RuntimeError):
    pass


class Deezer:
    def __init__(self, min_interval: float = 1.1, session: requests.Session | None = None):
        self.t = Throttle("deezer", max(1.1, min_interval))
        self.s = session or requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.requests = 0

    def api(self, path: str, params: dict | None = None, tries: int = 4):
        for i in range(tries):
            self.t.wait(); self.requests += 1
            try:
                r = self.s.get(API + path.lstrip("/"), params=params, timeout=20)
            except requests.RequestException as e:
                log.warning("deezer %s: %r", path, e); time.sleep(3 * (i + 1)); continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, int(r.headers.get("Retry-After", 5 * (i + 1))))); continue
            j = r.json()
            if isinstance(j, dict) and "error" in j:
                if j["error"].get("code") == 4:       # quota exceeded
                    time.sleep(5 * (i + 1)); continue
                if j["error"].get("code") == 800:     # no data
                    return None
                raise DeezerError(f"{path}: {j['error']}")
            return j
        raise DeezerError(f"{path}: retries exhausted")

    def _data(self, path, params=None) -> list[dict]:
        j = self.api(path, params)
        return (j or {}).get("data", []) if isinstance(j, dict) else []

    def search(self, q: str, limit: int = 25, order: str | None = None) -> list[dict]:
        # NOTE: two fields together, artist:"x" track:"y", return 0 results (2026-10 experiment); a lone track:"y"
        # works (2026-10-23). order: RANKING (most popular first), RATING_DESC, ….
        return self._data("search", {"q": q, "limit": limit, **({"order": order} if order else {})})

    def track(self, track_id: int) -> dict | None:
        return self.api(f"track/{int(track_id)}")

    def track_by_isrc(self, isrc: str) -> dict | None:
        try:
            return self.api(f"track/isrc:{isrc}")
        except DeezerError:
            return None

    def chart_tracks(self, editorial_id: int = 0, limit: int = 100) -> list[dict]:
        j = self.api(f"chart/{int(editorial_id)}/tracks", {"limit": limit})
        return (j or {}).get("data", [])

    def editorial_releases(self, editorial_id: int = 0, limit: int = 50) -> list[dict]:
        """New releases picked by Deezer's editors. /editorial/{id}/releases returns an empty list as of
        2026-10, so /editorial/{id}/selection (the editorial new-release selection) is used first."""
        return self._data(f"editorial/{int(editorial_id)}/selection", {"limit": limit}) or \
            self._data(f"editorial/{int(editorial_id)}/releases", {"limit": limit})

    def album(self, album_id: int) -> dict | None:
        """An album with its label, release date, record type, genres and fans (None: Deezer has no such album)."""
        return self.api(f"album/{int(album_id)}")

    def artist(self, artist_id: int) -> dict | None:
        """An artist with its fan count (nb_fan)."""
        return self.api(f"artist/{int(artist_id)}")

    def album_tracks(self, album_id: int, limit: int = 3) -> list[dict]:
        return self._data(f"album/{int(album_id)}/tracks", {"limit": limit})

    def search_artist(self, name: str, limit: int = 5) -> list[dict]:
        return self._data("search/artist", {"q": name, "limit": limit})

    def artist_radio(self, artist_id: int, limit: int = 25) -> list[dict]:
        return self._data(f"artist/{int(artist_id)}/radio", {"limit": limit})

    def artist_related(self, artist_id: int, limit: int = 10) -> list[dict]:
        return self._data(f"artist/{int(artist_id)}/related", {"limit": limit})

    def artist_top(self, artist_id: int, limit: int = 3) -> list[dict]:
        return self._data(f"artist/{int(artist_id)}/top", {"limit": limit})

    def search_album(self, q: str, limit: int = 50) -> list[dict]:
        """Albums; Deezer's advanced search takes label:"<name>" here (checked 2026-10), but can't sort by date."""
        return self._data("search/album", {"q": q, "limit": limit})

    def search_playlist(self, q: str, limit: int = 6) -> list[dict]:
        return self._data("search/playlist", {"q": q, "limit": limit})

    def playlist(self, playlist_id: int) -> dict | None:
        """A playlist with its fans and its tracks (the first page, in playlist order)."""
        try:
            j = self.api(f"playlist/{int(playlist_id)}")
        except DeezerError:                   # a playlist made private since the search
            return None
        return j if isinstance(j, dict) and j.get("id") else None

    def preview_bytes(self, url: str) -> bytes:
        """Fetch a 30 s preview into memory (never to disk)."""
        self.t.wait(); self.requests += 1
        with self.s.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            buf = bytearray()
            for chunk in r.iter_content(65536):
                buf += chunk
                if len(buf) > MAX_PREVIEW_BYTES:
                    raise DeezerError("preview too large")
        return bytes(buf)
