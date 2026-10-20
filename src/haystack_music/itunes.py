"""Apple's iTunes Search API (no key): a second source of 30 s previews, for a Deezer track that has none (ADR 0020).

The same safeguards as Deezer's: at most one request per harvest.itunes_min_interval (3 s; Apple allows about 20 a
minute) across all processes, the preview CDN included; the AAC clip is held in memory only, decoded through the
same ffmpeg pipe, embedded and dropped; only the vectors are kept. `lookup?isrc=` finds nothing (checked 2026-10),
so a song is found by a text search, checked with matching.compare (artist, title, duration)."""
from __future__ import annotations

import logging
import time

import requests

from . import USER_AGENT
from .matching import TrackRef, compare
from .textnorm import split_title
from .throttle import Throttle

log = logging.getLogger("haystack_music.itunes")
SEARCH = "https://itunes.apple.com/search"
MAX_PREVIEW_BYTES = 3_000_000


class ItunesError(RuntimeError):
    pass


class Itunes:
    def __init__(self, min_interval: float = 3.0, session: requests.Session | None = None):
        self.t = Throttle("itunes", max(3.0, min_interval))
        self.s = session or requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.requests = 0

    def search(self, term: str, limit: int = 10, tries: int = 3) -> list[dict]:
        for i in range(tries):
            self.t.wait(); self.requests += 1
            try:
                r = self.s.get(SEARCH, params={"term": term, "media": "music", "entity": "song", "limit": limit},
                               timeout=20)
            except requests.RequestException as e:
                log.warning("itunes search: %r", e); time.sleep(5 * (i + 1)); continue
            if r.status_code in (403, 429) or r.status_code >= 500:     # Apple answers 403 when over its limit
                time.sleep(min(60, 20 * (i + 1))); continue
            if r.status_code != 200:
                return []
            try:
                return [x for x in r.json().get("results", []) if x.get("kind") == "song"]
            except ValueError:
                return []
        raise ItunesError("search: retries exhausted")

    def find(self, artist: str, title: str, duration, tol: float = 3.0) -> dict | None:
        """The iTunes song with a preview that matches artist, title and duration (high or medium confidence)."""
        q = TrackRef(None, artist, title, duration)
        best = None
        for x in self.search(f"{artist} {split_title(title)[0]}"):
            if not x.get("previewUrl"):
                continue
            secs = (x.get("trackTimeMillis") or 0) / 1000 or None
            m = compare(q, TrackRef(x.get("trackId"), x.get("artistName", ""), x.get("trackName", ""), secs), tol)
            if m and m.confidence in ("high", "medium") and (best is None or m.title_sim + m.artist_sim > best[0]):
                best = (m.title_sim + m.artist_sim, x)
        return best[1] if best else None

    def preview_bytes(self, url: str) -> bytes:
        """Fetch a 30 s preview into memory (never to disk)."""
        self.t.wait(); self.requests += 1
        with self.s.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            buf = bytearray()
            for chunk in r.iter_content(65536):
                buf += chunk
                if len(buf) > MAX_PREVIEW_BYTES:
                    raise ItunesError("preview too large")
        return bytes(buf)
