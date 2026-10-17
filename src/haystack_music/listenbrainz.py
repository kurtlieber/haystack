"""ListenBrainz labs endpoints (no key; MetaBrainz blocks cloud IPs, so the harvest must run from a home connection).

api.listenbrainz.org metadata/lookup and lb-radio now require a user token (401), so v1 uses the keyless
labs dataset hoster: recording-search (MBID resolution) + similar-recordings (session-based similarity)."""
from __future__ import annotations

import logging
import time

import requests

from . import USER_AGENT
from .textnorm import key, split_title
from .throttle import Throttle

log = logging.getLogger("haystack_music.listenbrainz")
LABS = "https://labs.api.listenbrainz.org/"


class ListenBrainz:
    def __init__(self, min_interval: float = 1.2, algorithm: str | None = None, session=None, contact: str = ""):
        self.t = Throttle("listenbrainz", min_interval)
        self.s = session or requests.Session()
        # MetaBrainz asks for a way to reach the operator (harvest.listenbrainz_contact)
        self.s.headers["User-Agent"] = (USER_AGENT[:-1] + f"; {contact})") if contact else USER_AGENT
        self.algorithm = algorithm or "session_based_days_7500_session_300_contribution_5_threshold_15_limit_50_skip_30"
        self.requests = 0

    def _get(self, path: str, params: dict, tries: int = 3):
        for i in range(tries):
            self.t.wait(); self.requests += 1
            try:
                r = self.s.get(LABS + path, params=params, timeout=30)
            except requests.RequestException as e:
                log.warning("listenbrainz %s: %r", path, e); time.sleep(3 * (i + 1)); continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, int(r.headers.get("Retry-After", 5 * (i + 1))))); continue
            if r.status_code != 200:
                return []
            try:
                return r.json()
            except ValueError:
                return []
        return []

    def recording_search(self, artist: str, title: str) -> list[dict]:
        return self._get("recording-search/json", {"query": f"{artist} {title}"}) or []

    def similar_recordings(self, mbid: str) -> list[dict]:
        return self._get("similar-recordings/json", {"recording_mbids": mbid, "algorithm": self.algorithm}) or []

    def similar_for(self, artist: str, title: str, tries: int = 3) -> tuple[str | None, list[dict]]:
        """Resolve artist/title to MBIDs and return the first non-empty similar-recordings list.
        (Only the canonical recording carries session data; DJ-mix/live duplicates return [].)"""
        akey, tkey = key(artist), key(split_title(title)[0])
        hits = [h for h in self.recording_search(artist, title)
                if key(split_title(h.get("recording_name", ""))[0]) == tkey
                and (akey in key(h.get("artist_credit_name", "")) or key(h.get("artist_credit_name", "")) in akey)]
        for h in hits[:tries]:
            sims = self.similar_recordings(h["recording_mbid"])
            if sims:
                return h["recording_mbid"], sims
        return (hits[0]["recording_mbid"] if hits else None), []
