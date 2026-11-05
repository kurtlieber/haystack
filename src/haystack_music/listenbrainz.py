"""ListenBrainz (MetaBrainz blocks cloud IPs, so the harvest must run from a home connection).

The keyless labs dataset hoster gives listenbrainz_similar its MBIDs (recording-search) and session-based
similarity (similar-recordings). The API gives listenbrainz_recs the listener's own weekly recommendations and the
names behind their MBIDs (cf/recommendation, metadata/recording: no token, checked 2026-10), and
listenbrainz_radio its LB Radio playlists (explore/lb-radio: a user token since 2026, the listenbrainz-token
secret). The keyless explore/fresh-releases, labs similar-artists and popularity/top-recordings-for-artist feed
listenbrainz_fresh and listenbrainz_artist_similar (ADR 0026). All of them share one throttle (ADR 0020)."""
from __future__ import annotations

import logging
import time

import requests

from . import USER_AGENT
from .textnorm import key, split_title
from .throttle import Throttle

log = logging.getLogger("haystack_music.listenbrainz")
LABS = "https://labs.api.listenbrainz.org/"
API = "https://api.listenbrainz.org/1/"
ARTIST_ALGORITHM = "session_based_days_9000_session_300_contribution_5_threshold_15_limit_50_skip_30"


class ListenBrainzError(RuntimeError):
    pass


def _mbid(identifier) -> str | None:
    """A JSPF track's recording MBID: its identifier is a MusicBrainz recording URL, or a list of them."""
    for x in ([identifier] if isinstance(identifier, str) else identifier or []):
        if isinstance(x, str) and "/recording/" in x:
            return x.rstrip("/").rsplit("/", 1)[-1]
    return None


class ListenBrainz:
    def __init__(self, min_interval: float = 1.2, algorithm: str | None = None, session=None, contact: str = "",
                 token: str | None = None, artist_algorithm: str | None = None):
        self.t = Throttle("listenbrainz", min_interval)
        self.token = token
        self.s = session or requests.Session()
        # MetaBrainz asks for a way to reach the operator (harvest.listenbrainz_contact)
        self.s.headers["User-Agent"] = (USER_AGENT[:-1] + f"; {contact})") if contact else USER_AGENT
        self.algorithm = algorithm or "session_based_days_7500_session_300_contribution_5_threshold_15_limit_50_skip_30"
        self.artist_algorithm = artist_algorithm or ARTIST_ALGORITHM
        self.requests = 0

    def _get(self, path: str, params: dict, tries: int = 3, base: str = LABS, auth: bool = False):
        headers = {"Authorization": f"Token {self.token}"} if auth and self.token else None
        for i in range(tries):
            self.t.wait(); self.requests += 1
            try:
                r = self.s.get(base + path, params=params, headers=headers, timeout=30)
            except requests.RequestException as e:
                log.warning("listenbrainz %s: %r", path, e); time.sleep(3 * (i + 1)); continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, int(r.headers.get("Retry-After", 5 * (i + 1))))); continue
            if r.status_code == 401 and auth:
                raise ListenBrainzError(f"{path}: the token was refused (401)")
            if r.status_code != 200:            # 204: nothing yet (no recommendations for a new account)
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

    def recommendations(self, user: str, count: int = 100) -> list[str]:
        """The recording MBIDs ListenBrainz recommends to `user` this week, best first ([] before it has any)."""
        j = self._get(f"cf/recommendation/user/{user}/recording", {"count": count}, base=API) or {}
        mbids = ((j.get("payload") or {}).get("mbids") or []) if isinstance(j, dict) else []
        return [m["recording_mbid"] for m in sorted(mbids, key=lambda m: -(m.get("score") or 0))
                if m.get("recording_mbid")]

    def recordings(self, mbids: list[str], per_request: int = 25) -> dict[str, dict]:
        """mbid -> {artist, title, isrc (or None)}; an MBID MusicBrainz doesn't know is left out."""
        out: dict[str, dict] = {}
        for s in range(0, len(mbids), per_request):
            j = self._get("metadata/recording/", {"recording_mbids": ",".join(mbids[s:s + per_request]),
                                                  "inc": "artist"}, base=API) or {}
            for mbid, v in (j.items() if isinstance(j, dict) else []):
                rec, art = (v or {}).get("recording") or {}, (v or {}).get("artist") or {}
                if rec.get("name") and art.get("name"):
                    out[mbid] = {"artist": art["name"], "title": rec["name"], "isrc": (rec.get("isrcs") or [None])[0]}
        return out

    def radio(self, prompt: str, mode: str = "easy") -> list[dict]:
        """An LB Radio playlist for `prompt`: [{artist, title, mbid}]. Needs the token (ListenBrainzError on 401)."""
        j = self._get("explore/lb-radio", {"prompt": prompt, "mode": mode}, base=API, auth=True) or {}
        pl = (((j.get("payload") or {}).get("jspf") or {}).get("playlist") or {}) if isinstance(j, dict) else {}
        return [{"artist": t.get("creator") or "", "title": t.get("title") or "", "mbid": _mbid(t.get("identifier"))}
                for t in pl.get("track") or [] if t.get("creator") and t.get("title")]

    def fresh_releases(self, days: int = 7) -> list[dict]:
        """Releases of the last `days` days: [{artist, title, release_mbid, tags}], tags as ListenBrainz writes them
        ('drum & bass', 'trip-hop'); most listened first."""
        j = self._get("explore/fresh-releases/", {"days": days, "past": "true", "future": "false"}, base=API) or {}
        rel = ((j.get("payload") or {}).get("releases") or []) if isinstance(j, dict) else []
        out = [{"artist": r.get("artist_credit_name") or "", "title": r.get("release_name") or "",
                "release_mbid": r.get("release_mbid") or "", "tags": [t for t in r.get("release_tags") or []
                                                                      if isinstance(t, str)],
                "listens": r.get("listen_count") or 0}
               for r in rel if isinstance(r, dict)]
        return sorted([r for r in out if r["artist"] and r["title"] and r["release_mbid"]], key=lambda r: -r["listens"])

    def artist_mbid(self, artist: str, title: str) -> str | None:
        """The MusicBrainz id of a song's artist: the song's recording (recording-search, as similar_for finds it),
        then its credited artists (metadata/recording); the one whose name is `artist`, else the first."""
        akey, tkey = key(artist), key(split_title(title)[0])
        hits = [h for h in self.recording_search(artist, title)
                if key(split_title(h.get("recording_name", ""))[0]) == tkey
                and (akey in key(h.get("artist_credit_name", "")) or key(h.get("artist_credit_name", "")) in akey)]
        if not hits:
            return None
        mbid = hits[0]["recording_mbid"]
        j = self._get("metadata/recording/", {"recording_mbids": mbid, "inc": "artist"}, base=API) or {}
        arts = (((j.get(mbid) or {}).get("artist") or {}).get("artists") or []) if isinstance(j, dict) else []
        arts = [a for a in arts if isinstance(a, dict) and a.get("artist_mbid")]
        for a in arts:
            if key(a.get("name")) == akey:
                return a["artist_mbid"]
        return arts[0]["artist_mbid"] if arts else None

    def similar_artists(self, mbid: str, limit: int = 6) -> list[dict]:
        """[{name, mbid}], most similar first (labs similar-artists, session based)."""
        j = self._get("similar-artists/json", {"artist_mbids": mbid, "algorithm": self.artist_algorithm}) or []
        rows = [x for x in j if isinstance(x, dict) and x.get("artist_mbid") and x.get("name")
                and x["artist_mbid"] != mbid] if isinstance(j, list) else []
        rows.sort(key=lambda x: -(x.get("score") or 0))
        return [{"name": x["name"], "mbid": x["artist_mbid"]} for x in rows[:limit]]

    def top_recordings(self, mbid: str, limit: int = 2) -> list[dict]:
        """[{artist, title, mbid}], the artist's most listened recordings first."""
        j = self._get(f"popularity/top-recordings-for-artist/{mbid}", {}, base=API) or []
        return [{"artist": x.get("artist_name") or "", "title": x.get("recording_name") or "",
                 "mbid": x.get("recording_mbid")}
                for x in (j if isinstance(j, list) else []) if isinstance(x, dict)
                and x.get("artist_name") and x.get("recording_name")][:limit]
