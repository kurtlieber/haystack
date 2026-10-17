"""Pure track-matching logic: Spotify ID tag -> ISRC -> fuzzy (artist + title + duration + version check).

Used for liked-song -> library mapping, To Rip detection, candidate exclusion and feedback matching."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .textnorm import key, split_artists, split_title, version_check


@dataclass
class TrackRef:
    ref: object                    # caller's id (local track id, spotify id, candidate id ...)
    artist: str
    title: str
    duration: float | None = None  # seconds
    spotify_id: str | None = None
    isrc: str | None = None
    artists: list[str] | None = None  # extra artist names (e.g. multi-valued ARTISTS tag / Spotify artists)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def artist_keys(self) -> list[str]:
        if "ak" not in self._cache:
            self._cache["ak"] = [k for k in (key(a) for a in split_artists(self.artist, self.artists)) if k]
        return self._cache["ak"]

    @property
    def base_suffix(self) -> tuple[str, str]:
        if "bs" not in self._cache:
            self._cache["bs"] = split_title(self.title)
        return self._cache["bs"]

    @property
    def title_key(self) -> str:
        return key(self.base_suffix[0])


@dataclass
class Match:
    ref: object
    method: str            # spotify_id | isrc | fuzzy
    confidence: str        # high | medium | low
    dt: float | None
    title_sim: float
    artist_sim: float
    version: str           # same | variant
    reason: str = ""

    @property
    def certain(self) -> bool:
        return self.confidence == "high"


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def compare(q: TrackRef, c: TrackRef, tol: float = 3.0, confident_dt: float = 2.0) -> Match | None:
    """Fuzzy comparison. Returns None when rejected (duration, text, or a different version)."""
    dt = None
    if q.duration and c.duration:
        dt = abs(float(q.duration) - float(c.duration))
        if dt > tol:
            return None
    qb, qs = q.base_suffix
    cb, cs = c.base_suffix
    ts = max(_sim(key(qb), key(cb)), _sim(key(q.title), key(c.title)))
    as_ = max((_sim(a, b) for a in q.artist_keys for b in c.artist_keys), default=0.0)
    if ts < 0.6 or as_ < 0.6:
        return None
    v = version_check(qs, cs)
    if v == "mismatch":
        return None
    if dt is not None and dt <= confident_dt and ts >= 0.9 and as_ >= 0.9 and v == "same":
        conf = "high"
    elif ts >= 0.75 and as_ >= 0.75:
        conf = "medium"
    else:
        conf = "low"
    reason = []
    if dt is None:
        reason.append("no duration")
    elif dt > confident_dt:
        reason.append(f"dt={dt:.1f}s")
    if v == "variant":
        reason.append(f"version differs ({qs or 'none'} vs {cs or 'none'})")
    if ts < 0.9:
        reason.append(f"title~{ts:.2f}")
    if as_ < 0.9:
        reason.append(f"artist~{as_:.2f}")
    return Match(c.ref, "fuzzy", conf, dt, round(ts, 3), round(as_, 3), v, "; ".join(reason))


_ORDER = {"high": 2, "medium": 1, "low": 0}


class Index:
    """In-memory index of target tracks (e.g. the local library) for match_one()."""

    def __init__(self, targets: list[TrackRef]):
        self.targets = targets
        self.by_sid: dict[str, TrackRef] = {}
        self.by_isrc: dict[str, TrackRef] = {}
        self.by_artist: dict[str, list[TrackRef]] = defaultdict(list)
        self.by_title: dict[str, list[TrackRef]] = defaultdict(list)
        for t in targets:
            if t.spotify_id:
                self.by_sid.setdefault(t.spotify_id, t)
            if t.isrc:
                self.by_isrc.setdefault(t.isrc.upper(), t)
            for a in t.artist_keys:
                self.by_artist[a].append(t)
            if t.title_key:
                self.by_title[t.title_key].append(t)

    def candidates(self, q: TrackRef) -> list[TrackRef]:
        seen, out = set(), []
        for a in q.artist_keys:
            for t in self.by_artist.get(a, ()):
                if id(t) not in seen:
                    seen.add(id(t)); out.append(t)
        for t in self.by_title.get(q.title_key, ()):
            if id(t) not in seen:
                seen.add(id(t)); out.append(t)
        return out

    def match(self, q: TrackRef, tol: float = 3.0, confident_dt: float = 2.0) -> Match | None:
        if q.spotify_id and q.spotify_id in self.by_sid:
            t = self.by_sid[q.spotify_id]
            return Match(t.ref, "spotify_id", "high", _dt(q, t), 1.0, 1.0, "same")
        if q.isrc and q.isrc.upper() in self.by_isrc:
            t = self.by_isrc[q.isrc.upper()]
            v = version_check(q.base_suffix[1], t.base_suffix[1])
            if v != "mismatch":
                return Match(t.ref, "isrc", "high" if v == "same" else "medium", _dt(q, t), 1.0, 1.0, v)
        best = None
        for t in self.candidates(q):
            m = compare(q, t, tol, confident_dt)
            if m is None:
                continue
            rank = (_ORDER[m.confidence], m.title_sim + m.artist_sim, -(m.dt if m.dt is not None else tol))
            if best is None or rank > best[0]:
                best = (rank, m)
        return best[1] if best else None


def _dt(a: TrackRef, b: TrackRef) -> float | None:
    if a.duration and b.duration:
        return round(abs(float(a.duration) - float(b.duration)), 2)
    return None


def use_local_vector(m: Match | None, confident_dt: float = 2.0) -> bool:
    """Taste-vector rule: use the local file's vector only for a confident, same-version match within 2 s."""
    if m is None or not m.certain or m.version != "same":
        return False
    if m.dt is None:
        return m.method != "fuzzy"
    return m.dt <= confident_dt
