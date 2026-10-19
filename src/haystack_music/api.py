"""needle's private HTTP API (ADR 0004, ADR 0006; the contract is docs/api.md).

Stdlib only (ThreadingHTTPServer): `haystack api` runs it as the user unit haystack-api. Every
request needs `Authorization: Bearer <token>` (the api-token secret, `haystack token new`) and must come from
loopback or a network in api.allow_networks (a host firewall should say the same). One worker thread makes the
Spotify writes that follow a verdict, measures tracks on demand and keeps the user's playlists current, so
requests never wait on Spotify or Essentia."""
from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import queue
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, db, playlists, review, verdicts

log = logging.getLogger("haystack_music.api")
VERSION = "1"
# Capabilities beyond /v1 as first shipped (docs/api.md, "Server and features"). A client shows a newer element
# only when its feature name is listed; a server without the `server` object has none of them.
FEATURES: tuple[str, ...] = ("why.band", "rerank", "why.reasons")
MAX_BODY = 1 << 20
TRACK_PATH = re.compile(r"^/v1/tracks/([0-9A-Za-z]{22})$")
PLAYLIST_PATH = re.compile(r"^/v1/playlists/([0-9A-Za-z]{22})$")
PLAYLIST_TRACKS_PATH = re.compile(r"^/v1/playlists/([0-9A-Za-z]{22})/tracks$")
QUEUES = ("discoveries", "likes", "playlists")
MEASURE_AHEAD = 20      # per queue; the harvest's backfill measures the rest
PLAYLISTS_EVERY = 600   # s: a snapshot read re-lists the playlists at most this often (one call if none changed)
MAX_PLAYLIST_BODY = 8 << 20      # a 10,000-song playlist read by needle's engine


def allowed(addr: str, networks) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback or any(ip in n for n in networks)


def parse_verdict(v: dict) -> dict:
    """One verdict from the outbox -> kwargs for verdicts.record. Raises ValueError with a reason."""
    if not isinstance(v, dict):
        raise ValueError("a verdict is an object")
    vid = v.get("id")
    if not isinstance(vid, str) or not 8 <= len(vid) <= 64:
        raise ValueError("id must be the client's verdict id (8-64 characters)")
    given = v.get("given_at")
    if not isinstance(given, str) or not re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$", given):
        raise ValueError("given_at must be UTC ISO 8601 (…Z)")
    q = v.get("queue")
    if q is not None and q not in QUEUES:
        raise ValueError(f"queue must be one of {', '.join(QUEUES)}")
    stars = verdicts.validate(v.get("track"), v.get("kind"), v.get("stars"))
    return {"spotify_id": v["track"], "kind": v["kind"], "stars": stars, "given_at": given, "id": vid, "queue": q}


def server_info() -> dict:
    """The `server` object of /v1/health and /v1/state: haystack's version, the API version and its features."""
    return {"version": __version__, "api": VERSION, "features": list(FEATURES)}


class Worker(threading.Thread):
    """Spotify writes and on-demand measuring, one at a time, off the request path. The Spotify writes that
    follow a verdict always go first: a long run of measuring must never hold back a heart or a playlist edit."""

    PRIORITY = {"effects": 0, "metadata": 1, "playlists": 1, "measure": 2}

    def __init__(self, cfg: dict, spotify_factory=None, retry_every: float = 600.0):
        super().__init__(daemon=True, name="api-worker")
        self.cfg = cfg
        self.q: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0
        self._lock = threading.Lock()
        self.measuring: set[str] = set()
        self._effects_queued = threading.Event()
        self._playlists_queued = threading.Event()
        self._playlists_at = 0.0
        self._spotify_factory = spotify_factory
        self._sp = None
        self.retry_every = retry_every

    def spotify(self):
        if self._sp is None:
            if self._spotify_factory:
                self._sp = self._spotify_factory()
            else:
                from .spotify import Spotify
                self._sp = Spotify(self.cfg)
        return self._sp

    def _put(self, kind: str, arg=None) -> None:
        with self._lock:
            self._seq += 1
            self.q.put((self.PRIORITY[kind], self._seq, kind, arg))

    def effects(self) -> None:
        if not self._effects_queued.is_set():
            self._effects_queued.set()
            self._put("effects")

    def playlists(self, every: float = 0.0) -> None:
        """Re-list the listener's playlists (and re-read the songs of any that changed), unless that ran less than
        `every` seconds ago or is already queued."""
        if self._playlists_queued.is_set() or time.monotonic() - self._playlists_at < every:
            return
        self._playlists_queued.set()
        self._put("playlists")

    def measure(self, sid: str) -> None:
        with self._lock:
            if sid in self.measuring:
                return
            self.measuring.add(sid)
        self._put("measure", sid)

    def run(self) -> None:
        con = db.connect()
        self._put("metadata")
        self.effects()
        self.playlists()
        while True:
            try:
                _pri, _seq, kind, arg = self.q.get(timeout=self.retry_every)
            except queue.Empty:
                if verdicts.pending(con):
                    self.effects()
                continue
            try:
                if kind == "effects":
                    time.sleep(2)                 # let a burst of verdicts land, then write once
                    self._effects_queued.clear()
                    if not self.cfg["api"].get("spotify_writes", True):
                        continue
                    out = verdicts.apply_effects(con, self.cfg, self.spotify())
                    if out["tracks"]:
                        log.info("verdict effects: %s", out)
                elif kind == "measure":
                    from . import traits
                    log.info("measuring %s on demand: %s", arg, traits.measure_track(con, self.cfg, arg, self.spotify()))
                elif kind == "metadata":
                    from .sync import fill_metadata
                    log.info("track metadata: %s", fill_metadata(con, self.spotify()))
                elif kind == "playlists":
                    self._playlists_at = time.monotonic()
                    self._playlists_queued.clear()
                    out = playlists.sync(con, self.spotify(), self.cfg)
                    if out["read"] or out["removed"] or out["errors"]:
                        log.info("playlists: %s", out)
            except Exception as e:  # noqa: BLE001 - the worker must outlive any one failure
                log.warning("worker %s %s failed: %r", kind, arg or "", e)
            finally:
                if kind == "measure":
                    self.measuring.discard(arg)


class Handler(BaseHTTPRequestHandler):
    server_version = "haystack-api/" + VERSION
    protocol_version = "HTTP/1.1"

    # ---- plumbing ----
    def log_message(self, fmt, *args):
        log.info("%s %s", self.client_address[0], fmt % args)

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, msg: str) -> None:
        self._send(code, {"error": msg})

    def _authorised(self) -> bool:
        h = self.headers.get("Authorization", "")
        tok = h[7:] if h.startswith("Bearer ") else ""
        return hmac.compare_digest(tok.encode(), self.server.token.encode())

    def _body(self, limit: int = MAX_BODY):
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError("body too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _con(self):
        local = self.server.local
        if not hasattr(local, "con"):
            local.con = db.connect(self.server.db_path)
        return local.con

    def _gate(self) -> bool:
        if not allowed(self.client_address[0], self.server.networks):
            self._error(403, "not from an allowed network")
            return False
        if not self._authorised():
            self._error(401, "missing or wrong bearer token")
            return False
        return True

    # ---- routes ----
    def do_GET(self):
        if not self._gate():
            return
        path = self.path.split("?", 1)[0]
        if path == "/v1/health":
            return self._send(200, {"ok": True, "version": VERSION, "time": db.now(), "server": server_info()})
        if path == "/v1/state":
            cards = review.Cards(self._con(), self.server.cfg, self.server.classes)
            cards.pending = set(self.server.worker.measuring) if self.server.worker else set()
            st = review.state(self._con(), self.server.cfg, cards)
            for qname in ("discoveries", "likes"):   # the next tracks with no Readings: measure them in the background
                for c in [c for c in st["queues"][qname]["items"] if c["measure"] == "unknown"][:MEASURE_AHEAD]:
                    if self.server.worker:
                        self.server.worker.measure(c["id"])
            if self.server.worker:
                self.server.worker.playlists(PLAYLISTS_EVERY)
            st["server"] = server_info()
            return self._send(200, st)
        m = PLAYLIST_PATH.match(path)
        if m:
            cards = review.Cards(self._con(), self.server.cfg, self.server.classes)
            cards.pending = set(self.server.worker.measuring) if self.server.worker else set()
            pl = playlists.cards(self._con(), cards, m.group(1))
            if pl is None:
                return self._error(404, "no such playlist in the listener's library")
            return self._send(200, pl)
        m = TRACK_PATH.match(path)
        if m:
            cards = review.Cards(self._con(), self.server.cfg, self.server.classes)
            offer = review.discoveries(self._con())[1].get(m.group(1))
            card = cards.build([m.group(1)], {m.group(1): offer} if offer else None)[0]
            if card["measure"] == "unknown" and self.server.worker:
                self.server.worker.measure(m.group(1))
                card["measure"] = "pending"
                return self._send(202, card)
            return self._send(200, card)
        self._error(404, "no such endpoint")

    def do_POST(self):
        if not self._gate():
            return
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(400, f"bad JSON: {e}")
        path = self.path.split("?", 1)[0]
        if path == "/v1/verdicts":
            items = body.get("verdicts") if isinstance(body, dict) else None
            if not isinstance(items, list) or len(items) > 500:
                return self._error(400, "expected {\"verdicts\": [...]} (at most 500)")
            accepted, rejected = [], []
            con = self._con()
            for v in items:
                try:
                    kw = parse_verdict(v)
                except ValueError as e:
                    rejected.append({"id": v.get("id") if isinstance(v, dict) else None, "error": str(e)})
                    continue
                verdicts.record(con, kw.pop("spotify_id"), kw.pop("kind"), kw.pop("stars"), source="needle", **kw)
                accepted.append(kw["id"])
            if accepted and self.server.worker:
                self.server.worker.effects()
            return self._send(200, {"accepted": accepted, "rejected": rejected})
        if path == "/v1/plays":
            items = body.get("plays") if isinstance(body, dict) else None
            if not isinstance(items, list) or len(items) > 2000:
                return self._error(400, "expected {\"plays\": [...]}")
            accepted, rejected = [], []
            con = self._con()
            for p in items:
                if not (isinstance(p, dict) and isinstance(p.get("track"), str) and verdicts.SPOTIFY_ID.match(p["track"])
                        and isinstance(p.get("started_at"), str) and isinstance(p.get("id"), str)):
                    rejected.append({"id": p.get("id") if isinstance(p, dict) else None, "error": "malformed play"})
                    continue
                # plays is keyed by (track, time): a replayed play is a no-op
                con.execute("INSERT OR IGNORE INTO plays(spotify_id, played_at, context) VALUES(?,?,?)",
                            (p["track"], p["started_at"], "needle"))
                accepted.append(p["id"])
            return self._send(200, {"accepted": accepted, "rejected": rejected})
        self._error(404, "no such endpoint")

    def do_PUT(self):
        if not self._gate():
            return
        path = self.path.split("?", 1)[0]
        m = PLAYLIST_TRACKS_PATH.match(path)
        try:                          # read first, as for POST: the connection stays usable whatever the answer
            body = self._body(MAX_PLAYLIST_BODY if m else MAX_BODY)
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(400, f"bad JSON: {e}")
        if not m:
            return self._error(404, "no such endpoint")
        try:                          # the songs of a playlist the server can't list, as needle's engine read them
            return self._send(200, playlists.from_engine(self._con(), m.group(1), body))
        except LookupError:
            return self._error(404, "no such playlist in the listener's library")
        except ValueError as e:
            return self._error(400, str(e))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, addr, cfg: dict, token: str, worker: Worker | None = None, db_path=None,
                 classes: list[str] | None = None):
        if len(token) < 32:
            raise ValueError("the API token must be at least 32 characters")
        super().__init__(addr, Handler)
        self.cfg, self.token, self.worker, self.db_path = cfg, token, worker, db_path
        self.networks = [ipaddress.ip_network(n) for n in cfg["api"]["allow_networks"]]
        self.local = threading.local()
        self.classes = classes


def load_token(cfg: dict) -> str:
    """The bearer token: the api-token secret."""
    from . import secrets
    tok = secrets.store(cfg).find("api-token")
    if not tok:
        raise secrets.SecretError("no API token: run `haystack token new`")
    return tok.strip()


def serve(cfg: dict) -> None:
    a = cfg["api"]
    worker = Worker(cfg)
    try:
        from . import embed
        classes = embed.style_classes()
    except Exception as e:  # noqa: BLE001
        log.warning("no style class names (%r): cards will have no styles", e)
        classes = []
    srv = Server((a["host"], a["port"]), cfg, load_token(cfg), worker, classes=classes)
    worker.start()
    log.info("needle API on %s:%s (allowed: %s)", a["host"], a["port"], ", ".join(a["allow_networks"]))
    srv.serve_forever()
