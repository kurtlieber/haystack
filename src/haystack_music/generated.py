"""The generated-song store (ADR 0029): the songs a client such as spindle has a music service make, and what the
listener thinks of them.

An Idea is what the client asked the service for: the idea in the client's own form, the reason it chose it and the
exact request it sent. A Take is one piece of audio the service made, named by the service and the service's id. A
Verdict on a take is 1 to 5 stars or a Dislike, with an optional note. A take is Kept once the listener has filed it
in their own collection.

Optional: `[generated] enabled` (default false). Off, haystack creates none of these tables and the API has no
/v1/generated/ routes (api.py). Apart on purpose: a take is not a Song and its verdict is not a Verdict on a Song.
Nothing that harvests, scores, re-ranks or builds the Baseline or the Review reads these tables, and no module but
this one names them (tests/test_generated.py), so a verdict on a take never changes what haystack recommends.

Verdicts on takes follow ADR 0005: an append-only log with a client-made id, so a resend is a no-op, and a take's
current verdict is its latest by given_at. A verdict or a keep may name a take haystack has not seen (one made
before the client recorded its ideas): it becomes a take with no idea, which an idea recorded later can claim."""
from __future__ import annotations

import json
import re

from . import db

SCHEMA = """
CREATE TABLE IF NOT EXISTS generated_ideas (
    id TEXT PRIMARY KEY,                 -- client-made
    service TEXT NOT NULL,               -- the music service asked, e.g. its name in lower case
    created_at TEXT NOT NULL,            -- the client's clock
    received_at TEXT NOT NULL,
    idea_json TEXT NOT NULL,             -- the idea in the client's own, service-neutral form
    reason TEXT,                         -- why the client chose it
    request_json TEXT                    -- exactly what was sent to the service; NULL: not known
);
CREATE TABLE IF NOT EXISTS generated_takes (
    service TEXT NOT NULL,
    service_id TEXT NOT NULL,            -- the service's id for the take
    idea_id TEXT,                        -- NULL: no idea recorded for it (yet)
    title TEXT,
    model TEXT,                          -- the service's model that made it
    received_at TEXT NOT NULL,
    kept_at TEXT,                        -- the listener filed it; the first mark stays
    PRIMARY KEY (service, service_id)
);
CREATE INDEX IF NOT EXISTS generated_takes_idea ON generated_takes(idea_id);
CREATE TABLE IF NOT EXISTS generated_verdicts (
    id TEXT PRIMARY KEY,                 -- client-made
    service TEXT NOT NULL,
    service_id TEXT NOT NULL,
    kind TEXT NOT NULL,                  -- rating | dislike
    stars INTEGER,                       -- 1..5 for a rating
    note TEXT,
    given_at TEXT NOT NULL,              -- the client's clock: orders the history
    received_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS generated_verdicts_take ON generated_verdicts(service, service_id, given_at);
"""

KINDS = ("rating", "dislike")
SERVICE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
TIME = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$")
MAX_JSON = 64 << 10           # bytes, an idea or a request as stored
MAX_TAKES = 20                # per idea
MAX_NOTE, MAX_REASON, MAX_TITLE, MAX_MODEL, MAX_TAKE_ID = 4000, 4000, 500, 200, 128
NOTES_LIMIT, NOTES_MOST = 50, 500


def enabled(cfg: dict | None) -> bool:
    """Whether the store is switched on (`[generated] enabled`)."""
    return bool(((cfg or {}).get("generated") or {}).get("enabled"))


def ensure(con) -> None:
    """The store's tables. Only the API runs this, and only when the store is on."""
    con.executescript(SCHEMA)


# ---- what a client sends ----

def _client_id(item: dict, what: str) -> str:
    v = item.get("id")
    if not isinstance(v, str) or not 8 <= len(v) <= 64:
        raise ValueError(f"id must be the client's {what} id (8-64 characters)")
    return v


def _time(item: dict, key: str) -> str:
    v = item.get(key)
    if not isinstance(v, str) or not TIME.match(v):
        raise ValueError(f"{key} must be UTC ISO 8601 (…Z)")
    return v


def _service(item: dict) -> str:
    v = item.get("service")
    if not isinstance(v, str) or not SERVICE.match(v):
        raise ValueError("service must be the music service's short name: a-z, 0-9, _ and - (at most 32)")
    return v


def _take_id(v) -> str:
    if not isinstance(v, str) or not 1 <= len(v) <= MAX_TAKE_ID or any(ord(c) < 32 or ord(c) == 127 for c in v):
        raise ValueError(f"take must be the service's id for it (1-{MAX_TAKE_ID} printable characters)")
    return v


def _text(item: dict, key: str, most: int) -> str | None:
    """An optional text: absent, null or empty is None."""
    v = item.get(key)
    if v is None or v == "":
        return None
    if not isinstance(v, str) or len(v) > most:
        raise ValueError(f"{key} must be text of at most {most} characters")
    return v


def _json(item: dict, key: str, required: bool) -> str | None:
    v = item.get(key)
    if v is None and not required:
        return None
    if not isinstance(v, dict):
        raise ValueError(f"{key} must be an object" + ("" if required else " or null"))
    s = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    if len(s.encode()) > MAX_JSON:
        raise ValueError(f"{key} is over {MAX_JSON // 1024} KiB")
    return s


def parse_idea(item) -> dict:
    """One idea with its takes, as a client sends it. Raises ValueError with a reason."""
    if not isinstance(item, dict):
        raise ValueError("an idea is an object")
    takes = item.get("takes", [])
    if not isinstance(takes, list) or len(takes) > MAX_TAKES:
        raise ValueError(f"takes must be a list of at most {MAX_TAKES}")
    parsed = []
    for t in takes:
        if not isinstance(t, dict):
            raise ValueError("a take is an object")
        parsed.append((_take_id(t.get("id")), _text(t, "title", MAX_TITLE), _text(t, "model", MAX_MODEL)))
    return {"id": _client_id(item, "idea"), "created_at": _time(item, "created_at"), "service": _service(item),
            "idea": _json(item, "idea", True), "reason": _text(item, "reason", MAX_REASON),
            "request": _json(item, "request", False), "takes": parsed}


def parse_verdict(item) -> dict:
    """One verdict on a take. Raises ValueError with a reason."""
    if not isinstance(item, dict):
        raise ValueError("a verdict is an object")
    vid, service, take = _client_id(item, "verdict"), _service(item), _take_id(item.get("take"))
    kind, stars = item.get("kind"), item.get("stars")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if kind == "rating" and (isinstance(stars, bool) or not isinstance(stars, int) or not 1 <= stars <= 5):
        raise ValueError("a rating needs stars from 1 to 5")
    if kind == "dislike" and stars is not None:
        raise ValueError("a dislike has no stars")
    return {"id": vid, "service": service, "take": take, "kind": kind, "stars": stars,
            "note": _text(item, "note", MAX_NOTE), "given_at": _time(item, "given_at")}


def parse_kept(item) -> dict:
    """One take marked kept. Raises ValueError with a reason."""
    if not isinstance(item, dict):
        raise ValueError("a keep is an object")
    return {"id": _client_id(item, "keep"), "service": _service(item), "take": _take_id(item.get("take")),
            "kept_at": _time(item, "kept_at")}


# ---- storing it ----

def _take(con, service: str, take: str, now: str) -> None:
    con.execute("INSERT OR IGNORE INTO generated_takes(service, service_id, received_at) VALUES(?,?,?)",
                (service, take, now))


def _record_idea(con, p: dict, now: str) -> None:
    """The idea as first sent stays. Each take not yet known is added; a take that is already there gets the idea,
    title and model it lacks, never new ones."""
    con.execute("INSERT OR IGNORE INTO generated_ideas(id, service, created_at, received_at, idea_json, reason, "
                "request_json) VALUES(?,?,?,?,?,?,?)",
                (p["id"], p["service"], p["created_at"], now, p["idea"], p["reason"], p["request"]))
    for take, title, model in p["takes"]:
        _take(con, p["service"], take, now)
        con.execute("""UPDATE generated_takes SET idea_id=COALESCE(idea_id, :idea), title=COALESCE(title, :title),
                       model=COALESCE(model, :model)
                       WHERE service=:service AND service_id=:take
                         AND ((idea_id IS NULL AND :idea IS NOT NULL) OR (title IS NULL AND :title IS NOT NULL)
                              OR (model IS NULL AND :model IS NOT NULL))""",
                    {"idea": p["id"], "title": title, "model": model, "service": p["service"], "take": take})


def _record_verdict(con, p: dict, now: str) -> None:
    _take(con, p["service"], p["take"], now)
    con.execute("INSERT OR IGNORE INTO generated_verdicts(id, service, service_id, kind, stars, note, given_at, "
                "received_at) VALUES(?,?,?,?,?,?,?,?)",
                (p["id"], p["service"], p["take"], p["kind"], p["stars"], p["note"], p["given_at"], now))


def _record_kept(con, p: dict, now: str) -> None:
    _take(con, p["service"], p["take"], now)
    con.execute("UPDATE generated_takes SET kept_at=? WHERE service=? AND service_id=? AND kept_at IS NULL",
                (p["kept_at"], p["service"], p["take"]))


OUTBOX = {"ideas": (parse_idea, _record_idea), "verdicts": (parse_verdict, _record_verdict),
          "kept": (parse_kept, _record_kept)}


def receive(con, kind: str, items: list) -> dict:
    """One outbox POST of ideas, verdicts or kept marks -> {"accepted": [ids], "rejected": [{"id", "error"}]}.
    Each item stands alone: a malformed one is rejected with its reason, the rest are stored, in one transaction.
    A resent item (the same id) is accepted and changes nothing."""
    parse, record = OUTBOX[kind]
    accepted, rejected, now = [], [], db.now()
    con.execute("BEGIN IMMEDIATE")
    try:
        for item in items:
            try:
                p = parse(item)
            except ValueError as e:
                vid = item.get("id") if isinstance(item, dict) else None
                rejected.append({"id": vid if isinstance(vid, str) else None, "error": str(e)})
                continue
            record(con, p, now)
            accepted.append(p["id"])
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return {"accepted": accepted, "rejected": rejected}


# ---- reading it back ----

def _verdicts(con) -> list:
    """Every verdict on a take, oldest first: the order ADR 0005 resolves a take's current one by."""
    return con.execute("SELECT id, service, service_id, kind, stars, note, given_at FROM generated_verdicts "
                       "ORDER BY given_at, received_at, rowid").fetchall()


def _current(rows) -> dict:
    """(service, take) -> its current verdict row."""
    return {(r["service"], r["service_id"]): r for r in rows}


def takes(con) -> dict:
    """`GET /v1/generated/takes`: every take, newest idea first (a take with no idea by when haystack first heard of
    it), with its current verdict, and every idea."""
    current = _current(_verdicts(con))
    out = []
    for r in con.execute("""SELECT t.service, t.service_id, t.idea_id, t.title, t.model, t.kept_at
                            FROM generated_takes t LEFT JOIN generated_ideas i ON i.id = t.idea_id
                            ORDER BY COALESCE(i.created_at, t.received_at) DESC, t.received_at DESC, t.rowid"""):
        v = current.get((r["service"], r["service_id"]))
        out.append({"service": r["service"], "id": r["service_id"], "idea": r["idea_id"], "title": r["title"],
                    "model": r["model"], "kept_at": r["kept_at"],
                    "verdict": None if v is None else {"kind": v["kind"], "stars": v["stars"], "note": v["note"],
                                                       "given_at": v["given_at"]}})
    ideas = {r["id"]: {"created_at": r["created_at"], "service": r["service"], "idea": json.loads(r["idea_json"]),
                       "reason": r["reason"],
                       "request": json.loads(r["request_json"]) if r["request_json"] is not None else None}
             for r in con.execute("SELECT * FROM generated_ideas ORDER BY created_at DESC, rowid")}
    return {"takes": out, "ideas": ideas}


def notes(con, limit: int = NOTES_LIMIT) -> dict:
    """`GET /v1/generated/notes`: the newest verdicts that carry a note, current or not (`current` says which)."""
    rows = _verdicts(con)
    current = {r["id"] for r in _current(rows).values()}
    takes_ = {(r["service"], r["service_id"]): r for r in con.execute(
        "SELECT service, service_id, idea_id, title FROM generated_takes")}
    out = []
    for r in reversed(rows):
        if r["note"] is None:
            continue
        t = takes_.get((r["service"], r["service_id"]))
        out.append({"service": r["service"], "take": r["service_id"], "idea": t["idea_id"] if t else None,
                    "title": t["title"] if t else None, "kind": r["kind"], "stars": r["stars"], "note": r["note"],
                    "given_at": r["given_at"], "current": r["id"] in current})
        if len(out) == limit:
            break
    return {"notes": out}
