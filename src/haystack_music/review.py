"""What needle shows for a Review: track cards and the Review queues (the `/v1/state` payload).

A track card carries everything the Concept A screen needs for one track, so needle can prefetch every queue and
Review offline: metadata (with the record label Deezer gives) and cover, the current verdict, why the server picked
it, its top Discogs styles and its Readings. Read-only: nothing here writes."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import numpy as np

from . import baseline, calibrate, db, metadata, playlists, traits, verdicts

SOURCE_LABELS = {
    "deezer_chart": "the Deezer chart",
    "deezer_editorial": "a Deezer genre chart",
    "deezer_new_releases": "Deezer's new releases",
    "deezer_artist_radio": "Deezer artist radio",
    "deezer_related": "Deezer's related artists",
    "listenbrainz_similar": "ListenBrainz similar recordings",
}
STYLES_SHOWN = 3


def _first(artists_json) -> str:
    a = json.loads(artists_json or "[]")
    return a[0] if a else ""


def _year(release_date) -> int | None:
    try:
        return int(str(release_date)[:4])
    except (TypeError, ValueError):
        return None


def nearest_label(con, owner: str | None) -> dict | None:
    """The positive a candidate sounded most like (candidates.nearest) as {title, artist}."""
    if not owner or ":" not in owner:
        return None
    kind, ref = owner.split(":", 1)
    if kind == "local":
        r = con.execute("SELECT title_raw, artists_json, artist_raw FROM local_tracks WHERE id=?", (ref,)).fetchone()
        return {"title": r[0], "artist": _first(r[1]) or r[2]} if r else None
    if kind == "manual":
        r = con.execute("SELECT title, artist FROM candidates WHERE id=?", (ref,)).fetchone()
        return {"title": r[0], "artist": r[1]} if r else None
    r = con.execute("SELECT name, artists_json FROM spotify_tracks WHERE spotify_id=?", (ref,)).fetchone()
    return {"title": r[0], "artist": _first(r[1])} if r else None


def style_names(classes: list[str]) -> list[str]:
    """Discogs 'Parent---Style' class names -> the style alone ('Electronic---Dubstep' -> 'Dubstep')."""
    return [c.split("---", 1)[-1] for c in classes]


class Cards:
    """Builds track cards; loads the style class names once."""

    def __init__(self, con, cfg: dict, classes: list[str] | None = None):
        from .vectors import get_styles, track_vectors
        self.con, self.cfg = con, cfg
        self._get_styles, self._track_vectors = get_styles, track_vectors
        if classes is None:
            try:
                from . import embed
                classes = embed.style_classes()
            except Exception:  # noqa: BLE001 - no model metadata (tests): cards just have no styles
                classes = []
        self.names = style_names(classes)
        self.pending: set[str] = set()      # tracks the API is measuring right now

    def build(self, sids: list[str], offers: dict | None = None, picks: dict | None = None) -> list[dict]:
        """Cards for these tracks; offers (track id: offers row) and picks (track id: calibrations row) give the
        why of a Discoveries offer and of a Calibrate pick."""
        con = self.con
        offers, picks = offers or {}, picks or {}
        cur = verdicts.current(con, sids)
        tv = self._track_vectors(con, sids)
        dids = [t["deezer_id"] for t in tv.values()]
        rd = traits.get(con, dids)
        labels = metadata.labels_for_cards(con, dids)
        st = self._get_styles(con, dids) if self.names else {}
        meta = {}
        for i in range(0, len(sids), 500):
            ch = sids[i:i + 500]
            for r in con.execute(f"SELECT * FROM spotify_tracks WHERE spotify_id IN ({','.join('?' * len(ch))})", ch):
                meta[r["spotify_id"]] = r
        unmeasurable = {r[0] for r in con.execute(
            "SELECT spotify_id FROM liked_map WHERE vector_status IN ('no_preview','not_found') "
            "UNION SELECT spotify_id FROM spotify_deezer WHERE status IN ('no_preview','not_found')")}
        out = []
        for sid in sids:
            m = meta.get(sid)
            t = tv.get(sid)
            did = t["deezer_id"] if t else None
            readings = rd.get(did) if did is not None else None
            if readings:
                measure = "ok"
            elif sid in self.pending:
                measure = "pending"
            elif (did is not None and did in rd) or sid in unmeasurable:
                measure = "no_preview"
            else:
                measure = "unknown"
            v = cur.get(sid)
            card = {
                "id": sid,
                "title": m["name"] if m else None,
                "artists": json.loads(m["artists_json"] or "[]") if m else [],
                "album": m["album"] if m else None,
                "album_type": m["album_type"] if m else None,
                "year": _year(m["release_date"]) if m else None,
                "label": labels.get(did),
                "duration_ms": int(round(m["duration"] * 1000)) if m and m["duration"] else None,
                "cover": m["image_url"] if m else None,
                "cover_small": m["image_url_small"] if m else None,
                "verdict": {"kind": v.kind, "stars": v.stars, "given_at": v.given_at} if v else None,
                "why": self._why_pick(picks[sid]) if sid in picks else self._why(offers.get(sid)),
                "styles": self._styles(st.get(did)) if did is not None else [],
                "readings": readings or None,
                "measure": measure,
            }
            out.append(card)
        return out

    def _why(self, offer) -> dict | None:
        if offer is None:
            return None
        c = self.con.execute("SELECT nearest, raw_cos, source, rerank_score, rerank_p, rerank_reason FROM candidates "
                             "WHERE id=?", (offer["candidate_id"],)).fetchone()
        if c is None:
            return None
        # the re-ranker's view (ADR 0011): only for a top pick it ordered; exploration picks go by similarity
        ranked = c["rerank_score"] is not None and c["rerank_p"] is not None and not offer["exploration"]
        return {"sounds_like": nearest_label(self.con, c["nearest"]),
                "similarity": round(float(c["raw_cos"]), 2) if c["raw_cos"] is not None else None,
                "source": c["source"], "source_label": SOURCE_LABELS.get(c["source"], c["source"]),
                "exploration": bool(offer["exploration"]),
                "band": {"kind": "explore" if offer["exploration"] else "top",
                         "percentile": offer["score_percentile"]},
                "model": "rerank" if ranked else "similarity",
                "confidence": round(float(c["rerank_p"]), 2) if ranked else None,
                "reasons": json.loads(c["rerank_reason"] or "[]") if ranked else []}

    def _why_pick(self, pick) -> dict | None:
        """A Calibrate pick's why (ADR 0015): as for an offer, with no band (it is neither a top nor an exploration
        pick) and with `calibrate`: its band, percentile, the re-ranker's chance and the card's phrase. The
        re-ranker's view shows whenever a model was fitted at the pick, used for the batch or not."""
        c = self.con.execute("SELECT nearest, raw_cos, source FROM candidates WHERE id=?",
                             (pick["candidate_id"],)).fetchone()
        if c is None:
            return None
        p = pick["rerank_p"]
        return {"sounds_like": nearest_label(self.con, c["nearest"]),
                "similarity": round(float(c["raw_cos"]), 2) if c["raw_cos"] is not None else None,
                "source": c["source"], "source_label": SOURCE_LABELS.get(c["source"], c["source"]),
                "exploration": False,
                "model": "rerank" if p is not None else "similarity",
                "confidence": round(float(p), 2) if p is not None else None,
                "reasons": json.loads(pick["rerank_reason"] or "[]") if p is not None else [],
                "calibrate": {"band": pick["band"], "percentile": pick["percentile"],
                              "chance": round(float(p), 2) if p is not None else None, "reason": pick["reason"]}}

    def _styles(self, vec) -> list:
        if vec is None or not self.names:
            return []
        idx = np.argsort(-vec)[:STYLES_SHOWN]
        return [[self.names[i], round(float(vec[i]), 2)] for i in idx]


def discoveries(con) -> tuple[list[str], dict]:
    """The latest batch's offers plus any older offer still open, in batch order. -> (track ids, offer rows)."""
    rows = con.execute("""SELECT * FROM offers WHERE batch_id=(SELECT MAX(id) FROM batches) OR removed_at IS NULL
                          ORDER BY batch_id, id""").fetchall()
    sids, offers = [], {}
    for r in rows:
        if r["spotify_id"] not in offers:
            sids.append(r["spotify_id"]); offers[r["spotify_id"]] = r
    return sids, offers


def calibrate_picks(con, ts: str | None = None) -> tuple[list[str], dict]:
    """The Calibrate queue (ADR 0015): every open pick plus the picks judged this ISO week, in pick order. A pick
    hearted in Spotify with no Verdict waits in Rate your likes for its stars instead.
    -> (track ids, calibrations rows). Empty on a database from before 0.5.0."""
    wk = calibrate.week(ts)
    try:
        rows = con.execute("SELECT * FROM calibrations ORDER BY id").fetchall()
    except Exception:  # noqa: BLE001 - no calibrations table yet
        return [], {}
    hearted = {r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL AND spotify_id "
                                         "NOT IN (SELECT spotify_id FROM verdicts)")}
    sids, picks = [], {}
    for r in rows:
        if r["spotify_id"] in picks or r["spotify_id"] in hearted or (
                r["judged_at"] is not None and calibrate.week(r["judged_at"]) != wk):
            continue
        sids.append(r["spotify_id"]); picks[r["spotify_id"]] = r
    return sids, picks


def unrated_likes(con) -> list[str]:
    cur = verdicts.current(con)
    return [r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL ORDER BY added_at DESC, spotify_id")
            if r[0] not in cur]


def next_batch_at(now: datetime | None = None, schedule: str = "Mon 08:00") -> str | None:
    """When the batch runs next (UTC ISO): the batch timer's next time from systemd; without systemd (`haystack
    serve`), the next time of `schedule`, schedule.batch in the config. None when that is off."""
    try:
        r = subprocess.run(["systemctl", "--user", "show", "haystack-batch.timer", "-p",
                            "NextElapseUSecRealtime", "--value", "--timestamp=unix"],
                           capture_output=True, text=True, timeout=5)
        v = r.stdout.strip().lstrip("@")
        if r.returncode == 0 and v.isdigit():
            return datetime.fromtimestamp(int(v), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, subprocess.SubprocessError):
        pass
    from .scheduler import next_after, parse
    spec = parse(schedule)
    if not spec:
        return None
    local = (now or datetime.now()).astimezone().replace(tzinfo=None)
    return next_after(spec, local).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def state(con, cfg: dict, cards: Cards | None = None) -> dict:
    """The whole /v1/state payload: the Review queues (Discoveries, Calibrate, Rate your likes) with full cards, the
    Baseline, the listener's playlists (the songs' ids only: needle fetches a playlist's cards when a Playlist review
    starts) and Spotify write status."""
    cards = cards or Cards(con, cfg)
    d_sids, offers = discoveries(con)
    d_cards = cards.build(d_sids, offers)
    reviewed = [c for c in d_cards if c["verdict"]]
    d_cards = [c for c in d_cards if not c["verdict"]] + reviewed          # unreviewed first
    c_sids, picks = calibrate_picks(con)
    c_cards = cards.build(c_sids, picks=picks)
    c_done = [c for c in c_cards if c["verdict"]]
    c_cards = [c for c in c_cards if not c["verdict"]] + c_done
    l_sids = unrated_likes(con)
    pend = con.execute("SELECT COUNT(DISTINCT spotify_id) FROM verdicts WHERE effects='pending'").fetchone()[0]
    err = con.execute("SELECT effects_error FROM verdicts WHERE effects='pending' AND effects_error IS NOT NULL "
                      "ORDER BY received_at DESC LIMIT 1").fetchone()
    return {
        "generated_at": db.now(),
        "baseline": baseline.compute(con, cfg),
        "queues": {
            "discoveries": {"total": len(d_cards), "done": len(reviewed),
                            "next_batch_at": next_batch_at(schedule=cfg["schedule"]["batch"]),
                            "items": d_cards},
            "calibrate": {"total": len(c_cards), "done": len(c_done), "items": c_cards},
            "likes": {"total": len(l_sids), "items": cards.build(l_sids)},
        },
        "playlists": playlists.summaries(con),
        "spotify": {"pending_writes": pend, "error": err[0] if err else None},
    }
