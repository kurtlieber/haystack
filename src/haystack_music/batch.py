"""Weekly discovery batch (needs Spotify auth): clear Discoveries, then score the pool, pick ~15
(about 20 % exploration, moved by the outcomes; 1 per artist; spread across styles and liked songs, ADR 0010),
map to Spotify (ISRC search, then artist/title + duration + version check), drop anything already liked, in the
library or offered before, and add to the Discoveries playlist."""
from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from dataclasses import replace

import numpy as np

from . import db, library, rerank, vectors
from .deezer import Deezer
from .matching import Index, TrackRef, compare
from .scoring import Pick, adapt_exploration, l2norm, select_batch
from .spotify import Spotify
from .clear import clear_discoveries
from .sync import bot_add, ensure_playlists, upsert_track
from .textnorm import key, split_title, version_check

log = logging.getLogger("haystack_music.batch")


def map_to_spotify(con, sp: Spotify, dz: Deezer, cand, tol: float = 3.0) -> str | None:
    """ISRC first (fetched from Deezer's track object), then artist/title search with duration + version check."""
    isrc = cand["isrc"]
    if not isrc and cand["deezer_id"]:
        t = dz.track(cand["deezer_id"]) or {}
        isrc = (t.get("isrc") or "").upper() or None
        if isrc:
            con.execute("UPDATE candidates SET isrc=? WHERE id=?", (isrc, cand["id"]))
    q = TrackRef(cand["id"], cand["artist"], cand["title"], cand["duration"])
    if isrc:
        for t in sp.search_tracks(f"isrc:{isrc}", 10):
            dt = abs((t.get("duration_ms") or 0) / 1000 - (cand["duration"] or 0))
            if dt <= tol and version_check(split_title(cand["title"])[1], split_title(t.get("name", ""))[1]) != "mismatch":
                upsert_track(con, t)
                return t["id"]
    base, _ = split_title(cand["title"])
    best = None
    for t in sp.search_tracks(f'track:"{base}" artist:"{cand["artist"]}"', 10):
        r = TrackRef(t["id"], (t.get("artists") or [{}])[0].get("name", ""), t.get("name", ""),
                     (t.get("duration_ms") or 0) / 1000, artists=[a.get("name", "") for a in t.get("artists", [])])
        m = compare(q, r, tol)
        if m and m.confidence in ("high", "medium") and (best is None or m.title_sim + m.artist_sim > best[0]):
            best = (m.title_sim + m.artist_sim, t)
    if best:
        upsert_track(con, best[1])
        return best[1]["id"]
    return None


def embedding_progress(con, cfg: dict) -> tuple[bool, str, dict]:
    """Is the full-library embedding complete enough for a real batch? (positives are thin otherwise)
    Without a library (library.root unset) there is nothing to wait for."""
    if not cfg["library"]["root"]:
        return True, "no local library: the positives are Liked Songs and Ratings", {
            "embedded": 0, "total": 0, "unembeddable": 0, "fraction": None}
    total = con.execute("SELECT COUNT(DISTINCT content_hash) FROM local_tracks WHERE present=1").fetchone()[0]
    ok = con.execute("""SELECT COUNT(DISTINCT t.content_hash) FROM local_tracks t JOIN local_embeddings e
                        ON e.content_hash=t.content_hash AND e.status='ok' WHERE t.present=1""").fetchone()[0]
    gave_up = con.execute("""SELECT COUNT(DISTINCT t.content_hash) FROM local_tracks t JOIN local_embeddings e
                             ON e.content_hash=t.content_hash AND e.status='error' AND e.attempts>=?
                             WHERE t.present=1""", (cfg["library"]["max_attempts"],)).fetchone()[0]
    need = cfg["batch"]["min_embedded_fraction"]
    frac = ok / total if total else 0.0
    done = total > 0 and (ok + gave_up) >= total * need
    msg = (f"library embedding {'complete' if done else 'incomplete'}: {ok}/{total} embedded ({frac:.1%}), "
           f"{gave_up} unembeddable; batch needs >= {need:.0%}")
    return done, msg, {"embedded": ok, "total": total, "unembeddable": gave_up, "fraction": round(frac, 4)}


def run(con, cfg: dict, sp: Spotify | None = None, size: int | None = None, dry_run: bool = False,
        rng: random.Random | None = None, force: bool = False, variant: str | None = None,
        clear_min_age_hours: float | None = None, clear_recent: str | None = None,
        no_penalty: bool | None = None) -> dict:
    b = cfg["batch"]
    size = size or b["size"]
    done, msg, prog = embedding_progress(con, cfg)
    if not done and not dry_run and not force:
        log.warning("batch skipped: %s", msg)
        return {"skipped": True, "reason": msg, **prog}
    if not done:
        log.warning("%s (continuing: %s)", msg, "dry run" if dry_run else "--force")
    sp = sp or Spotify(cfg)
    pl = ensure_playlists(con, sp, cfg, create=not dry_run)
    cleared = clear_discoveries(con, cfg, sp, playlist_id=pl.get("discoveries"), resolve_playlist=False,
                                min_age_hours=clear_min_age_hours, recent=clear_recent, no_penalty=no_penalty,
                                dry_run=dry_run)
    choice = _rechoose(con, cfg, write=not dry_run) if variant is None else None
    variant = variant or (choice or {}).get("used") or vectors.active_variant(con, cfg)
    res = vectors.compute_scores(con, cfg, variant)
    scored = vectors.score_candidates(con, cfg, variant=variant, result=res)
    rr = _rerank(con, cfg, variant, res, write=not dry_run)
    dz = Deezer(cfg["harvest"]["deezer_min_interval"])
    tol = cfg["matching"]["duration_tolerance"]

    def resolve(c):
        sid = c["spotify_id"] or map_to_spotify(con, sp, dz, c, tol)
        if not sid:
            con.execute("UPDATE candidates SET spotify_status='not_found' WHERE id=?", (c["id"],))
            return None
        con.execute("UPDATE candidates SET spotify_id=?, spotify_status='ok' WHERE id=?", (sid, c["id"]))
        return sid

    def on_known(c):
        con.execute("UPDATE candidates SET state='excluded', exclude_reason='known on spotify' WHERE id=?", (c["id"],))

    explore = exploration_share(con, cfg)
    chosen = select_for_batch(con, cfg, rng, size=size, resolve=resolve, on_known=on_known,
                              explore_fraction=explore["fraction"], ranks=rr.get("ranks"))
    # final guard: Spotify's own saved-state check
    if chosen:
        saved = sp.library_contains([p.spotify_id for p in chosen])
        for p in [p for p in chosen if saved.get(p.spotify_id)]:
            con.execute("UPDATE candidates SET state='excluded', exclude_reason='liked' WHERE id=?", (p.cand_id,))
            chosen.remove(p)
    out = {"scored": scored, "picked": len(chosen), "exploration": sum(p.exploration for p in chosen),
           "by_source": {}, "dry_run": dry_run, "embedding": prog,
           "variant": variant, "variant_choice": choice, "clear": cleared,
           "exploration_fraction_used": explore["fraction"], "exploration_share": explore, "spread": spread(chosen),
           "rerank": rerank.summary(rr)}
    for p in chosen:
        out["by_source"][p.source] = out["by_source"].get(p.source, 0) + 1
    if dry_run or not chosen:
        out["tracks"] = _describe(con, chosen, top=rr.get("top"))
        return out
    sids = [p.spotify_id for p in chosen]
    snap = bot_add(con, sp, "discoveries", pl["discoveries"], sids)
    ts = db.now()
    bid = con.execute("INSERT INTO batches(created_at, size, n_explore) VALUES(?,?,?)",
                      (ts, len(chosen), out["exploration"])).lastrowid
    for p in chosen:
        con.execute("""INSERT INTO offers(candidate_id, spotify_id, batch_id, source, score, exploration, added_at,
                       score_percentile) VALUES(?,?,?,?,?,?,?,?)""",
                    (p.cand_id, p.spotify_id, bid, p.source, p.score, int(p.exploration), ts, p.percentile))
        con.execute("UPDATE candidates SET state='offered' WHERE id=?", (p.cand_id,))
    out["batch_id"] = bid; out["snapshot_id"] = snap
    out["tracks"] = _describe(con, chosen, top=rr.get("top"))
    return out


def _rerank(con, cfg: dict, variant: str, res: dict | None, write: bool) -> dict:
    """The second stage (ADR 0011) over the freshly scored pool: token-free, and never blocks the batch (on an
    error the batch goes by score alone). With write the fit is recorded and the pool's re-ranker columns set."""
    try:
        r = rerank.run(con, cfg, variant, res or {}, "batch")
        if write:
            rerank.record(con, r)
            rerank.write_pool(con, r)
        if r["used"]:
            log.info("re-ranked the top of the pool: %s", r["reason"])
        else:
            log.info("stage 1 only: %s", r["reason"])
        return r
    except Exception as e:  # noqa: BLE001
        log.exception("re-ranking failed; the batch goes by score alone")
        return {"used": False, "reason": f"error: {e!r}"}


def _rechoose(con, cfg: dict, write: bool) -> dict | None:
    """The weekly re-choice of the scoring variant (ADR 0009) when scoring.variant is auto, and the check that a
    preview alignment is not worse (ADR 0012) when there is one: token-free, takes seconds, and never blocks the
    batch. Without write it only says which variant the batch would use."""
    from . import align, evaluate
    if cfg["scoring"].get("variant", "auto") != "auto" and align.candidate(con, cfg) is None:
        return None
    try:
        return evaluate.summary(evaluate.choose(con, cfg, "weekly", write=write))
    except Exception as e:  # noqa: BLE001
        log.exception("scoring eval failed; keeping the stored variant")
        return {"error": repr(e)}


def exploration_share(con, cfg: dict) -> dict:
    """This batch's exploration share (ADR 0010): the prior batch.exploration_fraction, moved by how the exploration
    picks did against the top picks over the last batch.explore_window_batches batches (kept over judged offers,
    as `haystack report` counts them). An exploration_fraction of 0 or a window of 0 keeps the prior."""
    from .report import offer_rows
    b = cfg["batch"]
    prior, window = b["exploration_fraction"], b["explore_window_batches"]
    if not prior or not window:
        return {"prior": prior, "fraction": prior, "why": "prior: adaptation off"}
    rows = offer_rows(con)
    recent = set(sorted({r["batch"] for r in rows if r["batch"] is not None})[-window:])
    judged = [r for r in rows if r["batch"] in recent and r["judged"]]

    def side(explore):
        rs = [r for r in judged if r["exploration"] == explore]
        return sum(r["kept"] for r in rs), len(rs)
    return adapt_exploration(prior, side(True), side(False), b["explore_min_judged"],
                             tuple(b["exploration_bounds"])) | {"batches": len(recent)}


def spread(chosen: list[Pick]) -> dict:
    """How spread a batch is: the largest number of top picks sharing a top-1 style, and of picks sharing a nearest
    positive."""
    styles = Counter(p.style for p in chosen if not p.exploration and p.style is not None)
    near = Counter(p.nearest for p in chosen if p.nearest is not None)
    return {"top_picks": sum(not p.exploration for p in chosen),
            "max_same_style": max(styles.values(), default=0), "max_same_nearest": max(near.values(), default=0)}


def select_for_batch(con, cfg: dict, rng: random.Random | None = None, *, size: int | None = None,
                     scores: dict | None = None, resolve=None, on_known=None,
                     explore_fraction: float | None = None, ranks: dict | None = None) -> list[Pick]:
    """The batch's selection over the scored pool, shared by `batch` and `batch --plan`. It only reads; the live
    batch's side effects come in as callbacks.
    scores: {candidate id: (score, nearest owner)} in place of the stored scores (`--plan` scores in memory).
    resolve(row) -> Spotify id, or None to drop the candidate (the live batch maps it and records the result).
    Without resolve a candidate keeps its stored spotify_id, and None means not mapped yet.
    on_known(row): the candidate's track is already liked, offered or in the library, so it is skipped.
    explore_fraction: this batch's exploration share (exploration_share), default batch.exploration_fraction.
    ranks: {candidate id: rank} from the re-ranker (rerank.run); the top picks follow it instead of the score.
    Each pick carries its top-1 style and its score's percentile in the scored pool; the batch is spread across
    styles and nearest positives (scoring.select_batch, ADR 0010)."""
    b = cfg["batch"]
    fraction = b["exploration_fraction"] if explore_fraction is None else explore_fraction
    size = size or b["size"]
    rng = rng or random.Random()
    retired = set(cfg["harvest"].get("retired_sources", []))
    rows = {}
    for r in con.execute("SELECT * FROM candidates WHERE state='pool' AND vector_status='ok' "
                         "AND COALESCE(spotify_status,'') != 'not_found'"):
        r = dict(r)
        if r["source"] in retired:
            continue
        if scores is not None:
            if r["id"] not in scores:
                continue
            r["score"], r["nearest"] = scores[r["id"]]
        if r["score"] is None or math.isnan(r["score"]):
            continue
        rows[r["id"]] = r
    ps = np.sort(np.array([r["score"] for r in rows.values()], np.float64))
    for r in rows.values():
        r["percentile"] = round(100.0 * float(np.searchsorted(ps, r["score"], "left")) / len(ps), 2)
    dids = [r["deezer_id"] for r in rows.values()]
    top_style = {d: int(np.argmax(s)) for d, s in vectors.get_styles(con, dids).items()}
    unit = None
    if b["diversity"] > 0 and rows:
        vec = vectors.get_deezer_vectors(con, dids)
        ids = [r["id"] for r in rows.values() if r["deezer_id"] in vec]
        if ids:
            u = l2norm(np.vstack([vec[rows[i]["deezer_id"]] for i in ids]))
            unit = dict(zip(ids, l2norm(u - u.mean(0))))    # centred on the pool: raw cosines are all close to 1
    known = {r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL")}
    known |= {r[0] for r in con.execute("SELECT spotify_id FROM offers")}
    known |= {r[0] for r in con.execute("SELECT spotify_id FROM local_tracks WHERE present=1 AND spotify_id IS NOT NULL")}
    active_artists = {r[0] for r in con.execute(
        "SELECT c.artist_key FROM offers o JOIN candidates c ON c.id=o.candidate_id WHERE o.removed_at IS NULL")}
    chosen: list[Pick] = []
    attempts = 0
    while len(chosen) < size and rows and attempts < size * b["map_attempts_factor"]:
        pool = [Pick(r["id"], r["artist_key"], r["score"], r["source"], nearest=r["nearest"],
                     style=top_style.get(r["deezer_id"]), percentile=r["percentile"],
                     rank=(ranks or {}).get(r["id"])) for r in rows.values()]
        need = size - len(chosen)
        explore_have = sum(p.exploration for p in chosen)
        frac = max(0.0, (round(size * fraction) - explore_have) / need) if need else 0
        picks = select_batch(pool, need, frac, tuple(b["explore_band"]), b["max_per_artist"], active_artists, rng,
                             taken=chosen, vectors=unit, diversity=b["diversity"],
                             max_style_share=b["max_style_share"], max_per_nearest=b["max_per_nearest"])
        if not picks:
            break
        for p in picks:
            attempts += 1
            c = rows.pop(p.cand_id)
            sid = resolve(c) if resolve else c["spotify_id"]
            if resolve and not sid:
                continue
            if sid and sid in known:
                if on_known:
                    on_known(c)
                continue
            chosen.append(replace(p, spotify_id=sid))
            if len(chosen) >= size:
                break
    return chosen


def plan(con, cfg: dict, size: int | None = None, variant: str | None = None,
         rng: random.Random | None = None, classes: list[str] | None = None, stage1: bool = False) -> dict:
    """`batch --plan`: the batch's scoring and selection with no Spotify calls and no writes. Like the batch it runs
    the weekly variant re-choice (in memory) unless a variant is given, and scores the pool in memory under the
    variant the batch would use; a pick with no stored Spotify id is marked unmapped (the live batch maps it first
    and may drop it). The re-ranker runs in memory too; stage1 leaves it out (to compare the two orders)."""
    choice = _rechoose(con, cfg, write=False) if variant is None else None
    variant = variant or (choice or {}).get("used") or vectors.active_variant(con, cfg)
    r = vectors.compute_scores(con, cfg, variant)
    scores = {}
    for (cid, _), sc, i in zip(r["rows"], r.get("score", ()), r.get("idx", ())):
        scores[cid] = (None if math.isnan(sc) else float(sc), r["owners"][int(i)])
    rr = {"used": False, "reason": "--stage1"} if stage1 else _rerank(con, cfg, variant, r, write=False)
    done, msg, prog = embedding_progress(con, cfg)
    explore = exploration_share(con, cfg)
    chosen = select_for_batch(con, cfg, rng, size=size, scores=scores, explore_fraction=explore["fraction"],
                              ranks=rr.get("ranks"))
    out = {"plan": True, "variant": variant, "variant_choice": choice, "positives": r["n_pos"],
           "pool": sum(1 for s, _ in scores.values() if s is not None),
           "picked": len(chosen), "exploration": sum(p.exploration for p in chosen),
           "unmapped": sum(1 for p in chosen if not p.spotify_id), "by_source": {},
           "embedding": prog, "embedding_note": None if done else msg,
           "exploration_fraction_used": explore["fraction"], "exploration_share": explore, "spread": spread(chosen),
           "rerank": rerank.summary(rr)}
    for p in chosen:
        out["by_source"][p.source] = out["by_source"].get(p.source, 0) + 1
    out["tracks"] = _describe(con, chosen, classes, top=rr.get("top"))
    return out


def _describe(con, chosen: list[Pick], classes: list[str] | None = None, top: dict | None = None) -> list[dict]:
    """top: the re-ranker's rows (rerank.run); a re-ranked top pick gets its chance, reasons and held-back."""
    from .review import nearest_label, style_names
    names = style_names(classes or [])
    out = []
    for p in chosen:
        rr = None if p.exploration else (top or {}).get(p.cand_id)
        c = con.execute("SELECT artist, title FROM candidates WHERE id=?", (p.cand_id,)).fetchone()
        style = None if p.style is None else (names[p.style] if p.style < len(names) else f"style {p.style}")
        out.append(dict(artist=c["artist"], title=c["title"], source=p.source, score=round(p.score, 4),
                        exploration=p.exploration, spotify_id=p.spotify_id, nearest=p.nearest,
                        sounds_like=nearest_label(con, p.nearest), style=style, percentile=p.percentile,
                        confidence=rr["p"] if rr else None, reasons=rr["reasons"] if rr else [],
                        held_back=rr["held_back"] if rr else []))
    return out
