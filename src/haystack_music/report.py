"""`haystack report`: how discovery is doing on this installation (read-only, no Spotify calls).

Counts and shares from the database and status.json. Track names only with --tracks, so the text can be pasted
into an issue without showing anyone's taste. The offer columns:
- decided: the offer has left Discoveries (outcome set), as in `haystack sources`.
- judged: decided, leaving out tracks that went unplayed and have no verdict (no signal either way).
- kept: the track's current verdict is a Rating of 3 stars or more, or it has no verdict and the offer closed as
  liked.
- liked/decided: outcome liked, over decided (the like rate in status.json). `haystack sources` shows
  kept/decided, the rate adaptive caps use (sources.py).

Plays per ISO week by context (needle, or Spotify's recently played) with the share of needle plays under 30 s;
the labels the re-ranker learns from per ISO week (hard and soft, ADR 0016); and the gates (`gates`, also written
to status.json after every job): for the AUC objective, the re-ranker, the adaptive exploration share and the
adaptive caps, the threshold, the current value, whether it is open and when it first opened."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np

from . import db, scoring, vectors, verdicts
from .review import nearest_label, style_names

WEEKS = 8       # plays and labels per ISO week: this many weeks, the current one last
SHORT_MS = 30000    # a needle play under this is "short" in the plays table
TOP = 100       # sample for the nearest-positive domain
HEAD = 30       # the head of the pool a batch draws from
LIBRARY = "local"   # owner prefix of a library window (vectors.positives); every other prefix is a preview


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def offer_rows(con) -> list[dict]:
    cur = verdicts.current(con)
    rows = []
    for r in con.execute("""SELECT o.spotify_id, o.candidate_id, o.source, o.exploration, o.outcome, o.batch_id,
                            b.created_at FROM offers o LEFT JOIN batches b ON b.id = o.batch_id ORDER BY o.id"""):
        v = cur.get(r["spotify_id"])
        decided = r["outcome"] is not None
        rows.append({"source": r["source"] or "unknown", "candidate": r["candidate_id"],
                     "exploration": bool(r["exploration"]),
                     "batch": r["batch_id"], "batch_at": r["created_at"], "decided": decided,
                     "judged": decided and (r["outcome"] != "unplayed" or v is not None),
                     "kept": decided and (v.keeps_heart if v else r["outcome"] == "liked"),
                     "liked": r["outcome"] == "liked"})
    return rows


def _summary(rows: list[dict]) -> dict:
    d = [r for r in rows if r["decided"]]
    judged = sum(r["judged"] for r in d)
    kept = sum(r["kept"] for r in d)
    return {"offered": len(rows), "decided": len(d), "judged": judged, "kept": kept,
            "kept_of_decided": _rate(kept, len(d)), "kept_of_judged": _rate(kept, judged),
            "liked_of_decided": _rate(sum(r["liked"] for r in d), len(d))}


def offers(con) -> dict:
    rows = offer_rows(con)
    out = {"all": _summary(rows),
           "exploit": _summary([r for r in rows if not r["exploration"]]),
           "exploration": _summary([r for r in rows if r["exploration"]]),
           "batches": [], "sources": {}}
    for b in sorted({r["batch"] for r in rows}, key=lambda x: (x is None, x or 0)):
        rs = [r for r in rows if r["batch"] == b]
        out["batches"].append({"batch": b, "at": rs[0]["batch_at"], "all": _summary(rs),
                               "exploit": _summary([r for r in rs if not r["exploration"]])})
    for s in sorted({r["source"] for r in rows}):
        out["sources"][s] = _summary([r for r in rows if r["source"] == s])
    return out


def verdict_counts(con) -> dict:
    cur = verdicts.current(con)
    kinds = Counter(v.kind for v in cur.values())
    stars = Counter(v.stars for v in cur.values() if v.kind == "rating")
    likes = [r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL")]
    return {"tracks": len(cur), "ratings": {str(s): stars.get(s, 0) for s in range(1, 6)},
            "pass": kinds.get("pass", 0), "dislike": kinds.get("dislike", 0),
            "liked_songs": len(likes), "liked_songs_rated": sum(1 for s in likes if s in cur and cur[s].kind == "rating"),
            "liked_songs_judged": sum(1 for s in likes if s in cur)}


def _ranked(con, limit: int) -> list:
    return con.execute("""SELECT id, deezer_id, artist, title, source, score, nearest FROM candidates
                          WHERE state='pool' AND vector_status='ok' AND score IS NOT NULL
                          ORDER BY score DESC LIMIT ?""", (limit,)).fetchall()


def pool(con, cfg: dict) -> dict:
    by_source: Counter = Counter()
    n = embedded = scored = 0
    for r in con.execute("SELECT source, vector_status, score FROM candidates WHERE state='pool'"):
        n += 1
        if r["vector_status"] == "ok":
            embedded += 1
            if r["score"] is not None:
                scored += 1
                by_source[r["source"]] += 1
    try:
        active = vectors.active_variant(con, cfg)
    except ValueError as e:
        active = f"invalid ({e})"
    # a database no 0.7.0 job has opened yet (the report opens it read-only) has no provider column
    cols = {r[1] for r in con.execute("PRAGMA table_info(deezer_vectors)")}
    prov = "v.provider" if "provider" in cols else "'deezer'"
    providers = dict(con.execute(f"""SELECT {prov} p, COUNT(*) FROM candidates c JOIN deezer_vectors v
                                     ON v.deezer_id=c.deezer_id WHERE c.state='pool' AND c.vector_status='ok'
                                     GROUP BY p ORDER BY p""").fetchall())
    kinds = Counter((r["nearest"] or "none:").split(":", 1)[0] for r in _ranked(con, TOP))
    top = sum(kinds.values())
    preview = top - kinds.get(LIBRARY, 0) - kinds.get("none", 0)
    return {"pool": n, "embedded": embedded, "embedded_by_provider": providers, "scored": scored,
            "scored_by_source": dict(sorted(by_source.items())),
            "active_variant": active, "scored_variant": db.kv_get(con, "scored_variant"),
            "scored_at": con.execute("SELECT MAX(scored_at) FROM candidates WHERE state='pool'").fetchone()[0],
            "top": top, "top_library": kinds.get(LIBRARY, 0), "top_preview": preview,
            "top_preview_share": _rate(preview, top),
            "top_preview_by_kind": {k: c for k, c in sorted(kinds.items()) if k not in (LIBRARY, "none")}}


def head(con, classes: list[str] | None = None, tracks: bool = False) -> dict:
    rows = _ranked(con, HEAD)
    if not rows:
        return {"n": 0}
    names = style_names(classes or [])
    styles = vectors.get_styles(con, [r["deezer_id"] for r in rows])
    top1: Counter = Counter()
    for r in rows:
        st = styles.get(r["deezer_id"])
        if st is None:
            top1["(no style vector)"] += 1
            continue
        i = int(np.argmax(st))
        top1[names[i] if i < len(names) else f"style #{i}"] += 1
    near = Counter(r["nearest"] for r in rows if r["nearest"])
    out = {"n": len(rows), "score_min": round(rows[-1]["score"], 4), "score_max": round(rows[0]["score"], 4),
           "distinct_nearest": len(near), "top3_nearest": sum(c for _, c in near.most_common(3)),
           "by_source": dict(Counter(r["source"] for r in rows).most_common()),
           "top1_styles": dict(top1.most_common())}
    if tracks:
        out["tracks"] = [{"score": round(r["score"], 4), "artist": r["artist"], "title": r["title"],
                          "source": r["source"], "sounds_like": nearest_label(con, r["nearest"])} for r in rows]
    return out


def eval_table(status_doc: dict | None) -> dict | None:
    """The latest eval (status.json scoring_eval): per-variant AUC (kept vs turned-down offers), mean percentile,
    recall and the judged tracks' mean percentile. The judged tracks are counted, never named."""
    e = (status_doc or {}).get("scoring_eval") or {}
    if not e.get("variants"):
        return None
    k = e.get("recall_k", 15)
    jc = e.get("judged_check") or {}
    jv = jc.get("variants") or {}
    n = Counter(t.get("verdict") for t in jc.get("tracks") or [])
    out = {"at": e.get("at"), "chosen": e.get("chosen"), "used": e.get("used"), "pinned": e.get("pinned"),
           "queries": sum((e.get("queries_by_kind") or {}).values()), "pool": e.get("pool"), "recall_k": k,
           "judged_yes": n.get("yes", 0), "judged_no": n.get("no", 0), "objective": e.get("objective_used"),
           "n_pos": e.get("n_pos"), "n_neg": e.get("n_neg"), "variants": {}}
    order = [v for v in scoring.VARIANTS if v in e["variants"]] + sorted(set(e["variants"]) - set(scoring.VARIANTS))
    for v in order:
        x, j = e["variants"][v], jv.get(v) or {}
        out["variants"][v] = {"auc": x.get("auc"), "degenerate": x.get("degenerate"),
                              "mean_pct": x.get("mean_pct"), "recall": x.get(f"recall_at_{k}"),
                              "judged_yes": j.get("mean_yes"), "judged_no": j.get("mean_no"),
                              "judged_yes_pen": j.get("mean_yes_pen"), "judged_no_pen": j.get("mean_no_pen")}
    return out


def _weeks(now: datetime | None = None) -> list[str]:
    """The last WEEKS ISO weeks, oldest first, e.g. ["2026-W36", ..., "2026-W43"]."""
    from .calibrate import week
    now = now or datetime.now(timezone.utc)
    return [week((now - timedelta(weeks=k)).strftime("%Y-%m-%dT%H:%M:%SZ")) for k in range(WEEKS - 1, -1, -1)]


def plays(con) -> dict:
    """Plays per ISO week by context (needle, or spotify: recently played), and of needle's: how many carry their
    length (ms_played, haystack 0.5.0) and how many of those stopped before 30 s."""
    from .calibrate import week
    weeks = _weeks()
    out = {w: {"needle": 0, "spotify": 0, "needle_with_length": 0, "needle_short": 0} for w in weeks}
    since = (datetime.now(timezone.utc) - timedelta(weeks=WEEKS, days=1)).strftime("%Y-%m-%d")
    for r in con.execute("SELECT played_at, context, ms_played FROM plays WHERE played_at >= ?", (since,)):
        try:
            w = week(r["played_at"])
        except ValueError:
            continue
        if w not in out:
            continue
        if r["context"] == "needle":
            out[w]["needle"] += 1
            if r["ms_played"] is not None:
                out[w]["needle_with_length"] += 1
                out[w]["needle_short"] += r["ms_played"] < SHORT_MS
        else:
            out[w]["spotify"] += 1
    n = sum(x["needle_with_length"] for x in out.values())
    return {"weeks": out, "needle_with_length": n,
            "needle_short_share": _rate(sum(x["needle_short"] for x in out.values()), n)}


def label_weeks(con, cfg: dict, labels: list[dict] | None = None) -> dict:
    """The re-ranker's labels by the ISO week they were given: {week: {hard, soft}}, the last WEEKS weeks."""
    from .calibrate import week
    from .rerank import labels as rerank_labels
    out = {w: {"hard": 0, "soft": 0} for w in _weeks()}
    for lb in labels if labels is not None else rerank_labels(con, cfg):
        w = week(lb["at"]) if lb.get("at") else None
        if w in out:
            out[w]["soft" if lb["soft"] else "hard"] += 1
    return out


def _first_run(con, job: str, test) -> str | None:
    """When the first successful run of job whose summary passes test finished."""
    for r in con.execute("SELECT finished_at, summary FROM runs WHERE job=? AND ok=1 ORDER BY id", (job,)):
        try:
            if test(json.loads(r["summary"] or "{}")):
                return r["finished_at"]
        except (ValueError, TypeError, AttributeError):
            continue
    return None


def gates(con, cfg: dict, labels: list[dict] | None = None) -> dict:
    """The thresholds that keep haystack's adaptive parts off until it has enough evidence (the dashboard of plan 2):
    each {threshold, value, open, opened_at}, opened_at the first batch (or harvest, for the caps) it was open."""
    from .batch import exploration_share
    from .rerank import labels as rerank_labels
    from .sources import rates
    out = {}
    e, ev = cfg["eval"], con.execute("SELECT n_pos, n_neg FROM eval_runs ORDER BY id DESC LIMIT 1").fetchone()
    have = None if ev is None or ev["n_pos"] is None else min(ev["n_pos"], ev["n_neg"] or 0)
    first = con.execute("SELECT MIN(at) FROM eval_runs WHERE objective='auc' AND trigger IN ('weekly','set')"
                        ).fetchone()[0]
    out["auc_objective"] = {"what": "kept and turned-down offers, the fewer of the two (eval.min_verdicts)",
                            "threshold": e["min_verdicts"], "value": have,
                            "open": e["objective"] == "auc" and have is not None and have >= e["min_verdicts"],
                            "opened_at": first}
    rr = cfg["rerank"]
    lbs = labels if labels is not None else rerank_labels(con, cfg)
    hard = sum(not lb["soft"] for lb in lbs)
    last = con.execute("SELECT loo_auc_model, loo_auc_stage1, used FROM rerank_runs ORDER BY id DESC LIMIT 1"
                       ).fetchone()
    gain = None if last is None or last["loo_auc_model"] is None or last["loo_auc_stage1"] is None \
        else round(last["loo_auc_model"] - last["loo_auc_stage1"], 3)
    first = con.execute("SELECT MIN(at) FROM rerank_runs WHERE used=1 AND trigger='batch'").fetchone()[0]
    out["rerank"] = {"what": "hard labels (rerank.min_labels) and the leave-one-out AUC gain (rerank.min_gain)",
                     "threshold": rr["min_labels"], "value": hard, "gain_threshold": rr["min_gain"], "gain": gain,
                     "soft": len(lbs) - hard,
                     "open": bool(rr["enabled"]) and hard >= rr["min_labels"] and gain is not None
                     and gain >= rr["min_gain"], "opened_at": first}
    b = cfg["batch"]
    x = exploration_share(con, cfg)
    moved = "moved by the recent kept rates"
    judged = None if "explore_judged" not in x else min(x["explore_judged"], x["exploit_judged"])
    out["exploration"] = {"what": "judged offers of each kind, exploration and top picks (batch.explore_min_judged)",
                          "threshold": b["explore_min_judged"], "value": judged, "open": x.get("why") == moved,
                          "opened_at": _first_run(con, "batch", lambda s: (s.get("exploration_share") or {})
                                                  .get("why") == moved)}
    h, need = cfg["harvest"], cfg["sources"]["min_offers_before_retire"]
    live = [s for s in h["sources"] if s not in h.get("retired_sources", [])]
    per = rates(con)["per_source"]
    best = max((per.get(s, {}).get("decided", 0) for s in live), default=0)
    out["adaptive_caps"] = {"what": "decided offers of the busiest source (sources.min_offers_before_retire)",
                            "threshold": need, "value": best,
                            "open": bool(h.get("adaptive_caps", True)) and best >= need,
                            "opened_at": _first_run(con, "harvest", lambda s: any(
                                str(c.get("why", "")).startswith("kept rate") for c in (s.get("caps") or {}).values()))}
    return out


def build(con, cfg: dict, status_doc: dict | None, classes: list[str] | None = None, tracks: bool = False) -> dict:
    from .rerank import labels as rerank_labels
    lbs = rerank_labels(con, cfg)
    return {"generated_at": db.now(), "offers": offers(con), "verdicts": verdict_counts(con), "pool": pool(con, cfg),
            "head": head(con, classes, tracks), "eval": eval_table(status_doc), "plays": plays(con),
            "labels": label_weeks(con, cfg, lbs), "gates": gates(con, cfg, lbs)}


# ---------------- text ----------------

def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def _num(x, fmt: str = "") -> str:
    return "-" if x is None else format(x, fmt)


def _row(label: str, s: dict) -> str:
    return (f"  {label:30} {s['offered']:7} {s['decided']:7} {s['judged']:6} {s['kept']:5} "
            f"{_pct(s['kept_of_decided']):>9} {_pct(s['kept_of_judged']):>9} {_pct(s['liked_of_decided']):>9}")


def text(r: dict) -> str:
    o, v, p, h, e = r["offers"], r["verdicts"], r["pool"], r["head"], r["eval"]
    out = [f"haystack report, {r['generated_at']}", ""]
    out += ["Offers",
            "  kept = current verdict a Rating of 3 stars or more, or liked with no verdict",
            "  judged = decided, leaving out unplayed tracks with no verdict",
            "  liked/dec = closed as liked, over decided (the like rate in status.json)",
            f"  {'':30} {'offered':>7} {'decided':>7} {'judged':>6} {'kept':>5} {'kept/dec':>9} {'kept/jdg':>9} "
            f"{'liked/dec':>9}",
            _row("all", o["all"]), _row("exploit", o["exploit"]), _row("exploration", o["exploration"])]
    for b in o["batches"]:
        name = f"batch {b['batch']} ({(b['at'] or '')[:10]})" if b["batch"] is not None else "no batch"
        out += [_row(name, b["all"]), _row("  exploit", b["exploit"])]
    for s, x in o["sources"].items():
        out.append(_row(f"source {s}", x))
    stars = "  ".join(f"{s}*: {n}" for s, n in v["ratings"].items())
    out += ["", "Verdicts (current, one per track)",
            f"  {v['tracks']} tracks. Ratings {stars}. Pass: {v['pass']}. Dislike: {v['dislike']}.",
            f"  Liked Songs: {v['liked_songs']}, {v['liked_songs_rated']} rated, "
            f"{v['liked_songs'] - v['liked_songs_judged']} with no verdict"]
    srcs = ", ".join(f"{s} {n}" for s, n in p["scored_by_source"].items()) or "-"
    kinds = ", ".join(f"{k} {n}" for k, n in p["top_preview_by_kind"].items()) or "-"
    prov = ", ".join(f"{k} {n}" for k, n in p["embedded_by_provider"].items())
    out += ["", "Pool",
            f"  {p['pool']} candidates, {p['embedded']} embedded{f' (previews: {prov})' if prov else ''}, "
            f"{p['scored']} scored ({srcs})",
            f"  active variant {p['active_variant']}; stored scores from {p['scored_variant'] or '-'} "
            f"at {p['scored_at'] or '-'}",
            f"  nearest positive of the top {p['top']}: library window {p['top_library']}, preview "
            f"{p['top_preview']} ({_pct(p['top_preview_share'])}; {kinds})"]
    if h.get("n"):
        styles = ", ".join(f"{s} {n}" for s, n in h["top1_styles"].items())
        hs = ", ".join(f"{s} {n}" for s, n in h["by_source"].items())
        out += ["", f"Top {h['n']} by stored score",
                f"  score {h['score_min']:.4f} to {h['score_max']:.4f}; sources: {hs}",
                f"  {h['distinct_nearest']} distinct nearest positives; the 3 most common are nearest for "
                f"{h['top3_nearest']} of {h['n']}",
                f"  top-1 style: {styles}"]
        for i, t in enumerate(h.get("tracks") or [], 1):
            sl = t["sounds_like"]
            like = f"  (sounds like {sl['artist']} - {sl['title']})" if sl else ""
            out.append(f"  {i:2}. {t['score']:.4f}  {t['artist']} - {t['title']}  [{t['source']}]{like}")
    out += ["", "Latest eval (status.json)"]
    if not e:
        out.append("  none yet (haystack eval)")
    else:
        out += [f"  at {e['at']}; chosen {e['chosen']}, used {e['used']}"
                f"{'' if not e['pinned'] else ', pinned ' + str(e['pinned'])}; {e['queries']} held-out queries, "
                f"pool {e['pool']}; judged tracks: {e['judged_yes']} yes, {e['judged_no']} no",
                f"  chosen by {e['objective'] or 'mean pct'}"
                + ("" if e["n_neg"] is None else f"; offers: {e['n_pos']} kept, {e['n_neg']} turned down"),
                f"  {'variant':14} {'AUC':>11} {'mean pct':>8} {'recall@' + str(e['recall_k']):>9} "
                f"{'judged yes/no':>15} {'penalised':>13}"]
        for name, x in e["variants"].items():
            jd = f"{_num(x['judged_yes'])} / {_num(x['judged_no'])}"
            jp = f"{_num(x['judged_yes_pen'])} / {_num(x['judged_no_pen'])}"
            a = _num(x["auc"], ".3f") + (" deg" if x["degenerate"] else "")
            out.append(f"  {name:14} {a:>11} {_num(x['mean_pct']):>8} {_num(x['recall'], '.3f'):>9} {jd:>15} {jp:>13}")
    pl, lb = r["plays"], r["labels"]
    out += ["", "Plays per week (needle short = stopped before 30 s, of the needle plays with a length)",
            f"  {'week':9} {'needle':>6} {'spotify':>7} {'short':>11}   {'labels':>6} {'soft':>5}"]
    for w, x in pl["weeks"].items():
        sh = f"{x['needle_short']}/{x['needle_with_length']}" if x["needle_with_length"] else "-"
        out.append(f"  {w:9} {x['needle']:6} {x['spotify']:7} {sh:>11}   {lb[w]['hard']:6} {lb[w]['soft']:5}")
    out.append(f"  needle plays with a length: {pl['needle_with_length']}, "
               f"{_pct(pl['needle_short_share'])} under 30 s")
    out += ["", "Gates (open = the adaptive part is in use)"]
    for name, g in r["gates"].items():
        val = f"{_num(g['value'])} of {g['threshold']}"
        if "gain" in g:
            val += f", gain {_num(g['gain'], '+.3f')} of {g['gain_threshold']}, {g['soft']} soft"
        state = "open" if g["open"] else "closed"
        since = f" (first open {g['opened_at']})" if g["opened_at"] else ""
        out.append(f"  {name:14} {state:6} {val}{since}")
        out.append(f"  {'':14} {g['what']}")
    return "\n".join(out) + "\n"
