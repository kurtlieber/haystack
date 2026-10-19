"""`haystack report`: how discovery is doing on this installation (read-only, no Spotify calls).

Counts and shares from the database and status.json. Track names only with --tracks, so the text can be pasted
into an issue without showing anyone's taste. The offer columns:
- decided: the offer has left Discoveries (outcome set), as in `haystack sources`.
- judged: decided, leaving out tracks that went unplayed and have no verdict (no signal either way).
- kept: the track's current verdict is a Rating of 3 stars or more, or it has no verdict and the offer closed as
  liked.
- liked/decided: the `haystack sources` like rate (outcome liked, over decided)."""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import db, scoring, vectors, verdicts
from .review import nearest_label, style_names

TOP = 100       # sample for the nearest-positive domain
HEAD = 30       # the head of the pool a batch draws from
LIBRARY = "local"   # owner prefix of a library window (vectors.positives); every other prefix is a preview


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def _offer_rows(con) -> list[dict]:
    cur = verdicts.current(con)
    rows = []
    for r in con.execute("""SELECT o.spotify_id, o.source, o.exploration, o.outcome, o.batch_id, b.created_at
                            FROM offers o LEFT JOIN batches b ON b.id = o.batch_id ORDER BY o.id"""):
        v = cur.get(r["spotify_id"])
        decided = r["outcome"] is not None
        rows.append({"source": r["source"] or "unknown", "exploration": bool(r["exploration"]),
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
    rows = _offer_rows(con)
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
    kinds = Counter((r["nearest"] or "none:").split(":", 1)[0] for r in _ranked(con, TOP))
    top = sum(kinds.values())
    preview = top - kinds.get(LIBRARY, 0) - kinds.get("none", 0)
    return {"pool": n, "embedded": embedded, "scored": scored, "scored_by_source": dict(sorted(by_source.items())),
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
    """The latest `haystack eval` (status.json scoring_eval): per-variant mean percentile, recall and the judged
    tracks' mean percentile. The judged tracks are counted, never named."""
    e = (status_doc or {}).get("scoring_eval") or {}
    if not e.get("variants"):
        return None
    k = e.get("recall_k", 15)
    jc = e.get("judged_check") or {}
    jv = jc.get("variants") or {}
    n = Counter(t.get("verdict") for t in jc.get("tracks") or [])
    out = {"at": e.get("at"), "chosen": e.get("chosen"), "used": e.get("used"), "pinned": e.get("pinned"),
           "queries": sum((e.get("queries_by_kind") or {}).values()), "pool": e.get("pool"), "recall_k": k,
           "judged_yes": n.get("yes", 0), "judged_no": n.get("no", 0), "variants": {}}
    order = [v for v in scoring.VARIANTS if v in e["variants"]] + sorted(set(e["variants"]) - set(scoring.VARIANTS))
    for v in order:
        x, j = e["variants"][v], jv.get(v) or {}
        out["variants"][v] = {"mean_pct": x.get("mean_pct"), "recall": x.get(f"recall_at_{k}"),
                              "judged_yes": j.get("mean_yes"), "judged_no": j.get("mean_no"),
                              "judged_yes_pen": j.get("mean_yes_pen"), "judged_no_pen": j.get("mean_no_pen")}
    return out


def build(con, cfg: dict, status_doc: dict | None, classes: list[str] | None = None, tracks: bool = False) -> dict:
    return {"generated_at": db.now(), "offers": offers(con), "verdicts": verdict_counts(con), "pool": pool(con, cfg),
            "head": head(con, classes, tracks), "eval": eval_table(status_doc)}


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
            "  liked/dec = the `haystack sources` like rate",
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
    out += ["", "Pool",
            f"  {p['pool']} candidates, {p['embedded']} embedded, {p['scored']} scored ({srcs})",
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
                f"  {'variant':14} {'mean pct':>8} {'recall@' + str(e['recall_k']):>9} {'judged yes/no':>15} "
                f"{'penalised':>13}"]
        for name, x in e["variants"].items():
            jd = f"{_num(x['judged_yes'])} / {_num(x['judged_no'])}"
            jp = f"{_num(x['judged_yes_pen'])} / {_num(x['judged_no_pen'])}"
            out.append(f"  {name:14} {_num(x['mean_pct']):>8} {_num(x['recall'], '.3f'):>9} {jd:>15} {jp:>13}")
    return "\n".join(out) + "\n"
