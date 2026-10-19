"""Preview vectors aligned to the library's space before scoring (ADR 0012; token-free).

A Deezer preview and the listener's own file of the same song land in slightly different places in the embedding
space, so a candidate (a preview) tends to find its nearest positive among the other previews (liked songs without
a local file, liked discoveries, Rated tracks) rather than among the library's files. The liked songs that have
both a preview vector and a confidently matched local file are pairs. A linear map from preview space to the
library's window space is fitted on them and applied to every preview vector, positives and candidates alike
(vectors.preview_vectors), so all comparisons happen in library space.

The map works on L2-normalised vectors: x -> (x - mp) @ M + mt, with mp and mt the pairs' mean preview and mean
target. M is a ridge fit shrunk toward the identity (min |XM - Y|^2 + lam |M - I|^2; lam = inf leaves the mean
shift alone) or an orthogonal Procrustes rotation, and the target is the own file's best-matching window or the
mean of its windows. They are chosen by 5-fold cross-validation on self-retrieval: is the held-out preview's own
file the nearest (top 1), or among the 10 nearest (top 10), of the library's files and the other liked songs'
previews? Ties go to the simpler fit (the mean target, then the larger lam, then Procrustes).

scoring.align = "auto" uses the map when it was fitted on at least scoring.align_min_pairs pairs, improved the
cross-validated self-retrieval and the latest eval did not find it worse (evaluate.choose); "off" never does.
Without a library there are no pairs and no map, and scoring is the same as without this module."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np

from . import config, db
from .scoring import l2norm
from .textnorm import song_key

log = logging.getLogger("haystack_music.align")
FILE = "preview_align.npz"
LAMBDAS = (float("inf"), 1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0)    # simplest first
TARGETS = ("mean", "best")
FOLDS = 5
SEED = 20261019
REFIT_CHANGE = 0.10        # the library job refits when the pair count moved by more than this share
TOP = 10
PAIRS_SQL = """FROM liked_map m JOIN likes l ON l.spotify_id=m.spotify_id
               JOIN local_tracks t ON t.id=m.local_id AND t.present=1
               JOIN local_embeddings e ON e.content_hash=t.content_hash AND e.status='ok'
               JOIN deezer_vectors d ON d.deezer_id=m.deezer_id
               WHERE l.removed_at IS NULL AND m.use_local=1 AND m.vector_status='ok'"""


@dataclass(frozen=True)
class Map:
    M: np.ndarray
    mp: np.ndarray
    mt: np.ndarray
    meta: dict = field(default_factory=dict)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((l2norm(x) - self.mp) @ self.M + self.mt).astype(np.float32)

    @property
    def n_pairs(self) -> int:
        return int(self.meta.get("n_pairs", 0))

    @property
    def improved(self) -> bool:
        return bool(self.meta.get("improved"))


def path():
    return config.EMB_DIR / FILE


_cache: dict = {}


def load() -> Map | None:
    """The stored map, or None. Cached until the file changes."""
    p = path()
    try:
        st = p.stat()
    except FileNotFoundError:
        return None
    key = (str(p), st.st_mtime_ns, st.st_size)
    if _cache.get("key") != key:
        with np.load(p, allow_pickle=False) as z:
            m = Map(z["M"].astype(np.float32), z["mp"].astype(np.float32), z["mt"].astype(np.float32),
                    json.loads(str(z["meta"])))
        _cache.update(key=key, map=m)
    return _cache["map"]


def gate(con) -> dict | None:
    """The latest eval's verdict on the aligned scorer (evaluate.choose), or None before the first."""
    g = db.kv_get(con, "preview_align_gate")
    return json.loads(g) if g else None


def active(con, cfg: dict) -> Map | None:
    """The map scoring uses now, or None (scoring then works on the stored preview vectors as they are).
    scoring.align "candidate" is internal: the eval's aligned run, which ignores the eval's own gate."""
    mode = cfg["scoring"].get("align", "auto")
    if mode not in ("auto", "off", "candidate"):
        raise ValueError(f"scoring.align {mode!r} is not auto or off")
    if mode == "off" or not cfg["library"].get("root"):
        return None
    m = load()
    if m is None or not m.improved or m.n_pairs < cfg["scoring"]["align_min_pairs"]:
        return None
    if mode == "auto" and (gate(con) or {}).get("ok") is False:
        return None
    return m


def with_mode(cfg: dict, mode: str) -> dict:
    return {**cfg, "scoring": {**cfg["scoring"], "align": mode}}


def candidate(con, cfg: dict) -> Map | None:
    """The map the eval weighs against unaligned scoring, whatever its last verdict; None when align is off."""
    if cfg["scoring"].get("align", "auto") == "off":
        return None
    return active(con, with_mode(cfg, "candidate"))


def count_pairs(con) -> int:
    return con.execute(f"SELECT COUNT(DISTINCT m.spotify_id) {PAIRS_SQL}").fetchone()[0]


def _pairs(con) -> dict | None:
    """The pairs and everything self-retrieval needs: the library's windows (one run per file), each pair's own
    files (the matched file and any file of the same song), and the other liked songs' previews."""
    from . import vectors
    ids, wins = vectors._library_windows(con)
    if not ids:
        return None
    at = {i: j for j, i in enumerate(ids)}
    key = {r[0]: song_key(r[1], r[2]) for r in con.execute(
        "SELECT id, artist_raw, title_raw FROM local_tracks WHERE present=1")}
    same: dict = {}
    for i in ids:
        if key.get(i) not in (None, "|"):
            same.setdefault(key[i], set()).add(at[i])
    rows = con.execute(f"SELECT DISTINCT m.spotify_id, m.local_id, m.deezer_id {PAIRS_SQL} "
                       "ORDER BY m.spotify_id").fetchall()
    vec = vectors.get_deezer_vectors(con, [r[2] for r in rows])
    rows = [r for r in rows if r[1] in at and r[2] in vec]
    liked = con.execute("""SELECT m.spotify_id, m.deezer_id FROM liked_map m JOIN likes l ON l.spotify_id=m.spotify_id
                           WHERE l.removed_at IS NULL AND m.vector_status='ok' ORDER BY m.spotify_id""").fetchall()
    cvec = vectors.get_deezer_vectors(con, [r[1] for r in liked])
    liked = [r for r in liked if r[1] in cvec]
    own = [at[r[1]] for r in rows]
    return dict(P=l2norm(np.vstack([vec[r[2]] for r in rows])) if rows else np.zeros((0, 512), np.float32),
                sid=[r[0] for r in rows], own=own,
                own_sets=[same.get(key.get(r[1]), set()) | {o} for r, o in zip(rows, own)],
                wins=[l2norm(w) for w in wins], W=l2norm(np.vstack(wins)),
                starts=np.concatenate([[0], np.cumsum([len(w) for w in wins])[:-1]]).astype(np.int64),
                C=l2norm(np.vstack([cvec[r[1]] for r in liked])) if liked else np.zeros((0, 512), np.float32),
                c_sid=np.array([r[0] for r in liked], dtype=object))


def ranks(d: dict, Q: np.ndarray, idx, C: np.ndarray | None = None, chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Self-retrieval ranks of the pairs idx, with queries Q (their previews, mapped or not) and C the other liked
    songs' previews in the same space: -> (rank among the library's files, rank among the files and the previews).
    1 = the own file is the nearest; another file of the same song counts as the own."""
    Q = l2norm(Q)
    C = d["C"] if C is None else l2norm(C)
    idx = list(idx)
    lib, cross = np.empty(len(idx), np.int64), np.empty(len(idx), np.int64)
    for s in range(0, len(idx), chunk):
        T = np.maximum.reduceat(Q[s:s + chunk] @ d["W"].T, d["starts"], axis=1)
        O = Q[s:s + chunk] @ C.T
        for r, i in enumerate(idx[s:s + chunk]):
            best = max(T[r, j] for j in d["own_sets"][i])
            o = O[r][d["c_sid"] != d["sid"][i]]
            lib[s + r] = int((T[r] > best).sum()) + 1
            cross[s + r] = lib[s + r] + int((o > best).sum())
    return lib, cross


def _targets(d: dict, kind: str) -> np.ndarray:
    if kind == "mean":
        return l2norm(np.vstack([d["wins"][o].mean(0) for o in d["own"]]))
    return np.vstack([d["wins"][o][int(np.argmax(d["wins"][o] @ p))] for o, p in zip(d["own"], d["P"])])


def fit_map(P: np.ndarray, T: np.ndarray, method: str, lam: float | None = None) -> Map:
    """One map from the previews P to the targets T (both L2-normalised rows)."""
    mp, mt = P.mean(0), T.mean(0)
    X, Y = (P - mp).astype(np.float64), (T - mt).astype(np.float64)
    eye = np.eye(P.shape[1])
    if method == "procrustes":
        u, _, vt = np.linalg.svd(X.T @ Y)
        M = u @ vt
    elif lam is None or np.isinf(lam):
        M = eye
    else:
        M = np.linalg.solve(X.T @ X + lam * eye, X.T @ Y + lam * eye)
    return Map(M.astype(np.float32), mp.astype(np.float32), mt.astype(np.float32))


def _summary(lib: np.ndarray, cross: np.ndarray) -> dict:
    return {"top1": round(float(np.mean(cross <= 1)), 4), f"top{TOP}": round(float(np.mean(cross <= TOP)), 4),
            "median_rank": float(np.median(cross)),
            "library_only": {"top1": round(float(np.mean(lib <= 1)), 4),
                             f"top{TOP}": round(float(np.mean(lib <= TOP)), 4), "median_rank": float(np.median(lib))}}


def _score(s: dict) -> tuple:
    return s[f"top{TOP}"], s["top1"]


def fit(con, cfg: dict, write: bool = True) -> dict:
    """Fit the map on the pairs, choose it by cross-validation and store it (with write) -> a summary.
    Too few pairs (or no library) removes a stored map: scoring is then as without one."""
    min_pairs = cfg["scoring"]["align_min_pairs"]
    if not cfg["library"].get("root"):
        out = {"status": "no library", "n_pairs": 0}
    else:
        d = _pairs(con)
        n = 0 if d is None else len(d["own"])
        out = {"status": "too few pairs", "n_pairs": n} if n < min_pairs else None
    if out is not None:
        out["reason"] = (f"{out['status']}: {out['n_pairs']} liked songs have both a preview and a local file "
                         f"(need {min_pairs})")
        if write and path().exists():
            path().unlink()
            db.kv_set(con, "preview_align_version", None)
        return out
    n = len(d["own"])
    folds = np.random.default_rng(SEED).permutation(n) % FOLDS
    base = _summary(*ranks(d, d["P"], range(n)))
    tried = []
    for tk in TARGETS:
        T = _targets(d, tk)
        for method, lam in [("ridge", x) for x in LAMBDAS] + [("procrustes", None)]:
            lib, cross = np.empty(n, np.int64), np.empty(n, np.int64)
            for f in range(FOLDS):
                tr, te = np.nonzero(folds != f)[0], np.nonzero(folds == f)[0]
                m = fit_map(d["P"][tr], T[tr], method, lam)
                lib[te], cross[te] = ranks(d, m(d["P"][te]), te, m(d["C"]))
            tried.append(dict(target=tk, method=method, lam=lam, cv=_summary(lib, cross)))
    best = max(tried, key=lambda t: _score(t["cv"]))           # max keeps the first of equals: the simplest
    m = fit_map(d["P"], _targets(d, best["target"]), best["method"], best["lam"])
    version = db.now()
    meta = {"version": version, "n_pairs": n, "method": best["method"],
            "lam": None if best["lam"] is None or np.isinf(best["lam"]) else best["lam"],
            "shift_only": bool(best["method"] == "ridge" and (best["lam"] is None or np.isinf(best["lam"]))),
            "target": best["target"], "folds": FOLDS, "unaligned": base, "cv": best["cv"],
            "improved": _score(best["cv"]) > _score(base),
            "tried": [{"target": t["target"], "method": t["method"],
                       "lam": None if t["lam"] is None or np.isinf(t["lam"]) else t["lam"],
                       "top1": t["cv"]["top1"], f"top{TOP}": t["cv"][f"top{TOP}"]} for t in tried]}
    out = {"status": "fitted", **{k: v for k, v in meta.items() if k != "tried"}}
    out["reason"] = (f"{n} pairs; self-retrieval top {TOP} {base[f'top{TOP}']:.3f} -> {best['cv'][f'top{TOP}']:.3f}, "
                     f"top 1 {base['top1']:.3f} -> {best['cv']['top1']:.3f} (cross-validated); "
                     + ("used" if meta["improved"] else "not used: no better than unaligned"))
    if write:
        config.EMB_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path().with_name("preview_align.tmp.npz")
        np.savez(tmp, M=m.M, mp=m.mp, mt=m.mt, meta=np.array(json.dumps(meta)))
        os.replace(tmp, path())
        db.kv_set(con, "preview_align_version", version)
    log.info("preview alignment: %s", out["reason"])
    return out


def refit(con, cfg: dict) -> dict:
    """The library job's step: fit when there is no map yet, or the pair count moved by more than 10 %."""
    if cfg["scoring"].get("align", "auto") == "off":
        return {"skipped": True, "reason": "scoring.align is off"}
    m = load()
    n = count_pairs(con) if cfg["library"].get("root") else 0
    if m is not None and m.n_pairs and abs(n - m.n_pairs) <= REFIT_CHANGE * m.n_pairs:
        return {"skipped": True, "reason": f"{n} pairs, fitted on {m.n_pairs}"}
    if m is None and n < cfg["scoring"]["align_min_pairs"]:
        return {"skipped": True, "reason": f"{n} pairs (need {cfg['scoring']['align_min_pairs']})"}
    return fit(con, cfg)


def show(con, cfg: dict) -> dict:
    m = load()
    a = active(con, cfg)
    return {"used": a is not None, "mode": cfg["scoring"].get("align", "auto"), "pairs_now": count_pairs(con),
            "map": None if m is None else {k: v for k, v in m.meta.items() if k != "tried"},
            "tried": [] if m is None else m.meta.get("tried", []), "gate": gate(con)}
