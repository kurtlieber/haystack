#!/usr/bin/env python3
"""Does a self-supervised music model tell the listener's yes from no better than haystack's Discogs-EffNet
512-d vectors? An offline experiment: it reads haystack's database read-only and changes nothing in haystack.

  select   pick the tracks once and write tracks.json into --out
  embed    embed them with one or more models (local files from the library, Deezer previews in memory only)
  report   print one markdown table for every model embedded so far, against the stored effnet512 vectors

The tracks: the labelled ones (the re-ranker's hard labels plus the eval's kept and turned-down offers), the
liked songs with a Deezer preview, the liked songs' local files (the same 30 s windows haystack embedded), and a
sample of the candidate pool stratified by its stage-1 score. Positives are the liked previews plus the liked
files; every score leaves out the track's own song, as `haystack eval` does. See README.md."""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from haystack_music import config, db, embed, evaluate, rerank, vectors
from haystack_music.scoring import l2norm, owner_max, topk_mean
from haystack_music.textnorm import song_key

log = logging.getLogger("embedding_compare")
WIN = 30.0                 # seconds per clip, as haystack's windows and Deezer's previews
MIN_CLIP = 5.0             # a shorter preview is skipped
DEEZER_INTERVAL = 1.2      # seconds between Deezer requests (haystack's own minimum is 1.1)
BASELINE = "effnet512"
BAR = 0.05                 # the AUC margin a model must win by, on raw and on topk_mean
MAX_SECS = 5.0             # the most a model may cost per 30 s clip
BOOT = 2000

# Pinned revisions: MERT runs its model code from the Hub (trust_remote_code), so it must not change underneath.
MODELS = {
    "mert-95m": dict(kind="mert", repo="m-a-p/MERT-v1-95M", rev="12af15fef9d0ac838c3f475bfbbf26d2060dd4f5", sr=24000),
    "mert-330m": dict(kind="mert", repo="m-a-p/MERT-v1-330M", rev="5240c2708a5acaee1007f43fb9735c7dcd0b78c9", sr=24000),
    "muq-large": dict(kind="muq", repo="OpenMuQ/MuQ-large-msd-iter", rev="0562a57814f6f8bbd9fdea0a25921a2fce1a841a",
                      sr=24000),
    "clap-music": dict(kind="clap", repo="laion/larger_clap_music", rev="a0b4534a14f58e20944452dff00a22a06ce629d1",
                       sr=48000),
}


def variants_of(name: str) -> list[str]:
    """A layered model gives two vectors per clip: its last layer and the mean of its middle third of layers."""
    return [name] if MODELS[name]["kind"] == "clap" else [name, f"{name}-mid"]


def _first_artist(artists_json) -> str:
    a = json.loads(artists_json or "[]")
    return a[0] if a else ""


# ---- select ---------------------------------------------------------------------------------------------------

def _labels(con, cfg) -> dict[str, dict]:
    """spotify id -> {did, y}: the re-ranker's hard labels, plus the eval's kept and turned-down offers."""
    fn = rerank.labels
    labs = fn(con, cfg) if "cfg" in inspect.signature(fn).parameters else fn(con)
    out = {x["sid"]: dict(did=int(x["deezer_id"]), y=int(x["y"])) for x in labs if not x.get("soft")}
    liked_ids = {r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL")}
    vq, _ = evaluate._verdict_queries(con, cfg, [], liked_ids, _known(con))
    for x in vq:
        out.setdefault(x["sid"], dict(did=int(x["did"]), y=int(x["kind"] == "kept")))
    return out


def _liked(con) -> list:
    return con.execute("""SELECT m.spotify_id, m.local_id, m.deezer_id, m.vector_status, s.name, s.artists_json
                          FROM likes l JOIN liked_map m ON m.spotify_id=l.spotify_id
                          LEFT JOIN spotify_tracks s ON s.spotify_id=l.spotify_id
                          WHERE l.removed_at IS NULL""").fetchall()


def _known(con) -> set:
    keys = {song_key(_first_artist(r["artists_json"]), r["name"]) for r in _liked(con) if r["name"]}
    lib = con.execute("SELECT artist_raw, title_raw FROM local_tracks WHERE present=1")
    keys |= {song_key(r[0], r[1]) for r in lib}
    return keys


def select(args) -> int:
    cfg = config.load()
    con = db.connect_readonly()
    rnd = random.Random(args.seed)
    labs = _labels(con, cfg)
    n_yes = sum(x["y"] for x in labs.values())
    if n_yes < args.min_labels or len(labs) - n_yes < args.min_labels:
        print(f"not enough labels for the experiment: {n_yes} yes and {len(labs) - n_yes} no "
              f"(need {args.min_labels} of each)", file=sys.stderr)
        return 1
    keys = evaluate._track_keys(con, labs)
    labels = [dict(sid=s, did=x["did"], y=x["y"], keys=sorted(keys.get(s, ()))) for s, x in sorted(labs.items())]
    previews, files = [], []
    for r in _liked(con):
        k = [song_key(_first_artist(r["artists_json"]), r["name"])] if r["name"] else []
        if r["vector_status"] == "ok" and r["deezer_id"] is not None:
            previews.append(dict(sid=r["spotify_id"], did=int(r["deezer_id"]), keys=k))
        if r["local_id"] is None:
            continue
        t = con.execute("""SELECT t.path, t.artist_raw, t.title_raw, e.file FROM local_tracks t JOIN local_embeddings e
                           ON e.content_hash=t.content_hash AND e.status='ok' WHERE t.id=? AND t.present=1""",
                        (r["local_id"],)).fetchone()
        if t is None or not Path(t["file"]).exists():
            continue
        with np.load(t["file"], allow_pickle=False) as z:
            starts = [float(s) for s in z["starts"]]
        files.append(dict(sid=r["spotify_id"], local_id=int(r["local_id"]), path=t["path"], emb=t["file"],
                          starts=starts, keys=sorted(set(k) | {song_key(t["artist_raw"], t["title_raw"])})))
    known = _known(con)
    liked_ids = {r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL")}
    _, _, rows = vectors.pool_vectors(con)
    pool = [r for r in rows if r["spotify_id"] not in liked_ids and r["spotify_id"] not in labs
            and song_key(r["artist"], r["title"]) not in known]
    scored = sorted((r for r in pool if r["score"] is not None), key=lambda r: r["score"])
    if scored:      # equal draws from each tenth of the stage-1 score
        pct = [(r, 100.0 * (i + 0.5) / len(scored)) for i, r in enumerate(scored)]
        bands = [pct[len(pct) * b // 10: len(pct) * (b + 1) // 10] for b in range(10)]
        picks = [x for b in bands for x in rnd.sample(b, min(len(b), args.pool // 10))]
    else:           # a pool never scored: a plain random sample
        picks = [(r, None) for r in rnd.sample(pool, min(len(pool), args.pool))]
    sample = [dict(cid=r["id"], did=int(r["deezer_id"]), pct=None if p is None else round(p, 1)) for r, p in picks]
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    doc = dict(created=datetime.now(timezone.utc).isoformat(timespec="seconds"), seed=args.seed, labels=labels,
               liked_previews=previews, liked_files=files, pool=sample)
    (out / "tracks.json").write_text(json.dumps(doc, indent=1))
    n_win = sum(len(f["starts"]) for f in files)
    print(f"tracks.json: {len(labels)} labels ({n_yes} yes, {len(labels) - n_yes} no), {len(previews)} liked "
          f"previews, {len(files)} liked files ({n_win} windows), {len(sample)} pool tracks")
    return 0


def _tracks(out: Path) -> dict:
    f = out / "tracks.json"
    if not f.exists():
        sys.exit(f"no {f}: run select first")
    return json.loads(f.read_text())


# ---- embed ----------------------------------------------------------------------------------------------------

class Store:
    """One variant's vectors, key -> vector, saved atomically as <out>/<variant>.npz."""

    def __init__(self, out: Path, variant: str):
        self.path = out / f"{variant}.npz"
        self.vec: dict[str, np.ndarray] = {}
        self.secs: dict[str, float] = {}
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as z:
                self.vec = dict(zip(z["keys"].tolist(), z["vecs"]))
                self.secs = dict(zip(z["keys"].tolist(), z["secs"].tolist()))

    def save(self) -> None:
        if not self.vec:
            return
        keys = list(self.vec)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, keys=np.array(keys), vecs=np.vstack([self.vec[k] for k in keys]).astype(np.float32),
                 secs=np.array([self.secs[k] for k in keys], np.float32))
        os.replace(tmp, self.path)


def _in_pause(pause, now=None) -> float:
    """Seconds until the pause window (UTC 'HH:MM-HH:MM') ends, 0 outside it."""
    if not pause:
        return 0.0
    now = now or datetime.now(timezone.utc)
    m = now.hour * 60 + now.minute
    a, b = ((int(x[:2]) * 60 + int(x[3:])) for x in pause.split("-"))
    inside = a <= m < b if a <= b else (m >= a or m < b)
    return (((b - m) % 1440) * 60 - now.second) if inside else 0.0


class Previews(threading.Thread):
    """Fetches the previews in the background, one Deezer request at a time, at most one per DEEZER_INTERVAL
    across every process sharing haystack's throttle. The audio stays in memory and is dropped at the end."""

    def __init__(self, dids: list[int], pause: str | None):
        super().__init__(daemon=True)
        self.dids, self.pause = dids, pause
        self.data: dict[int, bytes] = {}
        self.missing: set[int] = set()
        self.cv = threading.Condition()

    def run(self) -> None:
        from haystack_music.deezer import Deezer
        dz = Deezer(min_interval=DEEZER_INTERVAL)
        for did in self.dids:
            while (wait := _in_pause(self.pause)) > 0:
                log.info("pause window: no Deezer requests for %d min", wait // 60 + 1)
                time.sleep(min(wait, 600))
            data = None
            try:
                url = (dz.track(did) or {}).get("preview")
                data = dz.preview_bytes(url) if url else None
            except Exception as e:  # noqa: BLE001
                log.warning("preview %s: %r", did, e)
            with self.cv:
                if data:
                    self.data[did] = data
                else:
                    self.missing.add(did)
                self.cv.notify_all()
        log.info("previews: %d fetched, %d missing, %d Deezer requests", len(self.data), len(self.missing), dz.requests)

    def get(self, did: int) -> bytes | None:
        with self.cv:
            self.cv.wait_for(lambda: did in self.data or did in self.missing)
            return self.data.get(did)


def _decode(path: str, sr: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True, timeout=600).stdout, np.float32).copy()


def load_model(name: str):
    """-> run(audio) -> {variant: vector}, one forward pass per clip."""
    import torch
    spec = MODELS[name]
    sr, repo, rev = spec["sr"], spec["repo"], spec["rev"]

    def layered(hs) -> dict[str, np.ndarray]:
        n = len(hs) - 1                                   # hs[0] is the input to the first layer
        means = [h[0].mean(0) for h in hs]
        mid = torch.stack(means[n // 3: 2 * n // 3 + 1]).mean(0)
        return {name: means[-1].numpy(), f"{name}-mid": mid.numpy()}

    if spec["kind"] == "mert":
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        m = AutoModel.from_pretrained(repo, revision=rev, trust_remote_code=True).eval()
        fe = Wav2Vec2FeatureExtractor.from_pretrained(repo, revision=rev)

        def run(a):
            x = fe(a, sampling_rate=sr, return_tensors="pt")["input_values"]
            return layered(m(input_values=x, output_hidden_states=True).hidden_states)
    elif spec["kind"] == "muq":
        from muq import MuQ
        m = MuQ.from_pretrained(repo, revision=rev).eval()

        def run(a):
            return layered(m(torch.from_numpy(a)[None], output_hidden_states=True).hidden_states)
    else:
        from transformers import ClapModel, ClapProcessor
        m = ClapModel.from_pretrained(repo, revision=rev).eval()
        p = ClapProcessor.from_pretrained(repo, revision=rev)

        def run(a):
            return {name: m.get_audio_features(**p(audio=[a], sampling_rate=sr, return_tensors="pt"))[0].numpy()}

    def timed(a):
        with torch.inference_mode():
            t = time.perf_counter()
            v = run(a)
            return v, time.perf_counter() - t
    return timed


def embed_cmd(args) -> int:
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out).expanduser()
    doc = _tracks(out)
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    for n in names:
        if n not in MODELS:
            print(f"unknown model {n!r} (known: {', '.join(MODELS)})", file=sys.stderr)
            return 2
    stores = {v: Store(out, v) for n in names for v in variants_of(n)}
    dids = list(dict.fromkeys([x["did"] for x in doc["labels"]] + [x["did"] for x in doc["liked_previews"]]
                              + [x["did"] for x in doc["pool"]]))
    need = [d for d in dids if any(f"dz:{d}" not in s.vec for s in stores.values())]
    prev = Previews(need, args.pause)
    if need:
        log.info("fetching %d previews from Deezer in the background (about %d min)", len(need),
                 len(need) * 2 * DEEZER_INTERVAL / 60 + 1)
        prev.start()
    for n in names:
        vs = [stores[v] for v in variants_of(n)]
        todo_files = [f for f in doc["liked_files"]
                      if any(f"local:{f['local_id']}:{w}" not in s.vec for s in vs for w in range(len(f["starts"])))]
        todo_dz = [d for d in dids if any(f"dz:{d}" not in s.vec for s in vs)]
        if not todo_files and not todo_dz:
            continue
        log.info("%s: %d files, %d previews to embed (%d threads)", n, len(todo_files), len(todo_dz),
                 torch.get_num_threads())
        run, sr, done = load_model(n), MODELS[n]["sr"], 0

        def put(key, a):
            nonlocal done
            v, secs = run(a)
            for s in vs:
                s.vec[key], s.secs[key] = v[s.path.stem].astype(np.float32), secs
            done += 1
            if done % 50 == 0:
                for s in vs:
                    s.save()
                log.info("%s: %d clips, median %.2f s per clip", n, done, float(np.median(list(vs[0].secs.values()))))
        for f in todo_files:
            try:
                a = _decode(f["path"], sr)
            except (subprocess.SubprocessError, OSError) as e:
                log.warning("cannot decode local track %s: %r", f["local_id"], e)
                continue
            for w, st in enumerate(f["starts"]):
                put(f"local:{f['local_id']}:{w}", a[int(st * sr): int((st + WIN) * sr)])
        for d in todo_dz:
            data = prev.get(d)
            if data is None:
                continue
            a = embed.decode_bytes(data, sr)[: int(WIN * sr)]
            if len(a) >= MIN_CLIP * sr:
                put(f"dz:{d}", a)
        for s in vs:
            s.save()
        del run
        gc.collect()
    prev.data.clear()
    for v, s in stores.items():
        med = float(np.median(list(s.secs.values()))) if s.secs else 0.0
        print(f"{v}: {len(s.vec)} clips, median {med:.2f} s per clip")
    return 0


# ---- report ---------------------------------------------------------------------------------------------------

def _baselines(doc, cfg) -> dict[str, tuple[dict, dict]]:
    """effnet512 from what haystack stored (no recomputation), and effnet512-aligned when scoring aligns previews."""
    con = db.connect_readonly()
    dids = {x["did"] for k in ("labels", "liked_previews", "pool") for x in doc[k]}
    files = {}
    for f in doc["liked_files"]:
        try:
            for w, v in enumerate(embed.load_windows(f["emb"])):
                files[f"local:{f['local_id']}:{w}"] = v
        except (OSError, KeyError, ValueError):
            pass
    raw = {f"dz:{d}": v for d, v in vectors.get_deezer_vectors(con, dids).items()}
    out = {BASELINE: ({**raw, **files}, {})}
    from haystack_music import align
    if align.active(con, cfg) is not None:
        al = {f"dz:{d}": v for d, v in vectors.preview_vectors(con, cfg, dids).items()}
        out[f"{BASELINE}-aligned"] = ({**al, **files}, {})
    return out


def _measure(doc, vec, cfg, keep_files, keep_dids) -> dict:
    ev = cfg["eval"]
    kt, K = cfg["scoring"]["topk"], ev["recall_k"]
    V = lambda keys: l2norm(np.vstack([vec[k] for k in keys]).astype(np.float32))  # noqa: E731
    prev = [x for x in doc["liked_previews"] if x["did"] in keep_dids]
    files = [f for f in doc["liked_files"] if f["local_id"] in keep_files]
    labels = [x for x in doc["labels"] if x["did"] in keep_dids]
    pool = [x for x in doc["pool"] if x["did"] in keep_dids]
    # positives: one row per liked preview, one per window of each liked file; owners are contiguous runs
    owners = [(x["sid"], set(x["keys"]), [f"dz:{x['did']}"]) for x in prev]
    owners += [(f["sid"], set(f["keys"]), [f"local:{f['local_id']}:{w}" for w in range(len(f["starts"]))])
               for f in files]
    rows = [k for _, _, ks in owners for k in ks]
    starts = np.cumsum([0] + [len(ks) for _, _, ks in owners])[:-1]
    P = V(rows)

    def same(sid, keys):
        return np.array([s == sid or bool(k & keys) for s, k, _ in owners])
    SP = owner_max(V([f"dz:{x['did']}" for x in pool]), P, None, starts)
    SH = owner_max(V([f"dz:{x['did']}" for x in prev]), P, None, starts)
    SL = owner_max(V([f"dz:{x['did']}" for x in labels]), P, None, starts)
    own_h = np.vstack([same(x["sid"], set(x["keys"])) for x in prev])
    own_l = np.vstack([same(x["sid"], set(x["keys"])) for x in labels])
    y = np.array([x["y"] for x in labels])
    nh = max(1, int(round(ev["holdout"] * len(prev))))
    res = {a: dict(auc=[], rec=[], lab=[]) for a in ("raw", "topk_mean")}
    for seed in ev["seeds"]:
        held = random.Random(seed).sample(range(len(prev)), nh)
        keep = ~own_h[held].any(0)                       # the held-out songs leave the positives, files too
        L = SL[:, keep].copy()
        L[own_l[:, keep]] = -np.inf                      # each label leaves out its own song
        for a, f in (("raw", lambda M: M.max(1)), ("topk_mean", lambda M: topk_mean(M, kt))):
            ps, hs, ls = f(SP[:, keep]), f(SH[held][:, keep]), f(L)
            res[a]["auc"].append(evaluate.auc(ls[y == 1], np.ones(int(y.sum())), ls[y == 0]))
            res[a]["rec"].append(float(((ps[None, :] > hs[:, None]).sum(1) < K).mean()))
            res[a]["lab"].append(ls)
    # the domain gap: does a song's preview find its own local file first among the liked files?
    fsid = [f["sid"] for f in files]
    pairs = [(i, fsid.index(x["sid"])) for i, x in enumerate(prev) if x["sid"] in fsid]
    top1 = None
    if pairs:
        fo = len(prev)
        S = owner_max(V([f"dz:{prev[i]['did']}" for i, _ in pairs]), P[starts[fo]:], None, starts[fo:] - starts[fo])
        top1 = float(np.mean([S[r].argmax() == j for r, (_, j) in enumerate(pairs)]))
    return dict(dim=int(P.shape[1]), y=y, n=dict(labels=len(labels), yes=int(y.sum()), previews=len(prev),
                files=len(files), windows=len(rows) - len(prev), pool=len(pool), pairs=len(pairs)),
                auc={a: float(np.mean(r["auc"])) for a, r in res.items()},
                recall=float(np.mean(res["raw"]["rec"])), recall_topk=float(np.mean(res["topk_mean"]["rec"])),
                lab={a: r["lab"] for a, r in res.items()}, top1=top1, K=K, seeds=len(ev["seeds"]), nh=nh)


def _delta_ci(y, a, b, rnd) -> tuple[float, float]:
    """5th and 95th percentile of the AUC gain of a over b (each the mean over the seeds, as reported), over a
    paired bootstrap of the labels: yes and no resampled apart, the same draw for both models and every seed."""
    yi, ni = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    one = np.ones(len(yi))
    d = []
    for _ in range(BOOT):
        ys, ns = rnd.choice(yi, len(yi)), rnd.choice(ni, len(ni))
        d.append(np.mean([evaluate.auc(sa[ys], one, sa[ns]) - evaluate.auc(sb[ys], one, sb[ns])
                          for sa, sb in zip(a, b)]))
    return float(np.percentile(d, 5)), float(np.percentile(d, 95))


def report(args) -> int:
    cfg = config.load()
    out = Path(args.out).expanduser()
    doc = _tracks(out)
    stores = {p.stem: Store(out, p.stem) for p in sorted(out.glob("*.npz")) if not p.stem.endswith(".tmp")}
    found = {**_baselines(doc, cfg), **{v: (s.vec, s.secs) for v, s in stores.items()}}
    # compare like with like: only the tracks every model has a vector for
    keep_dids = {x["did"] for k in ("labels", "liked_previews", "pool") for x in doc[k]}
    keep_files = {f["local_id"] for f in doc["liked_files"]}
    for vec, _ in found.values():
        keep_dids &= {int(k[3:]) for k in vec if k.startswith("dz:")}
        keep_files = {f["local_id"] for f in doc["liked_files"] if f["local_id"] in keep_files
                      and all(f"local:{f['local_id']}:{w}" in vec for w in range(len(f["starts"])))}
    m = {v: _measure(doc, vec, cfg, keep_files, keep_dids) for v, (vec, _) in found.items()}
    b = m[BASELINE]
    n = b["n"]
    rnd = np.random.default_rng(0)
    K = b["K"]
    print(f"Embedding comparison, {datetime.now().astimezone():%Y-%m-%d}: {n['labels']} labels ({n['yes']} yes, "
          f"{n['labels'] - n['yes']} no), positives {n['previews']} liked previews + {n['files']} liked files "
          f"({n['windows']} windows), pool sample {n['pool']}, {b['seeds']} seeds holding out {b['nh']} liked previews "
          f"each, {n['pairs']} preview/file pairs. Tracks selected {doc['created'][:10]}.\n")
    print(f"| model | dim | AUC raw | AUC topk_mean | Δ raw vs {BASELINE} (90 % interval) "
          f"| Δ topk_mean (90 % interval) "
          f"| recall@{K} | self-retrieval top-1 | s per clip | meets the bar |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for v, r in m.items():
        secs = found[v][1]
        spc = float(np.median(list(secs.values()))) if secs else None
        if v.startswith(BASELINE):
            d_raw = d_top = "—"
            bar = "baseline"
        else:
            dr = r["auc"]["raw"] - b["auc"]["raw"]
            dt = r["auc"]["topk_mean"] - b["auc"]["topk_mean"]
            cr = _delta_ci(r["y"], r["lab"]["raw"], b["lab"]["raw"], rnd)
            ct = _delta_ci(r["y"], r["lab"]["topk_mean"], b["lab"]["topk_mean"], rnd)
            d_raw = f"{dr:+.3f} ({cr[0]:+.3f} to {cr[1]:+.3f})"
            d_top = f"{dt:+.3f} ({ct[0]:+.3f} to {ct[1]:+.3f})"
            checks = (("raw", dr >= BAR), ("topk_mean", dt >= BAR), (f"recall@{K}", r["recall"] >= b["recall"]),
                      ("cost", spc is not None and spc <= MAX_SECS))
            why = [w for w, ok in checks if not ok]
            bar = "yes" if not why else "no: " + ", ".join(why)
        top1 = "—" if r["top1"] is None else f"{r['top1']:.2f}"
        print(f"| {v} | {r['dim']} | {r['auc']['raw']:.3f} | {r['auc']['topk_mean']:.3f} | {d_raw} | {d_top} | "
              f"{r['recall']:.3f} | {top1} | {'stored' if spc is None else f'{spc:.2f}'} | {bar} |")
    print(f"\nThe bar: at least +{BAR} AUC on both raw and topk_mean, recall@{K} not below {BASELINE}'s, at most "
          f"{MAX_SECS:.0f} s per 30 s clip. A result counts only if it holds again on a later run with more labels. "
          f"With {n['labels']} labels an AUC's standard error is about "
          f"{np.sqrt(1 / (12 * n['yes']) + 1 / (12 * (n['labels'] - n['yes']))):.2f}.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="~/haystack-experiments", help="folder for tracks.json and the vectors")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="pick the tracks and write tracks.json")
    s.add_argument("--pool", type=int, default=600, help="pool sample size (default 600)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--min-labels", type=int, default=10, help="at least this many yes and no labels (default 10)")
    e = sub.add_parser("embed", help="embed the tracks with one or more models")
    e.add_argument("--models", default="mert-95m,muq-large,clap-music",
                   help=f"comma-separated, from {', '.join(MODELS)} (default mert-95m,muq-large,clap-music)")
    e.add_argument("--pause", help="UTC window with no Deezer requests, e.g. 04:30-07:30: the hours around your "
                   "daily harvest")
    e.add_argument("--threads", type=int, help="torch CPU threads (default: torch's choice)")
    sub.add_parser("report", help="print the comparison table")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return {"select": select, "embed": embed_cmd, "report": report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
