"""Essentia Discogs-EffNet (track variant) embeddings. One forward pass gives both heads:
  PartitionedCall:0 -> 512-d track-similarity head (primary vector)
  PartitionedCall:1 -> 1280-d embedding (kept as float16 per window, cheap)
Patches: 128 mel frames (~2.05 s), hop 62 frames (~0.99 s), 16 kHz mono (same as TensorflowPredictEffnetDiscogs)."""
from __future__ import annotations

import os
import subprocess

import numpy as np

from .windows import window_starts

SR = 16000
PATCH = 128
HOP = 62
BS = 64
FPS = SR / 256.0
FFMPEG = "/usr/bin/ffmpeg"
MODEL_NAME = "discogs_track_embeddings-effnet-bs64-1"

_state: dict = {}


def init(model_path: str, threads: int | None = None) -> None:
    if threads:
        for k in ("TF_NUM_INTRAOP_THREADS", "OMP_NUM_THREADS"):
            os.environ[k] = str(threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    _state["model_path"] = model_path


def _es():
    if "es" not in _state:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import essentia
        essentia.log.infoActive = False
        essentia.log.warningActive = False
        import essentia.standard as es
        _state["essentia"] = essentia
        _state["es"] = es
        _state["mel"] = es.TensorflowInputMusiCNN()
    return _state["es"]


def _model():
    if "pred" not in _state:
        if "model_path" not in _state:          # not initialised by a worker pool: use the configured model
            from . import config
            init(config.load()["embedding"]["model"], threads=2)
        es = _es()
        _state["pred"] = es.TensorflowPredict(graphFilename=_state["model_path"],
                                              inputs=["serving_default_melspectrogram"],
                                              outputs=["PartitionedCall:0", "PartitionedCall:1"])
    return _state["pred"]


STYLE_BACKBONE = "discogs-effnet-bs64-1"            # the official style backbone (its 1280-d embeddings feed the head)
STYLE_HEAD = "genre_discogs400-discogs-effnet-1"     # 400 Discogs styles, sigmoid
# NB: the track-similarity model's 1280-d output is NOT compatible with the genre head (tested: corr ~0 with the
# official pipeline), so styles need their own backbone pass over the same mel patches.


def _model_dir():
    from pathlib import Path
    if "model_path" in _state:
        return Path(_state["model_path"]).parent
    from . import config
    return Path(config.load()["embedding"]["model"]).parent


def _style_models():
    if "style_bb" not in _state:
        if "model_path" not in _state:
            from . import config
            init(config.load()["embedding"]["model"], threads=2)
        es = _es()
        d = _model_dir()
        _state["style_bb"] = es.TensorflowPredict(graphFilename=str(d / f"{STYLE_BACKBONE}.pb"),
                                                  inputs=["serving_default_melspectrogram"],
                                                  outputs=["PartitionedCall:1"])
        _state["style_head"] = es.TensorflowPredict2D(graphFilename=str(d / f"{STYLE_HEAD}.pb"),
                                                      input="serving_default_model_Placeholder",
                                                      output="PartitionedCall:0")
    return _state["style_bb"], _state["style_head"]


def style_classes() -> list[str]:
    import json
    if "style_classes" not in _state:
        _state["style_classes"] = json.loads((_model_dir() / f"{STYLE_HEAD}.json").read_text())["classes"]
    return _state["style_classes"]


def effnet_embeddings(p: np.ndarray) -> np.ndarray:
    """discogs-effnet-bs64-1 backbone embeddings per mel patch [patches, 1280]: the input of the style head and
    of the Trait classifier heads (danceability, acousticness, instrumentalness)."""
    essentia = _state.get("essentia") or (_es() and _state["essentia"])
    bb, _head = _style_models()
    embs = []
    for b in range(0, len(p), BS):
        batch = p[b:b + BS]; k = len(batch)
        if k < BS:
            batch = np.vstack([batch, np.zeros((BS - k, PATCH, 96), np.float32)])
        pool = essentia.Pool()
        pool.set("serving_default_melspectrogram", batch[:, None, :, :])
        embs.append(np.asarray(bb(pool)["PartitionedCall:1"]).reshape(BS, -1)[:k])
    return np.ascontiguousarray(np.vstack(embs), dtype=np.float32)


def style_from_embeddings(emb: np.ndarray) -> np.ndarray:
    """genre_discogs400 head over backbone embeddings -> mean over patches [400]."""
    _bb, head = _style_models()
    act = np.asarray(head(np.ascontiguousarray(emb, dtype=np.float32))).reshape(-1, 400)
    return act.mean(0).astype(np.float32)


def style_from_patches(p: np.ndarray) -> np.ndarray:
    """Official genre_discogs400 pipeline: backbone embeddings per patch -> head -> mean over patches [400]."""
    return style_from_embeddings(effnet_embeddings(p))


def decode_file(path: str) -> np.ndarray:
    cmd = [FFMPEG, "-v", "error", "-nostdin", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True, timeout=600).stdout, dtype=np.float32).copy()


def decode_bytes(data: bytes, sr: int = SR, channels: int = 1) -> np.ndarray:
    """Decode in-memory audio through an ffmpeg pipe: preview audio never touches the disk. Interleaved samples
    when channels > 1."""
    cmd = [FFMPEG, "-v", "error", "-nostdin", "-i", "pipe:0", "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, input=data, capture_output=True, check=True, timeout=120).stdout,
                         dtype=np.float32).copy()


def mel_frames(audio: np.ndarray) -> np.ndarray:
    es = _es(); mel = _state["mel"]
    out = [mel(fr) for fr in es.FrameGenerator(audio, frameSize=512, hopSize=256, startFromZero=True,
                                               validFrameThresholdRatio=1)]
    return np.array(out, dtype=np.float32).reshape(-1, 96)


def patches(m: np.ndarray, hop: int = HOP) -> np.ndarray:
    if len(m) < PATCH:
        m = np.vstack([m, np.zeros((PATCH - len(m), 96), np.float32)])
    return np.stack([m[s:s + PATCH] for s in range(0, len(m) - PATCH + 1, hop)])


def run(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    essentia = _state.get("essentia") or (_es() and _state["essentia"])
    pred = _model(); o0, o1 = [], []
    for b in range(0, len(p), BS):
        batch = p[b:b + BS]; k = len(batch)
        if k < BS:
            batch = np.vstack([batch, np.zeros((BS - k, PATCH, 96), np.float32)])
        pool = essentia.Pool()
        pool.set("serving_default_melspectrogram", batch[:, None, :, :])
        r = pred(pool)
        o0.append(np.asarray(r["PartitionedCall:0"]).reshape(BS, -1)[:k])
        o1.append(np.asarray(r["PartitionedCall:1"]).reshape(BS, -1)[:k])
    return np.vstack(o0), np.vstack(o1)


def embed_track(path: str, n_windows: int = 9, win: float = 30.0) -> dict:
    """Whole-file decode once, mel once, then ~9 x 30 s windows -> per-window mean vectors."""
    a = decode_file(path)
    dur = len(a) / SR
    m = mel_frames(a)
    starts = window_starts(dur, n_windows, win)
    wp = [patches(m[int(round(s * FPS)): int(round((s + win) * FPS))]) for s in starts]
    n = [len(x) for x in wp]
    e5, e12 = run(np.vstack(wp))
    idx = np.cumsum([0] + n)
    w5 = np.stack([e5[idx[i]:idx[i + 1]].mean(0) for i in range(len(n))]).astype(np.float32)
    w12 = np.stack([e12[idx[i]:idx[i + 1]].mean(0) for i in range(len(n))])
    return {"dur": dur, "starts": np.array(starts, np.float32), "win512": w5, "win1280": w12.astype(np.float16),
            "mean1280": e12.mean(0).astype(np.float32), "model": MODEL_NAME}


def embed_audio(audio: np.ndarray, style: bool = False):
    """Embed a short clip (a 30 s preview): -> (mean 512, mean 1280, n_patches[, style 400]).
    The mel patches are computed once and shared by the track model and the style backbone."""
    p = patches(mel_frames(audio))
    e5, e12 = run(p)
    out = (e5.mean(0).astype(np.float32), e12.mean(0).astype(np.float32), len(e5))
    return out + (style_from_patches(p),) if style else out


# ---- local embedding store (one .npz per audio-content hash) ----

def npz_path(emb_dir, content_hash: str):
    from pathlib import Path
    return Path(emb_dir) / "local" / content_hash[:2] / f"{content_hash}.npz"


def save_npz(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **data)
    os.replace(tmp, path)


def load_windows(path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        return z["win512"].astype(np.float32)
