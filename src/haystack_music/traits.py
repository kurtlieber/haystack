"""Traits (ADR 0001): seven Readings per track, measured on the server from the 30 s Deezer preview.

    danceability, acousticness, instrumentalness  classifier heads on the discogs-effnet-bs64-1 embeddings
                                                  (the same backbone pass the genre styles use)
    valence, energy                               deam-msd-musicnn-2 on msd-musicnn-1 embeddings; DEAM's 1-9
                                                  scale mapped to 0-1, energy = arousal
    tempo                                         RhythmExtractor2013 (multifeature), BPM, at 44.1 kHz; Deezer's
                                                  own bpm when it agrees up to an octave (it counts the way people
                                                  do: drum & bass at 170, not 85)
    loudness                                      LoudnessEBUR128 integrated loudness, LUFS, stereo 44.1 kHz

Readings are model estimates from a clip, not Spotify's numbers. They are stored per Deezer track, like the
preview vectors; spotify_deezer maps Spotify tracks the server had never seen (measured on demand)."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import db

log = logging.getLogger("haystack_music.traits")

TRAITS = ("tempo", "valence", "energy", "danceability", "acousticness", "instrumentalness", "loudness")
MODELS_VERSION = "effnet-heads-1+deam-msd-musicnn-2+rhythm2013+ebur128"

BASE = "https://essentia.upf.edu/models/"
# Every Essentia model haystack runs: name -> (path under BASE, the class whose probability is the Reading, or
# None). The first three are the embeddings (embed.py); the Readings need the style backbone too.
MODELS = {
    "discogs_track_embeddings-effnet-bs64-1": ("feature-extractors/discogs-effnet/", None),
    "discogs-effnet-bs64-1": ("feature-extractors/discogs-effnet/", None),
    "genre_discogs400-discogs-effnet-1": ("classification-heads/genre_discogs400/", None),
    "danceability-discogs-effnet-1": ("classification-heads/danceability/", "danceable"),
    "mood_acoustic-discogs-effnet-1": ("classification-heads/mood_acoustic/", "acoustic"),
    "voice_instrumental-discogs-effnet-1": ("classification-heads/voice_instrumental/", "instrumental"),
    "msd-musicnn-1": ("feature-extractors/musicnn/", None),
    "deam-msd-musicnn-2": ("classification-heads/deam/", None),
}
HEADS = {"danceability": "danceability-discogs-effnet-1", "acousticness": "mood_acoustic-discogs-effnet-1",
         "instrumentalness": "voice_instrumental-discogs-effnet-1"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    deezer_id INTEGER PRIMARY KEY,
    tempo REAL, tempo_confidence REAL, valence REAL, energy REAL, danceability REAL, acousticness REAL,
    instrumentalness REAL, loudness REAL, models TEXT, measured_at TEXT
);
"""

_state: dict = {}


# ---------------- pure helpers ----------------

def deam_unit(x: float) -> float:
    """DEAM's 1..9 valence/arousal scale -> 0..1."""
    return float(min(1.0, max(0.0, (float(x) - 1.0) / 8.0)))


def class_index(meta: dict, wanted: str) -> int:
    classes = meta.get("classes") or []
    if wanted not in classes:
        raise ValueError(f"model {meta.get('name')} has no class {wanted!r} (classes: {classes})")
    return classes.index(wanted)


def reconcile_tempo(estimate: float, deezer_bpm) -> float:
    """Our clip estimate, or Deezer's track bpm when it is the same tempo counted at another metrical level
    (x2, x1/2, x3/2, x2/3) or the same within 4%. On 20 liked songs (2026-10-11) both RhythmExtractor2013 and
    TempoCNN matched Deezer within 4% 16 times and up to an octave 20 times; the misses were octave choices."""
    try:
        hint = float(deezer_bpm or 0)
    except (TypeError, ValueError):
        return estimate
    if hint <= 0 or not estimate:
        return estimate
    for f in (1.0, 2.0, 0.5, 1.5, 2 / 3):
        if abs(estimate * f - hint) <= 0.04 * hint:
            return hint
    return estimate


def finite(readings: dict) -> dict:
    """Drop NaN/inf Readings (a silent clip has no loudness): a missing Reading beats a wrong one."""
    out = {}
    for k, v in readings.items():
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        out[k] = v
    return out


# ---------------- models ----------------

def model_dir() -> Path:
    from . import embed
    return embed._model_dir()


def missing_models(d: Path | None = None) -> list[str]:
    d = d or model_dir()
    return [n for n in MODELS if not ((d / f"{n}.pb").exists() and (d / f"{n}.json").exists())]


def fetch_models(d: Path | None = None, session=None) -> list[str]:
    """Download the models that are missing (Essentia's published files, metadata included)."""
    import requests
    d = d or model_dir()
    d.mkdir(parents=True, exist_ok=True)
    s = session or requests.Session()
    got = []
    for name in missing_models(d):
        sub, wanted = MODELS[name]
        for ext in ("json", "pb"):
            r = s.get(f"{BASE}{sub}{name}.{ext}", timeout=120)
            r.raise_for_status()
            tmp = d / f".{name}.{ext}.part"
            tmp.write_bytes(r.content)
            os.replace(tmp, d / f"{name}.{ext}")
        if wanted:
            class_index(json.loads((d / f"{name}.json").read_text()), wanted)    # the class order we rely on
        got.append(name)
    return got


def _models():
    if "heads" not in _state:
        from . import embed
        es = embed._es()
        d = model_dir()
        if missing_models(d):
            raise FileNotFoundError(f"Essentia models missing in {d}: {missing_models(d)} "
                                    "(run `haystack models fetch`)")
        heads = {}
        for trait, name in HEADS.items():
            meta = json.loads((d / f"{name}.json").read_text())
            heads[trait] = (es.TensorflowPredict2D(graphFilename=str(d / f"{name}.pb"), input="model/Placeholder",
                                                   output="model/Softmax"), class_index(meta, MODELS[name][1]))
        _state["heads"] = heads
        _state["musicnn"] = es.TensorflowPredictMusiCNN(graphFilename=str(d / "msd-musicnn-1.pb"),
                                                         output="model/dense/BiasAdd")
        _state["deam"] = es.TensorflowPredict2D(graphFilename=str(d / "deam-msd-musicnn-2.pb"),
                                                input="model/Placeholder", output="model/Identity")
        _state["rhythm"] = es.RhythmExtractor2013(method="multifeature")
        _state["loudness"] = es.LoudnessEBUR128(sampleRate=44100.0)
    return _state


# ---------------- measuring ----------------

def measure(audio16: np.ndarray, stereo44: np.ndarray, backbone: np.ndarray, bpm_hint=None) -> dict:
    """The seven Readings of a clip. audio16: mono 16 kHz; stereo44: [n, 2] at 44.1 kHz; backbone: the
    discogs-effnet-bs64-1 patch embeddings [patches, 1280] the styles already computed."""
    m = _models()
    out = {}
    emb = np.ascontiguousarray(backbone, dtype=np.float32)
    for trait, (head, idx) in m["heads"].items():
        out[trait] = float(np.asarray(head(emb)).reshape(len(emb), -1)[:, idx].mean())
    mus = np.asarray(m["musicnn"](np.ascontiguousarray(audio16, dtype=np.float32)))
    va = np.asarray(m["deam"](np.ascontiguousarray(mus.reshape(-1, 200), dtype=np.float32))).reshape(-1, 2).mean(0)
    out["valence"], out["energy"] = deam_unit(va[0]), deam_unit(va[1])
    mono = np.ascontiguousarray(stereo44.mean(axis=1), dtype=np.float32)
    bpm, _ticks, confidence, _est, _iv = m["rhythm"](mono)
    out["tempo"], out["tempo_confidence"] = reconcile_tempo(float(bpm), bpm_hint), float(confidence)
    _mom, _short, integrated, _range = m["loudness"](np.ascontiguousarray(stereo44, dtype=np.float32))
    out["loudness"] = float(integrated)
    return finite(out)


@dataclass
class Analysis:
    """Everything the server keeps from one preview, from one download and one decode."""
    e512: np.ndarray
    e1280: np.ndarray
    n_patches: int
    seconds: float
    style: np.ndarray | None
    readings: dict | None


def analyse_bytes(data: bytes, style: bool = True, readings: bool = True, bpm_hint=None) -> Analysis:
    from . import embed
    audio = embed.decode_bytes(data)
    stereo = embed.decode_bytes(data, sr=44100, channels=2).reshape(-1, 2) if readings else None
    p = embed.patches(embed.mel_frames(audio))
    e5, e12 = embed.run(p)
    backbone = embed.effnet_embeddings(p) if (style or readings) else None
    sty = embed.style_from_embeddings(backbone) if style else None
    r = measure(audio, stereo, backbone, bpm_hint) if readings else None
    return Analysis(e5.mean(0).astype(np.float32), e12.mean(0).astype(np.float32), len(e5), len(audio) / embed.SR,
                    sty, r)


def analyse_preview(dz, url: str, style: bool = True, readings: bool = True, bpm_hint=None) -> Analysis:
    """Fetch a preview into memory (never to disk) and analyse it; the bytes go out of scope at once.
    bpm_hint: the Deezer track's own bpm, when the track object has one."""
    data = dz.preview_bytes(url)
    try:
        return analyse_bytes(data, style, readings, bpm_hint)
    finally:
        del data


# ---------------- storage ----------------

def ensure(con) -> None:
    con.executescript(SCHEMA)


def put(con, deezer_id: int, r: dict) -> None:
    con.execute(f"""INSERT INTO readings(deezer_id, {', '.join(TRAITS)}, tempo_confidence, models, measured_at)
                    VALUES(?, {', '.join('?' * len(TRAITS))}, ?, ?, ?) ON CONFLICT(deezer_id) DO UPDATE SET
                    {', '.join(f'{t}=excluded.{t}' for t in TRAITS)}, tempo_confidence=excluded.tempo_confidence,
                    models=excluded.models, measured_at=excluded.measured_at""",
                (int(deezer_id), *[r.get(t) for t in TRAITS], r.get("tempo_confidence"), MODELS_VERSION, db.now()))


def get(con, deezer_ids) -> dict[int, dict]:
    ids = [int(i) for i in deezer_ids if i is not None]
    out = {}
    for s in range(0, len(ids), 500):
        ch = ids[s:s + 500]
        for r in con.execute(f"SELECT * FROM readings WHERE deezer_id IN ({','.join('?' * len(ch))})", ch):
            out[r["deezer_id"]] = {t: r[t] for t in TRAITS if r[t] is not None}
    return out


def store(con, deezer_id: int, a: Analysis) -> None:
    """Keep a preview's vectors, styles and Readings."""
    from . import vectors
    vectors.put_deezer_vector(con, deezer_id, a.e512, a.e1280, a.n_patches, a.seconds, a.style)
    if a.readings:
        put(con, deezer_id, a.readings)


# ---------------- coverage ----------------

def backfill_order(con, limit: int) -> list[int]:
    """Stored previews without Readings, most needed first: Discoveries (the latest batch and any open offer),
    liked songs (the Baseline and Rate your likes), Rated tracks, then the pool by score."""
    rows = con.execute("""
        SELECT deezer_id, MIN(pri) FROM (
          SELECT c.deezer_id, 0 AS pri FROM offers o JOIN candidates c ON c.id=o.candidate_id
            WHERE o.removed_at IS NULL OR o.batch_id=(SELECT MAX(id) FROM batches)
          UNION ALL SELECT m.deezer_id, 1 FROM liked_map m JOIN likes l ON l.spotify_id=m.spotify_id
            WHERE l.removed_at IS NULL AND m.vector_status='ok'
          UNION ALL SELECT c.deezer_id, 2 FROM candidates c JOIN verdicts v ON v.spotify_id=c.spotify_id
            WHERE c.vector_status='ok'
          UNION ALL SELECT d.deezer_id, 2 FROM spotify_deezer d WHERE d.status='ok'
          UNION ALL SELECT c.deezer_id, 3 - COALESCE(c.score, 0) FROM candidates c
            WHERE c.state='pool' AND c.vector_status='ok')
        WHERE deezer_id IS NOT NULL AND deezer_id IN (SELECT deezer_id FROM deezer_vectors)
          AND deezer_id NOT IN (SELECT deezer_id FROM readings)
        GROUP BY deezer_id ORDER BY MIN(pri), deezer_id LIMIT ?""", (limit,)).fetchall()
    return [r[0] for r in rows]


def backfill(con, dz, limit: int = 150, analyse=None) -> dict:
    """Readings (and any missing style vector) for stored previews, in backfill_order. Needs a fresh preview
    URL (/track/{id}) plus the preview itself: two throttled Deezer requests per track."""
    analyse = analyse or analyse_preview
    ids = backfill_order(con, limit)
    c = dict(todo=len(ids), done=0, no_preview=0, errors=0)
    for did in ids:
        try:
            t = dz.track(did) or {}
            if not t.get("preview"):
                c["no_preview"] += 1
                put(con, did, {})                 # measured: nothing to measure (keeps it out of the queue)
                continue
            store(con, did, analyse(dz, t["preview"], bpm_hint=t.get("bpm")))
            c["done"] += 1
        except Exception as e:  # noqa: BLE001
            c["errors"] += 1
            log.warning("readings for deezer %s failed: %r", did, e)
    return c


def find_deezer(dz, ref, tol: float = 3.0) -> tuple[dict | None, bool, str | None]:
    """The Deezer release of a Spotify track that has a preview: ISRC first, then a text search with the
    artist/title/duration/version check. -> (deezer track or None, whether any release matched, method)."""
    from .matching import TrackRef, compare
    from .textnorm import split_title

    def dzref(t):
        return TrackRef(t.get("id"), (t.get("artist") or {}).get("name", ""), t.get("title", ""), t.get("duration"))

    found_any = False
    if ref.isrc:
        t = dz.track_by_isrc(ref.isrc)
        if t and t.get("id") and compare(ref, dzref(t), tol):
            found_any = True
            if t.get("preview"):
                return t, True, "isrc"
    for cand in dz.search(f"{ref.artist} {split_title(ref.title)[0]}", 10):
        m = compare(ref, dzref(cand), tol)
        if m and m.confidence in ("high", "medium"):
            found_any = True
            if cand.get("preview"):
                return cand, True, "search"
    return None, found_any, None


def measure_track(con, cfg: dict, spotify_id: str, sp=None, dz=None, analyse=None) -> dict:
    """On demand: Readings for any Spotify track. -> {"status": ok | no_preview | not_found, "deezer_id"}."""
    from .deezer import Deezer
    from .sync import spotify_ref, upsert_track
    from .vectors import track_vectors
    analyse = analyse or analyse_preview
    known = track_vectors(con, [spotify_id]).get(spotify_id)
    dz = dz or Deezer(cfg["harvest"]["deezer_min_interval"])
    if known:
        did = known["deezer_id"]
        if not get(con, [did]):
            t = dz.track(did) or {}
            if not t.get("preview"):
                put(con, did, {})
                return {"status": "no_preview", "deezer_id": did}
            store(con, did, analyse(dz, t["preview"], bpm_hint=t.get("bpm")))
        return {"status": "ok", "deezer_id": did}
    ref = spotify_ref(con, spotify_id)
    if ref is None:
        if sp is None:
            from .spotify import Spotify
            sp = Spotify(cfg)
        upsert_track(con, sp.track(spotify_id))
        ref = spotify_ref(con, spotify_id)
    t, found_any, method = find_deezer(dz, ref, cfg["matching"]["duration_tolerance"])
    status = "ok" if t else ("no_preview" if found_any else "not_found")
    if t:
        store(con, t["id"], analyse(dz, t["preview"], bpm_hint=t.get("bpm")))
    con.execute("""INSERT INTO spotify_deezer(spotify_id, deezer_id, status, method, checked_at) VALUES(?,?,?,?,?)
                   ON CONFLICT(spotify_id) DO UPDATE SET deezer_id=excluded.deezer_id, status=excluded.status,
                   method=excluded.method, checked_at=excluded.checked_at""",
                (spotify_id, t["id"] if t else None, status, method, db.now()))
    return {"status": status, "deezer_id": t["id"] if t else None}
