"""Taste sanity check: for N random liked songs, their K nearest library tracks by 512-d cosine
(best window pair), written as markdown to read."""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from . import embed
from .scoring import l2norm
from .vectors import _library_windows, get_deezer_vectors


def _liked_vectors(con) -> list[dict]:
    rows = con.execute("""SELECT m.spotify_id, m.use_local, m.local_id, m.deezer_id, m.vector_status, s.name,
                          s.artists_json, e.file FROM liked_map m JOIN likes l ON l.spotify_id=m.spotify_id
                          JOIN spotify_tracks s ON s.spotify_id=m.spotify_id
                          LEFT JOIN local_tracks t ON t.id=m.local_id
                          LEFT JOIN local_embeddings e ON e.content_hash=t.content_hash AND e.status='ok'
                          WHERE l.removed_at IS NULL""").fetchall()
    out = []
    dz = get_deezer_vectors(con, [r["deezer_id"] for r in rows if not r["use_local"] and r["vector_status"] == "ok"])
    for r in rows:
        v, kind = None, None
        if r["use_local"] and r["file"]:
            v, kind = embed.load_windows(r["file"]), "local file"
        elif not r["use_local"] and r["deezer_id"] in dz:
            v, kind = dz[r["deezer_id"]][None, :], "Deezer preview"
        if v is not None:
            out.append(dict(sid=r["spotify_id"], name=r["name"], artists=", ".join(json.loads(r["artists_json"] or "[]")),
                            vec=v, kind=kind, local_id=r["local_id"] if r["use_local"] else None))
    return out


def write(con, path: Path, n: int = 10, k: int = 10, rng: random.Random | None = None) -> dict:
    rng = rng or random.Random()
    liked = _liked_vectors(con)
    ids, mats = _library_windows(con)
    if not liked or not ids:
        path.write_text("# Taste sanity check\n\nNot enough data yet (no liked vectors or no library embeddings).\n")
        return {"liked_with_vectors": len(liked), "library_tracks": len(ids)}
    lib = l2norm(np.vstack(mats))
    starts = np.cumsum([0] + [len(m) for m in mats])[:-1]
    meta = {r["id"]: r for r in con.execute(
        f"SELECT id, artist_raw, title_raw FROM local_tracks WHERE id IN ({','.join('?' * len(ids))})", ids)}
    picks = rng.sample(liked, min(n, len(liked)))
    lines = [f"# Taste sanity check ({datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')})", "",
             f"{len(picks)} random liked songs (of {len(liked)} with a vector) and their {k} nearest library tracks by "
             f"512-d cosine (best window pair). Library: {len(ids)} embedded tracks. The liked song's own local file "
             "is excluded.", ""]
    for x in picks:
        sims = l2norm(x["vec"]) @ lib.T                       # [q, rows]
        per_track = np.maximum.reduceat(sims.max(0), starts)  # best window pair per library track
        order = [i for i in np.argsort(-per_track) if ids[i] != x["local_id"]][:k]
        lines += [f"## {x['artists']} - {x['name']}", f"vector: {x['kind']} · spotify:track:{x['sid']}", "",
                  "| # | artist | title | cosine |", "|---|---|---|---|"]
        for rank, i in enumerate(order, 1):
            m = meta[ids[i]]
            lines.append(f"| {rank} | {m['artist_raw']} | {m['title_raw']} | {per_track[i]:.4f} |")
        lines.append("")
    path.write_text("\n".join(lines))
    return {"liked_with_vectors": len(liked), "library_tracks": len(ids), "written": str(path)}
