"""Pure window planning for local-file embeddings."""
from __future__ import annotations


def window_starts(duration: float, n: int = 9, win: float = 30.0) -> list[float]:
    """~n windows of `win` seconds, one every ~10 % of the track (starts at 5 %, 15 %, ... 85 %),
    clamped so each window fits; duplicates (short tracks) collapse. Tracks shorter than a window -> [0]."""
    if duration <= win:
        return [0.0]
    out: list[float] = []
    for k in range(n):
        s = round(min((k + 0.5) / (n + 1) * duration, duration - win), 2)
        if not out or s - out[-1] >= 1.0:
            out.append(s)
    return out
