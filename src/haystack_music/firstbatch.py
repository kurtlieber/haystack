"""The one-time FIRST real batch, triggered by OnSuccess= of the library unit once embedding is complete.
Before the batch it runs the offline scoring eval and (unless scoring.variant is pinned) stores the winning
variant; every later batch re-checks it (ADR 0009). The batch itself clears Discoveries first. Runs only once:
on success it writes first-batch.done (the unit has ConditionPathExists=! on it)."""
from __future__ import annotations

import logging

from . import batch, config, db, evaluate, sanity, status, vectors

log = logging.getLogger("haystack_music.firstbatch")
DONE = "first-batch.done"


def run(con, cfg: dict, clear_min_age_hours: float | None = None, clear_recent: str | None = None,
        no_penalty: bool | None = None) -> dict:
    done_file = config.STATE_DIR / DONE
    if done_file.exists():
        return {"skipped": True, "reason": f"first batch already done ({done_file.read_text().strip()})"}
    ok, msg, prog = batch.embedding_progress(con, cfg)
    if not ok:
        log.warning("first batch waiting: %s", msg)
        return {"skipped": True, "reason": msg, **prog}
    path = config.STATE_DIR / "sanity-check.md"
    out = {"sanity_check": sanity.write(con, path)}
    try:
        ev = choose(con, cfg)
    except Exception as e:  # noqa: BLE001  (the eval must never block the first batch)
        log.exception("scoring eval failed")
        ev = {"skipped": True, "chosen": None, "reason": f"eval failed: {e!r}", "used": vectors.active_variant(con, cfg)}
        status.update(None, scoring_eval=ev)
    out["eval"] = ev
    with path.open("a") as f:
        f.write("\n" + evaluate.markdown(ev, "Scoring variant eval (first batch)"))
    out["batch"] = batch.run(con, cfg, variant=ev["used"], clear_min_age_hours=clear_min_age_hours,
                              clear_recent=clear_recent, no_penalty=no_penalty)
    if out["batch"].get("picked"):
        done_file.write_text(f"{db.now()} batch {out['batch'].get('batch_id')} ({out['batch']['picked']} tracks)\n")
        status.notify("Spotify discoveries ready", f"{out['batch']['picked']} new tracks in your Discoveries playlist.")
    return out


def choose(con, cfg: dict) -> dict:
    """Run the offline eval; unless scoring.variant is pinned, its choice becomes the variant (evaluate.choose).
    Records the result in eval_runs and status.json (scoring_eval)."""
    return evaluate.choose(con, cfg, "set")
