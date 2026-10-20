"""haystack: personal Spotify music discovery (see README.md)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import traceback
from pathlib import Path

from . import config, db, status

log = logging.getLogger("haystack_music")
EXIT_NEEDS_AUTH = 3


def _logging(log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)


def job(name: str, fn, needs_spotify: bool = False) -> int:
    """Run a job, record it in the runs table and status.json. Spotify jobs fail fast without auth."""
    from .spotify import AuthMissing
    con = db.connect()
    started = db.now(); t0 = time.time()
    status.job(name, last_started=started, running=True)
    try:
        summary = fn(con)
        skipped = isinstance(summary, dict) and summary.get("skipped")
        status.job(name, last_finished=db.now(), running=False, ok=True, error=None,
                   seconds=round(time.time() - t0, 1), summary=summary,
                   message=("skipped: " + str(summary.get("reason"))) if skipped else None)
        con.execute("INSERT INTO runs(job, started_at, finished_at, ok, summary) VALUES(?,?,?,1,?)",
                    (name, started, db.now(), json.dumps(summary, default=str)))
        log.info("%s done: %s", name, json.dumps(summary, default=str))
        if needs_spotify:
            status.update("auth", {"ok": True, "message": "authorised"})
        _like_rate(con)
        _gates(con)
        return 0
    except AuthMissing as e:
        msg = f"{e}"
        log.error("%s: %s", name, msg)
        status.job(name, last_finished=db.now(), running=False, ok=False, error=msg)
        status.update("auth", {"ok": False, "message": "run `haystack auth`"})
        _auth_alert(con, name, msg)
        con.execute("INSERT INTO runs(job, started_at, finished_at, ok, error) VALUES(?,?,?,0,?)",
                    (name, started, db.now(), msg))
        return EXIT_NEEDS_AUTH
    except Exception as e:  # noqa: BLE001
        log.error("%s failed: %r\n%s", name, e, traceback.format_exc())
        status.job(name, last_finished=db.now(), running=False, ok=False, error=repr(e)[:500])
        con.execute("INSERT INTO runs(job, started_at, finished_at, ok, error) VALUES(?,?,?,0,?)",
                    (name, started, db.now(), repr(e)[:2000]))
        return 1


def _auth_alert(con, name: str, msg: str) -> None:
    from . import alert
    if alert.once_a_day(con, "auth_fail_alerted_on"):
        from .auth import reauth_hint
        cfg = config.load()
        alert.send(cfg, "auth", f"auth.ok=false ({name}: {msg}). {reauth_hint(cfg)}", "fail", EXIT_NEEDS_AUTH)


def cmd_alert_failure(cfg, a) -> int:
    from . import alert
    alert.unit_failure(cfg, a.unit)
    return 0


def _like_rate(con) -> None:
    from .scoring import like_rates
    rows = [dict(r) for r in con.execute("SELECT source, outcome, played FROM offers")]
    status.update(None, like_rate=like_rates(rows))


def _gates(con) -> None:
    """status.json gates (report.gates): never fails the job."""
    try:
        from .report import gates
        status.update(None, gates=gates(con, config.load()))
    except Exception as e:  # noqa: BLE001
        log.warning("gates not written: %r", e)


def _have_auth(cfg) -> bool:
    try:
        from .secrets import SpotifySecrets
        tok, _ = SpotifySecrets(cfg).refresh_token()
        return bool(tok)
    except Exception:  # noqa: BLE001
        return False


# ---------------- commands ----------------

def cmd_auth(cfg, a) -> int:
    from . import auth
    return auth.run(cfg, timeout=a.timeout, ready_file=a.ready_file, log_file=a.log_file)


def cmd_library(cfg, a) -> int:
    from . import library

    def scan(con):
        return library.scan(con, cfg)

    def emb(con):
        return library.embed_pending(con, cfg, workers=a.workers, limit=a.limit)

    def update(con):
        out = {"scan": library.scan(con, cfg)}
        log.info("scan: %s", out["scan"])
        out["embed"] = library.embed_pending(con, cfg, workers=a.workers, limit=a.limit)
        try:
            from . import align
            out["align"] = align.refit(con, cfg)
        except Exception as e:  # noqa: BLE001
            log.exception("preview alignment refit failed; the stored map stays")
            out["align"] = {"error": repr(e)}
        if _have_auth(cfg):
            from .spotify import Spotify
            from .sync import torip_check
            out["to_rip"] = torip_check(con, cfg, Spotify(cfg))
        else:
            out["to_rip"] = "skipped: Spotify not authorised yet"
        return out
    return job("library", {"scan": scan, "embed": emb, "update": update}[a.action])


def cmd_harvest(cfg, a) -> int:
    import random
    from . import vectors
    from datetime import date
    from .harvest import Harvester, broad_liked_seeds, stratified_seeds

    def run(con):
        seeds = liked = None
        out = {}
        if a.broad:
            # the week's seeds: the same for every shard started this ISO week, different from last week's
            year, week, _ = date.today().isocalendar()
            rng = random.Random(a.seed_rng or f"{cfg['harvest']['broad_seed_rng']}:{year}-W{week:02d}")
            seeds = stratified_seeds(con, a.broad, rng)
            liked = broad_liked_seeds(con, cfg, rng)
            buckets = {}
            for x in seeds:
                buckets[x["bucket"]] = buckets.get(x["bucket"], 0) + 1
            out["seed_buckets"] = dict(sorted(buckets.items(), key=lambda kv: -kv[1]))
            if a.shard:
                i, n = (int(x) for x in a.shard.split("/"))
                seeds, liked = seeds[i::n], liked[i::n]
            out["seeds"] = len(seeds); out["liked_seeds"] = len(liked)
        if a.per_seed:
            for k in ("broad_radio_per_seed", "broad_related_per_seed", "broad_lb_per_seed"):
                cfg["harvest"][k] = a.per_seed
        h = Harvester(con, cfg, seeds=seeds, liked_seeds=liked)
        out.update(h.run(a.sources.split(",") if a.sources else None, a.limit_per_source, retry=not a.no_retry))
        if a.styles:
            out["style_backfill_extra"] = h.backfill_styles(a.styles)
        if not a.no_score:
            out["scoring"] = vectors.score_candidates(con, cfg)
        out["pool"] = con.execute("SELECT COUNT(*) FROM candidates WHERE state='pool' AND vector_status='ok'").fetchone()[0]
        status.update("candidates", _candidate_stats(con))
        return out
    name = "harvest_broad" if a.broad else "harvest"
    return job(name + (f"-{a.shard.replace('/', 'of')}" if a.shard else ""), run)


def cmd_score(cfg, a) -> int:
    from . import vectors
    return job("score", lambda con: vectors.score_candidates(con, cfg))


def cmd_sync(cfg, a) -> int:
    from . import sync
    return job("sync", lambda con: sync.run(con, cfg), needs_spotify=True)


def _print_clear(c: dict) -> None:
    print(f"clear: items {c.get('items', 0)} tracks {c.get('tracks', 0)} liked {c.get('liked', 0)} "
          f"no {c.get('no', 0)} unplayed {c.get('unplayed', 0)} recent_removed {c.get('recent_removed', 0)} "
          f"recent_kept {c.get('recent_kept', 0)} removed {c.get('removed', 0)} feedback {c.get('feedback', 0)} "
          f"reconciled {c.get('reconciled', 0)} dry_run {bool(c.get('dry_run'))} "
          f"min_age_hours {c.get('min_age_hours')} recent {c.get('recent')} no_penalty {bool(c.get('no_penalty'))}")


def _readonly_db():
    try:
        return db.connect_readonly()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return None


def cmd_batch(cfg, a) -> int:
    from . import batch
    if a.plan:
        return _batch_plan(cfg, a)
    if a.stage1:
        print("--stage1 goes with --plan", file=sys.stderr)
        return 2
    def run(con):
        out = batch.run(con, cfg, size=a.size, dry_run=a.dry_run, force=a.force, variant=a.variant,
                        clear_min_age_hours=a.clear_min_age_hours, clear_recent=a.clear_recent,
                        no_penalty=a.no_penalty)
        if out.get("clear"):
            _print_clear(out["clear"])
        print(f"variant: {out.get('variant')}{_choice_note(out.get('variant_choice'))}")
        if a.dry_run or out.get("tracks"):
            for t in out.get("tracks", []):
                print(f"  {t['score']:.4f}  {'explore' if t['exploration'] else 'top    '}  {t['source']:22} "
                      f"{t['artist']} - {t['title']}")
        return out
    return job("batch", run, needs_spotify=True)


def _choice_note(c: dict | None) -> str:
    if not c:
        return ""
    if c.get("error"):
        return f" (the variant eval failed: {c['error']})"
    auc = "" if c.get("auc_used") is None else f", AUC {c['auc_used']:.3f}"
    return f" (eval by {c.get('objective_used') or '-'}{auc}; {c.get('decision')})"


def _batch_plan(cfg, a) -> int:
    """batch --plan: no Spotify session, no writes, no runs row (a read-only database connection)."""
    from . import batch
    con = _readonly_db()
    if con is None:
        return 1
    out = batch.plan(con, cfg, size=a.size, variant=a.variant, classes=_style_classes(), stage1=a.stage1)
    if out["embedding_note"]:
        print(f"note: {out['embedding_note']} (the weekly batch skips unless --force)")
    print(f"plan: variant {out['variant']}{_choice_note(out['variant_choice'])}")
    print(f"plan: {out['positives']} positive rows, {out['pool']} scored in the pool; "
          f"{out['picked']} picks, {out['exploration']} exploration, {out['unmapped']} unmapped")
    print(f"plan: exploration share {out['exploration_fraction_used']:.2f} ({out['exploration_share']['why']})")
    rr = out["rerank"]
    print(f"plan: {'re-ranked' if rr['used'] else 'stage 1 only'} ({rr['reason']})")
    s = out["spread"]
    print(f"plan: at most {s['max_same_style']} of {s['top_picks']} top picks share a style, "
          f"at most {s['max_same_nearest']} picks share a nearest liked song")
    for t in out["tracks"]:
        sl = t["sounds_like"]
        near = f"{sl['artist']} - {sl['title']}" if sl else (t["nearest"] or "-")
        pct = "" if t["percentile"] is None else f"p{t['percentile']:.1f}  "
        print(f"  {t['score']:.4f}  {pct}{'explore' if t['exploration'] else 'top    '}  {t['source']:22} "
              f"{t['artist']} - {t['title']}{'' if t['spotify_id'] else '  [unmapped]'}  "
              f"(style: {t['style'] or '-'}; nearest: {near})")
        if t["confidence"] is not None:
            held = f"; held back: {', '.join(t['held_back'])}" if t["held_back"] else ""
            why = f": {', '.join(t['reasons'])}" if t["reasons"] else ""
            print(f"           chance {t['confidence']:.2f}{why}{held}")
    return 0


def cmd_rerank(cfg, a) -> int:
    from . import rerank, vectors
    if a.action == "show":
        con = _readonly_db()
        if con is None:
            return 1
        try:
            m, runs = rerank.last(con), rerank.runs(con, a.limit)
        except sqlite3.OperationalError:          # a database from before 0.3.0: no re-ranker yet
            m, runs = None, []
        if m is None:
            print("no re-ranker fit yet (the next batch or `haystack rerank fit` makes one)")
            return 0
        print(_rerank_line(m))
        if m.get("beta"):
            w = sorted(zip(m["features"], m["beta"][1:]), key=lambda x: -abs(x[1]))
            print("weights (per pool sd, largest first): " + ", ".join(f"{f} {b:+.2f}" for f, b in w[:a.limit]))
        for r in runs:
            print(f"  {r['at']}  {r['trigger']:5}  {r['variant']:12}  {'used' if r['used'] else 'off '}  "
                  f"{r['n_pos']}+/{r['n_neg']}-  AUC {r['loo_auc_model']} vs {r['loo_auc_stage1']}  {r['reason']}")
        return 0

    def run(con):
        variant = vectors.active_variant(con, cfg)
        r = rerank.run(con, cfg, variant, vectors.compute_scores(con, cfg, variant), "fit")
        rerank.record(con, r)
        print(_rerank_line(r))
        return rerank.summary(r)
    return job("rerank", run)


def cmd_align(cfg, a) -> int:
    from . import align
    if a.action == "show":
        con = _readonly_db()
        if con is None:
            return 1
        try:
            s = align.show(con, cfg)
        except sqlite3.OperationalError:          # a database from before the library tables
            print("no library yet: no pairs and no alignment")
            return 0
        m = s["map"]
        print(f"scoring.align = {s['mode']}; {s['pairs_now']} pairs now; "
              + ("the map is used" if s["used"] else "scoring uses the preview vectors as they are"))
        if m is None:
            print("no map (the library job fits one once there are enough pairs, or `haystack align fit`)")
            return 0
        print(_align_line(m))
        if s["gate"]:
            print(f"last eval: {s['gate']['why']} ({s['gate']['at']})")
        for t in s["tried"][:a.limit] if a.limit else s["tried"]:
            lam = "-" if t["lam"] is None else t["lam"]
            print(f"  {t['target']:4}  {t['method']:10}  lambda {lam!s:6}  top1 {t['top1']:.3f}  "
                  f"top{align.TOP} {t[f'top{align.TOP}']:.3f}")
        return 0

    def run(con):
        r = align.fit(con, cfg)
        print(r["reason"])
        return r
    return job("align", run)


def _align_line(m: dict) -> str:
    lam = "" if m["method"] != "ridge" else (" (mean shift)" if m["lam"] is None else f" lambda {m['lam']}")
    u, cv = m["unaligned"], m["cv"]
    return (f"{m['method']}{lam}, target the {m['target']} window, {m['n_pairs']} pairs, {m['version']}: "
            f"self-retrieval top 1 {u['top1']:.3f} -> {cv['top1']:.3f}, top 10 {u['top10']:.3f} -> {cv['top10']:.3f} "
            f"(cross-validated){'' if m['improved'] else '; not used: no better than unaligned'}")


def _rerank_line(r: dict) -> str:
    auc = ("" if r.get("loo_auc_model") is None else
           f"; leave-one-out AUC {r['loo_auc_model']:.3f} vs stage 1 {r['loo_auc_stage1']:.3f}")
    soft = f", {r['n_soft']} soft" if r.get("n_soft") else ""
    return (f"re-ranker ({r.get('variant')}): {'used' if r.get('used') else 'not used'}, {r.get('n_labels')} labels "
            f"({r.get('n_pos')} yes, {r.get('n_neg')} no{soft}){auc}" + ("" if r.get("used") else f". {r.get('reason')}"))


def _style_classes() -> list[str]:
    try:
        from . import embed
        return embed.style_classes()
    except Exception:  # noqa: BLE001 - no model metadata yet: styles print by number
        return []


def cmd_report(cfg, a) -> int:
    from . import report
    con = _readonly_db()
    if con is None:
        return 1
    r = report.build(con, cfg, status.read(), _style_classes(), tracks=a.tracks)
    print(json.dumps(r, indent=1, default=str) if a.json else report.text(r), end="\n" if a.json else "")
    return 0


def cmd_clear(cfg, a) -> int:
    from .clear import clear_discoveries

    def run(con):
        out = clear_discoveries(con, cfg, min_age_hours=a.clear_min_age_hours, recent=a.clear_recent,
                                no_penalty=a.no_penalty, dry_run=a.dry_run)
        _print_clear(out)
        return out
    return job("clear_discoveries", run, needs_spotify=True)


def cmd_first_batch(cfg, a) -> int:
    from . import firstbatch
    return job("first_batch", lambda con: firstbatch.run(
        con, cfg, clear_min_age_hours=a.clear_min_age_hours, clear_recent=a.clear_recent,
        no_penalty=a.no_penalty), needs_spotify=True)


def cmd_eval(cfg, a) -> int:
    from . import evaluate
    if a.seeds:
        cfg["eval"]["seeds"] = [int(x) for x in a.seeds.split(",")]
    if a.objective:
        cfg["eval"]["objective"] = a.objective

    def run(con):
        ev = evaluate.choose(con, cfg, "set" if a.choose else "report")
        print(evaluate.markdown(ev))
        if cfg["scoring"]["verdicts"].get("unrated_like_plays"):
            print(_eval_without_plays(con, cfg, ev))
        return ev
    return job("eval", run)


def _eval_without_plays(con, cfg, ev: dict) -> str:
    """The eval again with Unrated likes at a flat unrated_like_stars (ADR 0016): the effect of grading them by
    plays, per variant. Shown for one release (0.5.0); nothing is written."""
    import copy
    from . import evaluate
    off = copy.deepcopy(cfg)
    off["scoring"]["verdicts"]["unrated_like_plays"] = False
    if ev.get("align"):
        off = evaluate._with_align(off, "candidate" if ev["align"]["gate"]["ok"] else "off")
    ev_off = evaluate.run(con, off)
    k = ev.get("recall_k", 15)
    out = ["", "Unrated likes graded by plays (scoring.verdicts.unrated_like_plays), with and without:",
           f"  {'variant':14} {'AUC with':>9} {'without':>8} {'mean pct with':>14} {'without':>8}"]
    for v, x in (ev.get("variants") or {}).items():
        y = (ev_off.get("variants") or {}).get(v) or {}
        f = (lambda n, fmt: "-" if n is None else format(n, fmt))
        out.append(f"  {v:14} {f(x.get('auc'), '.3f'):>9} {f(y.get('auc'), '.3f'):>8} "
                   f"{f(x.get('mean_pct'), '.1f'):>14} {f(y.get('mean_pct'), '.1f'):>8}")
    out.append(f"  chosen with: {ev.get('chosen')}; without: {ev_off.get('chosen')} (recall@{k} in the table above)")
    return "\n".join(out)


def cmd_calibrate(cfg, a) -> int:
    """The Calibrate queue (ADR 0015): plan prints what the next top-up would pick (token-free, no writes); top-up
    picks and maps them now (needs auth; the hourly sync does the same)."""
    from . import calibrate
    if a.action == "plan":
        con = _readonly_db()
        if con is None:
            return 1
        out = calibrate.plan(con, cfg, n=a.n)
        rr = out["rerank"] or {}
        print(f"calibrate: {'on' if out['enabled'] else 'off (calibrate.enabled)'}; week {out['week']}: "
              f"{out['this_week']} picked, {out['outstanding']} open, the queue takes {out['need']} now")
        print(f"calibrate: variant {out['variant']}, {out['scored']} scored, {out['eligible']} eligible; re-ranker "
              f"{'fitted' if out['fitted'] else 'not fitted'}"
              f" ({rr.get('n_labels', 0)} labels, {rr.get('n_soft', 0)} soft)")
        print("calibrate: by band " + (", ".join(f"{b} {n}" for b, n in sorted(out["by_band"].items())) or "-"))
        for p in out["picks"]:
            ch = "-" if p["rerank_p"] is None else f"{p['rerank_p']:.2f}"
            dis = "-" if p["disagreement"] is None else f"{p['disagreement']:.1f}"
            name = f"  {p['artist']} - {p['title']}" if a.tracks else ""
            print(f"  {p['band']:7} p{p['percentile']:5.1f}  chance {ch:>4}  disagree {dis:>5}  {p['source']:22}"
                  f"{name}  ({p['reason']})")
        return 0

    def run(con):
        from .spotify import Spotify
        return calibrate.top_up(con, cfg, Spotify(cfg))
    return job("calibrate", run, needs_spotify=True)


def cmd_feedback(cfg, a) -> int:
    """Old yes/no command, kept as an alias of `verdict` on candidates."""
    from . import feedback, verdicts

    def run(con):
        out = {}
        if a.verdict in ("yes", "no"):
            out["recorded"] = feedback.record(con, a.verdict, a.artist, a.title, a.strength, a.note)
        if a.verdict == "rip" or a.to_rip:
            from .spotify import Spotify
            out["effects"] = verdicts.apply_effects(con, cfg, Spotify(cfg))
        if a.verdict == "list":
            out["judged"] = feedback.judged(con, cfg)
        print(json.dumps(out, indent=1, default=str))
        return out
    return job("feedback", run, needs_spotify=(a.verdict == "rip" or a.to_rip))


def cmd_verdict(cfg, a) -> int:
    """Record a verdict from the shell (needle is the usual way), list them, or apply pending Spotify writes."""
    from . import verdicts
    if a.action in ("export", "import"):
        return _verdict_file(cfg, a)

    def run(con):
        out = {}
        if a.action == "list":
            cur = verdicts.current(con)
            names = {r[0]: f"{', '.join(json.loads(r[2] or '[]'))} - {r[1]}" for r in con.execute(
                "SELECT spotify_id, name, artists_json FROM spotify_tracks")}
            for sid, v in sorted(cur.items(), key=lambda kv: kv[1].given_at):
                label = f"{v.stars}*" if v.kind == "rating" else v.kind
                print(f"{v.given_at}  {label:8} {sid}  {names.get(sid, '')}")
            out["verdicts"] = len(cur)
            out["pending_effects"] = len(verdicts.pending(con))
            return out
        if a.action == "give":
            if not (a.track and a.value):
                raise SystemExit("usage: haystack verdict give TRACK_ID 1-5|pass|dislike")
            kind, stars = ("rating", int(a.value)) if a.value.isdigit() else (a.value, None)
            out["recorded"] = verdicts.record(con, a.track, kind, stars, source="cli")
        if not a.no_apply:
            from .spotify import Spotify
            out["effects"] = verdicts.apply_effects(con, cfg, Spotify(cfg))
        print(json.dumps(out, indent=1, default=str))
        return out
    return job("verdict", run, needs_spotify=a.action != "list" and not a.no_apply)


def _verdict_file(cfg, a) -> int:
    """verdict export FILE (0600, every verdict) / verdict import FILE [--pending]."""
    from . import verdicts
    if not a.track:
        raise SystemExit(f"usage: haystack verdict {a.action} FILE")
    path = Path(a.track).expanduser()
    try:
        if a.action == "export":
            doc = verdicts.export_doc(db.connect())
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(doc, f, indent=1, ensure_ascii=False)
                f.write("\n")
            out = {"verdicts": len(doc["verdicts"]), "file": str(path)}
        else:
            out = verdicts.import_doc(db.connect(), json.loads(path.read_text()),
                                      effects="pending" if a.pending else "skipped")
    except (OSError, ValueError) as e:
        print(f"verdict {a.action}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=1))
    return 0


def cmd_models(cfg, a) -> int:
    from . import traits
    if a.action == "fetch":
        got = traits.fetch_models()
        print(json.dumps({"fetched": got, "missing": traits.missing_models()}))
    else:
        print(json.dumps({"dir": str(traits.model_dir()), "missing": traits.missing_models()}))
    return 1 if traits.missing_models() else 0


def cmd_traits(cfg, a) -> int:
    from . import traits
    from .deezer import Deezer

    def run(con):
        if a.action == "backfill":
            return traits.backfill(con, Deezer(cfg["harvest"]["deezer_min_interval"]), a.limit)
        if a.action == "measure":
            out = traits.measure_track(con, cfg, a.track)
        else:
            from .vectors import track_vectors
            t = track_vectors(con, [a.track]).get(a.track)
            out = {"track": a.track, **(t or {"deezer_id": None})}
        if out.get("deezer_id"):
            out["readings"] = traits.get(con, [out["deezer_id"]]).get(out["deezer_id"])
        print(json.dumps(out, indent=1))
        return out
    return job("traits", run, needs_spotify=a.action == "measure")


def cmd_baseline(cfg, a) -> int:
    from . import baseline
    print(json.dumps(baseline.compute(db.connect(), cfg), indent=1))
    return 0


def cmd_api(cfg, a) -> int:
    from . import api
    api.serve(cfg)
    return 0


def cmd_serve(cfg, a) -> int:
    """needle's API and the job schedule in one process: the container's entry point, where there is no systemd."""
    import signal
    from . import api, scheduler, secrets, traits
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if traits.missing_models():
        log.warning("Essentia models missing (%s): run `haystack models fetch`; the library and harvest jobs "
                    "fail until then", ", ".join(traits.missing_models()))
    try:
        sch = scheduler.Scheduler(cfg)
    except ValueError as e:
        log.error("%s (in %s)", e, config.CONFIG_FILE)
        return 2
    sch.start()
    try:
        if not a.no_api:
            try:
                api.serve(cfg)
            except secrets.SecretError as e:
                log.warning("needle's API is off: %s, then restart haystack", e)
        while True:
            time.sleep(3600)
    finally:
        sch.stop()


def cmd_genres(cfg, a) -> int:
    from . import evaluate
    print(json.dumps(evaluate.liked_genres(db.connect(), a.top), indent=1))
    return 0


def cmd_sanity(cfg, a) -> int:
    from . import sanity
    con = db.connect()
    out = sanity.write(con, config.STATE_DIR / "sanity-check.md", a.n, a.k)
    print(json.dumps(out))
    return 0


def _candidate_stats(con) -> dict:
    st = {r[0]: r[1] for r in con.execute("SELECT state, COUNT(*) FROM candidates GROUP BY state")}
    src = {}
    for r in con.execute("""SELECT source, COUNT(*), SUM(vector_status='ok'), SUM(state='excluded')
                            FROM candidates GROUP BY source"""):
        src[r[0]] = {"total": r[1], "embedded": r[2] or 0, "excluded": r[3] or 0}
    return {"by_state": st, "by_source": src}


def cmd_status(cfg, a) -> int:
    st = status.read()
    con = db.connect()
    st["live"] = {
        "library_files": con.execute("SELECT COUNT(*) FROM local_tracks WHERE present=1").fetchone()[0],
        "embedded": con.execute("""SELECT COUNT(DISTINCT t.content_hash) FROM local_tracks t JOIN local_embeddings e
                                   ON e.content_hash=t.content_hash AND e.status='ok' WHERE t.present=1""").fetchone()[0],
        "candidates": _candidate_stats(con),
        "review_pending": con.execute("SELECT COUNT(*) FROM liked_map WHERE review='pending'").fetchone()[0],
        "active_offers": con.execute("SELECT COUNT(*) FROM offers WHERE removed_at IS NULL").fetchone()[0],
        "to_rip_open": con.execute("SELECT COUNT(*) FROM to_rip WHERE removed_at IS NULL").fetchone()[0],
    }
    print(json.dumps(st, indent=1, sort_keys=True, default=str))
    return 0


def cmd_review(cfg, a) -> int:
    con = db.connect()
    if a.action == "list":
        rows = con.execute("""SELECT m.spotify_id, m.local_id, m.method, m.confidence, m.dt, m.version, m.reason,
                              s.name, s.artists_json, s.duration, t.artist_raw, t.title_raw, t.duration AS ldur, t.path
                              FROM liked_map m JOIN spotify_tracks s ON s.spotify_id=m.spotify_id
                              LEFT JOIN local_tracks t ON t.id=m.local_id WHERE m.review='pending'
                              ORDER BY m.confidence DESC""").fetchall()
        if not rows:
            print("No uncertain matches to review.")
        for r in rows:
            arts = ", ".join(json.loads(r["artists_json"] or "[]"))
            print(f"{r['spotify_id']}  [{r['confidence']}, {r['reason'] or r['method']}]\n"
                  f"   liked: {arts} - {r['name']} ({(r['duration'] or 0):.0f}s)\n"
                  f"   local: {r['artist_raw']} - {r['title_raw']} ({(r['ldur'] or 0):.0f}s)  #{r['local_id']}\n"
                  f"          {r['path']}")
        if rows:
            print(f"\n{len(rows)} pending. Confirm with: haystack review accept <spotify_id> | reject <spotify_id>")
        return 0
    from .matching import Match, use_local_vector
    for sid in a.ids:
        r = con.execute("SELECT * FROM liked_map WHERE spotify_id=?", (sid,)).fetchone()
        if not r:
            print(f"{sid}: not in the review list"); continue
        if a.action == "accept":
            m = Match(r["local_id"], r["method"] or "fuzzy", "high", r["dt"], 1.0, 1.0, r["version"] or "same")
            con.execute("UPDATE liked_map SET review='accepted', confidence='high', use_local=? WHERE spotify_id=?",
                        (1 if use_local_vector(m, cfg["matching"]["confident_dt"]) else 0, sid))
        else:
            con.execute("UPDATE liked_map SET review='rejected', local_id=NULL, use_local=0 WHERE spotify_id=?", (sid,))
        print(f"{sid}: {a.action}ed")
    return 0


def cmd_sources(cfg, a) -> int:
    from .scoring import can_retire, like_rates
    con = db.connect()
    rates = like_rates([dict(r) for r in con.execute("SELECT source, outcome, played FROM offers")])
    if a.action == "list":
        from . import sources
        cand = _candidate_stats(con)["by_source"]
        kept = sources.rates(con)
        h = cfg["harvest"]
        live = [x for x in h["sources"] if x not in h.get("retired_sources", [])]
        caps = sources.caps(cfg, live, kept)

        def pct(v):
            return "-" if v is None else f"{v:.0%}"

        print(f"{'source':24} {'cands':>6} {'embedded':>8} {'offered':>7} {'decided':>7} {'kept':>5} {'kept rate':>9}"
              f"  cap")
        for s in sorted(set(cand) | set(kept["per_source"]) | set(h["sources"])):
            c = cand.get(s, {}); r = kept["per_source"].get(s, {})
            cap = (f"{caps[s]['cap']} (x{caps[s]['multiplier']:.2f})" if s in caps else
                   "retired" if s in h.get("retired_sources", []) else "-")
            print(f"{s:24} {c.get('total', 0):6} {c.get('embedded', 0):8} {r.get('offered', 0):7} "
                  f"{r.get('decided', 0):7} {r.get('kept', 0):5} {pct(r.get('kept_rate')):>9}  {cap}")
        print(f"\n{'seed':24} {'offered':>7} {'decided':>7} {'kept':>5} {'kept rate':>9}")
        for s, r in kept["per_seed"].items():
            print(f"{s:24} {r['offered']:7} {r['decided']:7} {r['kept']:5} {pct(r['kept_rate']):>9}")
        o = kept["overall"]
        print(f"\noverall: offered {o['offered']}, decided {o['decided']}, kept {o['kept']}, "
              f"kept rate {pct(o['kept_rate'])}")
        print("kept: a Rating of 3 stars or more, or liked without a Rating. cap: new candidates per harvest"
              + (f", scaled by the kept rate from {cfg['sources']['min_offers_before_retire']} decided offers."
                 if h.get("adaptive_caps", True) else "."))
        return 0
    s = a.name
    n = cfg["sources"]["min_offers_before_retire"]
    if not can_retire(rates["per_source"].get(s, {}), n):
        print(f"refusing: {s} has {rates['per_source'].get(s, {}).get('offered', 0)} offered tracks; needs >= {n} "
              "before it can be retired")
        return 2
    print(f"{s} is eligible. Add it to [harvest] retired_sources in {config.CONFIG_FILE} to retire it.")
    return 0


def cmd_check(cfg, a) -> int:
    """Which secrets are set, without revealing any (lengths only)."""
    from . import secrets
    st = secrets.store(cfg)
    print(f"backend: {cfg['secrets']['backend']} ({st.where})")
    bad = 0
    for name in secrets.NAMES:
        try:
            v = st.find(name)
            print(f"{name}: " + (f"set ({len(v)} characters)" if v else "not set"))
        except secrets.SecretError as e:
            print(f"{name}: unreadable: {e}"); bad += 1
    s = secrets.SpotifySecrets(cfg, st)
    try:
        tok, meta = s.refresh_token()
    except secrets.SecretError as e:
        print(f"spotify refresh token: unreadable: {e}")
        return 1
    print(f"spotify refresh token: {'set' if tok else 'not set (run haystack auth)'}; issued at: {meta.get('issued at')}")
    if a.write:
        print(f"refresh token record writable: yes (last write check {s.write_check()})")
    return 1 if bad else 0


def cmd_secret(cfg, a) -> int:
    """Store one secret, read from stdin or a hidden prompt (never from the command line)."""
    import getpass
    from . import secrets
    value = (getpass.getpass(f"{a.name}: ") if sys.stdin.isatty() else sys.stdin.read()).strip()
    if not value:
        print("nothing stored: the value is empty", file=sys.stderr)
        return 2
    st = secrets.store(cfg)
    st.put(a.name, value)
    print(f"stored {a.name} ({st.where})", file=sys.stderr)
    return 0


def cmd_token(cfg, a) -> int:
    """needle's API token: new stores a random one, show prints it for a client's token file."""
    import secrets as pysecrets
    from . import secrets
    st = secrets.store(cfg)
    if a.action == "show":
        tok = st.find("api-token")
        if not tok:
            print("no API token yet: run `haystack token new`", file=sys.stderr)
            return 1
        print(tok)
        return 0
    if st.find("api-token") and not a.force:
        print("an API token is already set; --force replaces it, and every client then needs the new one",
              file=sys.stderr)
        return 2
    st.put("api-token", pysecrets.token_urlsafe(32))
    print("stored a new API token. Restart haystack-api if it runs, then give each needle the token:\n"
          "  (umask 077; ssh <this server> haystack token show > ~/.config/needle/token)", file=sys.stderr)
    return 0


def _add_clear_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--clear-min-age-hours", type=float, default=None,
                   help="tracks added more recently than this get no verdict "
                        "(default: batch.clear_min_age_hours, 24)")
    p.add_argument("--clear-recent", choices=["keep", "remove"], default=None,
                   help="keep: leave recent tracks in Discoveries; "
                        "remove: take them off as unplayed, back in the pool, no negative "
                        "(default: batch.clear_recent, remove)")
    p.add_argument("--no-penalty", action="store_true", default=None,
                   help="close every removed offer except likes as unplayed (no 'no' verdict, no negative weight)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="haystack", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("auth", help="one-time Spotify login (prints a URL; paste back the address the browser "
                                    "lands on, or let the loopback listener catch it)")
    s.add_argument("--timeout", type=int, default=900)
    s.add_argument("--ready-file", help="write a markdown file with the authorize URL and instructions")
    s.add_argument("--log-file", help="also append log lines to this file")
    s.set_defaults(fn=cmd_auth)

    s = sub.add_parser("library", help="local library index (token-free)")
    s.add_argument("action", choices=["scan", "embed", "update"], help="update = scan + embed + To Rip check")
    s.add_argument("--workers", type=int)
    s.add_argument("--limit", type=int, help="embed at most N files this run")
    s.set_defaults(fn=cmd_library)

    s = sub.add_parser("harvest", help="harvest + embed candidates from Deezer / ListenBrainz (token-free)")
    s.add_argument("--sources", help="comma-separated subset of sources")
    s.add_argument("--limit-per-source", type=int, help="max new candidates per source this run")
    s.add_argument("--broad", type=int, nargs="?", const=150, metavar="N",
                   help="seed artist radio / related from N stratified seeds and ListenBrainz from all liked songs")
    s.add_argument("--shard", help="i/N: this process takes every N-th seed starting at i (parallel harvesters)")
    s.add_argument("--no-retry", action="store_true", help="skip retrying failed previews and the style backfill")
    s.add_argument("--per-seed", type=int, help="broad mode: new candidates per seed (overrides the config)")
    s.add_argument("--seed-rng", type=int,
                   help="broad mode: a different seed sample (default: harvest.broad_seed_rng and the ISO week)")
    s.add_argument("--no-score", action="store_true", help="don't re-score the pool afterwards")
    s.add_argument("--styles", type=int, metavar="N", help="also backfill style vectors for up to N older previews")
    s.set_defaults(fn=cmd_harvest)

    s = sub.add_parser("score", help="re-score the candidate pool (token-free)")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("sync", help="hourly: plays, likes, outcomes, To Rip, token expiry (needs auth)")
    s.set_defaults(fn=cmd_sync)

    s = sub.add_parser("batch", help="weekly: clear Discoveries, then add a batch (needs auth)")
    s.add_argument("--size", type=int)
    g = s.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="no Spotify writes; prints the clear plan and the would-be picks")
    g.add_argument("--plan", action="store_true",
                   help="no Spotify calls and no writes (token-free): score in memory and print the would-be picks "
                        "with their nearest positive; picks not mapped to Spotify yet are marked unmapped")
    s.add_argument("--stage1", action="store_true",
                   help="with --plan: leave the re-ranker out, to compare its order with stage 1 alone (ADR 0011)")
    s.add_argument("--force", action="store_true", help="run even if the library embedding is incomplete")
    s.add_argument("--variant", choices=["raw", "topk_mean", "centred", "csls", "csls_centred", "genre", "genre_blend"],
                   help="scoring variant for this run (default: configured / last chosen)")
    _add_clear_args(s)
    s.set_defaults(fn=cmd_batch)

    s = sub.add_parser("clear-discoveries", help="empty Discoveries and record nos for unliked tracks (needs auth)")
    s.add_argument("--dry-run", action="store_true", help="read the playlist and print counts; write nothing")
    _add_clear_args(s)
    s.set_defaults(fn=cmd_clear)

    s = sub.add_parser("feedback", help="alias of `verdict` on candidates: yes (3 stars) / no ARTIST TITLE "
                                        "(--strength strong = Dislike); rip = apply pending Spotify writes")
    s.add_argument("verdict", choices=["yes", "no", "rip", "list"])
    s.add_argument("artist", nargs="?")
    s.add_argument("title", nargs="?")
    s.add_argument("--strength", choices=["mild", "normal", "strong"], default="normal")
    s.add_argument("--note")
    s.add_argument("--to-rip", action="store_true", help="after recording, apply the Spotify writes (needs auth)")
    s.set_defaults(fn=cmd_feedback)

    s = sub.add_parser("verdict", help="verdicts on any Spotify track: give TRACK 1-5|pass|dislike; list; apply "
                                       "(pending Spotify writes: heart, Discoveries, To Rip); export|import FILE")
    s.add_argument("action", choices=["give", "list", "apply", "export", "import"])
    s.add_argument("track", nargs="?", help="give: the 22-character Spotify track id; export, import: the JSON file")
    s.add_argument("value", nargs="?", help="1-5 stars, pass or dislike")
    s.add_argument("--no-apply", action="store_true", help="record only; leave the Spotify writes for the sync")
    s.add_argument("--pending", action="store_true",
                   help="import: the sync makes the hearts and playlist writes the imported verdicts imply "
                        "(default: none, as for a restore to the same Spotify account)")
    s.set_defaults(fn=cmd_verdict)

    s = sub.add_parser("models", help="Essentia models: fetch the missing ones from essentia.upf.edu, or check")
    s.add_argument("action", choices=["fetch", "check"])
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("traits", help="Readings: backfill stored previews, measure TRACK on demand, show TRACK")
    s.add_argument("action", choices=["backfill", "measure", "show"])
    s.add_argument("track", nargs="?", help="Spotify track id (measure, show)")
    s.add_argument("--limit", type=int, default=150)
    s.set_defaults(fn=cmd_traits)

    s = sub.add_parser("baseline", help="the listener's Baseline per Trait (star-weighted mean and usual range)")
    s.set_defaults(fn=cmd_baseline)

    s = sub.add_parser("api", help="needle's private HTTP API (the haystack-api unit runs this)")
    s.set_defaults(fn=cmd_api)

    s = sub.add_parser("serve", help="needle's API plus the scheduled jobs ([schedule]) in one process: for hosts "
                                     "without systemd, such as the container")
    s.add_argument("--no-api", action="store_true", help="run only the scheduled jobs")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("genres", help="genre makeup of the liked songs (genre_discogs400 styles of their previews)")
    s.add_argument("--top", type=int, default=15)
    s.set_defaults(fn=cmd_genres)

    s = sub.add_parser("eval", help="offline scoring-variant eval on your verdicts and held-out liked songs "
                                    "(no Spotify calls)")
    s.add_argument("--seeds", help="comma-separated seeds (default from config)")
    s.add_argument("--objective", choices=("auc", "mean_pct"), help="how to choose (default eval.objective)")
    s.add_argument("--choose", action="store_true",
                   help="persist the winner as the scoring variant (ignored when scoring.variant is pinned)")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("rerank", help="the second stage (ADR 0011): fit it on the current Verdicts and record the "
                                      "result (token-free; the pool's order is left alone), or show the last fit")
    s.add_argument("action", choices=["fit", "show"])
    s.add_argument("--limit", type=int, default=10, help="show: how many weights and past fits")
    s.set_defaults(fn=cmd_rerank)

    s = sub.add_parser("calibrate", help="the Calibrate queue (ADR 0015): plan (what the next top-up would pick; "
                                         "token-free, no writes) or top-up (pick and map now; needs auth)")
    s.add_argument("action", choices=["plan", "top-up"])
    s.add_argument("--n", type=int, help="plan: this many picks (default what the queue takes now, else a week's)")
    s.add_argument("--tracks", action="store_true", help="plan: show artist and title")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("align", help="the preview alignment (ADR 0012): fit the map from preview vectors to the "
                                     "library's space on the liked songs that have both, or show it")
    s.add_argument("action", choices=["fit", "show"])
    s.add_argument("--limit", type=int, default=0, help="show: how many of the tried fits (default all)")
    s.set_defaults(fn=cmd_align)

    s = sub.add_parser("first-batch", help="one-time first real batch + taste sanity check (library OnSuccess hook)")
    _add_clear_args(s)
    s.set_defaults(fn=cmd_first_batch)

    s = sub.add_parser("sanity-check", help="write sanity-check.md: nearest library tracks for random liked songs")
    s.add_argument("-n", type=int, default=10)
    s.add_argument("-k", type=int, default=10)
    s.set_defaults(fn=cmd_sanity)

    s = sub.add_parser("status", help="print status.json plus live counts")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("review", help="uncertain liked-song -> library matches")
    s.add_argument("action", nargs="?", default="list", choices=["list", "accept", "reject"])
    s.add_argument("ids", nargs="*")
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("report", help="how discovery is doing: like rates, verdicts, pool, last eval "
                                      "(read-only, token-free; counts only unless --tracks)")
    s.add_argument("--tracks", action="store_true", help="also list the top of the pool by name")
    s.add_argument("--json", action="store_true", help="print the report as JSON")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("sources", help="kept rate per source and per seed, harvest caps; retirement check")
    s.add_argument("action", nargs="?", default="list", choices=["list", "retire"])
    s.add_argument("name", nargs="?")
    s.set_defaults(fn=cmd_sources)

    s = sub.add_parser("alert-failure", help="OnFailure= hook: send a failure alert for UNIT (webhook or outbox)")
    s.add_argument("unit")
    s.set_defaults(fn=cmd_alert_failure)

    s = sub.add_parser("check-secrets", help="which secrets are set (and --write: the token record is writable), "
                                             "without printing any")
    s.add_argument("--write", action="store_true")
    s.set_defaults(fn=cmd_check)

    from .secrets import NAMES
    s = sub.add_parser("secret", help="set NAME: store a secret, read from stdin or a hidden prompt")
    s.add_argument("action", choices=["set"])
    s.add_argument("name", choices=NAMES)
    s.set_defaults(fn=cmd_secret)

    s = sub.add_parser("token", help="needle's API token: new (store a random one; --force replaces) or show "
                                     "(print it, for a client's token file)")
    s.add_argument("action", choices=["new", "show"])
    s.add_argument("--force", action="store_true", help="new: replace the token that is set")
    s.set_defaults(fn=cmd_token)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    config.ensure_dirs()
    _logging(getattr(a, "log_file", None))
    cfg = config.load()
    return a.fn(cfg, a) or 0


if __name__ == "__main__":
    sys.exit(main())
