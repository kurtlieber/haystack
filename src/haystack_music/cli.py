"""haystack: personal Spotify music discovery (see README.md)."""
from __future__ import annotations

import argparse
import json
import logging
import os
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
    from .harvest import Harvester, stratified_seeds

    def run(con):
        seeds = liked = None
        out = {}
        if a.broad:
            rng = random.Random(a.seed_rng or cfg["harvest"]["broad_seed_rng"])
            seeds = stratified_seeds(con, a.broad, rng)
            liked = [(json.loads(r[0] or "[]")[0], r[1]) for r in con.execute(
                """SELECT s.artists_json, s.name FROM likes l JOIN spotify_tracks s ON s.spotify_id=l.spotify_id
                   WHERE l.removed_at IS NULL ORDER BY l.spotify_id""") if json.loads(r[0] or "[]") and r[1]]
            rng.shuffle(liked)
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
    return job("harvest" + (f"-{a.shard.replace('/', 'of')}" if a.shard else ""), run)


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


def cmd_batch(cfg, a) -> int:
    from . import batch
    def run(con):
        out = batch.run(con, cfg, size=a.size, dry_run=a.dry_run, force=a.force, variant=a.variant,
                        clear_min_age_hours=a.clear_min_age_hours, clear_recent=a.clear_recent,
                        no_penalty=a.no_penalty)
        if out.get("clear"):
            _print_clear(out["clear"])
        print(f"variant: {out.get('variant')}")
        if a.dry_run or out.get("tracks"):
            for t in out.get("tracks", []):
                print(f"  {t['score']:.4f}  {'explore' if t['exploration'] else 'top    '}  {t['source']:22} "
                      f"{t['artist']} - {t['title']}")
        return out
    return job("batch", run, needs_spotify=True)


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
    from . import evaluate, firstbatch
    if a.seeds:
        cfg["eval"]["seeds"] = [int(x) for x in a.seeds.split(",")]

    def run(con):
        ev = firstbatch.choose(con, cfg) if a.choose else evaluate.run(con, cfg)
        if not a.choose:
            status.update(None, scoring_eval=ev)
        print(evaluate.markdown(ev))
        return ev
    return job("eval", run)


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
        cand = _candidate_stats(con)["by_source"]
        print(f"{'source':24} {'cands':>6} {'embedded':>8} {'offered':>7} {'liked':>5} {'like rate':>9}")
        for s in sorted(set(cand) | set(rates["per_source"]) | set(cfg["harvest"]["sources"])):
            c = cand.get(s, {}); r = rates["per_source"].get(s, {})
            lr = r.get("like_rate")
            flag = " (retired)" if s in cfg["harvest"].get("retired_sources", []) else ""
            print(f"{s:24} {c.get('total', 0):6} {c.get('embedded', 0):8} {r.get('offered', 0):7} {r.get('liked', 0):5} "
                  f"{'-' if lr is None else f'{lr:.0%}':>9}{flag}")
        o = rates["overall"]
        lr = o["like_rate"]
        print(f"\noverall: offered {o['offered']}, liked {o['liked']}, like rate {'-' if lr is None else f'{lr:.0%}'}")
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

    s = sub.add_parser("auth", help="one-time Spotify login (prints a URL, listens on the loopback callback)")
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
    s.add_argument("--seed-rng", type=int, help="broad mode: a different seed sample (default from config)")
    s.add_argument("--no-score", action="store_true", help="don't re-score the pool afterwards")
    s.add_argument("--styles", type=int, metavar="N", help="also backfill style vectors for up to N older previews")
    s.set_defaults(fn=cmd_harvest)

    s = sub.add_parser("score", help="re-score the candidate pool (token-free)")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("sync", help="hourly: plays, likes, outcomes, To Rip, token expiry (needs auth)")
    s.set_defaults(fn=cmd_sync)

    s = sub.add_parser("batch", help="weekly: clear Discoveries, then add a batch (needs auth)")
    s.add_argument("--size", type=int)
    s.add_argument("--dry-run", action="store_true", help="no Spotify writes; prints the clear plan and the would-be picks")
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

    s = sub.add_parser("genres", help="genre makeup of the liked songs (genre_discogs400 styles of their previews)")
    s.add_argument("--top", type=int, default=15)
    s.set_defaults(fn=cmd_genres)

    s = sub.add_parser("eval", help="offline scoring-variant eval on held-out liked songs (no Spotify calls)")
    s.add_argument("--seeds", help="comma-separated seeds (default from config)")
    s.add_argument("--choose", action="store_true",
                   help="persist the winner as the scoring variant (ignored when scoring.variant is pinned)")
    s.set_defaults(fn=cmd_eval)

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

    s = sub.add_parser("sources", help="like rate per source; retirement check")
    s.add_argument("action", nargs="?", default="list", choices=["list", "retire"])
    s.add_argument("name", nargs="?")
    s.set_defaults(fn=cmd_sources)

    s = sub.add_parser("alert-failure", help="OnFailure= hook: POST a failure alert for UNIT to the webhook")
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
