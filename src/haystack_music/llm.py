"""An optional language model (ADR 0019): what a model knows about artists, labels and scenes, which no audio
embedding hears. Off until `[llm] enabled = true`; with it off neither provider is imported or built.

Phase 1 is a measurement, run by hand. `haystack llm judge` asks the model, blind, about every track the
re-ranker has a label for and compares its answers with the verdicts: the chance alone as a score, and the
re-ranker's leave-one-out AUC with and without the chance as one more feature. `haystack llm suggest --dry-run`
asks it for songs and counts how many Deezer has. Nothing here writes to the database, to Spotify or to a file.

What a request carries, and nothing else:
  per track     artist, title, album, year, label, its top three Discogs styles with their scores, and the
                haystack source that found it, under a number local to the request
  the profile   the artists of the listener's liked and rated tracks with the most stars (tracks, mean stars),
                the artists of their Passes and Dislikes, their top styles by share, and each Trait's usual range
Never audio, Spotify or Deezer ids, account names, verdicts of the tracks asked about, or tokens. The tracks of a
request, and every other id of the same songs, are left out of its profile (blind, leave-one-out). Every request
is logged at debug level; the key is never logged.

Providers: "anthropic" (the official SDK, the optional extra `haystack-music[llm]`; structured outputs, the
model's default thinking and effort, the SDK's own retries) and "openai_compatible" (a POST to
{base_url}/chat/completions with a JSON-schema response format: Ollama, llama.cpp's server, vLLM and others)."""
from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import Counter, defaultdict

import numpy as np

from . import baseline, db, metadata, rerank, secrets, traits, vectors, verdicts
from .evaluate import _chunks, _first_artist, _track_keys
from .review import SOURCE_LABELS, _year
from .textnorm import song_key

log = logging.getLogger("haystack_music.llm")

BATCH = 40                 # tracks per judge request
SHUFFLE_SEED = 20261020    # the judged tracks are shuffled before batching, the same way every run
MAX_TOKENS = 16000         # thinking counts against it (ADR 0019)
TOP_ARTISTS = 30
TURNED_DOWN_ARTISTS = 50
TOP_STYLES = 15
TRACK_STYLES = 3
CHANCE_CLIP = 0.01         # the chance's logit is taken on [0.01, 0.99]
JUDGE_AUC_BAR = 0.70       # the decision rule (ADR 0019): a rerank job if the chance alone reaches this AUC,
LOO_GAIN_BAR = 0.05        # or adds this much to the re-ranker's leave-one-out AUC
RESOLVE_BAR = 0.60         # a seeds job on its own if more than this share of the suggestions resolve on Deezer
TRAIT_UNITS = {"tempo": ("BPM", 0), "loudness": ("LUFS", 1)}   # the rest are 0-1, two decimals

JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["tracks"],
    "properties": {"tracks": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "chance", "reason"],
        "properties": {"id": {"type": "integer"}, "chance": {"type": "number"}, "reason": {"type": "string"}}}}}}
SUGGEST_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["suggestions"],
    "properties": {"suggestions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["artist", "title", "why"],
        "properties": {"artist": {"type": "string"}, "title": {"type": "string"}, "why": {"type": "string"}}}}}}

JUDGE_SYSTEM = """You help a music discovery tool guess which songs one listener will keep.

You get a profile of the listener's taste, built from the songs they liked and rated, then a numbered list of \
songs the tool found. For each song, from what you know about the artist, the label, the scene and the era, give \
the chance, from 0 to 1, that this listener keeps it (rates it 3 stars or more out of 5), and one short reason.

Judge every song in the list exactly once, by its number. Most songs a tool like this finds are not kept, so \
keep the chances honest rather than hopeful. If you don't know an artist, say so in the reason and judge from the \
rest: the label, the styles, the year. The styles and their scores come from an audio classifier and can be wrong."""

SUGGEST_SYSTEM = """You help a music discovery tool find songs one listener will keep.

You get a profile of the listener's taste, built from the songs they liked and rated. Name real songs that \
exist as released recordings, by artists who are not in the profile's list of artists they rate highest, that \
this listener is likely to keep. Give the artist and title exactly as released, and one short reason each. \
Prefer songs a streaming service has; never invent a song."""


class LLMError(RuntimeError):
    pass


class Refused(LLMError):
    """The model declined, ran out of tokens or gave no JSON: that request counts as missing."""

    def __init__(self, msg: str, usage: dict | None = None):
        super().__init__(msg)
        self.usage = usage or {"in": 0, "out": 0}


# ---------------- providers ----------------

class AnthropicProvider:
    """Claude through the official SDK: structured outputs (output_config.format), the model's default thinking
    and effort (no `thinking` or `budget_tokens` sent), streamed, with the SDK's retries. stop_reason is checked
    before the content is read."""
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, client=None, max_tokens: int = MAX_TOKENS):
        if client is None:
            if not api_key:
                raise LLMError("the llm-api-key secret is not set (`haystack secret set llm-api-key`)")
            try:
                import anthropic
            except ImportError as e:
                raise LLMError('the anthropic provider needs the optional extra: pip install "haystack-music[llm]"') \
                    from e
            client = anthropic.Anthropic(api_key=api_key)
        self.c, self.model, self.max_tokens = client, model, max_tokens

    def complete(self, system: str, user: str, schema: dict, name: str) -> tuple[dict, dict]:
        try:
            with self.c.messages.stream(model=self.model, max_tokens=self.max_tokens, system=system,
                                        messages=[{"role": "user", "content": user}],
                                        output_config={"format": {"type": "json_schema", "schema": schema}}) as s:
                m = s.get_final_message()
        except Exception as e:  # noqa: BLE001 - the SDK's errors, without importing it here
            raise LLMError(f"{self.name}: {type(e).__name__}: {e}") from e
        u = getattr(m, "usage", None)
        usage = {"in": int(getattr(u, "input_tokens", 0) or 0), "out": int(getattr(u, "output_tokens", 0) or 0)}
        if m.stop_reason == "refusal":
            raise Refused(f"the model declined ({getattr(m, 'stop_details', None)})", usage)
        if m.stop_reason == "max_tokens":
            raise Refused(f"the answer ran past max_tokens ({self.max_tokens})", usage)
        text = next((b.text for b in m.content if getattr(b, "type", None) == "text"), None)
        return _parse(text, usage), usage


class OpenAICompatibleProvider:
    """Any server with OpenAI's chat completions and JSON-schema response format: Ollama, llama.cpp's server, vLLM,
    LM Studio. The llm-api-key secret, when set, goes as the bearer token; a local server needs none."""
    name = "openai_compatible"

    def __init__(self, model: str, base_url: str, api_key: str | None = None, session=None, timeout: float = 600.0,
                 tries: int = 3):
        if not base_url:
            raise LLMError("llm.base_url is empty: set it to the server's /v1 address")
        if session is None:
            import requests
            session = requests.Session()
        self.s, self.model, self.url = session, model, base_url.rstrip("/") + "/chat/completions"
        self.key, self.timeout, self.tries = api_key, timeout, tries

    def complete(self, system: str, user: str, schema: dict, name: str) -> tuple[dict, dict]:
        import requests
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": name, "schema": schema, "strict": True}}}
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        for i in range(self.tries):
            try:
                r = self.s.post(self.url, json=body, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                if i + 1 == self.tries:
                    raise LLMError(f"{self.name}: {type(e).__name__}") from e
                time.sleep(3 * (i + 1))
                continue
            if (r.status_code == 429 or r.status_code >= 500) and i + 1 < self.tries:
                time.sleep(3 * (i + 1))
                continue
            if r.status_code >= 400:
                raise LLMError(f"{self.name}: HTTP {r.status_code}: {r.text[:200]}")
            break
        j = r.json()
        u = j.get("usage") or {}
        usage = {"in": int(u.get("prompt_tokens") or 0), "out": int(u.get("completion_tokens") or 0)}
        ch = (j.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        if msg.get("refusal"):
            raise Refused(f"the model declined ({msg['refusal'][:200]})", usage)
        if ch.get("finish_reason") == "length":
            raise Refused("the answer ran past the server's token limit", usage)
        return _parse(msg.get("content"), usage), usage


def _parse(text: str | None, usage: dict) -> dict:
    try:
        obj = json.loads(text or "")
    except ValueError:
        raise Refused("the answer was not JSON", usage) from None
    if not isinstance(obj, dict):
        raise Refused("the answer was not a JSON object", usage)
    return obj


def make_provider(cfg: dict, store=None):
    """The configured provider. LLMError when llm.enabled is false (before anything is imported), or when the
    provider is unknown or can't be built."""
    c = cfg["llm"]
    if not c["enabled"]:
        raise LLMError("llm.enabled is false: nothing is sent until you turn it on (README, "
                       "\"Optional: a language model\")")
    key = (store or secrets.store(cfg)).find("llm-api-key")
    if c["provider"] == "anthropic":
        return AnthropicProvider(c["model"], key)
    if c["provider"] == "openai_compatible":
        return OpenAICompatibleProvider(c["model"], c["base_url"], key)
    raise LLMError(f"unknown llm.provider {c['provider']!r} (anthropic or openai_compatible)")


def cost(cfg: dict, tokens_in: int, tokens_out: int) -> float:
    c = cfg["llm"]
    return round(tokens_in / 1e6 * float(c["price_in_per_mtok"]) + tokens_out / 1e6 * float(c["price_out_per_mtok"]),
                 4)


# ---------------- what is sent ----------------

def _style_names(classes: list[str] | None) -> list[str]:
    """Discogs 'Parent---Style' class names as 'Parent / Style' (the parent says which family a style is in)."""
    return [c.replace("---", " / ") for c in classes or []]


def _first_artists(con, sids) -> dict[str, str]:
    """spotify id -> its first artist: Spotify's, else a candidate's."""
    out: dict[str, str] = {}
    for ch in _chunks(sorted(set(sids))):
        ph = ",".join("?" * len(ch))
        for r in con.execute(f"SELECT spotify_id, artist FROM candidates WHERE spotify_id IN ({ph}) "
                             f"AND artist IS NOT NULL ORDER BY id DESC", ch):
            out[r[0]] = r[1]
        for r in con.execute(f"SELECT spotify_id, artists_json FROM spotify_tracks WHERE spotify_id IN ({ph})", ch):
            if _first_artist(r[1]):
                out[r[0]] = _first_artist(r[1])
    return out


def track_facts(con, labs: list[dict], classes: list[str] | None = None) -> dict[str, dict]:
    """spotify id -> what a judge request says about the track (rerank.labels rows: sid, cand_id, deezer_id):
    {artist, title, album, year, label, styles [[name, score]], source}; None for what isn't known."""
    names = _style_names(classes)
    sids = [x["sid"] for x in labs]
    sp: dict[str, dict] = {}
    for ch in _chunks(sids):
        for r in con.execute(f"SELECT * FROM spotify_tracks WHERE spotify_id IN ({','.join('?' * len(ch))})", ch):
            sp[r["spotify_id"]] = dict(r)
    cands: dict[int, dict] = {}
    for ch in _chunks([x["cand_id"] for x in labs]):
        for r in con.execute(f"SELECT id, artist, title, source FROM candidates WHERE id IN "
                             f"({','.join('?' * len(ch))})", ch):
            cands[r[0]] = {"artist": r[1], "title": r[2], "source": r[3]}
    dids = [x["deezer_id"] for x in labs]
    labels = metadata.labels_for_cards(con, dids)
    years = _album_years(con, [x["cand_id"] for x in labs])
    st = vectors.get_styles(con, dids) if names else {}
    out = {}
    for x in labs:
        s, c = sp.get(x["sid"]) or {}, cands.get(x["cand_id"]) or {}
        artists = json.loads(s.get("artists_json") or "[]")
        v = st.get(x["deezer_id"])
        styles = [] if v is None else [[names[i], round(float(v[i]), 2)] for i in np.argsort(-v)[:TRACK_STYLES]]
        out[x["sid"]] = {"artist": ", ".join(artists) or c.get("artist"), "title": s.get("name") or c.get("title"),
                         "album": s.get("album"), "year": _year(s.get("release_date")) or years.get(x["cand_id"]),
                         "label": labels.get(x["deezer_id"]), "styles": styles,
                         "source": SOURCE_LABELS.get(c.get("source"), c.get("source"))}
    return out


def _album_years(con, cand_ids) -> dict[int, int]:
    out: dict[int, int] = {}
    try:
        for ch in _chunks(sorted(set(cand_ids))):
            for r in con.execute(f"SELECT c.id, a.release_date FROM candidates c JOIN deezer_albums a "
                                 f"ON a.album_id=c.album_id WHERE c.id IN ({','.join('?' * len(ch))})", ch):
                if _year(r[1]):
                    out[r[0]] = _year(r[1])
    except Exception:  # noqa: BLE001 - a database from before 0.6.0: no album rows
        return {}
    return out


class Profile:
    """The listener's taste as a request gives it, from the tracks they like (baseline.liked_weights: Liked Songs
    and every Rating, by stars) and the ones they turned down (a current Pass or Dislike). without() leaves tracks
    out, and every other id of the same songs. Loaded once; without() is cheap but for the Traits' ranges."""

    def __init__(self, con, cfg: dict, classes: list[str] | None = None):
        self.con, self.cfg = con, cfg
        self.stars = baseline.liked_weights(con, cfg)
        self.down = {s for s, v in verdicts.current(con).items() if v.kind in ("pass", "dislike")}
        self.artist = _first_artists(con, set(self.stars) | self.down)
        self.keys = _track_keys(con, set(self.stars) | self.down)
        names = _style_names(classes)
        self.style: dict[str, str] = {}
        if names:
            sids, mat, _ = vectors.liked_styles(con, cfg)
            for s, row in zip(sids, mat):
                self.style[s] = names[int(np.argmax(row))]
        self.n = len(self.stars)

    def _drop(self, sids) -> set[str]:
        drop = set(sids)
        keys = set().union(*_track_keys(self.con, drop).values()) if drop else set()
        return drop | {s for s, k in self.keys.items() if k & keys}

    def without(self, sids=()) -> dict:
        """{tracks, artists [[name, tracks, mean stars]], turned_down [[name, tracks]], styles [[name, share]],
        traits {trait: [lo, hi]}} over the listener's tracks less these."""
        drop = self._drop(sids)
        per: dict[str, list[float]] = defaultdict(list)
        for s, w in self.stars.items():
            if s not in drop and self.artist.get(s):
                per[self.artist[s]].append(w)
        top = sorted(per.items(), key=lambda kv: (-sum(kv[1]), kv[0]))[:TOP_ARTISTS]
        down = Counter(self.artist[s] for s in self.down if s not in drop and self.artist.get(s))
        styles = Counter(st for s, st in self.style.items() if s not in drop)
        n_st = sum(styles.values())
        usual = baseline.compute(self.con, self.cfg, exclude=drop)["traits"]
        tr = {}
        for t in traits.TRAITS:
            u = usual.get(t)
            if u:
                nd = TRAIT_UNITS.get(t, ("", 2))[1]
                tr[t] = [round(u["lo"], nd), round(u["hi"], nd)]
        return {"tracks": sum(1 for s in self.stars if s not in drop),
                "artists": [[a, len(ws), round(float(np.mean(ws)), 1)] for a, ws in top],
                "turned_down": [[a, n] for a, n in sorted(down.items(), key=lambda kv: (-kv[1], kv[0]))
                                [:TURNED_DOWN_ARTISTS]],
                "styles": [[s, round(n / n_st, 3)] for s, n in sorted(styles.items(), key=lambda kv: (-kv[1], kv[0]))
                           [:TOP_STYLES]],
                "traits": tr}


def profile_text(p: dict) -> str:
    lines = ["Listener profile"]
    lines.append("Artists of the songs they liked and rated, most stars first (songs, mean stars of 5): " +
                 ("; ".join(f"{a} ({n}, {m})" for a, n, m in p["artists"]) or "none"))
    lines.append("Artists of songs they turned down (songs): " +
                 ("; ".join(f"{a} ({n})" for a, n in p["turned_down"]) or "none"))
    if p["styles"]:
        lines.append("Their top styles (share of their liked songs): " +
                     "; ".join(f"{s} {round(100 * x)}%" for s, x in p["styles"]))
    if p["traits"]:
        lines.append("Usual range of their liked songs (25th to 75th percentile): " +
                     "; ".join(_trait_range(t, lo, hi) for t, (lo, hi) in p["traits"].items()))
    return "\n".join(lines)


def _trait_range(t: str, lo: float, hi: float) -> str:
    unit, nd = TRAIT_UNITS.get(t, ("", 2))
    return f"{t} {lo:.{nd}f} to {hi:.{nd}f}" + (f" {unit}" if unit else "")


def _track_line(n: int, f: dict) -> str:
    parts = [f"{n}. {f['artist'] or '?'} - {f['title'] or '?'}"]
    if f["album"] or f["year"]:
        parts.append("album: " + " ".join(x for x in (f["album"] or "", f"({f['year']})" if f["year"] else "") if x))
    if f["label"]:
        parts.append(f"label: {f['label']}")
    if f["styles"]:
        parts.append("styles: " + ", ".join(f"{s} {x}" for s, x in f["styles"]))
    if f["source"]:
        parts.append(f"found by: {f['source']}")
    return " | ".join(parts)


def judge_prompt(profile: dict, facts: list[dict]) -> str:
    return (profile_text(profile) + "\n\nSongs\n" + "\n".join(_track_line(i + 1, f) for i, f in enumerate(facts)))


def suggest_prompt(profile: dict, n: int) -> str:
    return profile_text(profile) + f"\n\nName {n} songs this listener is likely to keep."


# ---------------- phase 1: the measurements ----------------

def _batches(labs: list[dict], size: int) -> list[list[dict]]:
    order = sorted(labs, key=lambda x: x["sid"])
    random.Random(SHUFFLE_SEED).shuffle(order)
    return [order[i:i + size] for i in range(0, len(order), size)]


def judge(con, cfg: dict, provider=None, *, classes: list[str] | None = None, dry_run: bool = False,
          size: int = BATCH) -> dict:
    """Ask the model about every labelled track (rerank.labels, soft ones too), BATCH at a time, each request's
    profile without its own tracks. dry_run: build the first request and send nothing. -> {at, provider, model,
    n_labels, n_pos, n_neg, n_soft, batches, n_sent, n_missing, refused, tokens_in, tokens_out, cost_usd,
    chance_auc, loo {...}, meets, tracks [{sid, kind, y, soft, chance, reason}], system, first_request, reason}."""
    labs = rerank.labels(con, cfg)
    hard = [x for x in labs if not x["soft"]]
    c = cfg["llm"]
    out = {"at": db.now(), "provider": getattr(provider, "name", c["provider"]), "model": c["model"],
           "n_labels": len(hard), "n_pos": sum(x["y"] for x in hard), "n_neg": sum(1 - x["y"] for x in hard),
           "n_soft": len(labs) - len(hard), "batches": 0, "n_sent": 0, "n_missing": 0, "refused": 0,
           "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "chance_auc": None, "loo": None, "meets": None,
           "tracks": [], "system": JUDGE_SYSTEM, "first_request": None, "reason": None}
    prof = Profile(con, cfg, classes)
    if prof.n < int(c["min_profile_tracks"]):
        return {**out, "reason": f"{prof.n} liked and rated tracks (needs llm.min_profile_tracks, "
                                 f"{c['min_profile_tracks']})"}
    if not labs:
        return {**out, "reason": "no labelled tracks to ask about"}
    facts = track_facts(con, labs, classes)
    batches = _batches(labs, size)
    out["batches"] = len(batches)
    first = judge_prompt(prof.without([x["sid"] for x in batches[0]]), [facts[x["sid"]] for x in batches[0]])
    out["first_request"] = first
    if dry_run:
        return {**out, "reason": "dry run: nothing was sent"}
    chance: dict[str, float] = {}
    why: dict[str, str] = {}
    for bi, b in enumerate(batches):
        user = first if bi == 0 else judge_prompt(prof.without([x["sid"] for x in b]), [facts[x["sid"]] for x in b])
        log.debug("llm judge request %d/%d to %s %s:\n%s", bi + 1, len(batches), out["provider"], out["model"], user)
        try:
            obj, u = provider.complete(JUDGE_SYSTEM, user, JUDGE_SCHEMA, "judge")
        except Refused as e:
            log.warning("llm judge request %d/%d: %s; its %d tracks count as missing", bi + 1, len(batches), e, len(b))
            out["refused"] += 1
            u, obj = e.usage, {"tracks": []}
        out["tokens_in"] += u["in"]
        out["tokens_out"] += u["out"]
        out["n_sent"] += len(b)
        log.debug("llm judge answer %d/%d: %s", bi + 1, len(batches), json.dumps(obj))
        for t in obj.get("tracks") or []:
            try:
                i, p = int(t["id"]), float(t["chance"])
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= i <= len(b) and math.isfinite(p) and b[i - 1]["sid"] not in chance:
                chance[b[i - 1]["sid"]] = min(1.0, max(0.0, p))
                why[b[i - 1]["sid"]] = str(t.get("reason") or "")
    out["n_missing"] = len(labs) - len(chance)
    out["cost_usd"] = cost(cfg, out["tokens_in"], out["tokens_out"])
    out["tracks"] = [{"sid": x["sid"], "kind": x["kind"], "y": x["y"], "soft": x["soft"],
                      "chance": chance.get(x["sid"]), "reason": why.get(x["sid"])} for x in labs]
    hc = [x for x in hard if x["sid"] in chance]
    out["chance_auc"] = _r(rerank.wauc([chance[x["sid"]] for x in hc], [x["y"] for x in hc], [x["w"] for x in hc]))
    out["loo"] = loo_gain(con, cfg, labs, chance)
    out["meets"] = decision(out)
    return out


def _r(x, nd: int = 4):
    return None if x is None else round(float(x), nd)


def loo_gain(con, cfg: dict, labs: list[dict], chance: dict[str, float]) -> dict:
    """The re-ranker's leave-one-out AUC over the hard labels the model answered for, without and with the
    chance's logit (clipped to [0.01, 0.99], standardised over those labels) as one more feature, and stage 1's on
    the same labels. Read-only: rerank.design on the active variant's scores, nothing recorded.
    -> {n, n_pos, n_neg, dropped, stage1, without, with, gain, reason}."""
    out = {"n": 0, "n_pos": 0, "n_neg": 0, "dropped": 0, "stage1": None, "without": None, "with": None, "gain": None,
           "reason": None}
    variant = vectors.active_variant(con, cfg)
    res = vectors.compute_scores(con, cfg, variant)
    d = rerank.design(con, cfg, variant, res, labs=labs) if res is not None else None
    if d is None:
        return {**out, "reason": "no positives or no scored pool"}
    rows = np.array([i for i, x in enumerate(d["labs"]) if x["sid"] in chance], int)
    out["dropped"] = len(d["labs"]) - len(rows)
    if not len(rows):
        return {**out, "reason": "the model answered for none of the labels"}
    Zl = d["Z"][d["lab_rows"][rows]]
    y, w, hard = d["y"][rows], d["w"][rows], d["hard"][rows]
    h = np.flatnonzero(hard)
    out.update(n=int(len(h)), n_pos=int(y[h].sum()), n_neg=int(len(h) - y[h].sum()))
    if out["n_pos"] < 2 or out["n_neg"] < 2:
        return {**out, "reason": f"{out['n_pos']} yes and {out['n_neg']} no labels: too few to fit"}
    p = np.clip([chance[d["labs"][i]["sid"]] for i in rows], CHANCE_CLIP, 1 - CHANCE_CLIP)
    lg = np.log(p / (1 - p))
    lg = (lg - lg.mean()) / (lg.std() or 1.0)
    l2 = cfg["rerank"]["l2"]
    out["stage1"] = _r(rerank.wauc(d["s1"][d["lab_rows"][rows]][h], y[h], w[h]))
    out["without"] = _r(rerank.wauc(rerank.loo_logits(Zl, y, w, l2, h)[h], y[h], w[h]))
    out["with"] = _r(rerank.wauc(rerank.loo_logits(np.column_stack([Zl, lg]), y, w, l2, h)[h], y[h], w[h]))
    if out["with"] is not None and out["without"] is not None:
        out["gain"] = _r(out["with"] - out["without"])
    return out


def decision(r: dict) -> dict:
    """ADR 0019's rule: a rerank job when the chance alone reaches JUDGE_AUC_BAR or it adds LOO_GAIN_BAR to the
    re-ranker's leave-one-out AUC."""
    gain = (r.get("loo") or {}).get("gain")
    auc = r.get("chance_auc")
    met = (auc is not None and auc >= JUDGE_AUC_BAR) or (gain is not None and gain >= LOO_GAIN_BAR)
    return {"rerank_job": bool(met), "auc_bar": JUDGE_AUC_BAR, "gain_bar": LOO_GAIN_BAR}


def known_songs(con) -> tuple[set[int], set[str]]:
    """Deezer ids of the candidates, and the song keys of the candidates, the liked songs and the library."""
    ids = {r[0] for r in con.execute("SELECT deezer_id FROM candidates WHERE deezer_id IS NOT NULL")}
    keys = {r[0] for r in con.execute("SELECT song_key FROM candidates WHERE song_key IS NOT NULL")}
    for r in con.execute("SELECT s.name, s.artists_json FROM likes l JOIN spotify_tracks s "
                         "ON s.spotify_id=l.spotify_id WHERE l.removed_at IS NULL"):
        keys.add(song_key(_first_artist(r[1]), r[0]))
    for r in con.execute("SELECT artist_raw, title_raw FROM local_tracks WHERE present=1"):
        keys.add(song_key(r[0], r[1]))
    return ids, keys - {None, "|"}


def suggest_dry_run(con, cfg: dict, provider, dz, n: int = 100, *, classes: list[str] | None = None) -> dict:
    """Ask for n songs with the whole profile and look each up on Deezer (one search each, through the shared
    throttle). Offers nothing and stores nothing. -> {n, returned, resolved, with_preview, known, new, resolve_rate,
    meets, tokens_in, tokens_out, cost_usd, suggestions [{artist, title, why, deezer, preview, known}], reason};
    resolve_rate is resolved / returned."""
    from .harvest import deezer_lookup
    c = cfg["llm"]
    out = {"provider": getattr(provider, "name", c["provider"]), "model": c["model"], "n": n, "returned": 0,
           "resolved": 0, "with_preview": 0, "known": 0, "new": 0, "resolve_rate": None, "meets": None,
           "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "suggestions": [], "reason": None}
    prof = Profile(con, cfg, classes)
    if prof.n < int(c["min_profile_tracks"]):
        return {**out, "reason": f"{prof.n} liked and rated tracks (needs llm.min_profile_tracks, "
                                 f"{c['min_profile_tracks']})"}
    user = suggest_prompt(prof.without(), n)
    log.debug("llm suggest request to %s %s:\n%s", out["provider"], out["model"], user)
    try:
        obj, u = provider.complete(SUGGEST_SYSTEM, user, SUGGEST_SCHEMA, "suggest")
    except Refused as e:
        u = e.usage
        out.update(tokens_in=u["in"], tokens_out=u["out"], cost_usd=cost(cfg, u["in"], u["out"]))
        return {**out, "reason": str(e)}
    out.update(tokens_in=u["in"], tokens_out=u["out"], cost_usd=cost(cfg, u["in"], u["out"]))
    ids, keys = known_songs(con)
    tol = cfg["matching"]["duration_tolerance"]
    for s in (obj.get("suggestions") or [])[:n]:
        artist, title = str(s.get("artist") or "").strip(), str(s.get("title") or "").strip()
        hit = deezer_lookup(dz, artist, title, tol) if artist and title else None
        known = hit is not None and (hit.get("id") in ids or
                                     song_key((hit.get("artist") or {}).get("name", ""), hit.get("title", "")) in keys)
        known = known or song_key(artist, title) in keys
        out["suggestions"].append({"artist": artist, "title": title, "why": str(s.get("why") or ""),
                                   "deezer": hit is not None, "preview": bool(hit and hit.get("preview")),
                                   "known": known})
    sg = out["suggestions"]
    out.update(returned=len(sg), resolved=sum(x["deezer"] for x in sg), with_preview=sum(x["preview"] for x in sg),
               known=sum(x["known"] for x in sg),
               new=sum(x["deezer"] and x["preview"] and not x["known"] for x in sg))
    out["resolve_rate"] = _r(out["resolved"] / len(sg)) if sg else None
    out["meets"] = out["resolve_rate"] is not None and out["resolve_rate"] > RESOLVE_BAR
    return out
