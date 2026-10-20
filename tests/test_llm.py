"""The optional language model (ADR 0019): what a request carries (blind, each request's tracks left out of its
profile), the measurements from a fake model, the providers against fake clients, and that nothing is built or
sent while llm.enabled is false. No network."""
import json
import re
import socket
import sys
from types import SimpleNamespace

import pytest

from haystack_music import baseline, cli, config, llm, rerank, verdicts
from test_rerank import _world, sid

CLASSES = [f"Family{i // 40}---Style{i}" for i in range(400)]
LINE = re.compile(r"^(\d+)\. Artist (lab\d+) - ")


class FakeLLM:
    """Answers judge requests from the track's name: an oracle (lab i is a yes when i is even) or a constant."""
    name = "fake"

    def __init__(self, chance=None, refuse=(), usage=(1000, 200), suggestions=None):
        self.calls, self.refuse, self.usage = [], set(refuse), {"in": usage[0], "out": usage[1]}
        self.chance = chance or (lambda name: 0.9 if int(name[3:]) % 2 == 0 else 0.1)
        self.suggestions = suggestions or []

    def complete(self, system, user, schema, name):
        self.calls.append({"system": system, "user": user, "schema": schema, "name": name})
        if len(self.calls) in self.refuse:
            raise llm.Refused("declined", dict(self.usage))
        if name == "suggest":
            return {"suggestions": self.suggestions}, dict(self.usage)
        tracks = []
        for line in user.split("\n"):
            m = LINE.match(line)
            if m:
                tracks.append({"id": int(m[1]), "chance": self.chance(m[2]), "reason": f"about {m[2]}"})
        return {"tracks": tracks}, dict(self.usage)


def _cfg(cfg, **kw):
    cfg["llm"].update(kw)
    return cfg


def _sections(user: str) -> tuple[str, str]:
    profile, songs = user.split("\n\nSongs\n")
    return profile, songs


# ---------------- what a request carries ----------------

def test_a_judge_request_is_blind_and_leaves_its_own_tracks_out_of_the_profile(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    fake = FakeLLM()
    r = llm.judge(con, cfg, fake, classes=CLASSES)
    labs = rerank.labels(con, cfg)
    assert len(fake.calls) == r["batches"] == -(-len(labs) // llm.BATCH)
    ids = [x["sid"] for x in labs]
    down = {s for s, v in verdicts.current(con).items() if v.kind in ("pass", "dislike")}
    seen_down = set()
    for c in fake.calls:
        u = c["user"]
        assert not any(s in u for s in ids)                       # no Spotify id
        assert not re.search(r"\bpass\b|dislike|stars?\b(?! of 5)|★|verdict", _sections(u)[1], re.I)
        profile, songs = _sections(u)
        asked = re.findall(r"^\d+\. Artist (lab\d+) - ", songs, re.M)
        assert asked and len(asked) == len(set(asked))
        for name in asked:                                        # leave-one-out: not in its own profile
            assert f"Artist {name} (" not in profile
        seen_down |= {sid(n) for n in re.findall(r"Artist (lab\d+) \(\d+\)", profile.split("turned down")[1])}
    assert seen_down and seen_down <= down
    # the rated tracks of the other requests are there
    assert re.search(r"Artist lab\d+ \(1, [345]\.0\)", fake.calls[0]["user"])


def test_track_facts_take_spotify_first_then_the_candidate_and_name_styles_with_their_family(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, album, release_date) VALUES(?,?,?,?,?)",
                (sid("lab0"), "Real Title", json.dumps(["One", "Two"]), "An Album", "2019-05-01"))
    labs = rerank.labels(con, cfg)
    f = llm.track_facts(con, labs, CLASSES)
    a, b = f[sid("lab0")], f[sid("lab1")]
    assert (a["artist"], a["title"], a["album"], a["year"]) == ("One, Two", "Real Title", "An Album", 2019)
    assert (b["artist"], b["title"], b["album"], b["year"]) == ("Artist lab1", "Title lab1", None, None)
    assert len(a["styles"]) == 3 and all(" / " in s for s, _ in a["styles"])
    assert a["styles"][0][1] >= a["styles"][1][1] >= a["styles"][2][1]
    assert b["source"] == "src_b" or b["source"] == "src_a"      # an unknown source goes by its own name
    line = llm._track_line(1, a)
    assert line.startswith("1. One, Two - Real Title | album: An Album (2019) | styles: Family")
    assert llm.track_facts(con, labs)[sid("lab0")]["styles"] == []    # no class names: no styles


def test_the_profile_counts_stars_styles_and_trait_ranges_without_the_tracks_asked_about(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    p = llm.Profile(con, cfg, CLASSES)
    full = p.without()
    assert full["tracks"] == len(baseline.liked_weights(con, cfg)) == p.n
    assert sum(x for _, x in full["styles"]) <= 1.0 and len(full["styles"]) <= llm.TOP_STYLES
    assert set(full["traits"]) == {"valence"} and full["traits"]["valence"][0] <= full["traits"]["valence"][1]
    rated = [s for s, v in verdicts.current(con).items() if v.kind == "rating" and v.stars >= 3]
    less = p.without(rated[:5])
    assert less["tracks"] == full["tracks"] - 5
    names = {a for a, _, _ in less["artists"]}
    assert not names & {f"Artist {s.rstrip('z')}" for s in rated[:5]}
    text = llm.profile_text(full)
    assert text.startswith("Listener profile\n") and "valence 0." in text and " to 0." in text and "Family" in text
    assert llm._trait_range("loudness", -10.94, -7.2) == "loudness -10.9 to -7.2 LUFS"
    assert llm._trait_range("tempo", 107.0, 136.0) == "tempo 107 to 136 BPM"


def test_the_schemas_close_every_object_and_set_no_bounds():
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert sorted(node["required"]) == sorted(node["properties"])
            assert not {"minimum", "maximum", "minLength", "maxLength"} & set(node)
            for v in node.values():
                walk(v)
    for s in (llm.JUDGE_SCHEMA, llm.SUGGEST_SCHEMA):
        walk(s)


# ---------------- the measurements ----------------

def test_the_chance_alone_and_the_re_ranker_with_it(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    # the verdicts follow valence in this world; flatten it so only the model's chance tells yes from no
    con.execute("UPDATE readings SET valence=0.5 WHERE deezer_id IN (SELECT deezer_id FROM candidates WHERE id < 1000)")
    r = llm.judge(con, cfg, FakeLLM(), classes=CLASSES)
    assert r["reason"] is None and r["n_missing"] == 0 and r["refused"] == 0
    assert r["chance_auc"] == 1.0
    lo = r["loo"]
    assert lo["n"] == r["n_labels"] and lo["dropped"] == 0
    assert lo["with"] > lo["without"] + 0.2 and lo["gain"] == pytest.approx(lo["with"] - lo["without"], abs=1e-4)
    assert r["meets"]["rerank_job"] is True
    assert (r["tokens_in"], r["tokens_out"]) == (1000 * r["batches"], 200 * r["batches"])
    assert r["cost_usd"] == pytest.approx(r["batches"] * (1000 * 4 + 200 * 20) / 1e6)
    flat = llm.judge(con, cfg, FakeLLM(chance=lambda name: 0.5), classes=CLASSES)
    assert flat["chance_auc"] == 0.5 and flat["meets"]["rerank_job"] is False


def test_a_refused_request_counts_its_tracks_as_missing_and_still_costs(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    r = llm.judge(con, cfg, FakeLLM(refuse=(1,)), classes=CLASSES)
    assert r["refused"] == 1 and r["n_missing"] == llm.BATCH
    assert r["tokens_in"] == 1000 * r["batches"]
    assert sum(t["chance"] is None for t in r["tracks"]) == llm.BATCH
    assert r["loo"]["dropped"] == llm.BATCH and r["chance_auc"] == 1.0


def test_unknown_ids_are_ignored_and_chances_clamped(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch, n_lab=8)

    class Odd(FakeLLM):
        def complete(self, system, user, schema, name):
            self.calls.append(user)
            return {"tracks": [{"id": 0, "chance": 0.5, "reason": ""}, {"id": 99, "chance": 0.5, "reason": ""},
                               {"id": 1, "chance": 1.7, "reason": "a"}, {"id": 1, "chance": 0.2, "reason": "b"},
                               {"id": 2, "chance": -3, "reason": "c"}, {"id": "x", "chance": 0.5, "reason": ""}]}, \
                {"in": 1, "out": 1}
    cfg["llm"]["min_profile_tracks"] = 1
    r = llm.judge(con, cfg, Odd(), classes=CLASSES)
    got = [t for t in r["tracks"] if t["chance"] is not None]
    assert sorted(t["chance"] for t in got) == [0.0, 1.0] and r["n_missing"] == 6
    assert {t["reason"] for t in got} == {"a", "c"}


def test_too_small_a_profile_sends_nothing(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    fake = FakeLLM()
    r = llm.judge(con, _cfg(cfg, min_profile_tracks=10_000), fake)
    assert "llm.min_profile_tracks" in r["reason"] and fake.calls == []


def test_baseline_leaves_out_what_it_is_told_to(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    assert baseline.compute(con, cfg, exclude=set(baseline.liked_weights(con, cfg)))["tracks"] == 0


# ---------------- suggest --dry-run ----------------

class FakeDeezer:
    def __init__(self, tracks):
        self.tracks, self.queries = tracks, []

    def search(self, q, limit=25):
        self.queries.append(q)
        return [t for t in self.tracks if t["title"].lower() in q.lower()][:limit]


def test_suggest_dry_run_counts_what_deezer_has(tmp_path, monkeypatch):
    con, cfg, _ = _world(tmp_path, monkeypatch)
    known_did = con.execute("SELECT deezer_id FROM candidates WHERE id=1000").fetchone()[0]
    dz = FakeDeezer([
        {"id": 1, "title": "Fresh Song", "artist": {"name": "New Band"}, "duration": 200, "preview": "https://x/p"},
        {"id": known_did, "title": "Title pool0", "artist": {"name": "Artist pool0"}, "duration": 200,
         "preview": "https://x/q"},
        {"id": 3, "title": "Quiet Song", "artist": {"name": "Shy Band"}, "duration": 200, "preview": ""}])
    sugg = [{"artist": "New Band", "title": "Fresh Song", "why": "w"},
            {"artist": "Artist pool0", "title": "Title pool0", "why": "w"},
            {"artist": "Shy Band", "title": "Quiet Song", "why": "w"},
            {"artist": "Nobody", "title": "Imaginary", "why": "w"}]
    fake = FakeLLM(suggestions=sugg)
    r = llm.suggest_dry_run(con, cfg, fake, dz, n=4, classes=CLASSES)
    assert (r["returned"], r["resolved"], r["with_preview"], r["known"], r["new"]) == (4, 3, 2, 1, 1)
    assert r["resolve_rate"] == 0.75 and r["meets"] is True
    assert len(dz.queries) == 4 and fake.calls[0]["name"] == "suggest"
    assert "Name 4 songs" in fake.calls[0]["user"] and "Songs\n" not in fake.calls[0]["user"]


# ---------------- the providers ----------------

class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    def __init__(self, msg):
        self.kw, self.msg = None, msg
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kw):
        self.kw = kw
        return _Stream(self.msg)


def _msg(stop, blocks, details=None):
    return SimpleNamespace(stop_reason=stop, stop_details=details, content=blocks,
                           usage=SimpleNamespace(input_tokens=120, output_tokens=30))


def test_the_anthropic_provider_checks_stop_reason_before_it_reads_the_answer():
    text = SimpleNamespace(type="text", text='{"tracks": []}')
    thinking = SimpleNamespace(type="thinking", thinking="...")
    c = FakeClient(_msg("end_turn", [thinking, text]))
    p = llm.AnthropicProvider("claude-opus-5-5", client=c)
    obj, u = p.complete("sys", "user", llm.JUDGE_SCHEMA, "judge")
    assert obj == {"tracks": []} and u == {"in": 120, "out": 30}
    assert c.kw["output_config"] == {"format": {"type": "json_schema", "schema": llm.JUDGE_SCHEMA}}
    assert c.kw["model"] == "claude-opus-5-5" and c.kw["max_tokens"] == llm.MAX_TOKENS and c.kw["system"] == "sys"
    assert "thinking" not in c.kw and "budget_tokens" not in json.dumps(c.kw, default=str)
    for stop in ("refusal", "max_tokens"):
        with pytest.raises(llm.Refused) as e:
            llm.AnthropicProvider("m", client=FakeClient(_msg(stop, [text]))).complete("s", "u", {}, "judge")
        assert e.value.usage == {"in": 120, "out": 30}
    with pytest.raises(llm.Refused):
        llm.AnthropicProvider("m", client=FakeClient(_msg("end_turn", [thinking]))).complete("s", "u", {}, "j")


class _Resp:
    def __init__(self, status, body):
        self.status_code, self.body, self.text = status, body, json.dumps(body)

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, *resps):
        self.resps, self.posts = list(resps), []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return self.resps.pop(0)


def test_the_openai_compatible_provider_asks_for_the_schema_and_reads_usage():
    ok = _Resp(200, {"choices": [{"message": {"content": '{"tracks": []}'}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 50, "completion_tokens": 7}})
    s = FakeSession(ok)
    p = llm.OpenAICompatibleProvider("local", "http://127.0.0.1:11434/v1/", session=s)
    assert p.complete("sys", "user", llm.JUDGE_SCHEMA, "judge") == ({"tracks": []}, {"in": 50, "out": 7})
    post = s.posts[0]
    assert post["url"] == "http://127.0.0.1:11434/v1/chat/completions" and post["headers"] == {}
    assert post["json"]["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "judge", "schema": llm.JUDGE_SCHEMA, "strict": True}}
    assert [m["role"] for m in post["json"]["messages"]] == ["system", "user"]
    s = FakeSession(_Resp(200, {"choices": [{"message": {"content": "{"}, "finish_reason": "length"}]}))
    with pytest.raises(llm.Refused):
        llm.OpenAICompatibleProvider("m", "http://h/v1", api_key="k", session=s).complete("s", "u", {}, "j")
    assert s.posts[0]["headers"] == {"Authorization": "Bearer k"}
    with pytest.raises(llm.LLMError, match="HTTP 400"):
        llm.OpenAICompatibleProvider("m", "http://h/v1", session=FakeSession(_Resp(400, {"error": "x"}))) \
            .complete("s", "u", {}, "j")
    with pytest.raises(llm.LLMError, match="base_url"):
        llm.OpenAICompatibleProvider("m", "")


class _Store:
    def __init__(self, key=None):
        self.key, self.asked = key, []

    def find(self, name):
        self.asked.append(name)
        return self.key


def test_make_provider_builds_nothing_while_disabled(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)          # `import anthropic` fails
    cfg = config.load(None)
    st = _Store("k")
    with pytest.raises(llm.LLMError, match="llm.enabled is false"):
        llm.make_provider(cfg, st)
    assert st.asked == []                                         # not even the secret is read
    cfg["llm"]["enabled"] = True
    with pytest.raises(llm.LLMError, match="haystack-music\\[llm\\]"):
        llm.make_provider(cfg, st)
    with pytest.raises(llm.LLMError, match="llm-api-key"):
        llm.make_provider(cfg, _Store(None))
    cfg["llm"].update(provider="openai_compatible", base_url="http://127.0.0.1:1/v1")
    assert isinstance(llm.make_provider(cfg, _Store(None)), llm.OpenAICompatibleProvider)
    cfg["llm"]["provider"] = "other"
    with pytest.raises(llm.LLMError, match="unknown llm.provider"):
        llm.make_provider(cfg, st)


# ---------------- the command ----------------

def test_the_command_sends_nothing_while_disabled_and_its_dry_run_shows_the_request(tmp_path, monkeypatch, capsys):
    con, cfg, path = _world(tmp_path, monkeypatch)
    con.close()
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setitem(sys.modules, "anthropic", None)

    def no_network(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(llm, "make_provider", lambda *a, **k: no_network())
    p = cli.build_parser()
    assert cli.cmd_llm(cfg, p.parse_args(["llm", "judge"])) == 2
    assert "llm.enabled is false" in capsys.readouterr().err
    assert cli.cmd_llm(cfg, p.parse_args(["llm", "suggest", "--dry-run"])) == 2
    capsys.readouterr()
    assert cli.cmd_llm(cfg, p.parse_args(["llm", "suggest"])) == 2
    assert "only --dry-run" in capsys.readouterr().err
    assert cli.cmd_llm(cfg, p.parse_args(["llm", "judge", "--dry-run"])) == 0
    out = capsys.readouterr().out
    assert out.startswith("--- system ---\nYou help a music discovery tool")
    assert "--- request 1 of 2 ---\nListener profile\n" in out and "\n1. Artist lab" in out
    assert "60 labels (30 yes, 30 no) and 0 soft, 2 requests; dry run: nothing was sent" in out
    assert sid("lab0") not in out


def test_the_command_reports_the_measurements(tmp_path, monkeypatch, capsys):
    con, cfg, path = _world(tmp_path, monkeypatch)
    con.close()
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(llm, "make_provider", lambda cfg: FakeLLM())
    _cfg(cfg, enabled=True)
    p = cli.build_parser()
    assert cli.cmd_llm(cfg, p.parse_args(["llm", "judge", "--tracks"])) == 0
    out = capsys.readouterr().out
    assert "60 labels (30 yes, 30 no) and 0 soft, 2 requests to fake claude-opus-5-5" in out
    assert "the chance alone, AUC 1.0" in out and "re-ranker leave-one-out AUC" in out
    assert "2000 tokens in, 400 out: about $0.02" in out and "decision rule (AUC >= 0.7 or +0.05): met" in out
    assert "  0.90  4★" in out and "about lab" in out
    assert cli.cmd_llm(cfg, p.parse_args(["llm", "judge", "--json"])) == 0
    assert json.loads(capsys.readouterr().out)["chance_auc"] == 1.0
