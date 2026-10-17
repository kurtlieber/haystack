from datetime import datetime, timezone

from haystack_music import config
from haystack_music.status import token_expiry_info


def test_token_expiry_warning_after_five_months():
    now = datetime(2027, 3, 10, tzinfo=timezone.utc)
    i = token_expiry_info("2026-10-08T09:30:00Z", 150, 182, now)
    assert i["expiry_warning"] and i["expires_estimate"] == "2027-04-08"
    i = token_expiry_info("2026-10-08T09:30:00Z", 150, 182, datetime(2027, 2, 1, tzinfo=timezone.utc))
    assert not i["expiry_warning"]
    assert token_expiry_info(None, 150, 182)["expiry_warning"] is False


def test_config_override(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[batch]\nsize = 20\n")
    c = config.load(p)
    assert c["batch"]["size"] == 20 and c["batch"]["expiry_days"] == 14


def test_to_rip_is_off_by_default(tmp_path):
    from haystack_music.sync import slots
    cfg = config.load(tmp_path / "none.toml")
    assert cfg["spotify"]["to_rip"] is False and cfg["alert"]["enabled"] is False
    assert slots(cfg) == [("discoveries", "Discoveries")]
    cfg["spotify"]["to_rip"] = True
    assert slots(cfg) == [("discoveries", "Discoveries"), ("to_rip", "To Rip")]


def test_defaults_carry_no_personal_values(tmp_path):
    cfg = config.load(tmp_path / "none.toml")
    assert cfg["library"]["root"] == "" and cfg["secrets"]["backend"] == "files"
    assert not any(cfg["onepassword"][k] for k in ("vault", "token_file", "client_item", "token_item",
                                                   "api_token_item", "alert_item"))
    assert cfg["harvest"]["listenbrainz_contact"] == ""


def test_user_agent_names_the_project():
    from haystack_music import USER_AGENT
    from haystack_music.listenbrainz import ListenBrainz
    assert USER_AGENT.startswith("haystack/") and "github.com" in USER_AGENT
    assert ListenBrainz(session=__import__("requests").Session()).s.headers["User-Agent"] == USER_AGENT
    ua = ListenBrainz(contact="me@example.org").s.headers["User-Agent"]
    assert ua.endswith("; me@example.org)") and ua.startswith("haystack/")


def test_config_example_matches_the_defaults():
    import re
    import tomllib
    from pathlib import Path
    example = Path(__file__).parent.parent / "config.example.toml"
    assert config.load(example) == config.DEFAULTS      # every live line is a default, under a real key
    section = None
    for line in example.read_text().splitlines():       # and every commented example names a real key
        if m := re.match(r"\[(\w+)\]$", line):
            section = m[1]
        elif m := re.match(r"# (\w+ = .*)$", line):
            (key, value), = tomllib.loads(m[1]).items()
            assert type(value) is type(config.DEFAULTS[section][key]), f"{section}.{key}"
