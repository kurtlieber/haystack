import threading
import urllib.request

from haystack_music import auth, config

REDIRECT = "http://127.0.0.1:8518/spotify/callback"


def test_pasted_address():
    assert auth.pasted(f"  {REDIRECT}?code=abc&state=s1 \n", REDIRECT, "s1") == ("abc", None)
    assert auth.pasted(f"{REDIRECT}?code=abc&state=old", REDIRECT, "s1") == (None, auth.STALE)
    assert auth.pasted(f"{REDIRECT}?error=access_denied&state=s1", REDIRECT, "s1") == (None, "access_denied")
    assert auth.pasted("https://accounts.spotify.com/authorize?x=1", REDIRECT, "s1") == (None, auth.NOT_CALLBACK)
    assert auth.pasted("abc", REDIRECT, "s1") == (None, auth.NOT_CALLBACK)


def test_paste_prompt_asks_again_until_it_gets_a_code(monkeypatch, capsys):
    lines = iter(["", "https://example.com/", f"{REDIRECT}?code=c&state=old", f"{REDIRECT}?code=c&state=s1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    got = []
    auth._read_pasted(REDIRECT, "s1", lambda code, err: got.append((code, err)))
    assert got == [("c", None)]
    out = capsys.readouterr().out
    assert auth.NOT_CALLBACK in out and auth.STALE in out


def test_paste_prompt_stops_at_end_of_input(monkeypatch):
    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    got = []
    auth._read_pasted(REDIRECT, "s1", lambda code, err: got.append((code, err)))
    assert got == []


class FakeSecrets:
    stored = {}

    def __init__(self, cfg):
        self.store = type("S", (), {"where": "test"})()

    def client(self):
        return "cid", "csecret"

    def store_refresh_token(self, tok, issued, scopes):
        FakeSecrets.stored["refresh"] = tok

    def refresh_token(self):
        return FakeSecrets.stored.get("refresh"), {"issued at": "2026-10-19"}


def _world(monkeypatch, tmp_path, port):
    cfg = config.load(tmp_path / "none.toml")
    cfg["spotify"]["redirect_uri"] = f"http://127.0.0.1:{port}/spotify/callback"
    monkeypatch.setattr(auth, "SpotifySecrets", FakeSecrets)
    monkeypatch.setattr(auth.pysecrets, "token_urlsafe", lambda n: "s1")
    exchanged = []
    monkeypatch.setattr(auth, "exchange_code",
                        lambda cid, sec, code, redirect: exchanged.append(code) or {"refresh_token": "r", "scope": ""})
    monkeypatch.setattr(auth, "Spotify", lambda cfg, sec: type("Sp", (), {"me": lambda self: {"id": "u"}})())
    monkeypatch.setattr(auth.status, "update", lambda *a, **kw: None)
    monkeypatch.setattr(auth.status, "token_expiry_info", lambda *a: {})
    monkeypatch.setattr(auth.db, "connect", lambda: type("C", (), {})())
    monkeypatch.setattr(auth.db, "kv_set", lambda *a: None)
    return cfg, exchanged


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_pasted_address_finishes_the_login(monkeypatch, tmp_path):
    port = _free_port()
    cfg, exchanged = _world(monkeypatch, tmp_path, port)
    monkeypatch.setattr("builtins.input", 
                        lambda prompt="": f"http://127.0.0.1:{port}/spotify/callback?code=pc&state=s1")
    assert auth.run(cfg, timeout=5, paste=True) == 0
    assert exchanged == ["pc"] and FakeSecrets.stored["refresh"] == "r"


def test_the_listener_still_works_without_a_terminal(monkeypatch, tmp_path):
    port = _free_port()
    cfg, exchanged = _world(monkeypatch, tmp_path, port)

    def browser():
        import time
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/spotify/callback?code=lc&state=s1", timeout=1)
                return
            except OSError:
                time.sleep(0.05)
    threading.Thread(target=browser, daemon=True).start()
    assert auth.run(cfg, timeout=5, paste=False) == 0
    assert exchanged == ["lc"]


def test_a_busy_port_falls_back_to_pasting(monkeypatch, tmp_path):
    import socket
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0)); busy.listen()
        port = busy.getsockname()[1]
        cfg, exchanged = _world(monkeypatch, tmp_path, port)
        assert auth.run(cfg, timeout=1, paste=False) == 1                   # no terminal: nothing to fall back on
        monkeypatch.setattr("builtins.input",
                            
                        lambda prompt="": f"http://127.0.0.1:{port}/spotify/callback?code=pc&state=s1")
        assert auth.run(cfg, timeout=5, paste=True) == 0
        assert exchanged == ["pc"]
