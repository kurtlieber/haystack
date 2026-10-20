import numpy as np
import pytest

from haystack_music import baseline, config, db, traits, verdicts


def sid(n: str) -> str:
    return n.ljust(22, "0")


def test_deam_scale_maps_one_to_nine_onto_zero_to_one():
    assert traits.deam_unit(1) == 0 and traits.deam_unit(9) == 1 and traits.deam_unit(5) == 0.5
    assert traits.deam_unit(-3) == 0 and traits.deam_unit(12) == 1


def test_class_index_reads_the_model_metadata():
    meta = {"name": "voice_instrumental-discogs-effnet-1", "classes": ["instrumental", "voice"]}
    assert traits.class_index(meta, "instrumental") == 0
    with pytest.raises(ValueError):
        traits.class_index(meta, "danceable")


def test_non_finite_readings_are_dropped():
    assert traits.finite({"loudness": float("-inf"), "tempo": 120.0, "valence": float("nan")}) == {"tempo": 120.0}


def test_weighted_quantiles_and_mean():
    v, w = [0.1, 0.2, 0.9], [1, 1, 1]
    assert baseline.weighted_quantile(v, w, 0.5) == pytest.approx(0.2)
    s = baseline.stats([0.0, 1.0], [1, 3])                         # 3x the weight on 1.0
    assert s["mean"] == pytest.approx(0.75) and s["n"] == 2
    assert s["lo"] <= s["mean"] <= s["hi"]
    assert baseline.stats([], []) is None


def _con(tmp_path):
    con = db.connect(tmp_path / "t.db")
    rows = [("unrated", 1, 0.2, 100.0), ("five", 2, 0.8, 140.0), ("passed", 3, 0.5, 90.0)]
    for name, did, valence, tempo in rows:
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(name),))
        con.execute("INSERT INTO liked_map(spotify_id, use_local, deezer_id, vector_status) VALUES(?,0,?,'ok')",
                    (sid(name), did))
        con.execute("INSERT INTO deezer_vectors(deezer_id, e512) VALUES(?, x'00')", (did,))
        traits.put(con, did, {"valence": valence, "tempo": tempo})
    # a 1-star discovery that's not in Liked Songs still counts
    con.execute("INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status, spotify_id) "
                "VALUES(9, 9, 'A', 'T', 'x', 'offered', 'ok', ?)", (sid("onestar"),))
    con.execute("INSERT INTO deezer_vectors(deezer_id, e512) VALUES(9, x'00')")
    traits.put(con, 9, {"valence": 0.0, "tempo": 60.0})
    verdicts.record(con, sid("five"), "rating", 5)
    verdicts.record(con, sid("passed"), "pass")
    verdicts.record(con, sid("onestar"), "rating", 1)
    return con


def test_baseline_is_the_star_weighted_mean_without_passes(tmp_path):
    con = _con(tmp_path)
    b = baseline.compute(con, config.load(tmp_path / "none.toml"))
    assert b["tracks"] == 3
    # weights: unrated 3, five 5, one-star 1; the Pass is out
    assert b["traits"]["valence"]["mean"] == pytest.approx((3 * 0.2 + 5 * 0.8 + 1 * 0.0) / 9)
    assert b["traits"]["tempo"]["mean"] == pytest.approx((3 * 100 + 5 * 140 + 60) / 9)
    assert b["traits"]["energy"] is None


def test_backfill_measures_discoveries_and_likes_before_the_pool(tmp_path):
    con = db.connect(tmp_path / "t.db")
    for did in (1, 2, 3, 4):
        con.execute("INSERT INTO deezer_vectors(deezer_id, e512) VALUES(?, x'00')", (did,))
    con.execute("INSERT INTO candidates(id, deezer_id, artist, title, source, state, vector_status, score) "
                "VALUES(1, 1, 'a', 'pool-low', 'x', 'pool', 'ok', 0.1), (2, 2, 'b', 'pool-high', 'x', 'pool', 'ok', 0.9),"
                " (3, 3, 'c', 'offered', 'x', 'offered', 'ok', 0.5)")
    con.execute("INSERT INTO offers(candidate_id, spotify_id, added_at) VALUES(3, 'x', 'x')")
    con.execute("INSERT INTO likes(spotify_id, added_at) VALUES('L', 'x')")
    con.execute("INSERT INTO liked_map(spotify_id, use_local, deezer_id, vector_status) VALUES('L', 0, 4, 'ok')")
    assert traits.backfill_order(con, 10) == [3, 4, 2, 1]

    class Dz:
        def track(self, did):
            return {"id": did, "preview": None if did == 2 else f"https://cdn/{did}"}

    a = traits.Analysis(np.zeros(512, np.float32), np.zeros(1280, np.float32), 10, 30.0, None,
                        {"valence": 0.5, "tempo": 120.0})
    out = traits.backfill(con, Dz(), 10, analyse=lambda dz, url, **kw: a)
    assert out == {"todo": 4, "done": 3, "no_preview": 1, "errors": 0}
    assert traits.get(con, [3])[3]["tempo"] == 120.0
    assert traits.get(con, [2])[2] == {}                                   # measured: no preview
    assert traits.backfill_order(con, 10) == []


def test_unknown_track_is_measured_on_demand_through_its_isrc(tmp_path):
    con = db.connect(tmp_path / "t.db")
    track = sid("new")
    con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, duration, isrc) VALUES(?,?,?,?,?)",
                (track, "Song", '["Artist"]', 200.0, "USAAA0000001"))

    class Dz:
        def track_by_isrc(self, isrc):
            return {"id": 77, "title": "Song", "artist": {"name": "Artist"}, "duration": 200,
                    "preview": "https://cdn/77"}

        def search(self, q, limit):
            raise AssertionError("ISRC matched: no search needed")

    a = traits.Analysis(np.ones(512, np.float32), np.zeros(1280, np.float32), 10, 30.0, np.zeros(400, np.float32),
                        {"energy": 0.7})
    cfg = config.load(tmp_path / "none.toml")
    r = traits.measure_track(con, cfg, track, dz=Dz(), analyse=lambda dz, url, **kw: a)
    assert r == {"status": "ok", "deezer_id": 77}
    assert traits.get(con, [77])[77] == {"energy": 0.7}
    assert con.execute("SELECT status, method FROM spotify_deezer").fetchone()[:] == ("ok", "isrc")
    # known now: a second call measures nothing
    assert traits.measure_track(con, cfg, track, dz=Dz(), analyse=None)["status"] == "ok"


def test_tempo_takes_deezers_count_only_when_it_is_the_same_tempo():
    assert traits.reconcile_tempo(84.9, 170.1) == 170.1          # drum & bass counted at half time
    assert traits.reconcile_tempo(128.3, 128) == 128
    assert traits.reconcile_tempo(161.5, 0) == 161.5             # Deezer has no bpm for it
    assert traits.reconcile_tempo(120.0, 97) == 120.0            # unrelated: keep our measurement
    assert traits.reconcile_tempo(90.0, 135) == 135              # 3:2


DNB, HOUSE = "Electronic---Drum n Bass", "Electronic---House"


def styles(top: str, act: float = 0.8) -> np.ndarray:
    s = np.zeros(400, np.float32)
    s[[DNB, HOUSE].index(top)] = act
    return s


@pytest.fixture
def classes(monkeypatch):
    from haystack_music import embed
    names = [DNB, HOUSE] + [f"Other---{i}" for i in range(398)]
    monkeypatch.setattr(embed, "style_classes", lambda: names)
    return names


def test_drum_and_bass_read_at_half_or_two_thirds_is_folded_into_its_range():
    assert traits.fold_tempo(87.0, DNB) == 174.0                 # the half-time snare
    assert traits.fold_tempo(115.1, DNB) == pytest.approx(172.65)  # a triplet feel: 2/3 of the tempo
    assert traits.fold_tempo(172.3, DNB) == 172.3                # already counted the way people count it
    assert traits.fold_tempo(140.0, DNB) == 140.0                # no fold lands in range: keep it
    assert traits.fold_tempo(87.0, "Electronic---Jungle") == 174.0
    assert traits.fold_tempo(87.0, HOUSE) == 87.0                # any other style keeps its tempo
    assert traits.fold_tempo(118.1, DNB, 0.21) == 118.1          # too weak a style guess
    assert traits.fold_tempo(None, DNB) is None
    assert traits.fold_tempo(traits.fold_tempo(87.0, DNB), DNB) == 174.0    # idempotent


def test_top_style_is_the_strongest_class(classes):
    assert traits.top_style(styles(HOUSE, 0.6), classes) == (HOUSE, pytest.approx(0.6))
    assert traits.top_style(None, classes) == (None, 0.0)


def test_deezers_bpm_costs_a_request_only_when_the_track_object_has_none():
    class Dz:
        def __init__(self):
            self.asked = []

        def track(self, did):
            self.asked.append(did)
            if did == 3:
                raise RuntimeError("synthetic")
            return {"id": did, "bpm": 174.1}

    dz = Dz()
    assert traits.deezer_bpm(dz, {"id": 1, "bpm": 0}) == 0       # a /track object: Deezer has no bpm for it
    assert dz.asked == []
    assert traits.deezer_bpm(dz, {"id": 2, "title": "from a search"}) == 174.1
    assert traits.deezer_bpm(dz, {"id": 3}) is None              # failed: the clip's own estimate is used
    assert dz.asked == [2, 3]


def test_a_clip_is_folded_after_deezers_bpm_is_reconciled(monkeypatch, classes):
    seen = {}
    monkeypatch.setattr(traits, "_state", {
        "heads": {}, "musicnn": lambda a: np.zeros((1, 200), np.float32),
        "deam": lambda x: np.array([[5.0, 5.0]], np.float32),
        "rhythm": lambda mono: (seen["bpm"], None, 2.0, None, None),
        "loudness": lambda s: (None, None, -9.0, None)})
    audio, stereo, backbone = np.zeros(16000, np.float32), np.zeros((44100, 2), np.float32), np.zeros((1, 1280))

    def tempo(bpm, hint=None, style=None):
        seen["bpm"] = bpm
        return traits.measure(audio, stereo, backbone, hint, style)["tempo"]

    assert tempo(115.1, None, styles(DNB)) == pytest.approx(172.65)
    assert tempo(86.0, 86.0, styles(DNB)) == 172.0               # Deezer's own count is at half time too
    assert tempo(115.1, None, styles(HOUSE)) == 115.1
    assert tempo(115.1, None, None) == 115.1                     # no style: as before


def test_retempo_folds_the_stored_drum_and_bass_once(tmp_path, classes):
    con = db.connect(tmp_path / "t.db")
    for did, tempo, sty in ((1, 87.0, styles(DNB)), (2, 115.1, styles(DNB)), (3, 172.3, styles(DNB)),
                            (4, 87.0, styles(HOUSE)), (5, 118.1, styles(DNB, 0.21)), (6, 87.0, None)):
        con.execute("INSERT INTO deezer_vectors(deezer_id, e512, style400) VALUES(?, x'00', ?)",
                    (did, None if sty is None else sty.astype(np.float16).tobytes()))
        traits.put(con, did, {"tempo": tempo, "energy": 0.5})
    assert traits.retempo(con) == {"checked": 5, "folded": 2}
    got = {d: r["tempo"] for d, r in traits.get(con, range(1, 7)).items()}
    assert got == {1: 174.0, 2: pytest.approx(172.65), 3: 172.3, 4: 87.0, 5: 118.1, 6: 87.0}
    assert traits.get(con, [1])[1]["energy"] == 0.5
    assert traits.retempo(con) == {"checked": 5, "folded": 0}


def test_each_harvest_folds_the_stored_tempos_unless_told_not_to_retry(tmp_path, monkeypatch):
    from haystack_music import harvest, metadata

    class Dz:
        requests = 0

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    monkeypatch.setattr(harvest, "Deezer", Dz)
    monkeypatch.setattr(metadata, "backfill", lambda con, dz, limit: {})
    folds = []
    monkeypatch.setattr(traits, "retempo", lambda con: folds.append(con) or {"checked": 3, "folded": 1})
    h = harvest.Harvester(db.connect(tmp_path / "t.db"), config.load(tmp_path / "none.toml"))
    for name in ("retry_errors", "backfill_styles", "_readings_backfill", "fallback_backfill"):
        monkeypatch.setattr(h, name, lambda *a: {})
    monkeypatch.setattr(traits, "missing_models", lambda *a: [])
    assert h.run(["no_such_source"], 1)["tempo_fold"] == {"checked": 3, "folded": 1}
    assert h.run(["no_such_source"], 1, retry=False)["tempo_fold"] == {"skipped": True}
    monkeypatch.setattr(traits, "missing_models", lambda *a: ["mood_happy"])
    assert h.run(["no_such_source"], 1)["tempo_fold"] == {"skipped": "models missing"}
    assert folds == [h.con]


def test_a_liked_preview_asks_deezer_for_its_bpm_only_when_readings_are_on(tmp_path, monkeypatch):
    from haystack_music import sync

    class Dz:
        def __init__(self, *a, **k):
            self.asked = []

        def track(self, did):
            self.asked.append(did)
            return {"id": did, "bpm": 174.0}

    made, seen = [], []
    monkeypatch.setattr(sync, "Deezer", lambda *a, **k: made.append(Dz()) or made[-1])
    found = {"id": 7, "title": "from a search", "preview": "https://example.test/7.mp3"}     # no bpm key
    monkeypatch.setattr(traits, "find_deezer", lambda dz, ref, tol: (found, True, "search"))
    monkeypatch.setattr(traits, "analyse_preview",
                        lambda dz, url, readings=True, bpm_hint=None: seen.append((readings, bpm_hint)) or {})
    monkeypatch.setattr(traits, "store", lambda con, did, r: None)
    cfg = config.load(tmp_path / "none.toml")
    for name, missing in (("on", []), ("off", ["mood_happy"])):
        con = db.connect(tmp_path / f"{name}.db")
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid(name),))
        con.execute("INSERT INTO spotify_tracks(spotify_id, name, artists_json, duration) VALUES(?, 'Song', "
                    "'[\"Artist\"]', 200)", (sid(name),))
        con.execute("INSERT INTO liked_map(spotify_id) VALUES(?)", (sid(name),))
        monkeypatch.setattr(traits, "missing_models", lambda *a, m=missing: m)
        assert sync.liked_preview_vectors(con, cfg, 5)["embedded"] == 1
    assert seen == [(True, 174.0), (False, None)]
    assert [d.asked for d in made] == [[7], []]                   # readings off: no bpm, so no extra request

def test_models_fetch_gets_every_model_once(tmp_path):
    class Resp:
        def __init__(self, url):
            self.url = url
            name = url.rsplit("/", 1)[1].rsplit(".", 1)[0]
            wanted = traits.MODELS[name][1]
            self.content = (b'{"name": "%s", "classes": ["%s", "other"]}' % (name.encode(), (wanted or "x").encode())
                            if url.endswith(".json") else b"graph")

        def raise_for_status(self):
            pass

    class Session:
        def __init__(self):
            self.urls = []

        def get(self, url, timeout):
            self.urls.append(url)
            return Resp(url)

    s = Session()
    assert sorted(traits.fetch_models(tmp_path, session=s)) == sorted(traits.MODELS)
    assert all(u.startswith(traits.BASE) for u in s.urls) and len(s.urls) == 2 * len(traits.MODELS)
    assert traits.missing_models(tmp_path) == [] and not list(tmp_path.glob(".*.part"))
    assert traits.fetch_models(tmp_path, session=s) == [] and len(s.urls) == 2 * len(traits.MODELS)
