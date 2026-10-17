from haystack_music.matching import Index, Match, TrackRef, compare, use_local_vector

LIB = [
    TrackRef(1, "WEEDMANE, Suicideyear, airshade", "SUICIDE YEAR - airshade Remix", 150.0),
    TrackRef(2, "Disclosure", "When A Fire Starts To Burn", 273.0, spotify_id="3RmmCLIemRIt1vrqJHDv7a"),
    TrackRef(3, "Martin Garrix", "Animals - Extended", 303.0),
    TrackRef(4, "Linkin Park", "One Step Closer", 157.0, isrc="USWB10002407"),
]
IDX = Index(LIB)


def test_spotify_id_first():
    m = IDX.match(TrackRef("x", "Someone Else", "Other", 10, spotify_id="3RmmCLIemRIt1vrqJHDv7a"))
    assert m.ref == 2 and m.method == "spotify_id" and m.certain


def test_isrc_second():
    m = IDX.match(TrackRef("x", "Linkin Park", "One Step Closer", 157.4, isrc="uswb10002407"))
    assert m.ref == 4 and m.method == "isrc"


def test_fuzzy_rejects_wrong_remix_even_with_matching_duration():
    assert IDX.match(TrackRef("x", "WEEDMANE", "SUICIDE YEAR (RAIZHELL Remix)", 150.1)) is None


def test_fuzzy_accepts_same_song_within_tolerance():
    m = IDX.match(TrackRef("x", "Disclosure", "When a Fire Starts to Burn", 274.5))
    assert m.ref == 2 and m.method == "fuzzy" and m.certain


def test_fuzzy_duration_limits():
    assert IDX.match(TrackRef("x", "Disclosure", "When a Fire Starts to Burn", 277.5)) is None   # > 3 s
    m = IDX.match(TrackRef("x", "Disclosure", "When a Fire Starts to Burn", 275.5))             # 2.5 s
    assert m and not m.certain and not use_local_vector(m)


def test_soft_variant_is_uncertain_and_uses_preview():
    m = IDX.match(TrackRef("x", "Martin Garrix", "Animals", 302.0))
    assert m.ref == 3 and m.version == "variant" and not m.certain
    assert not use_local_vector(m)


def test_multi_artist_overlap():
    m = compare(TrackRef("a", "Zabo", "Ultima", 120), TrackRef("b", "ZABO, Social Kid", "Ultima", 121))
    assert m and m.certain


def test_use_local_vector_rule():
    assert use_local_vector(Match(1, "spotify_id", "high", None, 1, 1, "same"))
    assert use_local_vector(Match(1, "fuzzy", "high", 1.5, 1, 1, "same"))
    assert not use_local_vector(Match(1, "fuzzy", "high", 2.5, 1, 1, "same"))
    assert not use_local_vector(None)
