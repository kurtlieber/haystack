from haystack_music.textnorm import (is_variant_title, key, song_key, split_artists, split_title,
                                        version_check, version_signature)


def test_split_artists_comma_joined():
    assert split_artists("WEEDMANE, Suicideyear, airshade") == ["WEEDMANE", "Suicideyear", "airshade"]
    assert split_artists("Simon & Garfunkel") == ["Simon & Garfunkel"]
    assert split_artists("10,000 Maniacs") == ["10,000 Maniacs"]
    assert split_artists("2 Edit,Torro Torro") == ["2 Edit", "Torro Torro"]
    assert split_artists("A feat. B") == ["A", "B"]
    assert split_artists("A, B", ["B", "C"]) == ["A", "B", "C"]


def test_split_title_suffixes():
    assert split_title("Strobe - Radio Edit") == ("Strobe", "Radio Edit")
    assert split_title("Levels - Original Mix") == ("Levels", "Original Mix")
    assert split_title("Song (Remastered 2011)") == ("Song", "Remastered 2011")
    assert split_title("Mo Money Mo Problems (feat. Puff Daddy & Mase) - 2014 Remaster")[0] == "Mo Money Mo Problems"
    assert split_title("Mix Tape") == ("Mix Tape", "")
    assert split_title("Dub Be Good To Me") == ("Dub Be Good To Me", "")


def test_version_signature_and_check():
    assert version_signature("airshade Remix")[0] == frozenset({"remix:airshade"})
    assert version_check("airshade Remix", "RAIZHELL Remix") == "mismatch"   # the experiment's wrong remix
    assert version_check("", "Live at Wembley") == "mismatch"
    assert version_check("", "Radio Edit") == "variant"
    assert version_check("2014 Remaster", "") == "same"
    assert version_check("Original Mix", "") == "same"
    assert version_check("Remix", "airshade Remix") == "same"
    assert is_variant_title("Animals - Extended")
    assert not is_variant_title("Animals")


def test_keys():
    assert key("Beyoncé & JAY-Z") == "beyonce and jay z"
    assert song_key("ZABO, Social Kid", "Ultima") == song_key("Zabo", "Ultima")
