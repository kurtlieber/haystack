import io
import struct

from haystack_music.library import audio_hash, find_mdat, parse_tags


def atom(typ, payload):
    return struct.pack(">I4s", 8 + len(payload), typ) + payload


def test_find_mdat_and_hash_ignores_tag_changes():
    audio = bytes(range(256)) * 2000
    a = atom(b"ftyp", b"M4A " * 4) + atom(b"moov", b"x" * 100) + atom(b"mdat", audio)
    b = atom(b"ftyp", b"M4A " * 4) + atom(b"moov", b"y" * 400) + atom(b"mdat", audio)   # retagged, moov grew
    off, ln = find_mdat(io.BytesIO(a), len(a))
    assert ln == len(audio) and a[off:off + 4] == audio[:4]
    assert audio_hash(io.BytesIO(a), len(a)) == audio_hash(io.BytesIO(b), len(b))
    c = atom(b"ftyp", b"M4A " * 4) + atom(b"mdat", audio[:-1] + b"\x00")
    assert audio_hash(io.BytesIO(c), len(c)) != audio_hash(io.BytesIO(a), len(a))


class FF:
    def __init__(self, b):
        self.b = b

    def decode(self, *a):
        return self.b.decode(*a)

    def __bytes__(self):
        return self.b


def test_parse_tags_spotify_id_from_musicbrainz_field():
    t = {"\xa9ART": ["WEEDMANE, Suicideyear"], "\xa9nam": ["SUICIDE YEAR - airshade Remix"],
         "----:com.apple.iTunes:MusicBrainz Track Id": [FF(b"3RmmCLIemRIt1vrqJHDv7a")],
         "----:com.apple.iTunes:ARTISTS": [FF(b"WEEDMANE"), FF(b"airshade")],
         "----:com.apple.iTunes:ISRC": [FF(b"")]}
    r = parse_tags(t, 150.04)
    assert r["spotify_id"] == "3RmmCLIemRIt1vrqJHDv7a"
    assert r["title_base"] == "SUICIDE YEAR" and r["version_suffix"] == "airshade Remix"
    assert r["isrc"] is None and r["duration"] == 150.04
    assert "airshade" in r["artists_json"] and r["artist_key"] == "weedmane"


def test_parse_tags_rejects_real_mbid():
    t = {"\xa9ART": ["X"], "\xa9nam": ["Y"],
         "----:com.apple.iTunes:MusicBrainz Track Id": [FF(b"fb3afa92-e197-438e-a541-84ff3969d38a")]}
    assert parse_tags(t, 1.0)["spotify_id"] is None


# ---------------- scan guards: no library, and a root that isn't there ----------------

def _scan_cfg(tmp_path, root):
    from haystack_music import config
    cfg = config.load(tmp_path / "none.toml")
    cfg["library"]["root"] = root
    return cfg


def test_scan_without_a_library_is_skipped(tmp_path):
    from haystack_music import db, library
    con = db.connect(tmp_path / "t.db")
    assert library.scan(con, _scan_cfg(tmp_path, ""))["skipped"] is True


def test_scan_of_a_missing_root_fails_and_keeps_the_tracks(tmp_path):
    import pytest
    from haystack_music import db, library
    con = db.connect(tmp_path / "t.db")
    con.execute("INSERT INTO local_tracks(path, size, content_hash) VALUES('/m/a.m4a', 1, 'h')")
    with pytest.raises(library.LibraryMissing, match="not a directory"):
        library.scan(con, _scan_cfg(tmp_path, str(tmp_path / "gone")))
    (tmp_path / "empty").mkdir()
    with pytest.raises(library.LibraryMissing, match="1 tracks are known"):
        library.scan(con, _scan_cfg(tmp_path, str(tmp_path / "empty")))
    assert con.execute("SELECT present FROM local_tracks").fetchone()[0] == 1


def test_scan_of_a_new_empty_library_is_fine(tmp_path):
    from haystack_music import db, library
    con = db.connect(tmp_path / "t.db")
    (tmp_path / "music").mkdir()
    assert library.scan(con, _scan_cfg(tmp_path, str(tmp_path / "music")))["files"] == 0


def test_no_library_needs_no_embedding(tmp_path):
    from haystack_music import db
    from haystack_music.batch import embedding_progress
    con = db.connect(tmp_path / "t.db")
    ok, msg, prog = embedding_progress(con, _scan_cfg(tmp_path, ""))
    assert ok and "no local library" in msg and prog["fraction"] is None
