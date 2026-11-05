"""The listener's scene, for the sources that follow it (ADR 0026): their top Discogs styles as Last.fm and
ListenBrainz write them, Deezer's own genre editors for those styles, and the artists they already have.

The tables here hold no listener data. TAG_SPELLING maps a Discogs style to the Last.fm tag its listeners use
(probed 2026-11 by tag.getInfo: the spelling with the most taggings); any other style is its name lower-cased with
"&" as "and". DEEZER_GENRES maps a Discogs genre to Deezer's genres, and GENRE_EDITORS each Deezer genre to the
Deezer account that curates its playlists (one probe of /genre and the editorial playlists per genre)."""
from __future__ import annotations

import json
import logging

from .textnorm import key

log = logging.getLogger("haystack_music.scene")

TAG_SPELLING = {
    "Drum n Bass": "drum and bass",
    "Hip Hop": "hip-hop",
    "Trip Hop": "trip-hop",
    "Synth-pop": "synthpop",
    "Psy-Trance": "psytrance",
    "Brit Pop": "britpop",
    "RnB/Swing": "new jack swing",
    "Contemporary R&B": "rnb",
    "Hardcore Hip-Hop": "hardcore hip hop",
    "Jazzy Hip-Hop": "jazz hop",
    "Cut-up/DJ": "turntablism",
    "UK Garage": "uk garage",
    "Bassline": "bassline",
    "Halftime": "halftime",
}
# other ways a style is written as a tag, for matching ListenBrainz's release tags only
TAG_ALIASES = {"Drum n Bass": ("dnb",), "Contemporary R&B": ("r and b",)}

# Discogs genre (the part before " / " in a style's full name) -> Deezer genre ids
DEEZER_GENRES = {
    "Electronic": (106, 113),        # Electro, Dance
    "Hip Hop": (116,),               # Rap/Hip Hop
    "Rock": (152, 464, 85),          # Rock, Metal, Alternative
    "Pop": (132,),
    "Funk / Soul": (169, 165),       # Soul & Funk, R&B
    "Reggae": (144,),
    "Jazz": (129,),
    "Blues": (153,),
    "Latin": (197,),
    "Classical": (98,),
    "Folk, World, & Country": (466, 84),   # Folk, Country
    "Stage & Screen": (173,),        # Films/Games
}

# Deezer genre id -> the Deezer user id of its editor (a playlist's creator.id). Deezer's country editors (Brasil,
# France, Deutschland, UK, LATAM) are deliberately absent: their playlists follow a country, not a style.
GENRE_EDITORS = {
    106: 2834392844, 113: 2834392844,    # Dance & EDM editor
    116: 917475151,
    165: 2928001764,
    152: 753566875,
    464: 577457871,
    85: 753543485,
    132: 753546365,
    466: 753548415,
    84: 4294642822,
    144: 753547275,
    129: 902801015, 153: 902801015,      # Jazz & Blues editor
    169: 2862678264,
    197: 3115986664,
    98: 353978015,
    173: 796271495,
}


def lastfm_tag(style: str) -> str:
    """The Last.fm tag for a Discogs style name ("Drum n Bass" -> "drum and bass")."""
    return TAG_SPELLING.get(style) or style.lower().replace(" & ", " and ")


def tag_keys(style: str) -> set[str]:
    """The keys (textnorm.key) a tag naming this style can have: 'drum & bass', 'Drum and Bass' and 'dnb' all
    name Drum n Bass."""
    return {k for k in (key(style), key(lastfm_tag(style)), *map(key, TAG_ALIASES.get(style, ()))) if k}


def split_style(full: str) -> tuple[str, str]:
    """'Electronic / Bassline' -> ('Electronic', 'Bassline'); 'Funk / Soul / Disco' -> ('Funk / Soul', 'Disco')."""
    genre, _, style = full.rpartition(" / ")
    return genre, style


def editors(genres) -> set[int]:
    """The Deezer editors of these Discogs genres."""
    return {GENRE_EDITORS[g] for d in genres for g in DEEZER_GENRES.get(d, ()) if g in GENRE_EDITORS}


def top_styles(con, cfg: dict, n: int) -> list[tuple[str, set[str]]]:
    """[(style name, {its Discogs genres})] for the n styles with the most activation over the liked songs
    (evaluate.liked_genres), most first. A style name under two genres (Trip Hop is Electronic and Hip Hop) is one
    entry with both."""
    from .evaluate import liked_genres
    try:
        top = liked_genres(con, cfg, top=n).get("top_styles_mean_activation", [])
    except (OSError, ValueError, KeyError) as e:         # style vectors stored, but no style model here
        log.warning("no style names (%s)", e)
        return []
    out: dict[str, set[str]] = {}
    for full, _ in top:
        genre, name = split_style(full)
        if name in out or len(out) < n:
            out.setdefault(name, set()).add(genre)
    return list(out.items())


def positive_artists(con) -> set[str]:
    """Artist keys the listener already has: every artist of a liked song, of a Rating of 3 stars or more, of a
    present library file and of a discovery they kept."""
    from . import verdicts
    out: set[str] = set()

    def add(names):
        out.update(k for k in map(key, names) if k)

    cur = verdicts.current(con)
    keep = {sid for sid, v in cur.items() if v.keeps_heart}
    keep |= {r[0] for r in con.execute("SELECT spotify_id FROM likes WHERE removed_at IS NULL")}
    for sid, arts in con.execute("SELECT spotify_id, artists_json FROM spotify_tracks"):
        if sid in keep:
            add(json.loads(arts or "[]"))
    for arts, raw in con.execute("SELECT artists_json, artist_raw FROM local_tracks WHERE present=1"):
        add(json.loads(arts or "[]") or [raw or ""])
    for sid, artist, outcome in con.execute("""SELECT o.spotify_id, c.artist, o.outcome FROM offers o
                                               JOIN candidates c ON c.id=o.candidate_id
                                               WHERE o.outcome IN ('liked', 'rated')"""):
        v = cur.get(sid)
        if v.keeps_heart if v else outcome == "liked":     # kept, as the report counts it
            add([artist or ""])
    return out
