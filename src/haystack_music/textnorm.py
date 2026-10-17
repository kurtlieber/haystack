"""Pure text normalisation for catalogue matching (artist splitting, version suffixes, keys)."""
from __future__ import annotations

import re
import unicodedata

# Version markers. "Hard" markers mean a different recording (a coincidental duration once let a wrong
# remix through in the experiment), so a mismatch rejects the match. "Soft" markers are edits of the
# same recording: allowed when the duration agrees, but flagged as a variant.
_HARD = [
    ("remix", r"re-?mix(ed)?|rmx"), ("live", r"live|unplugged"), ("acoustic", r"acoustic"),
    ("instrumental", r"instrumental|karaoke"), ("slowed", r"slowed|reverb"), ("sped up", r"sped[\s-]*up|speed[\s-]*up"),
    ("nightcore", r"nightcore"), ("vip", r"vip"), ("bootleg", r"bootleg"), ("rework", r"re-?work|flip"),
    ("dub", r"dub"), ("cover", r"cover"), ("mashup", r"mash-?up"), ("demo", r"demo"), ("acapella", r"a\s?cappella|acapella"),
]
_SOFT = [("edit", r"(radio\s+)?edit"), ("extended", r"extended"), ("club", r"club\s+mix")]
_NEUTRAL = r"remaster(ed)?|original(\s+mix)?|album\s+version|single\s+version|explicit|clean|mono|stereo|deluxe|bonus(\s+track)?|anniversary|\d{4}|version|mix"

_MARK = "|".join(p for _, p in _HARD + _SOFT) + "|" + _NEUTRAL
SUFFIX_DASH = re.compile(r"\s+[-\u2013\u2014]\s+(.*\b(" + _MARK + r")\b.*)$", re.I)
SUFFIX_PAREN = re.compile(r"\s*[\(\[]([^\)\]]*\b(" + _MARK + r")\b[^\)\]]*)[\)\]]", re.I)
FEAT = re.compile(r"\s*[\(\[]\s*(feat\.?|ft\.?|featuring|with)\s+[^\)\]]*[\)\]]|\s+(feat\.?|ft\.?|featuring)\s+.*$", re.I)
_ARTIST_SPLIT = re.compile(r"(?<!\d),\s*|\s*,\s+|\s*;\s*|\s+feat\.?\s+|\s+ft\.?\s+|\s+featuring\s+", re.I)


def key(s: str | None) -> str:
    """Accent-, case- and punctuation-insensitive comparison key."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def split_artists(artist: str | None, extra: list[str] | None = None) -> list[str]:
    """Split comma/semicolon/feat-joined artist strings. '&' and 'x' are NOT split (Simon & Garfunkel).
    `extra` (e.g. the multi-valued ARTISTS tag) is merged in, order preserved, deduplicated by key."""
    out, seen = [], set()
    for part in list(_ARTIST_SPLIT.split(artist or "")) + list(extra or []):
        p = part.strip()
        k = key(p)
        if p and k and k not in seen:
            seen.add(k)
            out.append(p)
    return out


def first_artist(artist: str | None) -> str:
    a = split_artists(artist)
    return a[0] if a else (artist or "").strip()


def split_title(title: str | None) -> tuple[str, str]:
    """-> (base title without version suffix / feat., suffix string or '')."""
    t = (title or "").strip()
    t0, suffix = t, []
    m = SUFFIX_DASH.search(t0)
    if m:
        suffix.append(m.group(1).strip())
        t0 = t0[:m.start()]
    for m in SUFFIX_PAREN.finditer(t0):
        suffix.append(m.group(1).strip())
    t0 = SUFFIX_PAREN.sub("", t0)
    t0 = FEAT.sub("", t0).strip(" -")
    return (t0 or t), " / ".join(suffix)


def version_signature(suffix: str | None) -> tuple[frozenset, frozenset]:
    """-> (hard markers, soft markers). Remixes carry the remixer: 'remix:airshade'."""
    s = (suffix or "").lower()
    hard, soft = set(), set()
    for part in re.split(r"\s*/\s*", s):
        for name, pat in _HARD:
            m = re.search(r"\b(" + pat + r")\b", part)
            if m:
                if name == "remix":
                    who = key(part[:m.start()])
                    who = re.sub(r"\b(the|official|original)\b", "", who).strip()
                    hard.add("remix:" + who if who else "remix")
                else:
                    hard.add(name)
        for name, pat in _SOFT:
            if re.search(r"\b(" + pat + r")\b", part):
                soft.add(name)
    return frozenset(hard), frozenset(soft)


def version_check(suffix_a: str | None, suffix_b: str | None) -> str:
    """'same' | 'variant' (soft edit difference) | 'mismatch' (different recording: remix/live/...)."""
    ha, sa = version_signature(suffix_a)
    hb, sb = version_signature(suffix_b)
    if ha != hb:
        # 'remix' without a remixer name is compatible with 'remix:<name>'
        def loose(x):
            return {("remix" if v.startswith("remix") else v) for v in x}
        named_a = {v for v in ha if v.startswith("remix:")}
        named_b = {v for v in hb if v.startswith("remix:")}
        if loose(ha) != loose(hb) or (named_a and named_b and named_a != named_b):
            return "mismatch"
    return "same" if sa == sb else "variant"


def is_variant_title(title: str | None) -> bool:
    hard, soft = version_signature(split_title(title)[1])
    return bool(hard or soft)


def song_key(artist: str | None, title: str | None) -> str:
    return key(first_artist(artist)) + "|" + key(split_title(title)[0])
