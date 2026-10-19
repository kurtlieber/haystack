#!/usr/bin/env python3
"""haystack's brand kit: the SVG masters, PNG exports and site icons in docs/brand/ (`python3 docs/brand/build.py`).

haystack's look is needle's with straw beside the amber, so this builds on needle's docs/brand/build.py, which
holds both marks, the type, the shared layouts and the export. It needs a needle checkout next to this one, or
NEEDLE set to one. haystack's own motifs are its week (the timers in systemd/), the field of straws the mark
stands in, and the score mound."""
import importlib.util
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = Path(os.environ.get("NEEDLE") or HERE.parents[2] / "needle") / "docs/brand/build.py"
if not SOURCE.exists():
    sys.exit(f"no {SOURCE}: check out needle next to haystack, or set NEEDLE to a checkout")
spec = importlib.util.spec_from_file_location("needle_brand", SOURCE)
sys.modules["needle_brand"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules["needle_brand"])
from needle_brand import (AMBER, BG, BODY, BRIGHT, DIM, FG, HAIR, HAYSTACK_INK, LABEL, MUTED, PAPER,  # noqa: E402
                          STRAW, STRAW_DEEP, SURFACE, TILE, card, disc, exports, fade, glow, haystack_mark, header, kit,
                          lit, marker, n, place, scale, sheet, small_tile, svg, text, tile, web, words, write)

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
HARVEST, BATCH = 5, 8   # the hours of the daily harvest and of Monday's batch (systemd/*.timer)
ROWS = ((4, 9.5, 14.5), (8, 6.5, 17.5), (12, 4.8, 19.2), (16, 3.8, 20.2), (20, 3, 21))   # the mark's straws: y, x0, x1
ARM = "M12 14 19 6.5"
TONEARM = (f'<path d="{ARM}" stroke="{AMBER}" stroke-width="2" stroke-linecap="round"/>'
           f'<circle cx="12" cy="14" r="2.2" fill="{AMBER}"/>')
CUT = f'<path d="{ARM}" stroke="#000" stroke-width="4.6" stroke-linecap="round"/><circle cx="12" cy="14" r="3.5"/>'


# Straws

def straws(segs: list[tuple], width: float, colour: str = STRAW, attrs: str = "") -> str:
    """Straws (x0, y0, x1, y1, opacity) as round-capped strokes, one path for each opacity."""
    d = {}
    for x0, y0, x1, y1, o in segs:
        d[o] = d.get(o, "") + f"M{n(x0)} {n(y0)}L{n(x1)} {n(y1)}"
    return "".join(f'<path d="{p}" fill="none" stroke="{colour}" stroke-opacity="{n(o)}" stroke-width="{n(width)}" '
                   f'stroke-linecap="round"{attrs}/>' for o, p in sorted(d.items()))


def needle_through(uid: str, hay: str, x: float = 0, y: float = 0, size: float = 24, attrs: str = "") -> str:
    """`hay` with needle's spindle and tonearm knocked through it as in the mark, their 24-unit grid drawn `size`
    across from (x, y); `attrs` go on the needle (a glow)."""
    big = 'x="-9999" y="-9999" width="19999" height="19999"'
    return (f'<mask id="{uid}" maskUnits="userSpaceOnUse" {big}><rect {big} fill="#fff"/>{place(CUT, x, y, size)}'
            f'</mask><g mask="url(#{uid})">{hay}</g>' + place(f"<g{attrs}>{TONEARM}</g>", x, y, size))


def field(box: tuple, cx: float, cy: float, s: float, seed: str, pitch: float = .5, width: float = .25,
          reach: float = 1, strength: float = .16, colour: str = STRAW) -> str:
    """Loose straws in rows on the grid of a mark drawn `s` across round (cx, cy), filling `box` (x0, y0, x1, y1):
    brightest by the mark and fading out over `reach` x s, with the mark's own outline kept clear. It is to
    haystack what a record's grooves are to needle."""
    rnd, u = random.Random(seed), s / 24
    first, step = cy - s / 2 + 4 * u, pitch * u
    segs = []
    for j in range(math.floor((box[1] - first) / step), math.ceil((box[3] - first) / step) + 1):
        y, x = first + j * step, box[0] - rnd.uniform(0, 6 * u)
        while x < box[2]:
            run = rnd.uniform(1.5, 7) * u
            gy = (y - cy) / u + 12
            gx0, gx1 = (x - cx) / u + 12, (x + run - cx) / u + 12
            half = 5 + (min(max(gy, 3), 21) - 4) * 6.5 / 16   # the stook's half width and a margin
            clear = 1.5 <= gy <= 22.5 and gx1 > 12 - half and gx0 < 12 + half
            o = strength * (1 - math.dist((x + run / 2, y), (cx, cy)) / (reach * s))
            if not clear and o > .015:
                segs.append((x, y, x + run, y, round(o * 40) / 40))
            x += run + rnd.uniform(.6, 3) * u
    return straws(segs, width * u, colour)


def mound(x: float, y: float, width: float, height: float, cols: int, seed: str, k: float = 1,
          picks: int = 12, explore: int = 3) -> str:
    """A week's candidates as straws, one each, stacked in columns by score from low (left) to high so the stack
    takes a haystack's outline, warming as the score rises. The picks are amber: the top of the right-hand
    tail, and a few from the middle (the share the batch spends exploring)."""
    rnd, pitch = random.Random(seed), 6 * k
    cw, rows = width / cols, int(height // pitch)
    counts = []
    for i in range(cols):
        t = i / (cols - 1)
        z = (t - .38) / (.19 if t < .38 else .3)
        counts.append(max(1, min(rows, round(rows * math.exp(-z * z / 2) * rnd.uniform(.9, 1.06)))))
    found = set([(i, j) for i in reversed(range(cols)) for j in range(counts[i])][:picks])
    while len(found) < picks + explore:
        i = rnd.randrange(round(cols * .45), round(cols * .62))
        found.add((i, rnd.randrange(counts[i] // 3, counts[i])))
    hay, picked = [], []
    for i, count in enumerate(counts):
        for j in range(count):
            cy = y + height - pitch * (j + .5)
            x0, x1 = x + i * cw + 2 * k + rnd.uniform(0, k), x + (i + 1) * cw - 2 * k - rnd.uniform(0, k)
            (picked if (i, j) in found else hay).append((x0, cy, x1, cy, round(.3 + .4 * i / (cols - 1), 2)))
    return straws(hay, 2.6 * k) + straws([s[:4] + (1,) for s in picked], 2.6 * k, AMBER)


# The week

def week(x: float, y: float, width: float, k: float = 1, uid: str | None = None, size: float = 12) -> str:
    """The week from Monday 00:00 on needle's scale, a tick an hour (the sync): a straw at each morning's harvest,
    needle's marker at Monday's batch, and the days under them."""
    def at(hour: float) -> float:
        return x + hour / 168 * (width - k) + k / 2
    harvests = "".join(f"M{n(at(24 * d + HARVEST))} {n(y - 2 * k)}v{n(-12 * k)}" for d in range(7))
    return (scale(x, y, width, k, 168, 24)
            + f'<path d="{harvests}" stroke="{STRAW}" stroke-width="{n(2 * k)}" stroke-linecap="round"/>'
            + marker(at(BATCH), y, k, uid)
            + "".join(text(day.upper(), size, at(24 * d + 12), y + 14 * k + 1.2 * size, MUTED, LABEL, "middle")
                      for d, day in enumerate(DAYS)))


def week_legend(x: float, y: float, k: float = 1) -> str:
    """The week's legend, centred on y from x: an hour's tick, a harvest's straw and the batch's marker."""
    size, out = 12.5 * k, []
    glyphs = (f'<rect x="0" y="-5" width="1.5" height="10" fill="{MUTED}"/>',
              f'<path d="M1 6V-6" stroke="{STRAW}" stroke-width="2.4" stroke-linecap="round"/>',
              f'<path d="M-3-9h8l-4 6zM0-3h2v12h-2z" fill="{AMBER}"/>')
    for glyph, label in zip(glyphs, ("Sync, hourly", "Harvest, daily", "Discoveries, Mondays")):
        d, (_, _, x1, _) = words(label, size, x + 12 * k, y + .36 * size, BODY)
        out.append(f'<g transform="translate({n(x)} {n(y)}) scale({n(k)})">{glyph}</g><path d="{d}" fill="{MUTED}"/>')
        x = x1 + 26 * k
    return "".join(out)


# haystack's assets

def icon(rx: float = 14) -> str:
    """The app icon: the mark lit on needle's matte tile, in its field over a bloom of straw."""
    hay = field((4, 4, 60, 60), 32, 32, 48, "icon", reach=.62, strength=.2)
    return tile(f'<g clip-path="url(#round)">{hay}</g>' + place(haystack_mark("i", attrs=lit("lit")), 8, 8, 48),
                STRAW, "haystack", rx, glow("lit", 1, .6) + '<clipPath id="round"><circle cx="32" cy="32" r="28"/>'
                '</clipPath>')


def social() -> str:
    """The link preview: the mark lit in its field, and the week along the bottom."""
    cx, cy, s = 1010, 236, 360
    hero = (f'<circle cx="{cx}" cy="{cy}" r="330" fill="url(#warm)"/>'
            + field((820, 0, 1280, 500), cx, cy, s, "social", reach=.9)
            + place(haystack_mark("hero", attrs=lit("lit")), cx - s / 2, cy - s / 2, s))
    return card("haystack", "Finds the needle.",
                ("Music discovery that listens to the audio itself,", "with new Discoveries every Monday."), hero,
                week(96, 552, 1088, 2, "mark", 14), fade("warm", STRAW, .1) + glow("lit", .5, .6) + glow("mark", 6, .6))


def banner() -> str:
    """The header: a week's candidates stacked by score, the picks in amber."""
    return header("haystack", haystack_mark("b"), HAYSTACK_INK, "Finds the needle.",
                  mound(880, 104, 600, 158, 30, "banner")
                  + text("Candidates, by score", 15, 880, 294, MUTED, BODY)
                  + text("Discoveries", 15, 1480, 294, MUTED, BODY, "right"))


def wallpaper(w: int, h: int, cx: float, cy: float, s: float) -> str:
    """A wallpaper: the mark lit, big, in its field, which runs out to the corners and fades as it goes."""
    far = max(math.hypot(cx - x, cy - y) for x in (0, w) for y in (0, h))
    defs = (fade("bloom", STRAW, .08) + fade("glint", AMBER, .1) + glow("lit", .5, .6)
            + f'<radialGradient id="shade" cx="{n(cx)}" cy="{n(cy)}" r="{n(far)}" gradientUnits="userSpaceOnUse">'
              '<stop stop-color="#161616"/><stop offset="1" stop-color="#0b0b0b"/></radialGradient>')
    body = (f'<rect width="{w}" height="{h}" fill="url(#shade)"/>'
            f'<circle cx="{n(cx)}" cy="{n(cy)}" r="{n(s * 1.5)}" fill="url(#bloom)"/>'
            f'<circle cx="{n(cx)}" cy="{n(cy + s / 12)}" r="{n(s * .4)}" fill="url(#glint)"/>'
            + field((0, 0, w, h), cx, cy, s, "wallpaper", reach=far / s, strength=.1)
            + place(haystack_mark("wall", attrs=lit("lit")), cx - s / 2, cy - s / 2, s))
    return svg(w, h, body, "haystack", defs)


def sticker() -> str:
    """The round sticker: the mark lit, the week's hours round it with each harvest's straw and the batch's
    marker, the line round the edge."""
    c, r = 300, 176

    def spoke(hour: float, length: float) -> str:
        a = 2 * math.pi * hour / 168
        dx, dy = math.sin(a), -math.cos(a)
        return f"M{n(c + r * dx)} {n(c + r * dy)}l{n(length * dx)} {n(length * dy)}"
    extra = (f'<path d="{"".join(spoke(24 * d + HARVEST, 26) for d in range(7))}" stroke="{STRAW}" '
             f'stroke-width="3.6" stroke-linecap="round"/>'
             f'<g transform="rotate({n(BATCH / 168 * 360)} {c} {c})">{marker(c, c - r, 1.4, "lit")}</g>')
    return disc("haystack · finds the needle · every monday · ", haystack_mark("st", attrs=lit("lit")), 168, 24,
                "haystack sticker", extra=extra)


ANIMATION = """
.s{transform-box:fill-box;transform-origin:center;animation:grow .5s cubic-bezier(.3,.7,.3,1) both}
.arm,.dot,.halo{transform-box:view-box;transform-origin:12px 14px}
.arm{animation:find 1.2s .5s cubic-bezier(.3,.7,.3,1) both}
.dot{animation:land .3s .5s ease-out both}
.halo{animation:ping 2.4s 1.7s ease-out infinite}
@keyframes grow{from{transform:scaleX(0)}}
@keyframes find{from{transform:rotate(80deg);opacity:0}15%{opacity:1}60%{transform:rotate(-6deg)}
80%{transform:rotate(2deg)}to{transform:none}}
@keyframes land{from{transform:scale(0)}}
@keyframes ping{from{opacity:.5;transform:none}40%,to{opacity:0;transform:scale(3.6)}}
@media (prefers-reduced-motion:reduce){*{animation:none!important}}
"""


def animated() -> str:
    """The mark for sites, animated: the straws stack up from the bottom, needle's tonearm swings in through
    them, then the spindle pings every 2.4 s. Still with reduced motion."""
    hay = "".join(f'<path class="s" d="M{x0} {y}H{x1}" style="animation-delay:{n(i * .08)}s"/>'
                  for i, (y, x0, x1) in enumerate(reversed(ROWS)))
    body = (f'<style>{ANIMATION}</style><mask id="a" maskUnits="userSpaceOnUse" x="0" y="0" width="24" height="24">'
            f'<rect width="24" height="24" fill="#fff"/>' + CUT.replace("<path", '<path class="arm"')
            .replace("<circle", '<circle class="dot"') + '</mask>'
            f'<g fill="none" stroke="{STRAW}" stroke-width="2" stroke-linecap="round" mask="url(#a)">{hay}</g>'
            f'<circle class="halo" cx="12" cy="14" r="2.2" fill="{AMBER}" opacity="0"/>'
            + TONEARM.replace("<path", '<path class="arm"').replace("<circle", '<circle class="dot"'))
    return svg(24, 24, body, "haystack", px=256)


def pattern() -> str:
    """A tile for backgrounds, 480 x 480, seamless: stooks in offset rows, every one the mark's straws, and the
    needle in one of them."""
    k, d = 52 / 24, ""
    for j in range(6):
        for i in range(-1, 6):   # the stook off the left edge is the one that runs over the right
            x, y = 80 * i + 40 * (j % 2) + 14, 80 * j + 14
            d += "".join(f"M{n(x + x0 * k)} {n(y + row * k)}H{n(x + x1 * k)}" for row, x0, x1 in ROWS)
    hay = f'<path d="{d}" stroke="{STRAW}" stroke-opacity=".22" stroke-width="{n(2 * k)}" stroke-linecap="round"/>'
    body = f'<rect width="480" height="480" fill="{BG}"/>' + needle_through("m", hay, 254, 174, 52, lit("lit"))
    return svg(480, 480, body, "haystack pattern", glow("lit", .7, .6))


def favicon() -> str:
    """The small cut for browser tabs, its straws deepened when the browser is light."""
    mark = haystack_mark("f", small=True).replace(f'stroke="{STRAW}"', f'class="straw" stroke="{STRAW}"', 1)
    return svg(24, 24, f'<style>@media (prefers-color-scheme:light){{.straw{{stroke:{STRAW_DEEP}}}}}</style>{mark}',
               "haystack")


def brand_sheet() -> str:
    marks = {"colour": haystack_mark("c"), "small": haystack_mark("sm", small=True),
             "white": haystack_mark("w", "#ffffff", "#ffffff", straw_opacity=.45),
             "black": haystack_mark("k", BG, BG, straw_opacity=.45), "light": haystack_mark("l", STRAW_DEEP),
             "ink": HAYSTACK_INK}
    palette = [("Ground", BG), ("Tile", TILE), ("Surface", SURFACE), ("Hairline", HAIR), ("Dim", DIM),
               ("Muted", MUTED), ("Text", FG), ("Bright", BRIGHT), ("Straw", STRAW), ("Straw deep", STRAW_DEEP),
               ("Amber", AMBER), ("Paper", PAPER)]
    motifs = (week(130, 1050, 560, 1.2, "smark", 10) + week_legend(130, 1100, 1.1)
              + field((780, 952, 1000, 1108), 890, 1030, 104, "sheet", 1, .3, .9, .3)
              + place(haystack_mark("sn", attrs=lit("sneedle")), 838, 978, 104)
              + mound(1040, 966, 440, 110, 30, "sheet", .8)
              + text("Candidates, by score", 12, 1040, 1100, MUTED, BODY)
              + text("Discoveries", 12, 1480, 1100, MUTED, BODY, "right"))
    return sheet("haystack", marks, icon(), palette, ("Finds the needle.", "Music discovery that listens to the audio "
                 "itself.", "0.91", "SCORE"), motifs, glow("smark", 4, .6) + glow("sneedle", 1, .6))


def main() -> None:
    small = haystack_mark("m", small=True)
    out = {"haystack-mark": svg(24, 24, haystack_mark("m"), "haystack", px=256),
           "haystack-mark-small": svg(24, 24, small, "haystack", px=32),
           "haystack-mark-white": svg(24, 24, haystack_mark("m", "#ffffff", "#ffffff", straw_opacity=.45), "haystack",
                                      px=256),
           "haystack-mark-black": svg(24, 24, haystack_mark("m", BG, BG, straw_opacity=.45), "haystack", px=256),
           "haystack-mark-on-light": svg(24, 24, haystack_mark("m", STRAW_DEEP), "haystack", px=256),
           "haystack-mark-animated": animated(),
           "haystack-icon": icon(),
           "haystack-icon-small": small_tile(small, "haystack")}
    out |= kit("haystack", HAYSTACK_INK, haystack_mark("m"), haystack_mark("m", STRAW_DEEP))
    out["haystack-social"] = social()
    out["haystack-banner"] = banner()
    out["haystack-wallpaper"] = wallpaper(3840, 2160, 2470, 1240, 520)
    out["haystack-wallpaper-phone"] = wallpaper(1179, 2556, 590, 1460, 420)
    out["haystack-sticker"] = sticker()
    out["haystack-pattern"] = pattern()
    out["haystack-brand-sheet"] = brand_sheet()
    write(out, HERE)
    pngs = exports("haystack", out, (("haystack-social", 1), ("haystack-banner", 2), ("haystack-wallpaper", 1),
                                     ("haystack-wallpaper-phone", 1), ("haystack-sticker", 2),
                                     ("haystack-brand-sheet", 1)))
    web("haystack", out["haystack-icon"], icon(0), out["haystack-icon-small"], favicon(), pngs, HERE)


if __name__ == "__main__":
    main()
