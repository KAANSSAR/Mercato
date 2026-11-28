"""A club's or nation's two main colours, read from its crest or flag.

Feeds the colour layer (pitch deck, slides 41 and 42): primary and secondary hex stored next
to the entity, used as a 20 to 30% tint over the dark base. Wrong reads are fixed in
db/overrides/colours.csv, which always wins.
"""

import colorsys
from io import BytesIO

from PIL import Image, ImageFilter

PALETTE = 6
MIN_SHARE = 0.08      # a colour must cover this much of the crest to count
MIN_SAT = 0.25        # below this a colour is white, black or grey
MIN_DISTANCE = 80     # RGB distance between primary and secondary
BORDER = 0.12         # share of the crest's width treated as its outline and ignored


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _sat(rgb) -> float:
    r, g, b = (c / 255 for c in rgb)
    return colorsys.rgb_to_hsv(r, g, b)[1]


def _dist(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def palette(img: Image.Image) -> list[tuple[tuple[int, int, int], float]]:
    """Main colours of the visible pixels, with their share, most common first."""
    img = img.convert("RGBA")
    img.thumbnail((120, 120))
    alpha = img.getchannel("A").point(lambda a: 255 if a >= 200 else 0)
    # ignore the outer ring: gold and yellow outlines read as the main colour otherwise
    # (Barcelona came out gold and red, Arsenal brown)
    k = max(3, int(BORDER * min(img.size)) | 1)
    inner = alpha.filter(ImageFilter.MinFilter(k))
    data = list(img.get_flattened_data())
    keep = [p[:3] for p, a in zip(data, inner.get_flattened_data()) if a]
    if len(keep) < 0.25 * sum(1 for a in alpha.get_flattened_data() if a):
        keep = [p[:3] for p, a in zip(data, alpha.get_flattened_data()) if a]
    pixels = keep
    if not pixels:
        return []
    small = Image.new("RGB", (len(pixels), 1))
    small.putdata(pixels)
    q = small.quantize(colors=PALETTE, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()
    counts = sorted(q.getcolors(), reverse=True)
    total = sum(n for n, _ in counts)
    return [(tuple(pal[i * 3:i * 3 + 3]), n / total) for n, i in counts]


def main_colours(img: Image.Image) -> tuple[str | None, str | None]:
    """Primary: the most common real colour (not white, black or grey) if one covers enough
    of the crest, else the most common colour (Juventus, Newcastle). Secondary: the next
    common colour clearly different from it."""
    pal = [(c, s) for c, s in palette(img) if s >= MIN_SHARE]
    if not pal:
        return None, None
    coloured = [c for c, _ in pal if _sat(c) >= MIN_SAT]
    primary = coloured[0] if coloured else pal[0][0]
    rest = [c for c, _ in pal if _dist(c, primary) >= MIN_DISTANCE]
    # prefer a real colour for the secondary too, else white or black
    rest_coloured = [c for c in rest if _sat(c) >= MIN_SAT]
    secondary = (rest_coloured or rest or [None])[0]
    return _hex(primary), (_hex(secondary) if secondary else None)


def from_bytes(data: bytes) -> tuple[str | None, str | None]:
    return main_colours(Image.open(BytesIO(data)))
