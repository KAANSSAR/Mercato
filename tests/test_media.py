"""Colours from crests and flags, and flag codes."""

from io import BytesIO

from PIL import Image, ImageDraw


def _crest(colours, size=100, background=(0, 0, 0, 0)):
    """Vertical stripes of the given colours on a transparent background, with a margin."""
    img = Image.new("RGBA", (size, size), background)
    d = ImageDraw.Draw(img)
    w = (size - 20) // len(colours)
    for i, c in enumerate(colours):
        d.rectangle([10 + i * w, 10, 10 + (i + 1) * w - 1, size - 10], fill=c + (255,))
    return img


def _rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _close(h, rgb, tol=40):
    return sum((a - b) ** 2 for a, b in zip(_rgb(h), rgb)) ** 0.5 < tol


def test_blue_and_garnet_club():
    from mercato.media.colours import main_colours
    blue, garnet = (0, 77, 152), (165, 0, 68)
    p, s = main_colours(_crest([blue, garnet, blue, garnet, blue]))
    assert _close(p, blue) and _close(s, garnet)


def test_black_and_white_club_keeps_black_and_white():
    from mercato.media.colours import main_colours
    p, s = main_colours(_crest([(0, 0, 0), (255, 255, 255), (0, 0, 0)]))
    assert {p, s} == {"#000000", "#ffffff"}


def test_a_real_colour_beats_a_white_background_panel():
    from mercato.media.colours import main_colours
    red = (200, 16, 46)
    p, _ = main_colours(_crest([(255, 255, 255), (255, 255, 255), red]))
    assert _close(p, red)


def test_transparent_image_has_no_colours():
    from mercato.media.colours import main_colours
    assert main_colours(Image.new("RGBA", (50, 50), (0, 0, 0, 0))) == (None, None)


def test_from_png_bytes():
    from mercato.media.colours import from_bytes
    buf = BytesIO()
    _crest([(0, 120, 60), (255, 220, 0)]).save(buf, "PNG")
    p, s = from_bytes(buf.getvalue())
    assert p is not None and s is not None


def test_flag_codes():
    from mercato.media.nations import flag_code
    assert flag_code("England") == "gb-eng"
    assert flag_code("Spain") == "es"
    assert flag_code("Korea, South") == "kr"
    assert flag_code("Cote d'Ivoire") == "ci"
    assert flag_code("United States") == "us"
    assert flag_code(None) is None


def test_auth_headers_for_new_and_legacy_keys():
    from mercato.jobs.media import auth_headers
    assert auth_headers("sb_secret_abc") == {"apikey": "sb_secret_abc"}
    legacy = auth_headers("eyJhbGciOi.x.y")
    assert legacy["Authorization"] == "Bearer eyJhbGciOi.x.y"


def test_gold_outline_is_ignored():
    """Barcelona-style crest: a thick gold ring round blue and garnet stripes."""
    from mercato.media.colours import main_colours
    gold, blue, garnet = (237, 187, 0), (0, 77, 152), (165, 0, 68)
    img = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([0, 0, 119, 119], fill=gold + (255,))
    inner = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    di = ImageDraw.Draw(inner)
    for i in range(6):
        di.rectangle([18 + i * 14, 18, 31 + i * 14, 101], fill=(blue if i % 2 else garnet) + (255,))
    mask = Image.new("L", (120, 120), 0)
    ImageDraw.Draw(mask).ellipse([18, 18, 101, 101], fill=255)
    img.paste(inner, (0, 0), mask)
    p, s = main_colours(img)
    assert {_close(p, blue) or _close(p, garnet), _close(s, blue) or _close(s, garnet)} == {True}


def test_curated_colours_match_by_name_within_our_leagues():
    from mercato.jobs.media import club_override
    curated = {"Inter Milan": ("#0068A8", "#000000"), "AC Milan": ("#FB090B", "#000000"),
               "FC Barcelona": ("#004D98", "#A50044")}
    assert club_override("FC Internazionale Milano", "IT1", curated)[0] == "#0068A8"
    assert club_override("Associazione Calcio Milan", "IT1", curated)[0] == "#FB090B"
    assert club_override("FC Barcelona", "ES1", curated)[0] == "#004D98"
    assert club_override("Inter Miami CF", "MLS1", curated) is None


def test_curated_file_is_well_formed():
    import csv
    import re
    with open("db/overrides/colours.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 100
    for r in rows:
        assert r["kind"] in ("club", "nation")
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", r["primary"]), r
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", r["secondary"]), r
