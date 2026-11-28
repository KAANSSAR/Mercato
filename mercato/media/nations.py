"""Nation names (as Transfermarkt writes them) to flag codes.

Codes follow flag-icons (MIT), which the site bundles: ISO 3166-1 alpha-2 in lower case,
plus gb-eng, gb-sct, gb-wls and gb-nir for the home nations.
"""

import pycountry

SPECIAL = {
    "England": "gb-eng", "Scotland": "gb-sct", "Wales": "gb-wls",
    "Northern Ireland": "gb-nir", "Korea, South": "kr", "Korea, North": "kp",
    "South Korea": "kr", "North Korea": "kp", "Cote d'Ivoire": "ci", "Ivory Coast": "ci",
    "Türkiye": "tr", "Turkey": "tr", "DR Congo": "cd", "Congo DR": "cd", "Congo": "cg",
    "Kosovo": "xk", "Curacao": "cw", "Cape Verde": "cv", "Bosnia-Herzegovina": "ba",
    "Russia": "ru", "Iran": "ir", "Czech Republic": "cz", "Chinese Taipei": "tw",
    "The Gambia": "gm", "Gambia": "gm", "St. Kitts & Nevis": "kn",
    "St. Lucia": "lc", "St. Vincent & Grenadinen": "vc", "Antigua and Barbuda": "ag",
    "Trinidad and Tobago": "tt", "Laos": "la", "Syria": "sy", "Palestine": "ps",
    "Tahiti": "pf", "Macao": "mo", "Hongkong": "hk", "Hong Kong": "hk",
    "Southern Sudan": "ss", "Sao Tome and Principe": "st", "Moldova": "md",
    "Venezuela": "ve", "Bolivia": "bo", "Tanzania": "tz", "Vietnam": "vn",
    "Brunei Darussalam": "bn", "Neukaledonien": "nc", "Faroe Islands": "fo",
    "Guinea-Bissau": "gw", "Equatorial Guinea": "gq", "Central African Republic": "cf",
    "Netherlands Antilles": "cw", "Jugoslawien (SFR)": None, "Yugoslavia": None,
}


def flag_code(name: str | None) -> str | None:
    if not name:
        return None
    if name in SPECIAL:
        return SPECIAL[name]
    try:
        return pycountry.countries.lookup(name).alpha_2.lower()
    except LookupError:
        pass
    try:
        hits = pycountry.countries.search_fuzzy(name)
        return hits[0].alpha_2.lower() if hits else None
    except LookupError:
        return None
