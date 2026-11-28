"""Name normalising and similarity for matching clubs and players across sources."""

import re
import unicodedata

from rapidfuzz import fuzz

# Words that differ between sources but don't identify a club
CLUB_NOISE = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "us", "rc", "rcd", "cd", "ud", "sd", "ca",
    "sv", "vfb", "vfl", "tsg", "fsv", "bsc", "sk", "ogc", "losc", "club", "calcio", "de", "del",
    "la", "le", "1", "04", "05", "07", "1846", "1848", "1860", "1893", "1899", "1900", "1901",
    "1904", "1907", "1909", "1913", "hotspur", "and", "albion", "the", "m",
}
# Common English vs local spellings
CLUB_ALIASES = {
    "munchen": "munich", "koln": "cologne", "internazionale": "inter",
    "milano": "milan", "wolverhampton": "wolves", "wanderers": "",
    "hove": "", "nottm": "nottingham", "nott": "nottingham", "rennais": "rennes",
    "man": "manchester", "utd": "united", "saint": "st", "sankt": "st",
    "sint": "st", "monchengladbach": "gladbach", "olympiakos": "olympiacos",
    "truidense": "truiden", "brestois": "brest", "moskva": "moscow",
}
POSITION_GROUPS = {
    "Goalkeeper": {"GK"},
    "Defender": {"CB", "FB"},
    "Midfielder": {"DM", "CM", "AM", "W"},
    "Attacker": {"ST", "W", "AM"},
}


SPECIAL = {"đ": "dj", "ø": "o", "ß": "ss", "ł": "l", "æ": "ae", "œ": "oe", "ı": "i", "þ": "th"}


def ascii_lower(s: str) -> str:
    s = s.lower()  # first, so capitals like Đ and Ø are caught below
    for k, v in SPECIAL.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", ascii_lower(s))


def club_key(name: str) -> str:
    out = []
    for t in tokens(name):
        t = CLUB_ALIASES.get(t, t)
        if t and t not in CLUB_NOISE:
            out.append(t)
    return " ".join(out)


def club_similarity(a: str, b: str) -> float:
    ka, kb = club_key(a), club_key(b)
    if not ka or not kb:
        return 0.0
    return max(fuzz.token_set_ratio(ka, kb), fuzz.ratio(ka, kb))


def player_key(name: str) -> str:
    return " ".join(tokens(name))


def player_similarity(api_name: str, tm_name: str) -> float:
    """API-Football often uses short forms ('Kepa', 'B. Saka', 'Vinícius Júnior')."""
    a, t = player_key(api_name), player_key(tm_name)
    if not a or not t:
        return 0.0
    a_tok, t_tok = a.split(), t.split()
    score = max(fuzz.token_set_ratio(a, t), fuzz.ratio(a, t))
    full = [x for x in a_tok if len(x) > 1]
    initials = [x for x in a_tok if len(x) == 1]
    initials_ok = not initials or any(tt.startswith(i) for i in initials for tt in t_tok)
    if full and all(x in t_tok for x in full):
        # 'b saka' vs 'bukayo saka': surname matches and the initial agrees
        # 'j timber' vs 'quinten timber': surname matches but the initial rules it out
        score = max(score, 92.0) if initials_ok else min(score, 60.0)
    elif initials and initials_ok and any(x in t_tok for x in full):
        # double surnames: 'm bokele mputu' vs 'morgan bokele'
        score = max(score, 90.0)
    elif (not initials and len(a_tok) >= 2 and len(t_tok) >= 2
          and a_tok[0] == t_tok[0] and len(a_tok[0]) >= 4):
        # same first name, different surname used: 'alvaro fernandez' vs 'alvaro carreras'.
        # Only enough to pass within one club when age and position also agree.
        score = max(score, 75.0)
    # single-name players: 'kepa' within 'kepa arrizabalaga'
    if len(a_tok) == 1 and a_tok[0] in t_tok and len(a_tok[0]) >= 4:
        score = max(score, 88.0)
    # single names spelled differently: 'brugui' vs 'roger brugue'
    elif len(a_tok) == 1 and len(a_tok[0]) >= 5 and any(
            fuzz.ratio(a_tok[0], tt) >= 80 for tt in t_tok if len(tt) >= 5):
        score = max(score, 78.0)
    return float(score)


def position_compatible(api_position: str | None, group: str | None) -> bool:
    if not api_position or not group:
        return True
    return group in POSITION_GROUPS.get(api_position, {group})
