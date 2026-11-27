"""Understat loader via soccerdata. Free xG and xA for the top 5 leagues, 2014/15 on.

PARKED (27 Nov 2026): Understat's data endpoints return its error page and soccerdata
hasn't caught up. Model v1 runs without xG. Revisit later.

    uv run python -m mercato.sources.understat              # all seasons, ~55 page loads
    uv run python -m mercato.sources.understat 2024 2025    # just these seasons

soccerdata caches pages under ~/soccerdata, so reruns are fast.
"""

import sys

import pandas as pd

from mercato.config import CURRENT_SEASON, FIRST_SEASON

LEAGUES = {
    "ENG-Premier League": "GB1",
    "ESP-La Liga": "ES1",
    "ITA-Serie A": "IT1",
    "GER-Bundesliga": "L1",
    "FRA-Ligue 1": "FR1",
}
COLUMNS = ["position", "matches", "minutes", "goals", "np_goals", "xg", "np_xg", "assists",
           "xa", "shots", "key_passes", "yellow_cards", "red_cards", "xg_chain", "xg_buildup"]


def season_code(year: int) -> str:
    """2015 -> '1516', the code soccerdata uses."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def season_start(code) -> int:
    """Season start year. Ints above 1900 are years; 4-digit codes with consecutive halves
    ('1516', '2526', '2021') are soccerdata season codes; '2015-2016' works too."""
    if isinstance(code, int) and code > 1900:
        return code
    digits = "".join(c for c in str(code) if c.isdigit())
    if len(digits) == 8:  # '20152016'
        return int(digits[:4])
    if len(digits) == 4:
        a, b = int(digits[:2]), int(digits[2:])
        if (a + 1) % 100 == b:
            return (2000 if a < 50 else 1900) + a
        return int(digits)
    raise ValueError(f"unrecognised season {code!r}")


def to_rows(df: pd.DataFrame) -> list[dict]:
    df = df.reset_index()
    out = pd.DataFrame({
        "understat_player_id": df["player_id"].astype(int),
        "season": df["season"].map(season_start),
        "league": df["league"].map(LEAGUES),
        "team": df["team"].astype(str),
        "player": df["player"].astype(str),
    })
    for c in COLUMNS:
        out[c] = df.get(c)
    out = out.dropna(subset=["league"])
    return out.astype(object).where(out.notna(), None).to_dict("records")


def fetch(seasons: list[int]) -> pd.DataFrame:
    import soccerdata as sd

    us = sd.Understat(leagues=list(LEAGUES), seasons=[season_code(y) for y in seasons])
    return us.read_player_season_stats()


def load(seasons: list[int]) -> dict:
    from mercato.db import connect, job_run, upsert

    rows = to_rows(fetch(seasons))
    with connect() as conn, job_run(conn, "understat_load") as stats:
        stats["rows_written"] = upsert(
            conn, "understat_player_season", rows,
            ["understat_player_id", "season", "league", "team"],
            update=[c for c in COLUMNS + ["player"]])
        return dict(stats) | {"seasons": seasons}


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    print(load(args or list(range(FIRST_SEASON, CURRENT_SEASON + 1))))
