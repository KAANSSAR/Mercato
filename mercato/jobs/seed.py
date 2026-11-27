"""Seed competitions and seasons. Safe to rerun."""

from mercato.config import COMPETITIONS, CURRENT_SEASON, FIRST_SEASON
from mercato.db import connect, upsert

TM_CODES = {
    "premier_league": "GB1",
    "laliga": "ES1",
    "serie_a": "IT1",
    "bundesliga": "L1",
    "ligue_1": "FR1",
    "champions_league": "CL",
    "europa_league": "EL",
}
NAMES = {
    "premier_league": "Premier League",
    "laliga": "LaLiga",
    "serie_a": "Serie A",
    "bundesliga": "Bundesliga",
    "ligue_1": "Ligue 1",
    "champions_league": "Champions League",
    "europa_league": "Europa League",
}
CUPS = {"champions_league", "europa_league"}


def main():
    comps = [
        {
            "api_football_id": COMPETITIONS[k],
            "tm_code": TM_CODES[k],
            "name": NAMES[k],
            "kind": "cup" if k in CUPS else "league",
        }
        for k in COMPETITIONS
    ]
    seasons = [{"year": y} for y in range(FIRST_SEASON, CURRENT_SEASON + 1)]
    with connect() as conn:
        n = upsert(conn, "competitions", comps, ["api_football_id"],
                   update=["tm_code", "name", "kind"])
        m = upsert(conn, "seasons", seasons, ["year"])
    print(f"competitions: {n}, seasons: {m}")


if __name__ == "__main__":
    main()
