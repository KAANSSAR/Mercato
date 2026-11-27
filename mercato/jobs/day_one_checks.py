"""Day-one checks on the free plan. Costs about 6 requests.

1. Which seasons the free plan reaches, including the current one
2. Whether player stats include xG
3. League IDs for every tracked competition
"""

import httpx

from mercato.config import COMPETITIONS
from mercato.sources.api_football import ApiFootball


def main():
    api = ApiFootball(budget=10)
    print("status:", api.status().get("response"))

    # One call returns every league with its available seasons and coverage
    leagues = api.get("leagues")["response"]
    by_id = {lg["league"]["id"]: lg for lg in leagues}
    for name, lid in COMPETITIONS.items():
        lg = by_id.get(lid)
        if not lg:
            print(f"{name}: id {lid} NOT FOUND")
            continue
        years = [s["year"] for s in lg["seasons"]]
        current = next((s for s in lg["seasons"] if s.get("current")), None)
        print(f"{name} ({lg['league']['name']}): seasons {min(years)} to {max(years)}, "
              f"current {current and current['year']}, "
              f"coverage {current and current['coverage']}")

    # Try the current EPL season and two seasons back to find the free-plan wall
    for season in (2025, 2023, 2021):
        try:
            r = api.get("players", league=39, season=season, page=1)
            sample = r["response"][0]["statistics"][0] if r["response"] else {}
            print(f"EPL {season}: OK, keys {list(sample)}")
        except (httpx.HTTPError, RuntimeError, IndexError, KeyError) as e:
            print(f"EPL {season}: {e}")

    print("requests used:", api.requests_made)


if __name__ == "__main__":
    main()
