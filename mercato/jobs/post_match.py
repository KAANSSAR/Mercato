"""Post-match job: poll, detect, pull, upsert, refresh.

Idempotent: only matches at full time with no stats yet are pulled,
so reruns cost zero requests.
"""

import datetime as dt

from mercato.config import COMPETITIONS
from mercato.sources.api_football import ApiFootball

FINISHED = {"FT", "AET", "PEN"}
TRACKED = set(COMPETITIONS.values())


def finished_tracked(fixtures: list[dict], already_pulled: set[int]) -> list[dict]:
    """Fixtures in tracked competitions, newly at full time, not yet pulled."""
    return [
        f for f in fixtures
        if f["league"]["id"] in TRACKED
        and f["fixture"]["status"]["short"] in FINISHED
        and f["fixture"]["id"] not in already_pulled
    ]


def run(date: str | None = None, already_pulled: set[int] | None = None) -> dict:
    date = date or dt.datetime.now(dt.UTC).date().isoformat()
    already_pulled = already_pulled or set()  # TODO: read from matches.stats_pulled_at

    api = ApiFootball()
    fixtures = api.fixtures_by_date(date)
    todo = finished_tracked(fixtures, already_pulled)

    for f in todo:
        fid = f["fixture"]["id"]
        players = api.fixture_players(fid)
        teams = api.fixture_statistics(fid)
        # TODO: upsert matches, player_match_stats, team_match_stats; set stats_pulled_at
        _ = players, teams

    return {"date": date, "fixtures": len(fixtures), "pulled": len(todo),
            "requests": api.requests_made}


if __name__ == "__main__":
    print(run())
