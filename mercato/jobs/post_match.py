"""Post-match job: check plan, poll, detect, pull, upsert, refresh.

Idempotent: only matches at full time with no stats yet are pulled,
so reruns cost zero requests. Skips all API calls when live is off.
"""

import datetime as dt
import sys

from mercato import service_status
from mercato.config import COMPETITIONS
from mercato.ingest.store import Store
from mercato.sources.api_football import ApiFootball, BudgetExceeded, PlanRestricted

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


def pulled_ids(conn, fixture_ids: list[int]) -> set[int]:
    if not fixture_ids:
        return set()
    with conn.cursor() as cur:
        cur.execute("select api_football_id from matches "
                    "where api_football_id = any(%s) and stats_pulled_at is not null",
                    (fixture_ids,))
        return {r[0] for r in cur.fetchall()}


def dates_to_poll(now: dt.datetime) -> list[str]:
    """Today in UTC, plus yesterday in the first hours after midnight to catch late finishes.

    The free plan only reaches back to yesterday, so anything older is unrecoverable on free.
    """
    today = now.date()
    dates = [today.isoformat()]
    if now.hour < 3:
        dates.insert(0, (today - dt.timedelta(days=1)).isoformat())
    return dates


def run(conn, api: ApiFootball, date: str | None = None) -> dict:
    now = dt.datetime.now(dt.UTC)
    dates = [date] if date else dates_to_poll(now)

    status, change = service_status.check(conn, api)
    if not status["live_enabled"]:
        return {"dates": dates, "skipped": status["reason"], "change": change, "status": status}

    gap = service_status.hours_since_live_update(conn)
    if gap is not None and gap > service_status.GAP_WARN_HOURS:
        print(f"::warning title=Mercato update gap::Last live update {gap:.0f}h ago. "
              "On the free plan, matches older than yesterday can't be fetched.")

    store = Store(conn)
    tables: set[tuple[int, int]] = set()
    pulled = 0
    try:
        fixtures = [f for d in dates for f in api.fixtures_by_date(d)]
        todo = finished_tracked(fixtures, pulled_ids(conn, [f["fixture"]["id"] for f in fixtures]))
        for f in todo:
            fid = f["fixture"]["id"]
            players = api.fixture_players(fid)
            teams = api.fixture_statistics(fid)
            match_id = store.match(f)
            store.stats(match_id, players, teams)
            conn.commit()
            pulled += 1
            comp_id, _ = store.comps[f["league"]["id"]]
            tables.add((comp_id, store.season(f["league"]["season"])))
    except BudgetExceeded as e:
        print(f"::warning title=Mercato request budget::{e}")
    except PlanRestricted as e:
        change = service_status.mark_restricted(conn, str(e)) or change
        return {"dates": dates, "skipped": str(e), "change": change, "status": status,
                "requests": api.requests_made}

    for comp_id, season_id in tables:
        store.standings(comp_id, season_id)
    conn.commit()
    service_status.mark_live_update(conn)  # last healthy run, matches or not
    return {"dates": dates, "fixtures": len(fixtures), "pulled": pulled,
            "requests": api.requests_made, "change": change, "status": status}


def main() -> int:
    from mercato.db import connect, job_run

    api = ApiFootball()
    with connect() as conn:
        with job_run(conn, "post_match") as stats:
            result = run(conn, api)
            stats["requests"] = api.requests_made
        print({k: v for k, v in result.items() if k != "status"})
        alert = service_status.announce(result["change"], result["status"])
    return 1 if alert else 0


if __name__ == "__main__":
    sys.exit(main())
