"""Backfill one competition-season match by match. Resumable and budget-aware.

    uv run python -m mercato.jobs.backfill premier_league 2024 --max 40
    uv run python -m mercato.jobs.backfill all 2023              # every tracked competition

Order for a past season: `mercato.mapping.season players <comp> <season>` first (about 35
requests, matches players by birth date), then this. Clubs are mapped here automatically.

Costs 1 request for the season's clubs, 1 for the fixture list, then 2 per finished match. Matches already pulled
are skipped, and every API response is cached on disk, so stopping and rerunning is free.
Each match is committed on its own, so progress survives a crash or the daily limit.
"""

import argparse

from mercato.config import COMPETITIONS
from mercato.ingest.parse import FINISHED
from mercato.ingest.store import Store
from mercato.sources.api_football import ApiFootball, BudgetExceeded, PlanRestricted


def backfill(conn, api: ApiFootball, comp: str, season: int, limit: int | None = None) -> dict:
    # Map this season's clubs first (1 cached request), so promoted, relegated and European
    # clubs attach to their Transfermarkt records instead of becoming placeholders
    from mercato.mapping.season import map_clubs
    try:
        map_clubs(conn, api, comp, season)
    except PlanRestricted as e:
        return {"competition": comp, "season": season, "error": str(e)}
    store = Store(conn)
    try:
        fixtures = api.cached("fixtures", league=COMPETITIONS[comp], season=season)["response"]
    except PlanRestricted as e:
        return {"competition": comp, "season": season, "error": str(e)}

    done = store.pulled([f["fixture"]["id"] for f in fixtures])
    todo = sorted((f for f in fixtures
                   if f["fixture"]["status"]["short"] in FINISHED
                   and f["fixture"]["id"] not in done),
                  key=lambda f: f["fixture"]["date"])
    if limit is not None:
        todo = todo[:limit]

    pulled = rows = 0
    stopped = None
    for f in todo:
        fid = f["fixture"]["id"]
        try:
            players = api.cached("fixtures/players", fixture=fid)["response"]
            teams = api.cached("fixtures/statistics", fixture=fid)["response"]
        except BudgetExceeded as e:
            stopped = str(e)
            break
        match_id = store.match(f)
        rows += store.stats(match_id, players, teams)
        conn.commit()
        pulled += 1

    table = 0
    if pulled:
        comp_id, _ = store.comps[COMPETITIONS[comp]]
        table = store.standings(comp_id, store.season(season))
        conn.commit()

    return {"competition": comp, "season": season, "fixtures": len(fixtures),
            "already_done": len(done), "pulled": pulled,
            "left": len(todo) - pulled,
            "player_rows": rows, "new_clubs": store.new_clubs,
            "new_players": store.new_players, "standings_rows": table,
            "requests": api.requests_made, "stopped": stopped}


def main():
    from mercato.db import connect, job_run

    ap = argparse.ArgumentParser()
    ap.add_argument("competition", help=f"one of {', '.join(COMPETITIONS)} or 'all'")
    ap.add_argument("season", type=int)
    ap.add_argument("--max", type=int, default=None, help="pull at most this many matches")
    args = ap.parse_args()
    comps = list(COMPETITIONS) if args.competition == "all" else [args.competition]

    api = ApiFootball()
    with connect() as conn:
        for c in comps:
            with job_run(conn, f"backfill_{c}_{args.season}") as stats:
                r = backfill(conn, api, c, args.season, args.max)
                stats["requests"] = r.get("requests", 0)
                stats["rows_written"] = r.get("player_rows", 0)
            print(r)
            if r.get("stopped"):
                break


if __name__ == "__main__":
    main()
