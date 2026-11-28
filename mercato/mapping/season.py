"""Season-aware mapping: who played in each competition each season, and who they are.

    uv run python -m mercato.mapping.season clubs premier_league 2024     # 1 request
    uv run python -m mercato.mapping.season players premier_league 2024   # ~35 requests
    uv run python -m mercato.mapping.season all 2024                      # every competition

Clubs: API-Football's team list for the competition-season, matched against the clubs
Transfermarkt shows playing in it that season (about 20 for a league), so promotion,
relegation and European qualification come straight from the data. Writes
competition_entries.

Players: API-Football's season player list has full names and birth dates, so matching is
name plus exact date of birth. Also stores season totals in api_player_season.
Placeholder clubs and players from earlier ingestion are merged into the real records.
"""

import datetime as dt
import sys

from psycopg.types.json import Jsonb

from mercato.config import COMPETITIONS, CURRENT_SEASON
from mercato.ingest.parse import flatten_player_stats
from mercato.jobs.seed import TM_CODES
from mercato.mapping.match import assign
from mercato.mapping.merge import merge_club, set_player_api_id
from mercato.mapping.names import club_similarity, player_similarity
from mercato.sources.api_football import ApiFootball, BudgetExceeded

SEASON_AUTO = 85.0


def season_window(season: int) -> tuple[dt.date, dt.date]:
    return dt.date(season, 7, 1), dt.date(season + 1, 6, 30)


def ids(conn, comp: str, season: int) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute("select id from competitions where api_football_id = %s", (COMPETITIONS[comp],))
        comp_id = cur.fetchone()[0]
        cur.execute("insert into seasons (year) values (%s) on conflict (year) "
                    "do update set year = excluded.year returning id", (season,))
        return comp_id, cur.fetchone()[0]


def tm_participants(conn, comp: str, season: int) -> list[dict]:
    """Clubs Transfermarkt shows playing in this competition that season. Empty for seasons
    the dataset doesn't cover yet."""
    start, end = season_window(season)
    with conn.cursor() as cur:
        cur.execute("""select distinct c.id, c.tm_club_id, c.name, c.api_football_id
                       from team_elo e join clubs c on c.tm_club_id = e.tm_club_id
                       where e.competition_id = %s and e.date between %s and %s""",
                    (TM_CODES[comp], start, end))
        return [dict(zip(["id", "tm_id", "name", "api_id"], r)) for r in cur.fetchall()]


def fallback_candidates(conn, comp: str) -> list[dict]:
    """Seasons past the dataset: current top-5 clubs for leagues, every club for cups."""
    q = ("select id, tm_club_id, name, api_football_id from clubs where tm_club_id is not null "
         + ("and current_top5 and tm_competition = %s" if comp in ("premier_league", "laliga",
            "serie_a", "bundesliga", "ligue_1") else "and %s is not null"))
    with conn.cursor() as cur:
        cur.execute(q, (TM_CODES[comp],))
        return [dict(zip(["id", "tm_id", "name", "api_id"], r)) for r in cur.fetchall()]


def map_clubs(conn, api: ApiFootball, comp: str, season: int) -> dict:
    comp_id, season_id = ids(conn, comp, season)
    api_teams = [{"id": t["team"]["id"], "name": t["team"]["name"]}
                 for t in api.cached("teams", league=COMPETITIONS[comp], season=season)["response"]]
    candidates = tm_participants(conn, comp, season) or fallback_candidates(conn, comp)
    matches = assign(api_teams, candidates,
                     lambda a, c: club_similarity(c["name"], a["name"]), auto=SEASON_AUTO)

    mapped = merged = added = 0
    review = []
    with conn.cursor() as cur:
        for m in matches:
            api_id, api_name = m.left["id"], m.left["name"]
            cur.execute("select id, tm_club_id from clubs where api_football_id = %s", (api_id,))
            holder = cur.fetchone()
            if m.status == "auto":
                club_id = m.right["id"]
                if holder and holder[0] != club_id:
                    if holder[1] is None:  # placeholder from ingestion
                        merge_club(cur, holder[0], club_id)
                        merged += 1
                    else:
                        why = "api id already on another Transfermarkt club"
                        review.append((api_name, m.right["name"], round(m.score), why))
                        club_id = holder[0]
                cur.execute("update clubs set api_football_id = %s where id = %s "
                            "and api_football_id is null", (api_id, club_id))
                mapped += 1
            else:
                if m.status == "review":
                    review.append((api_name, m.right["name"], round(m.score), "check"))
                if holder:
                    club_id = holder[0]
                else:
                    cur.execute("insert into clubs (api_football_id, name) values (%s, %s) "
                                "returning id", (api_id, api_name))
                    club_id = cur.fetchone()[0]
                    added += 1
            cur.execute("insert into competition_entries (competition_id, season_id, club_id) "
                        "values (%s, %s, %s) on conflict do nothing", (comp_id, season_id, club_id))
    conn.commit()
    return {"competition": comp, "season": season, "api_teams": len(api_teams),
            "candidates": len(candidates), "mapped": mapped, "merged_placeholders": merged,
            "added_without_tm": added, "review": review, "requests": api.requests_made}


# ---------- players ----------

def parse_season_players(resp: list[dict]) -> list[dict]:
    rows = []
    for item in resp:
        p = item["player"]
        birth = (p.get("birth") or {}).get("date")
        full = " ".join(x for x in [p.get("firstname"), p.get("lastname")] if x)
        for s in item.get("statistics", []):
            games = s.get("games") or {}
            rows.append({
                "api_player_id": p["id"], "name": p.get("name") or full, "full_name": full,
                "birth": dt.date.fromisoformat(birth) if birth else None,
                "team_api_id": s["team"]["id"],
                "minutes": games.get("minutes"), "appearances": games.get("appearences"),
                "position": games.get("position"),
                "rating": float(games["rating"]) if games.get("rating") else None,
                "stats": flatten_player_stats({k: v for k, v in s.items()
                                               if k not in ("team", "league")}),
            })
    return rows


def fetch_season_players(api: ApiFootball, comp: str, season: int) -> tuple[list[dict], bool]:
    """All pages. Returns (rows, complete)."""
    rows, page = [], 1
    while True:
        try:
            body = api.cached("players", league=COMPETITIONS[comp], season=season, page=page)
        except BudgetExceeded:
            return rows, False
        rows += parse_season_players(body["response"])
        paging = body.get("paging") or {}
        if page >= int(paging.get("total") or 1):
            return rows, True
        page += 1


def name_score(api_row: dict, tm_name: str) -> float:
    return max(player_similarity(api_row["full_name"], tm_name),
               player_similarity(api_row["name"], tm_name))


def map_players(conn, api: ApiFootball, comp: str, season: int) -> dict:
    comp_id, season_id = ids(conn, comp, season)
    rows, complete = fetch_season_players(api, comp, season)
    with conn.cursor() as cur:
        cur.execute("""select p.mercato_id, p.name, p.birth_date from players p
                       join id_map i using (mercato_id)
                       where i.tm_player_id is not null and p.birth_date is not null""")
        by_birth: dict[dt.date, list[tuple[int, str]]] = {}
        for mid, name, birth in cur.fetchall():
            by_birth.setdefault(birth, []).append((mid, name))

    outcome = {"set": 0, "merged": 0, "moved": 0, "unchanged": 0}
    unmatched = []
    resolved: dict[int, int] = {}
    with conn.cursor() as cur:
        cur.execute("select api_football_id, mercato_id from id_map "
                    "where api_football_id is not null")
        current = dict(cur.fetchall())
        for r in rows:
            if r["api_player_id"] in resolved:
                continue
            cands = by_birth.get(r["birth"], []) if r["birth"] else []
            scored = sorted(((name_score(r, n), mid) for mid, n in cands), reverse=True)
            if scored and scored[0][0] >= 70 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 10):
                sc, mid = scored[0]
                if current.get(r["api_player_id"]) == mid:
                    outcome["unchanged"] += 1  # already right, no database call
                else:
                    outcome[set_player_api_id(cur, mid, r["api_player_id"], "season_dob",
                                              sc + 20)] += 1
                    current[r["api_player_id"]] = mid
                resolved[r["api_player_id"]] = mid
            else:
                unmatched.append(r)
        cur.execute("select api_football_id, mercato_id from id_map "
                    "where api_football_id is not null")
        api_to_mid = dict(cur.fetchall())
        with conn.pipeline():
            cur.executemany(
                """insert into api_player_season (api_player_id, competition_id, season_id,
                       team_api_id, mercato_id, minutes, appearances, position, rating, stats)
                   values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   on conflict (api_player_id, competition_id, season_id, team_api_id) do update
                   set mercato_id = excluded.mercato_id, minutes = excluded.minutes,
                       appearances = excluded.appearances, rating = excluded.rating,
                       stats = excluded.stats""",
                [(r["api_player_id"], comp_id, season_id, r["team_api_id"],
                  api_to_mid.get(r["api_player_id"]), r["minutes"], r["appearances"],
                  r["position"], r["rating"], Jsonb(r["stats"])) for r in rows])
    conn.commit()
    played = [r for r in unmatched if (r["minutes"] or 0) > 0]
    return {"competition": comp, "season": season, "complete": complete,
            "api_player_rows": len(rows), "matched_by_birth_date": len(resolved), **outcome,
            "unmatched_who_played": len(played),
            "unmatched_sample": [(r["name"], str(r["birth"])) for r in played[:10]],
            "requests": api.requests_made}


if __name__ == "__main__":
    from mercato.db import connect, job_run

    usage = ("usage: python -m mercato.mapping.season [clubs|players] <competition> <season>\n"
             "       python -m mercato.mapping.season all <season>")
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "all":
        plan = [(w, c, int(args[1])) for c in COMPETITIONS for w in ("clubs", "players")]
    elif len(args) == 3 and args[0] in ("clubs", "players") and args[1] in COMPETITIONS:
        plan = [(args[0], args[1], int(args[2]))]
    else:
        sys.exit(usage)
    if any(s > CURRENT_SEASON for _, _, s in plan):
        sys.exit(f"season can't be after {CURRENT_SEASON}")
    api = ApiFootball()
    with connect() as conn:
        for what, comp, season in plan:
            with job_run(conn, f"season_{what}_{comp}_{season}") as stats:
                fn = map_clubs if what == "clubs" else map_players
                r = fn(conn, api, comp, season)
                stats["requests"] = r["requests"]
            print(r)
            if r.get("complete") is False:
                print("Stopped at the request limit. Rerun the same command later to finish.")
                break
