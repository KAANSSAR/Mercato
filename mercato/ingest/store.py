"""Write parsed matches to Postgres. Shared by the post-match job and the backfill.

Unknown clubs (European opponents outside the top 5) are added on the spot.
Players with no id_map entry get a placeholder record (match_method 'api_only'),
so no stats are dropped; they can be merged into a Transfermarkt record later.
"""

import datetime as dt

from mercato.ingest.parse import (
    FINISHED,
    league_table,
    parse_fixture,
    parse_players,
    parse_team_stats,
)


class Store:
    def __init__(self, conn):
        self.conn = conn
        with conn.cursor() as cur:
            cur.execute("select api_football_id, id from clubs where api_football_id is not null")
            self.clubs = dict(cur.fetchall())
            cur.execute("select api_football_id, mercato_id from id_map "
                        "where api_football_id is not null")
            self.players = dict(cur.fetchall())
            cur.execute("select api_football_id, id, kind from competitions "
                        "where api_football_id is not null")
            self.comps = {a: (i, k) for a, i, k in cur.fetchall()}
            cur.execute("select year, id from seasons")
            self.seasons = dict(cur.fetchall())
        self.new_clubs = self.new_players = 0

    # ---------- lookups that create on demand ----------

    def club(self, api_id: int, name: str) -> int:
        if api_id not in self.clubs:
            with self.conn.cursor() as cur:
                cur.execute("insert into clubs (api_football_id, name) values (%s, %s) "
                            "on conflict (api_football_id) do update set name = clubs.name "
                            "returning id", (api_id, name))
                self.clubs[api_id] = cur.fetchone()[0]
            self.new_clubs += 1
        return self.clubs[api_id]

    def player(self, api_id: int, name: str, club_id: int) -> int:
        if api_id not in self.players:
            with self.conn.cursor() as cur:
                cur.execute("insert into players (name, club_id) values (%s, %s) "
                            "returning mercato_id", (name, club_id))
                mid = cur.fetchone()[0]
                cur.execute("insert into id_map (mercato_id, api_football_id, match_method) "
                            "values (%s, %s, 'api_only')", (mid, api_id))
            self.players[api_id] = mid
            self.new_players += 1
        return self.players[api_id]

    def season(self, year: int) -> int:
        if year not in self.seasons:
            with self.conn.cursor() as cur:
                cur.execute("insert into seasons (year) values (%s) on conflict (year) "
                            "do update set year = excluded.year returning id", (year,))
                self.seasons[year] = cur.fetchone()[0]
        return self.seasons[year]

    # ---------- writes ----------

    def match(self, fixture: dict) -> int | None:
        """Upsert a fixture. Returns the match id, or None for an untracked competition."""
        f = parse_fixture(fixture)
        if f["league_id"] not in self.comps:
            return None
        row = {
            "api_football_id": f["api_football_id"],
            "competition_id": self.comps[f["league_id"]][0],
            "season_id": self.season(f["season"]),
            "kickoff": f["kickoff"],
            "home_club_id": self.club(f["home"]["id"], f["home"]["name"]),
            "away_club_id": self.club(f["away"]["id"], f["away"]["name"]),
            "home_goals": f["home_goals"],
            "away_goals": f["away_goals"],
            "status": f["status"],
            "round": f["round"],
        }
        with self.conn.cursor() as cur:
            cur.execute(
                """insert into matches (api_football_id, competition_id, season_id, kickoff,
                       home_club_id, away_club_id, home_goals, away_goals, status, round)
                   values (%(api_football_id)s, %(competition_id)s, %(season_id)s, %(kickoff)s,
                       %(home_club_id)s, %(away_club_id)s, %(home_goals)s, %(away_goals)s,
                       %(status)s, %(round)s)
                   on conflict (api_football_id) do update set
                       kickoff = excluded.kickoff, home_goals = excluded.home_goals,
                       away_goals = excluded.away_goals, status = excluded.status,
                       round = excluded.round
                   returning id""", row)
            return cur.fetchone()[0]

    def stats(self, match_id: int, players_resp: list[dict], teams_resp: list[dict]) -> int:
        """Write player and team stats for a match and mark it pulled. Returns player rows."""
        p_rows = []
        for p in parse_players(players_resp):
            club_id = self.club(p["team_api_id"], p["team_name"])
            p_rows.append((match_id, self.player(p["api_player_id"], p["name"], club_id),
                           club_id, p["minutes"], _json(p["stats"]), p["api_player_id"],
                           p["position"], p["rating"], p["is_sub"]))
        t_rows = []
        for t in parse_team_stats(teams_resp):
            t_rows.append((match_id, self.clubs.get(t["team_api_id"]),
                           t["possession"], t["xg"], _json(t["stats"])))
        with self.conn.cursor() as cur, self.conn.pipeline():
            cur.executemany(
                """insert into player_match_stats (match_id, mercato_id, club_id, minutes, stats,
                       api_player_id, position, rating, is_sub)
                   values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   on conflict (match_id, mercato_id) do update set
                       minutes = excluded.minutes, stats = excluded.stats,
                       rating = excluded.rating, position = excluded.position""", p_rows)
            cur.executemany(
                """insert into team_match_stats (match_id, club_id, possession, xg, stats)
                   values (%s, %s, %s, %s, %s)
                   on conflict (match_id, club_id) do update set
                       possession = excluded.possession, xg = excluded.xg,
                       stats = excluded.stats""", [r for r in t_rows if r[1]])
            cur.execute("update matches set stats_pulled_at = now() where id = %s", (match_id,))
        return len(p_rows)

    def pulled(self, api_ids: list[int]) -> set[int]:
        if not api_ids:
            return set()
        with self.conn.cursor() as cur:
            cur.execute("select api_football_id from matches where api_football_id = any(%s) "
                        "and stats_pulled_at is not null", (api_ids,))
            return {r[0] for r in cur.fetchall()}

    def standings(self, competition_id: int, season_id: int) -> int:
        """Recompute a league table from stored results. Cups are skipped."""
        with self.conn.cursor() as cur:
            cur.execute("select kind from competitions where id = %s", (competition_id,))
            if cur.fetchone()[0] != "league":
                return 0
            cur.execute("""select home_club_id, away_club_id, home_goals, away_goals,
                                  kickoff::date
                           from matches where competition_id = %s and season_id = %s
                             and status = any(%s) and home_goals is not null""",
                        (competition_id, season_id, list(FINISHED)))
            res = [{"home": h, "away": a, "home_goals": hg, "away_goals": ag, "date": d}
                   for h, a, hg, ag, d in cur.fetchall()]
            if not res:
                return 0
            as_of = max(r["date"] for r in res)
            rows = [(competition_id, season_id, r["club"], as_of, r["position"], r["points"],
                     r["played"], r["goal_diff"], r["won"], r["drawn"], r["lost"],
                     r["goals_for"], r["goals_against"]) for r in league_table(res)]
            cur.executemany(
                """insert into standings (competition_id, season_id, club_id, as_of, position,
                       points, played, goal_diff, won, drawn, lost, goals_for, goals_against)
                   values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   on conflict (competition_id, season_id, club_id, as_of) do update set
                       position = excluded.position, points = excluded.points,
                       played = excluded.played, goal_diff = excluded.goal_diff,
                       won = excluded.won, drawn = excluded.drawn, lost = excluded.lost,
                       goals_for = excluded.goals_for, goals_against = excluded.goals_against""",
                rows)
            return len(rows)


def _json(d: dict):
    from psycopg.types.json import Jsonb
    return Jsonb(d)


def today() -> dt.date:
    return dt.datetime.now(dt.UTC).date()
