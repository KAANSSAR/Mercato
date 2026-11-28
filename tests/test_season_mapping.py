import datetime as dt
import os

import pytest

from mercato.mapping.season import name_score, parse_season_players

SEASON_PLAYERS = [
    {"player": {"id": 1460, "name": "B. Saka", "firstname": "Bukayo", "lastname": "Saka",
                "birth": {"date": "2001-09-05"}},
     "statistics": [{"team": {"id": 42, "name": "Arsenal"}, "league": {"id": 39},
                     "games": {"appearences": 35, "minutes": 2900, "position": "Attacker",
                               "rating": "7.45"},
                     "goals": {"total": 16, "assists": 9}, "shots": {"total": 80, "on": 40}}]},
    {"player": {"id": 7001, "name": "L. Delap", "firstname": "Liam", "lastname": "Delap",
                "birth": {"date": "2003-02-08"}},
     "statistics": [{"team": {"id": 57, "name": "Ipswich"}, "league": {"id": 39},
                     "games": {"appearences": 37, "minutes": 2800, "position": "Attacker",
                               "rating": "6.9"}, "goals": {"total": 12}}]},
    {"player": {"id": 7002, "name": "Nobody", "firstname": "No", "lastname": "Body",
                "birth": {"date": "1999-01-01"}},
     "statistics": [{"team": {"id": 57, "name": "Ipswich"}, "league": {"id": 39},
                     "games": {"appearences": 1, "minutes": 5, "position": "Midfielder"}}]},
]


def test_parse_season_players():
    rows = parse_season_players(SEASON_PLAYERS)
    saka = rows[0]
    assert saka["full_name"] == "Bukayo Saka" and saka["birth"] == dt.date(2001, 9, 5)
    assert saka["minutes"] == 2900 and saka["rating"] == 7.45
    assert saka["stats"]["goals_total"] == 16 and "team" not in saka["stats"]


def test_name_score_uses_full_name():
    r = {"name": "L. Delap", "full_name": "Liam Delap"}
    assert name_score(r, "Liam Delap") >= 90


class FakeApi:
    requests_made = 0

    def cached(self, path, **p):
        self.requests_made += 1
        if path == "teams":
            return {"response": [{"team": {"id": 42, "name": "Arsenal"}},
                                 {"team": {"id": 57, "name": "Ipswich"}}]}
        return {"response": SEASON_PLAYERS, "paging": {"current": 1, "total": 1}}


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="needs TEST_DATABASE_URL")
def test_season_mapping_merges_placeholders():
    import psycopg

    from mercato.mapping.season import map_clubs, map_players

    with psycopg.connect(os.environ["TEST_DATABASE_URL"]) as conn:
        for t in ["competition_entries", "api_player_season", "standings", "player_match_stats",
                  "team_match_stats", "matches", "team_elo", "id_map", "players", "clubs"]:
            conn.execute(f"delete from {t}")
        conn.execute("insert into competitions (api_football_id, name, kind, tm_code) values "
                     "(39, 'Premier League', 'league', 'GB1') on conflict (api_football_id) "
                     "do update set tm_code = 'GB1'")
        conn.execute("insert into seasons (year) values (2024) on conflict do nothing")
        # Transfermarkt side: Arsenal (current, mapped) and Ipswich (relegated, unmapped)
        conn.execute("""insert into clubs (id, tm_club_id, name, api_football_id, current_top5)
                        values (1, 11, 'Arsenal FC', 42, true), (2, 677, 'Ipswich Town', null, false),
                               (3, 999, 'Celtic FC', null, false)""")
        conn.execute("""insert into team_elo (game_id, tm_club_id, opponent_tm_club_id, date,
                            competition_id, is_home, elo_pre, opp_elo_pre, elo_post, expected, result)
                        values (1, 11, 677, '2024-08-17', 'GB1', true, 1800, 1500, 1805, 0.8, 1),
                               (1, 677, 11, '2024-08-17', 'GB1', false, 1500, 1800, 1495, 0.2, 0)""")
        # ingestion already created a placeholder Ipswich and a placeholder Delap with stats
        conn.execute("insert into clubs (id, api_football_id, name) values (50, 57, 'Ipswich')")
        conn.execute("""insert into players (mercato_id, name, birth_date) values
                        (1, 'Bukayo Saka', '2001-09-05'), (2, 'Liam Delap', '2003-02-08'),
                        (90, 'L. Delap', null)""")
        conn.execute("""insert into id_map (mercato_id, tm_player_id, api_football_id, match_method)
                        values (1, 901, 1460, 'auto'), (2, 902, null, null),
                               (90, null, 7001, 'api_only')""")
        conn.execute("""insert into matches (id, api_football_id, competition_id, season_id, kickoff,
                            home_club_id, away_club_id, home_goals, away_goals, status)
                        select 1, 5001, c.id, s.id, '2024-08-17', 1, 50, 2, 0, 'FT'
                        from competitions c, seasons s where c.api_football_id = 39 and s.year = 2024""")
        conn.execute("""insert into player_match_stats (match_id, mercato_id, club_id, minutes, stats)
                        values (1, 90, 50, 90, '{}')""")
        for seq, table, col in [("clubs_id_seq", "clubs", "id"),
                                ("players_mercato_id_seq", "players", "mercato_id"),
                                ("matches_id_seq", "matches", "id")]:
            conn.execute(f"select setval('{seq}', (select max({col}) from {table}))")
        conn.commit()

        api = FakeApi()
        rc = map_clubs(conn, api, "premier_league", 2024)
        assert rc["candidates"] == 2 and rc["mapped"] == 2 and rc["merged_placeholders"] == 1
        assert conn.execute("select count(*) from clubs where id = 50").fetchone()[0] == 0
        assert conn.execute("select away_club_id from matches where id = 1").fetchone()[0] == 2
        assert conn.execute("select api_football_id from clubs where id = 2").fetchone()[0] == 57
        entries = conn.execute("select club_id from competition_entries order by 1").fetchall()
        assert entries == [(1,), (2,)]  # Celtic not entered: it wasn't in this league

        rp = map_players(conn, api, "premier_league", 2024)
        assert rp["matched_by_birth_date"] == 2 and rp["merged"] == 1 and rp["unchanged"] == 1
        assert rp["unmatched_who_played"] == 1
        # Delap's stats moved from the placeholder to the Transfermarkt record
        assert conn.execute("select mercato_id, club_id from player_match_stats").fetchone() == (2, 2)
        assert conn.execute("select count(*) from players where mercato_id = 90").fetchone()[0] == 0
        assert conn.execute("select mercato_id from api_player_season "
                            "where api_player_id = 7001").fetchone()[0] == 2

        # rerun is a no-op
        rp2 = map_players(conn, FakeApi(), "premier_league", 2024)
        assert rp2["unchanged"] == 2 and rp2["merged"] == 0
