import os

import pytest

from mercato.ingest.parse import (
    _num,
    league_table,
    parse_fixture,
    parse_players,
    parse_team_stats,
)


def fixture(fid=1001, home=(42, "Arsenal"), away=(40, "Liverpool"), hg=2, ag=1,
            status="FT", league=39, season=2024, date="2024-08-17T14:00:00+00:00"):
    return {"fixture": {"id": fid, "date": date, "status": {"short": status}},
            "league": {"id": league, "season": season, "round": "Regular Season - 1"},
            "teams": {"home": {"id": home[0], "name": home[1]},
                      "away": {"id": away[0], "name": away[1]}},
            "goals": {"home": hg, "away": ag}}


def player(pid, name, minutes, rating="7.1", sub=False, **extra):
    stats = {"games": {"minutes": minutes, "position": "F", "rating": rating, "substitute": sub},
             "shots": {"total": 3, "on": 2}, "goals": {"total": 1, "assists": None},
             "passes": {"total": 30, "key": 2, "accuracy": "25"}, "offsides": None}
    stats.update(extra)
    return {"player": {"id": pid, "name": name}, "statistics": [stats]}


PLAYERS = [
    {"team": {"id": 42, "name": "Arsenal"},
     "players": [player(1460, "B. Saka", 90), player(9999, "New Kid", 12, sub=True),
                 player(5555, "Unused Sub", None, rating=None, sub=True)]},
    {"team": {"id": 40, "name": "Liverpool"}, "players": [player(290, "V. van Dijk", 90)]},
]
TEAMS = [
    {"team": {"id": 42, "name": "Arsenal"}, "statistics": [
        {"type": "Ball Possession", "value": "58%"}, {"type": "expected_goals", "value": "1.87"},
        {"type": "Total Shots", "value": 15}, {"type": "Passes %", "value": "86%"}]},
    {"team": {"id": 40, "name": "Liverpool"}, "statistics": [
        {"type": "Ball Possession", "value": "42%"}, {"type": "expected_goals", "value": "0.95"},
        {"type": "Total Shots", "value": 9}, {"type": "Passes %", "value": None}]},
]


def test_num():
    assert _num("58%") == 58 and _num("1.87") == 1.87 and _num("7.0") == 7.0
    assert _num(None) is None and _num("") is None and _num(3) == 3 and _num("n/a") is None


def test_parse_fixture():
    f = parse_fixture(fixture())
    assert f["api_football_id"] == 1001 and f["status"] == "FT" and f["season"] == 2024
    assert f["home"] == {"id": 42, "name": "Arsenal"} and f["kickoff"].year == 2024


def test_parse_players_skips_unused_subs_and_flattens():
    rows = parse_players(PLAYERS)
    assert [r["name"] for r in rows] == ["B. Saka", "New Kid", "V. van Dijk"]
    saka = rows[0]
    assert saka["minutes"] == 90 and saka["rating"] == 7.1 and not saka["is_sub"]
    assert saka["stats"]["shots_total"] == 3 and saka["stats"]["passes_key"] == 2
    assert saka["stats"]["passes_accuracy"] == 25 and saka["stats"]["offsides"] is None
    assert "games_minutes" not in saka["stats"]


def test_parse_team_stats():
    ars, liv = parse_team_stats(TEAMS)
    assert ars["possession"] == 58 and ars["xg"] == 1.87
    assert ars["stats"]["passes_pct"] == 86 and liv["stats"]["passes_pct"] is None


def test_league_table():
    res = [{"home": "A", "away": "B", "home_goals": 2, "away_goals": 0},
           {"home": "B", "away": "C", "home_goals": 1, "away_goals": 1},
           {"home": "C", "away": "A", "home_goals": 0, "away_goals": 3}]
    t = league_table(res)
    assert [r["club"] for r in t] == ["A", "B", "C"]
    a = t[0]
    assert (a["points"], a["played"], a["won"], a["goal_diff"]) == (6, 2, 2, 5)
    assert (t[1]["points"], t[2]["points"]) == (1, 1)  # B ahead of C on goal difference


class FakeApi:
    def __init__(self, fixtures):
        self.fixtures_list, self.requests_made = fixtures, 0

    def cached(self, path, **p):
        self.requests_made += 1
        if path == "teams":
            return {"response": [{"team": {"id": 42, "name": "Arsenal"}},
                                 {"team": {"id": 40, "name": "Liverpool"}}]}
        if path == "fixtures":
            return {"response": self.fixtures_list}
        return {"response": PLAYERS if path == "fixtures/players" else TEAMS}


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="needs TEST_DATABASE_URL")
def test_backfill_against_postgres():
    import psycopg

    from mercato.jobs.backfill import backfill

    with psycopg.connect(os.environ["TEST_DATABASE_URL"]) as conn:
        for t in ["standings", "player_match_stats", "team_match_stats", "matches"]:
            conn.execute(f"delete from {t}")
        conn.execute("delete from id_map where match_method = 'api_only'")
        conn.execute("insert into competitions (api_football_id, name, kind, tm_code) "
                     "values (39, 'Premier League', 'league', 'GB1') on conflict do nothing")
        conn.execute("insert into seasons (year) values (2024) on conflict do nothing")
        conn.commit()

        fx = [fixture(1001), fixture(1002, home=(40, "Liverpool"), away=(42, "Arsenal"), hg=0,
                                     ag=0, date="2024-08-24T14:00:00+00:00"),
              fixture(1003, status="NS", date="2024-08-31T14:00:00+00:00")]
        api = FakeApi(fx)
        r = backfill(conn, api, "premier_league", 2024)
        assert r["pulled"] == 2 and r["player_rows"] == 6 and r["standings_rows"] == 2
        assert api.requests_made == 6  # 1 clubs + 1 list + 2 per finished match

        # rerun: nothing to do, no stats requests
        api2 = FakeApi(fx)
        r2 = backfill(conn, api2, "premier_league", 2024)
        assert r2["pulled"] == 0 and api2.requests_made == 2

        top = conn.execute("""select c.name, s.points, s.played from standings s
                              join clubs c on c.id = s.club_id
                              order by s.as_of desc, s.position limit 1""").fetchone()
        assert top[0] in ("Arsenal", "Arsenal FC") and top[1:] == (4, 2)
        xg = conn.execute("select count(*) from team_match_stats where xg is not null").fetchone()
        assert xg[0] == 4


class LiveApi:
    requests_made = 0

    def status(self):
        return {"subscription": {"plan": "Free", "active": True},
                "requests": {"current": 1, "limit_day": 100}}

    def fixtures_by_date(self, date):
        self.requests_made += 1
        return [fixture(2001, date="2024-09-14T14:00:00+00:00"),
                fixture(2002, status="2H", date="2024-09-14T16:30:00+00:00"),
                fixture(2003, league=999)]

    def fixture_players(self, fid):
        self.requests_made += 1
        return PLAYERS

    def fixture_statistics(self, fid):
        self.requests_made += 1
        return TEAMS


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="needs TEST_DATABASE_URL")
def test_post_match_stores_finished_tracked_matches():
    import psycopg

    from mercato.jobs.post_match import run

    with psycopg.connect(os.environ["TEST_DATABASE_URL"]) as conn:
        conn.execute("insert into competitions (api_football_id, name, kind) "
                     "values (39, 'Premier League', 'league') on conflict do nothing")
        conn.commit()
        api = LiveApi()
        r = run(conn, api, "2024-09-14")
        assert r["pulled"] == 1 and api.requests_made == 3  # in-play and untracked skipped
        api = LiveApi()
        assert run(conn, api, "2024-09-14")["pulled"] == 0 and api.requests_made == 1
        assert conn.execute("select stats_pulled_at is not null from matches "
                            "where api_football_id = 2001").fetchone()[0]
