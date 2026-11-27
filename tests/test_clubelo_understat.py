import datetime as dt

import httpx
import pandas as pd
import pytest

from mercato.sources import clubelo, understat

SNAPSHOT = """Rank,Club,Country,Level,Elo,From,To
1,Man City,ENG,1,2050.1,2024-08-20,2024-09-02
2,Real Madrid,ESP,1,2010.4,2024-08-26,2024-09-14
None,Leeds,ENG,2,1650.0,2024-08-25,2024-09-01
None,Ajax,NED,1,1780.2,2024-08-25,2024-09-01
None,Jong Ajax,NED,2,1300.0,2024-08-25,2024-09-01
"""

HISTORY = """Rank,Club,Country,Level,Elo,From,To
None,Man City,ENG,1,1700.0,2010-07-01,2010-08-01
None,Man City,ENG,1,2040.0,2024-08-12,2024-08-19
None,Man City,ENG,1,2050.1,2024-08-20,2024-09-02
"""


def test_parse_snapshot():
    df = clubelo.parse(SNAPSHOT)
    assert list(df.columns) == ["clubelo_name", "country", "level", "elo", "valid_from", "valid_to"]
    assert df.loc[0, "valid_from"] == dt.date(2024, 8, 20)


def test_clubs_to_track_keeps_top_flights_and_top5_second_tiers():
    names = clubelo.clubs_to_track([clubelo.parse(SNAPSHOT)])
    assert names == {"Man City", "Real Madrid", "Leeds", "Ajax"}  # not Jong Ajax


def test_url_name():
    assert clubelo.url_name("Man City") == "ManCity"


def test_history_trims_old_rows_and_elo_on():
    def handler(req):
        assert req.url.path == "/ManCity"
        return httpx.Response(200, text=HISTORY)
    api = clubelo.ClubElo(transport=httpx.MockTransport(handler), pause=0)
    h = api.history("Man City")
    assert len(h) == 2  # 2010 row dropped
    assert clubelo.elo_on(h, "Man City", dt.date(2024, 8, 25)) == pytest.approx(2050.1)
    assert clubelo.elo_on(h, "Man City", dt.date(2023, 1, 1)) is None


@pytest.mark.parametrize("code,year", [("1516", 2015), (1516, 2015), ("2015-2016", 2015),
                                       (2015, 2015), ("2526", 2025), ("2021", 2020),
                                       ("9900", 1999), ("2015", 2015)])
def test_season_start(code, year):
    assert understat.season_start(code) == year


def test_season_code():
    assert understat.season_code(2015) == "1516"
    assert understat.season_code(2099) == "9900"


def test_understat_to_rows():
    df = pd.DataFrame({
        "league": ["ENG-Premier League", "NED-Eredivisie"],
        "season": ["2324", "2324"],
        "team": ["Arsenal", "Ajax"],
        "player": ["Bukayo Saka", "X"],
        "player_id": [7322, 1],
        "position": ["F M", "M"],
        "minutes": [2900, 100],
        "xg": [14.2, 0.1],
        "xa": [9.8, 0.0],
    }).set_index(["league", "season", "team", "player"])
    rows = understat.to_rows(df)
    assert len(rows) == 1  # untracked league dropped
    r = rows[0]
    assert (r["understat_player_id"], r["season"], r["league"]) == (7322, 2023, "GB1")
    assert r["xg"] == pytest.approx(14.2) and r["key_passes"] is None
