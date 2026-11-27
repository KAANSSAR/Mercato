import pandas as pd
import pytest

from mercato.sources import understat


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
