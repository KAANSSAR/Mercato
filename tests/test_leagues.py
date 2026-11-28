"""Match difficulty and the league check."""

import numpy as np
import pandas as pd


def _elo_rows(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    gap = rng.normal(0, 200, n)
    goals = rng.poisson(1.35 * np.exp(0.003 * gap))
    return pd.DataFrame({"elo_pre": 1600 + gap / 2, "opp_elo_pre": 1600 - gap / 2,
                         "is_home": False, "goals_for": goals})


def test_difficulty_rises_with_the_gap():
    from mercato.model.features import match_difficulty
    f = match_difficulty(_elo_rows())
    lo, mid, hi = f(np.array([-235.0, -65.0, 235.0]))
    assert lo < mid < hi
    assert 1.7 < hi / mid < 3.5  # roughly exp(0.003 * 300) on this data


def test_goal_in_a_mismatch_counts_less():
    from mercato.model.features import prepare_appearances
    te = pd.DataFrame({"game_id": [1, 1, 2, 2], "tm_club_id": [10, 20, 10, 30],
                       "elo_pre": [1900.0, 1400.0, 1700.0, 1700.0],
                       "opp_elo_pre": [1400.0, 1900.0, 1700.0, 1700.0],
                       "is_home": [True, False, True, False], "goals_for": [4, 0, 1, 1]})
    te = pd.concat([te, _elo_rows().assign(game_id=99, tm_club_id=99)], ignore_index=True)
    apps = pd.DataFrame({"game_id": [1, 2], "player_id": [5, 5], "player_club_id": [10, 10],
                         "date": ["2025-01-01", "2025-01-08"], "competition_id": "PO1",
                         "goals": [1, 1], "assists": [0, 0], "minutes_played": [90, 90]})
    a = prepare_appearances(apps, te)
    assert a.loc[0, "goals_w"] < a.loc[1, "goals_w"]


def test_league_factors_find_a_weaker_league():
    from mercato.model.leagues import factors
    rng = np.random.default_rng(2)
    rows = []
    worth = {"GB1": 1.0, "ES1": 1.0, "PO1": 0.6}  # output in PO1 is inflated by 1/0.6
    for pid in range(600):
        talent = rng.lognormal(-0.8, 0.4)
        path = rng.choice(["GB1", "ES1", "PO1"], size=2)
        for t, lg in enumerate(path):
            rate = talent / worth[lg] * np.exp(rng.normal(0, 0.15))
            rows.append({"player_id": pid, "season": 2020 + t, "league": lg,
                         "minutes": 2500, "ga90": rate})
    f = factors(pd.DataFrame(rows), "ga90").set_index("league")["factor"]
    assert f["PO1"] < 0.8 and abs(f["GB1"] - 1) < 0.15
