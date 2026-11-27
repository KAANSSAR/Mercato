import pandas as pd
import pytest

from mercato.features.elo import EloParams, brier, expected, latest, margin, run


def g(gid, date, home, away, hg, ag, comp="GB1", ctype="domestic_league"):
    return {"game_id": gid, "date": date, "competition_id": comp, "competition_type": ctype,
            "home_club_id": home, "away_club_id": away,
            "home_club_goals": hg, "away_club_goals": ag}


def test_expected_and_margin():
    assert expected(0) == pytest.approx(0.5)
    assert expected(400) == pytest.approx(10 / 11)
    assert expected(100) + expected(-100) == pytest.approx(1)
    assert [margin(x) for x in (0, 1, -1, 2, 3, 5)] == [1, 1, 1, 1.5, 1.75, 2]


def test_zero_sum_and_home_advantage():
    rated = run(pd.DataFrame([g(1, "2020-01-01", 1, 2, 1, 1)]))
    home, away = rated.iloc[0], rated.iloc[1]
    # equal teams draw at home: home side expected > 0.5, so it loses points
    assert home["expected"] > 0.5
    assert home["elo_post"] < 1500 < away["elo_post"]
    assert home["elo_post"] + away["elo_post"] == pytest.approx(3000)


def test_bigger_wins_move_more():
    one = run(pd.DataFrame([g(1, "2020-01-01", 1, 2, 1, 0)]))
    four = run(pd.DataFrame([g(1, "2020-01-01", 1, 2, 4, 0)]))
    assert four.iloc[0]["elo_post"] > one.iloc[0]["elo_post"] > 1500


def test_cups_count_less_and_cup_only_clubs_start_lower():
    lg = run(pd.DataFrame([g(1, "2020-01-01", 1, 2, 1, 0)]))
    cup = run(pd.DataFrame([g(1, "2020-01-01", 1, 2, 1, 0, comp="FAC", ctype="domestic_cup")]))
    assert cup.iloc[0]["elo_pre"] == EloParams().cup_init
    assert (cup.iloc[0]["elo_post"] - cup.iloc[0]["elo_pre"]
            < lg.iloc[0]["elo_post"] - lg.iloc[0]["elo_pre"])


def test_promoted_club_starts_low_in_its_league():
    # six established clubs, with club 1 winning a lot, then newcomer 99 arrives
    games, gid = [], 0
    for rnd in range(10):
        for home, away in [(1, 2), (3, 4), (5, 6)]:
            gid += 1
            games.append(g(gid, f"2020-0{1 + rnd // 4}-{1 + rnd % 4 * 7:02d}",
                           home if rnd % 2 else away, away if rnd % 2 else home,
                           3 if home == 1 else 1, 0 if home == 1 else 1))
    games.append(g(999, "2020-08-01", 99, 3, 0, 0))
    rated = run(pd.DataFrame(games))
    pool = latest(rated[rated["game_id"] < 999])
    newcomer = rated[(rated["game_id"] == 999) & (rated["tm_club_id"] == 99)].iloc[0]
    assert newcomer["elo_pre"] < pool["elo"].median()


def test_returning_club_is_pulled_down_after_a_gap():
    games = [g(1, "2018-01-01", 1, 2, 5, 0)]  # club 1 rated high
    games += [g(10 + i, "2021-06-01", a, b, 1, 1)
              for i, (a, b) in enumerate([(3, 4), (5, 6), (7, 2), (3, 5), (4, 6), (7, 3)])]
    games.append(g(50, "2020-09-01", 1, 3, 0, 0, comp="ES1"))  # moves league after a long gap
    games.append(g(51, "2021-09-01", 1, 4, 0, 0))              # back in GB1 after a gap
    rated = run(pd.DataFrame(games))
    before = rated[(rated["game_id"] == 50) & (rated["tm_club_id"] == 1)].iloc[0]["elo_post"]
    back = rated[(rated["game_id"] == 51) & (rated["tm_club_id"] == 1)].iloc[0]["elo_pre"]
    assert back < before


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing"):
        run(pd.DataFrame([{"game_id": 1}]))


def test_brier_and_latest():
    rated = run(pd.DataFrame([g(1, "2020-01-01", 1, 2, 1, 0), g(2, "2020-02-01", 2, 1, 0, 2)]))
    assert 0 < brier(rated, since="2019-01-01") < 0.5
    lt = latest(rated).set_index("tm_club_id")
    assert lt.loc[1, "elo"] > 1500 > lt.loc[2, "elo"]
