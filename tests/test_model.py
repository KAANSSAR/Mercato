"""End-to-end test of the valuation model on a synthetic football world where we know
what drives fees."""

import itertools

import numpy as np
import pandas as pd
import pytest

from mercato.features.elo import run as run_elo
from mercato.model.fee_index import build_index, deflate, season_of


def test_season_of():
    assert season_of("2019-08-01") == 2019 and season_of("2020-01-15") == 2019
    s = season_of(pd.Series(pd.to_datetime(["2021-07-01", "2021-06-30"])))
    assert list(s) == [2021, 2020]


def test_fee_index_and_deflate():
    moves = pd.DataFrame({"season": [2016] * 40 + [2017] * 40 + [2018] * 5,
                          "fee": [10e6] * 40 + [20e6] * 40 + [99e6] * 5})
    idx = build_index(moves)
    assert idx[2016] == 10e6 and idx[2017] == 20e6
    assert idx[2018] == 20e6  # too few moves: borrows the previous level
    out = deflate(pd.Series([10e6]), pd.Series([2016]), idx, 2017)
    assert out.iloc[0] == 20e6


def synthetic_world(seed: int = 7):
    rng = np.random.default_rng(seed)
    clubs = pd.DataFrame({"club_id": range(1, 41), "name": [f"Club {i}" for i in range(1, 41)],
                          "domestic_competition_id": ["GB1"] * 20 + ["NL1"] * 20,
                          "last_season": 2025})
    strength = {c: rng.normal(0.4 if c <= 20 else -0.4, 0.4) for c in clubs["club_id"]}
    groups = [("Centre-Forward", "Attack", 1.0), ("Right Winger", "Attack", 0.6),
              ("Attacking Midfield", "Midfield", 0.5), ("Central Midfield", "Midfield", 0.25),
              ("Centre-Back", "Defender", 0.08), ("Goalkeeper", "Goalkeeper", 0.0)]
    players = []
    pid = 0
    for c in clubs["club_id"]:
        for j in range(22):
            pid += 1
            sub, pos, scoring = groups[j % len(groups)]
            players.append({"player_id": pid, "name": f"P{pid}", "current_club_id": c,
                            "sub_position": sub, "position": pos, "scoring": scoring,
                            "q": rng.normal(strength[c], 0.6),
                            "date_of_birth": pd.Timestamp("1990-01-01")
                            + pd.Timedelta(days=int(rng.integers(0, 365 * 14)))})
    P = pd.DataFrame(players).set_index("player_id")
    club_of = P["current_club_id"].to_dict()

    games, apps, transfers = [], [], []
    gid = 0
    level = {s: (1.0 if s < 2017 else 1.8) * (0.8 if s in (2020, 2021) else 1.0)
             for s in range(2014, 2027)}
    for season in range(2014, 2026):
        # summer window: 45 moves into the GB1 clubs
        if season >= 2015:
            for _ in range(45):
                p = int(rng.choice(P.index))
                to = int(rng.integers(1, 21))
                if club_of[p] == to:
                    continue
                age = (pd.Timestamp(f"{season}-08-01") - P.loc[p, "date_of_birth"]).days / 365.25
                fee = (8e6 * np.exp(0.9 * P.loc[p, "q"] - 0.012 * (age - 25) ** 2)
                       * level[season] * np.exp(rng.normal(0, 0.35)))
                transfers.append({"player_id": p, "transfer_date": f"{season}-08-01",
                                  "from_club_id": club_of[p], "to_club_id": to,
                                  "from_club_name": "", "to_club_name": "",
                                  "transfer_fee": round(fee),
                                  "market_value_in_eur": round(fee * np.exp(rng.normal(0, 0.6)))})
                club_of[p] = to
        for league, ids in (("GB1", range(1, 21)), ("NL1", range(21, 41))):
            ids = list(ids)
            for rnd in range(30):
                rng.shuffle(ids)
                date = pd.Timestamp(f"{season}-08-15") + pd.Timedelta(days=7 * rnd)
                for h, a in zip(ids[::2], ids[1::2]):
                    gid += 1
                    hg = rng.poisson(np.exp(0.35 + 0.5 * (strength[h] - strength[a])))
                    ag = rng.poisson(np.exp(0.15 - 0.5 * (strength[h] - strength[a])))
                    games.append({"game_id": gid, "date": date, "competition_id": league,
                                  "competition_type": "domestic_league", "home_club_id": h,
                                  "away_club_id": a, "home_club_goals": hg, "away_club_goals": ag})
                    for club in (h, a):
                        squad = [p for p, c in club_of.items() if c == club]
                        squad = sorted(squad, key=lambda p: -P.loc[p, "q"])[:11]
                        for p in squad:
                            lam = 0.12 * P.loc[p, "scoring"] * np.exp(0.6 * P.loc[p, "q"])
                            apps.append({"game_id": gid, "player_id": p, "player_club_id": club,
                                         "date": date, "competition_id": league,
                                         "goals": rng.poisson(lam),
                                         "assists": rng.poisson(lam * 0.7),
                                         "minutes_played": 90})
    P["current_club_id"] = pd.Series(club_of)
    P["last_season"] = 2025
    P["market_value_in_eur"] = (8e6 * np.exp(0.9 * P["q"]) * 1.8).round()
    from mercato.model.train import Data
    games = pd.DataFrame(games)
    return Data(players=P.reset_index(), clubs=clubs, transfers=pd.DataFrame(transfers),
                apps=pd.DataFrame(apps), team_elo=run_elo(games)), P


@pytest.fixture(scope="module")
def trained():
    from mercato.model.train import train
    d, truth = synthetic_world()
    return d, truth, train(d, verbose=False, min_test_moves=30)


def test_model_beats_naive(trained):
    # (the synthetic "market value" is built from the real fee, so it isn't a fair rival here)
    _, _, r = trained
    m = r["metrics"]
    assert m["test"] >= 30 and m["test_season"] == 2025
    assert m["test_mdape"] < 0.9 * m["naive_median_mdape"]
    assert 0.5 <= m["test_coverage_p10_p90"] <= 0.97
    assert m["fee_index"][2017] > 1.4 * m["fee_index"][2016]  # the market jump is found


def test_current_valuations_track_true_quality(trained):
    from mercato.model.train import value_current
    d, truth, r = trained
    v = value_current(d, r)
    assert (v["p10"] <= v["p50"]).all() and (v["p50"] <= v["p90"]).all()
    q = truth.loc[v["player_id"], "q"].to_numpy()
    for group, bar in [("ST", 0.4), ("W", 0.4), ("CB", 0.4), ("GK", 0.3)]:
        m = (v["group"] == group).to_numpy()
        # attackers through goals; defenders through minutes and club strength; keepers
        # have the least signal until saves arrive from API-Football
        assert np.corrcoef(np.log(v.loc[m, "p50"]), q[m])[0, 1] > bar, group
    d0 = v["drivers"].iloc[0]
    assert len(d0) == 6 and all(isinstance(x, float) for x in d0.values())


def _row(trained):
    from mercato.model.features import design_matrix
    from mercato.model.train import current_rows, features_for
    d, _, r = trained
    f, _ = features_for(current_rows(d).head(1), d, r["group_means"])
    f["year"] = r["level"]
    return design_matrix(f)[r["columns"]], r


def test_value_never_rises_past_peak_age(trained):
    from mercato.model.train import predict
    X, r = _row(trained)
    vals = []
    for over in [0, 3, 6, 9, 11]:
        x = X.copy()
        x["age_under_peak"], x["age_over_peak"] = 0.0, float(over)
        vals.append(predict(r["model"], x, r["band"], [27 + over], r["age_rule"])["p50"].iloc[0])
    assert all(b <= a + 1e-6 for a, b in itertools.pairwise(vals)), vals
    # beyond the supported ages the decline is strict
    support = r["age_rule"][0]
    old = [predict(r["model"], X, r["band"], [support + k], r["age_rule"])["p50"].iloc[0]
           for k in (1, 3, 5)]
    assert old[0] > old[1] > old[2]


def test_band_never_crosses_and_matches_held_out_spread(trained):
    _, _, r = trained
    lo, hi = r["band"]
    assert lo < 0 < hi
    assert r["metrics"]["band_multipliers"][0] < 1 < r["metrics"]["band_multipliers"][1]


def test_one_injured_season_does_not_erase_a_player(trained):
    from mercato.model.train import predict
    X, r = _row(trained)
    healthy_then_injured, poor_both = X.copy(), X.copy()
    healthy_then_injured[["share_best", "share_1y"]] = [0.95, 0.1]
    poor_both[["share_best", "share_1y"]] = [0.12, 0.1]
    a = predict(r["model"], healthy_then_injured, r["band"])["p50"].iloc[0]
    b = predict(r["model"], poor_both, r["band"])["p50"].iloc[0]
    assert a >= b  # guaranteed by the monotonic constraint on share_best


def test_cap_holds(trained):
    from mercato.model.train import predict
    X, r = _row(trained)
    x = X.copy()
    x[["g90", "a90", "ga90"]] = 5.0  # absurd output: would extrapolate
    x["club_elo"] = 2400.0
    p = predict(r["model"], x, r["band"], cap=r["cap"])
    assert (p[["p10", "p50", "p90"]] <= r["cap"] + 1e-6).all(axis=None)


def test_minutes_best_takes_the_better_season():
    import pandas as pd

    from mercato.model.features import finish
    # injured last year (300 min), fit the year before (3000), patchy before that (1500)
    f = pd.DataFrame({"player_id": [1], "as_of": ["2025-08-01"], "birth": ["2000-08-01"],
                      "group": ["CB"], "league": ["GB1"], "club_id": [1], "club_elo": [1700.0],
                      "minutes_y0": [300.0], "minutes_y1": [3000.0], "minutes_y2": [1500.0],
                      "goals_y0": [0.0], "goals_y1": [3.0], "goals_y2": [1.0],
                      "assists_y0": [0.0], "assists_y1": [1.0], "assists_y2": [0.0],
                      "club_games_y0": [38], "club_games_y1": [38], "club_games_y2": [38],
                      "apps_1y": [4], "ucl_minutes_2y": [0.0], "uel_minutes_2y": [0.0],
                      "top5_minutes_2y": [3300.0]})
    out, _ = finish(f)
    assert out["minutes_best"].iloc[0] == 3000.0 and out["minutes_1y"].iloc[0] == 300.0
    assert out["share_best"].iloc[0] == pytest.approx(3000 / (90 * 38))
    assert out["share_1y"].iloc[0] == pytest.approx(300 / (90 * 38))
    assert out["top5_share_2y"].iloc[0] == pytest.approx(3300 / (2 * 38 * 90))
    # recency-weighted: (0 + 0.75*4 + 0.5*1) / (300 + 0.75*3000 + 0.5*1500) * 90
    assert out["ga90_raw"].iloc[0] == pytest.approx(3.5 / 3300 * 90)
    assert out["age_under_peak"].iloc[0] == pytest.approx(2.0, abs=0.01)
    assert out["age_over_peak"].iloc[0] == 0.0


def test_volume_is_capped():
    import pandas as pd

    from mercato.model.features import finish
    base = {"player_id": [1, 2], "as_of": ["2025-08-01"] * 2, "birth": ["2001-08-01"] * 2,
            "group": ["AM"] * 2, "league": ["GB1"] * 2, "club_id": [1, 1],
            "club_elo": [1800.0] * 2, "apps_1y": [40, 40], "uel_minutes_2y": [0.0, 0.0]}
    for k in range(3):
        base |= {f"minutes_y{k}": [3420.0, 3420.0], f"goals_y{k}": [8.0, 8.0],
                 f"assists_y{k}": [6.0, 6.0], f"club_games_y{k}": [50, 50]}
    # an ironman with a long European run against a regular with a normal one
    base |= {"top5_minutes_2y": [6840.0, 9000.0], "ucl_minutes_2y": [1200.0, 2500.0]}
    out, _ = finish(pd.DataFrame(base))
    assert out["top5_share_2y"].tolist() == [1.0, 1.0]
    assert out["ucl_share_2y"].tolist() == [1.0, 1.0]
