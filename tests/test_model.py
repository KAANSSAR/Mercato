"""End-to-end test of the valuation model on a synthetic football world where we know
what drives fees."""

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
    for group in ["ST", "W", "CB", "GK"]:
        m = (v["group"] == group).to_numpy()
        # attackers through goals; defenders and keepers through minutes and club strength
        assert np.corrcoef(np.log(v.loc[m, "p50"]), q[m])[0, 1] > 0.4, group
    d0 = v["drivers"].iloc[0]
    assert len(d0) == 6 and all(isinstance(x, float) for x in d0.values())
