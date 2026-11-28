"""Model v2: performance rating, value curve, euro scale, checks and backtest."""

import numpy as np
import pandas as pd
import pytest
from test_model import synthetic_world


@pytest.fixture(scope="module")
def world():
    from mercato.model.train import current_rows
    from mercato.model.value import fit, prepare
    d, truth = synthetic_world()
    prep = prepare(d, current_rows(d), min_test_moves=30)
    return d, truth, prep, fit(prep, verbose=False)


def test_game_features_window_and_recency():
    from mercato.model.rating import K_GAMES, game_features
    t = pd.Timestamp("2025-06-30")
    tg = pd.DataFrame({"game_id": [1, 2, 3], "tm_club_id": [9, 9, 9],
                       "result": [1.0, 0.0, 1.0], "expected": [0.5, 0.5, 0.5],
                       "goals_against": [0, 2, 0], "exp_ga": [1.0, 1.0, 1.0],
                       "elo_pre": [1600.0] * 3, "opp_elo_pre": [1700.0] * 3})
    apps = pd.DataFrame({"game_id": [1, 2, 3], "player_id": 5, "player_club_id": 9,
                         "date": [t - pd.Timedelta(days=100), t - pd.Timedelta(days=500),
                                  t - pd.Timedelta(days=1500)],  # the last is too old
                         "minutes_played": [90, 90, 90]})
    g = game_features(pd.DataFrame({"player_id": [5], "as_of": [t]}), apps, tg).iloc[0]
    # weights 1 (this year) and 0.75 (last year); the four-year-old game is ignored
    assert g["games_w"] == pytest.approx(1.75)
    assert g["impact"] == pytest.approx((0.5 - 0.75 * 0.5) / (1.75 + K_GAMES))
    assert g["defence"] == pytest.approx((1.0 - 0.75 * 1.0) / (1.75 + K_GAMES))


def test_expected_goals_against_falls_with_strength(world):
    from mercato.model.rating import team_games
    d, *_ = world
    tg = team_games(d.team_elo)
    gap = tg["elo_pre"] - tg["opp_elo_pre"]
    strong, weak = tg.loc[gap > 150, "exp_ga"].mean(), tg.loc[gap < -150, "exp_ga"].mean()
    assert strong < weak


def test_rating_is_within_position_and_tracks_quality(world):
    _, truth, _, r = world
    v = r["values"]
    assert v["rating"].between(0, 100).all()
    for g in ("ST", "W", "CB"):
        reg = v[(v["group"] == g) & (v["minutes_best"] >= 900)]
        assert reg["rating"].median() == pytest.approx(50, abs=8)
    q = truth.loc[v["player_id"], "q"].to_numpy()
    for g in ("ST", "W"):
        m = (v["group"] == g).to_numpy()
        assert np.corrcoef(np.log(v.loc[m, "p50"]), q[m])[0, 1] > 0.3, g


def test_value_uses_only_rating_age_and_position(world):
    from mercato.model.value import design
    _, _, _prep, r = world
    cols = list(design(r["values"].assign(age_under_peak=0, age_over_peak=0)).columns)
    assert cols[:3] == ["score", "age_under_peak", "age_over_peak"]
    assert all(c.startswith("pos_") for c in cols[3:])


def test_scale_never_changes_the_order(world):
    from mercato.model.value import check, fit
    _, _, prep, r = world
    c = check(r["values"])
    assert abs(float(c["median_gap"].rstrip("%")) / 100) < 0.12
    # line price lists: the order within each line is untouched
    v = r["values"]
    for mode in ("rank", "linear", "level"):
        other = fit(prep, scale=mode, verbose=False)["values"]
        for ln in ("ATT", "MID", "DEF", "GK"):
            m = v["line"] == ln
            assert np.corrcoef(v.loc[m, "p50"].rank(), other.loc[m, "p50"].rank())[0, 1] \
                > 0.995, (mode, ln)  # ties only where the scale clamps at the ends


def test_each_line_gets_its_own_price_list(world):
    from mercato.model.value import check
    _, _, _, r = world
    for ln, x in check(r["values"])["by_line"].items():
        assert abs(float(x["median_gap"].rstrip("%")) / 100) < 0.15, ln


def test_age_multiplies_performance_rather_than_interacting():
    from mercato.model.value import _constraints
    groups = _constraints(["score", "age_under_peak", "age_over_peak", "pos_ATT", "pos_GK"])
    assert not any("score" in g and "age_under_peak" in g for g in groups)


def test_rank_scale_uses_the_market_price_list():
    from mercato.model.value import euro_scale
    rng = np.random.default_rng(0)
    raw = rng.normal(16, 0.3, 500)              # a compressed curve with a floor
    tm = rng.normal(16, 1.2, 500)               # the market's wider price list
    f = euro_scale(raw, tm, "rank")
    assert f(np.quantile(raw, 0.9)) == pytest.approx(np.quantile(tm, 0.9), abs=0.02)
    assert f(np.quantile(raw, 0.1)) == pytest.approx(np.quantile(tm, 0.1), abs=0.02)
    assert np.all(np.diff(f(np.sort(raw))) >= 0)


def test_bands_and_flags(world):
    _, _, _, r = world
    v = r["values"]
    assert ((v["p10"] <= v["p50"]) & (v["p50"] <= v["p90"])).all()
    assert (v.loc[v["minutes_best"] < 900, "low_confidence"]).all()
    d0 = v["drivers"].iloc[0]
    assert set(d0) == {"current_level", "projected_years"}
    assert sum(d0.values()) == pytest.approx(v["p50"].iloc[0], rel=0.01, abs=2e4)
    assert set(v["components"].iloc[0]) == {"output", "impact", "defence", "minutes",
                                            "opposition", "team_level", "projected_rating"}
    assert len(v["components"].iloc[0]["projected_rating"]) == 6


def test_weights_sum_to_one():
    from mercato.model.rating import WEIGHTS
    for ln, w in WEIGHTS.items():
        assert sum(w.values()) == pytest.approx(1.0), ln


def test_backtest_uses_only_the_past():
    from mercato.model.value import backtest
    d, truth = synthetic_world(seed=11)
    rng = np.random.default_rng(3)
    rows = []
    for pid, p in truth.iterrows():
        for date, drift in (("2024-12-15", 0.0), ("2025-12-15", rng.normal(0.3 * p["q"], 0.2))):
            rows.append({"player_id": pid, "date": date, "current_club_id": p["current_club_id"],
                         "market_value_in_eur": 8e6 * np.exp(0.9 * p["q"] + drift)})
    d.valuations = pd.DataFrame(rows)
    out = backtest(d, min_test_moves=30, names=dict(zip(d.players["player_id"],
                                                        d.players["name"])))
    assert out["as_of"] == "2024-12-15"
    assert len(out["move_by_gap_quintile"]) == 5
    assert -1 <= out["rank_corr_gap_vs_move"] <= 1
    assert len(out["most_undervalued_then"]) > 0


def test_rating_separates_the_top():
    """v2.0 put every good player at 97 to 100. Far above typical must beat a bit above."""
    from mercato.model.rating import rate
    n = 200
    rng = np.random.default_rng(1)
    f = pd.DataFrame({"group": "ST", "g90": rng.normal(0.3, 0.1, n), "a90": 0.1,
                      "share_best": 0.8, "share_1y": 0.8, "minutes_best": 2000,
                      "impact": 0.0, "defence": 0.0, "opposition": 1600.0,
                      "team_level": 1600.0})
    f.loc[0, "g90"], f.loc[1, "g90"] = 1.0, 0.6  # Haaland-like, and merely very good
    out, _ = rate(f)
    assert out.loc[0, "score"] > out.loc[1, "score"] + 0.15


def test_fee_curve_method_still_runs(world):
    from mercato.model.value import fit
    _, _, prep, _ = world
    r = fit(prep, method="curve", verbose=False)
    assert set(r["values"]["drivers"].iloc[0]) == {"performance", "age", "position"}


def _snaps(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n // 6):
        age0, s = rng.uniform(18, 30), rng.normal(0, 1)
        for t in range(6):
            age = age0 + t
            s1 = s + rng.normal(0, 0.5)
            rows.append({"player_id": pid, "year": 2016 + t, "line": "ATT", "age": age,
                         "score3": s, "score1": s1})
            s = 0.8 * s - 0.08 * (age - 27) + rng.normal(0, 0.3)  # fades, ages
    return pd.DataFrame(rows)


def test_paths_learn_fading_and_ageing():
    from mercato.model.projection import fit_paths
    c = fit_paths(_snaps())["coefs"]["ATT"][1]  # [const, score3, age-27, (age-27)^2]
    assert 0.5 < c[1] < 1.0     # good seasons partly fade
    assert c[2] < 0             # older players decline


def test_projection_stops_at_retirement_and_values_youth_less_than_a_cliff():
    from mercato.model.projection import MAX_AGE, fit_paths, log_index, project
    paths = fit_paths(_snaps())
    f = pd.DataFrame({"group": ["ST", "ST", "ST"], "age": [22.0, 27.5, 35.0],
                      "score": [1.0, 1.0, 1.0]})
    P = project(f, paths)
    assert np.isnan(P[2, MAX_AGE - 35 + 1:]).all() and not np.isnan(P[2, :3]).any()
    li = log_index(P, 1.0, 0.8)
    assert li[0] > li[1] > li[2]
    # same level at 27.5 keeps a good share of its worth at 22 (no cliff after 26)
    assert np.exp(li[1] - li[0]) > 0.4


def test_dropouts_count_as_weak_seasons():
    from mercato.model.projection import fit_paths
    s = _snaps()
    full = fit_paths(s)["coefs"]["ATT"][3][0]
    # remove every season after 30: those players "drop out" and pull the path down
    gone = fit_paths(s[s["age"] < 30])["coefs"]["ATT"][3]
    assert gone[2] < fit_paths(s)["coefs"]["ATT"][3][2] or gone[0] < full


def test_current_season_always_counts_and_youth_gets_its_prime():
    from mercato.model.projection import fit_paths, log_index, project
    paths = fit_paths(_snaps())
    f = pd.DataFrame({"group": ["ST", "ST", "ST"], "age": [38.5, 20.0, 31.0],
                      "score": [1.0, 0.5, 0.5]})
    P = project(f, paths)
    assert not np.isnan(P[0, 0]) and np.isnan(P[0, 1:]).all()  # Lewandowski
    li = log_index(P, 0.75, 0.9)
    assert np.isfinite(li[0]) and li[0] > log_index(P * 0 - 5, 0.75, 0.9)[0]
    assert li[1] > li[2]  # same level: 20-year-old has his prime ahead
