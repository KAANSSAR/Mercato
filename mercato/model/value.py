"""Model v2: Mercato value from on-pitch performance.

    uv run python -m mercato.model.value --dry          # fit, check, print, write nothing
    uv run python -m mercato.model.value --dry --backtest
    uv run python -m mercato.model.value                # also save and write valuations

Three steps:

1. Rating (model.rating): how well he plays for his position, 0 to 100, stats only.
2. Projection (model.projection): his rating over every season he has left, from how
   ratings have actually moved with age since 2016, discounted into one index. Fees only
   tune how much an elite season counts and how much later years count. No club, league,
   Europe or buyer, so two players who play the same get the same value wherever they
   play. (v2.0 to v2.2 learned the age effect from fees instead, which over-priced youth
   and dropped players off a cliff after 26; method="curve" keeps that for comparison.)
3. Euro scale: Transfermarkt's price list for each line (attack, midfield, defence,
   keepers), measured on regulars. Mercato ranks every player in his line; the defender
   Mercato ranks 20th gets what the 20th most valuable regular defender on Transfermarkt
   is worth, whoever that is. The order is entirely Mercato's. No player's own
   Transfermarkt value is ever an input to his Mercato value. (One list for everyone left
   keepers 26% under, since the fee curve and the market price keepers differently.)
   (v2.0 matched only the level and spread, two numbers. Fees exist only for players
   someone paid for, so the fee curve has a floor of a few million and bench players came
   out at 15x Transfermarkt; a straight line can't fix a bent curve.)

Checks: rank agreement with Transfermarkt, the typical gap (near zero by construction),
the share of regulars more than 50% away, and the backtest: when Mercato rated a player
above Transfermarkt a year ago, did his Transfermarkt value rise afterwards?
"""

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm, spearmanr

from mercato.model.features import GROUP_LINE, RECENCY, finish, raw_features
from mercato.model.fee_index import build_index, deflate, season_of
from mercato.model.leagues import apply_league_factors, league_factors
from mercato.model.projection import (
    HORIZON,
    ONE_SEASON,
    fit_paths,
    fit_preferences,
    log_index,
    project,
    snapshot_rows,
)
from mercato.model.rating import COMPONENTS, _pct, game_features, rate, team_games
from mercato.model.train import (
    MODEL_DIR,
    Data,
    _player_info,
    age_decline,
    current_rows,
    load,
    mdape,
    report,
    residual_band,
    training_rows,
)
from mercato.sources.tm_datasets import TOP5

VERSION = "v2.6"
CURVE_PARAMS = {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 20,
                "subsample": 0.8, "colsample_bytree": 1.0, "reg_lambda": 5.0,
                "tree_method": "hist"}
MONOTONE = {"score": 1, "age_over_peak": -1}
REGULAR_MINUTES = 1500  # best season, for the scale and the checks
MIN_TM = 1_000_000
LOW_MINUTES = 900       # fewer than this in his best season: rating too thin to trust
SCALE_Q = (0.10, 0.90)  # linear scale: spread measured between these quantiles
SCALES = ("rank_line", "rank", "linear", "level")
LINE_NAMES = ("ATT", "MID", "DEF", "GK")


# ---------- features ----------

def _team_games(d: Data) -> pd.DataFrame:
    if getattr(d, "_tg", None) is None:
        d._tg = team_games(d.team_elo)
    return d._tg


def base_features(d: Data, rows: pd.DataFrame, gm: dict | None = None,
                  recency: tuple = RECENCY, raw: pd.DataFrame | None = None):
    """Per-90 output, shares and age (model.features) plus the game features.
    recency ONE_SEASON gives that season alone (the projection's targets)."""
    apps, games, elo = d.prepared()
    if raw is None:
        raw = raw_features(rows, apps, games, elo)
    f, gm = finish(raw, gm, recency)
    if tuple(recency) == ONE_SEASON:
        f["share_best"] = f["share_1y"]
    f = pd.concat([f, game_features(f, apps, _team_games(d), recency)], axis=1)
    return f, gm


def design(f: pd.DataFrame) -> pd.DataFrame:
    """Rating score, age and line. Lines, not the eight groups: a handful of giant
    historical fees for attacking midfielders made every good one look like Wirtz."""
    X = f[["score", "age_under_peak", "age_over_peak"]].astype(float).copy()
    line = f["group"].map(GROUP_LINE)
    for ln in LINE_NAMES:
        X[f"pos_{ln}"] = (line == ln).astype(float)
    return X


def _constraints(columns) -> list[list[str]]:
    """Performance and age each work with position but not with each other, so age is a
    multiplier on what a player's performance is worth. Freely interacting, the fee data
    taught v2.1 that a young player with a high rating is worth Wirtz money (Hinshelwood
    €156m)."""
    pos = [c for c in columns if c.startswith("pos_")]
    return [["score", *pos], ["age_under_peak", "age_over_peak", *pos]]


def _curve(X, y, X_val=None, y_val=None, rounds=None,
           additive: bool = True) -> tuple[xgb.XGBRegressor, int]:
    extra = {"interaction_constraints": _constraints(list(X.columns))} if additive else {}
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
                         n_estimators=rounds or 3000,
                         monotone_constraints=tuple(MONOTONE.get(c, 0) for c in X.columns),
                         early_stopping_rounds=100 if X_val is not None else None,
                         **CURVE_PARAMS, **extra)
    if X_val is not None:
        m.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        return m, int(m.best_iteration) + 1
    m.fit(X, y, verbose=False)
    return m, rounds


# ---------- fit ----------

def prepare(d: Data, pool_rows: pd.DataFrame, cutoff=None, min_test_moves: int = 200,
            projection: bool = True, league_factors_on: bool = True) -> dict:
    """Features for the pool being valued (also the rating's reference), for every paid
    move before the cutoff and, for the projection, for yearly snapshots of top-5
    regulars. Computed once, shared by variants."""
    cutoff = pd.Timestamp(cutoff if cutoff is not None else pool_rows["as_of"].max())
    lf = {}
    if league_factors_on:
        apps = d.prepared()[0]
        lf = league_factors(apps, d.clubs, d.players, before_season=int(season_of(cutoff)))
        apply_league_factors(apps, lf)
    pool, gm = base_features(d, pool_rows)
    rows = training_rows(d)
    rows = rows[rows["as_of"] < cutoff].reset_index(drop=True)
    index = build_index(rows[["season", "fee"]])
    counts = rows["season"].value_counts()
    test_season = int(max(s for s, n in counts.items() if n >= min_test_moves))
    rows["fee_adj"] = deflate(rows["fee"], rows["season"], index, test_season)
    rows["y"] = np.log(rows["fee_adj"])
    moves, _ = base_features(d, rows, gm)
    out = {"pool": pool, "moves": moves, "index": index, "test_season": test_season,
           "cutoff": cutoff, "league_factors": lf}
    if projection:
        snaps = snapshot_rows(d.apps, d.clubs, _player_info(d), cutoff)
        apps, games, elo = d.prepared()
        raw = raw_features(snaps, apps, games, elo)
        out["snap3"], _ = base_features(d, snaps, gm, raw=raw)
        out["snap1"], _ = base_features(d, snaps, gm, ONE_SEASON, raw=raw)
    return out


def _ols_x(li: np.ndarray, line: np.ndarray) -> np.ndarray:
    return np.column_stack([li] + [(line == ln).astype(float) for ln in LINE_NAMES])


def fit(prep: dict, weights: dict | None = None, scale: str = "rank_line",
        additive: bool = True, method: str = "projection", prefs: str = "fees",
        verbose: bool = True) -> dict:
    """method projection (v2.3): projected performance over six years, discounted.
    method curve (v2.2): rating and age through a curve learned from fees."""
    assert scale in SCALES, scale
    pool, ref = rate(prep["pool"], None, weights)
    moves, _ = rate(prep["moves"], ref)
    ts = prep["test_season"]
    tr, va, te = (moves["season"] <= ts - 2), (moves["season"] == ts - 1), (moves["season"] == ts)
    y = moves["y"].to_numpy()
    actual = moves.loc[te, "fee_adj"].to_numpy()
    result = {"ref": ref, "scale_mode": scale, "method": method,
              "cap": float(moves["fee_adj"].max()), "weights": ref["weights"]}
    metrics = {"moves": len(moves), "test_season": ts}

    if method == "projection":
        s3, _ = rate(prep["snap3"], ref)
        s1, _ = rate(prep["snap1"], ref)
        snaps = pd.DataFrame({"player_id": s3["player_id"], "year": s3["year"],
                              "line": s3["group"].map(GROUP_LINE), "age": s3["age"],
                              "score3": s3["score"], "score1": s1["score"].to_numpy()})
        paths = fit_paths(snaps)
        P = project(moves, paths)
        line = moves["group"].map(GROUP_LINE).to_numpy()
        trva = (tr | va).to_numpy()
        if prefs == "market":
            # how much an elite season and later years count, read from the market's
            # prices for today's regulars (two numbers for everyone, never per player)
            reg = regulars(pool.assign(line=pool["group"].map(GROUP_LINE))).to_numpy()
            Pp = project(pool[reg], paths)
            beta, delta, rho = fit_preferences(
                Pp, pool.loc[reg, "group"].map(GROUP_LINE).to_numpy(),
                np.log(pool.loc[reg, "tm_value"].to_numpy(float)))
        else:
            beta, delta, rho = fit_preferences(P[trva], line[trva], y[trva])
        li = log_index(P, beta, delta)
        X = _ols_x(li, line)
        c_tr = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        band = residual_band(y[va], X[va] @ c_tr)
        c_eval = np.linalg.lstsq(X[trva], y[trva], rcond=None)[0]
        log_te = X[te] @ c_eval
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        result |= {"paths": paths, "beta": beta, "delta": delta, "ols": coef}
        metrics |= {"snapshots": len(snaps), "beta": beta, "delta": delta, "prefs": prefs,
                    "prefs_rank_corr": round(rho, 3),
                    "fee_rank_corr_test": round(float(spearmanr(li[te], y[te])[0]), 3)}
    else:
        X = design(moves)
        m_tr, best = _curve(X[tr], y[tr], X[va], y[va], additive=additive)
        band = residual_band(y[va], m_tr.predict(X[va]))
        m_eval, _ = _curve(X[tr | va], y[tr | va], rounds=best, additive=additive)
        rule = age_decline(moves[tr | va])
        log_te = (m_eval.predict(X[te])
                  + rule[1] * np.clip(moves.loc[te, "age"] - rule[0], 0, None))
        model, _ = _curve(X, y, rounds=best, additive=additive)
        result |= {"model": model, "age_rule": age_decline(moves)}
        metrics["rounds"] = best

    result["band"] = band
    # for reference only: v2 values players, it doesn't predict fees
    metrics |= {"fee_mdape_test": mdape(np.exp(log_te), actual),
                "naive_fee_mdape": mdape(np.full(len(actual), np.exp(np.median(y[tr | va]))),
                                         actual)}
    result["metrics"] = metrics
    result["values"] = value(pool, result)
    if verbose:
        print(json.dumps(metrics, indent=2, default=float))
    return result


def raw_value(f: pd.DataFrame, result: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """Log raw value before the euro scale, and the projected path (projection only)."""
    if result["method"] == "projection":
        P = project(f, result["paths"])
        li = log_index(P, result["beta"], result["delta"])
        return _ols_x(li, f["group"].map(GROUP_LINE).to_numpy()) @ result["ols"], P
    support, slope = result["age_rule"]
    return (result["model"].predict(design(f))
            + slope * np.clip(f["age"].to_numpy(float) - support, 0, None)), None


def regulars(v: pd.DataFrame) -> pd.Series:
    return ((v["minutes_best"] >= REGULAR_MINUTES) & (v["tm_value"] >= MIN_TM)
            & v["tm_value"].notna())


def value(pool: pd.DataFrame, result: dict) -> pd.DataFrame:
    """Raw value, then the euro scale."""
    log_raw, P = raw_value(pool, result)
    v = pool[["player_id", "group", "age", "minutes_best", "minutes_1y", "tm_value", "rating",
              "score"] + [f"z_{c}" for c in COMPONENTS]].copy()
    v["line"] = v["group"].map(GROUP_LINE)
    reg = regulars(v).to_numpy()
    mode = result["scale_mode"]
    lo, hi = result["band"]
    logs = np.zeros((len(v), 3))
    slope = np.ones(len(v))
    parts = ([(v["line"] == ln).to_numpy() for ln in LINE_NAMES] if mode == "rank_line"
             else [np.ones(len(v), dtype=bool)])
    for m in parts:
        r = reg & m
        if r.sum() < 20:  # too few regulars in this line: use the whole market
            r = reg
        to_eur = euro_scale(log_raw[r], np.log(v.loc[r, "tm_value"].to_numpy(float)),
                            "rank" if mode == "rank_line" else mode)
        x = log_raw[m]
        logs[m] = np.column_stack([to_eur(x + lo), to_eur(x), to_eur(x + hi)])
        # local slope of the scale at each player, so the drivers are in euros
        slope[m] = (to_eur(x + 0.05) - to_eur(x - 0.05)) / 0.1
    v["p10"], v["p50"], v["p90"] = np.exp(logs[:, 0]), np.exp(logs[:, 1]), np.exp(logs[:, 2])
    cap = result["cap"]
    v["capped"] = v["p50"] > cap
    v[["p10", "p50", "p90"]] = v[["p10", "p50", "p90"]].clip(upper=cap)
    v["low_confidence"] = v["capped"] | (v["minutes_best"] < LOW_MINUTES)
    v["gap"] = v["p50"] / v["tm_value"] - 1
    v["components"] = [{c: round(float(norm.cdf(r[f"z_{c}"]) * 100), 1) for c in COMPONENTS}
                       for _, r in v.iterrows()]
    if P is None:
        v["drivers"] = drivers(result["model"], design(pool), v["p50"], slope)
    else:
        # now vs later: the share of the value index from this season's level
        disc = result["delta"] ** np.arange(P.shape[1])
        terms = np.where(np.isnan(P), 0.0, disc * np.exp(result["beta"] * np.nan_to_num(P)))
        total = terms.sum(axis=1)
        # players already past the last counted age have no seasons left: all current level
        now = np.divide(terms[:, 0], total, out=np.ones(len(total)), where=total > 0)
        v["drivers"] = [{"current_level": float(round(p * n, -4)),
                         "projected_years": float(round(p * (1 - n), -4))}
                        for p, n in zip(v["p50"], now)]
        ref = result["ref"]["score"]
        groups = v["group"].to_numpy()
        proj = np.full(P.shape, np.nan)
        for g in set(groups):
            m = groups == g
            proj[m] = 100 * _pct(ref.get(g, np.array([])),
                                 np.nan_to_num(P[m]).ravel()).reshape(P[m].shape)
        proj[np.isnan(P)] = np.nan
        for comp, row in zip(v["components"], proj):
            comp["projected_rating"] = [None if np.isnan(x) else round(float(x), 1)
                                        for x in row[1:HORIZON + 1]]
        has = ~np.isnan(proj).all(axis=1)
        v["projected_peak"] = np.nan
        v.loc[has, "projected_peak"] = np.nanmax(proj[has, :HORIZON + 1], axis=1)
    return v


def euro_scale(log_raw_reg: np.ndarray, log_tm_reg: np.ndarray, mode: str):
    """Map the curve's raw log values onto euros, using regulars on both sides.
    rank: quantile mapping (Mercato's order, Transfermarkt's price list).
    linear: match level and spread. level: match the median only."""
    if len(log_raw_reg) < 20:
        return lambda x: np.asarray(x, dtype=float)
    if mode == "rank":
        qs = np.linspace(0, 1, 201)
        xs, ys = np.quantile(log_raw_reg, qs), np.quantile(log_tm_reg, qs)
        xs = xs + np.arange(len(xs)) * 1e-9  # strictly increasing for interp
        return lambda x: np.interp(np.asarray(x, dtype=float), xs, ys)
    b = 1.0
    if mode == "linear":
        b = float(np.diff(np.quantile(log_tm_reg, SCALE_Q))[0]
                  / np.diff(np.quantile(log_raw_reg, SCALE_Q))[0])
    a = float(np.median(log_tm_reg) - b * np.median(log_raw_reg))
    return lambda x: a + b * np.asarray(x, dtype=float)


def drivers(model, X: pd.DataFrame, p50: pd.Series, slope) -> list[dict]:
    """TreeSHAP in log space, through the euro scale: performance, age, position."""
    c = (model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
         * np.asarray(slope, dtype=float)[:, None])
    c = pd.DataFrame(c, columns=X.columns, index=X.index)
    parts = pd.DataFrame({"performance": c["score"],
                          "age": c["age_under_peak"] + c["age_over_peak"],
                          "position": c[[x for x in X.columns if x.startswith("pos_")]].sum(axis=1)})
    eur = (p50.to_numpy()[:, None] * (1 - np.exp(-parts.to_numpy())))
    return [{k: float(round(x, -4)) for k, x in zip(parts.columns, row)} for row in eur]


# ---------- checks ----------

def check(v: pd.DataFrame, names: dict | None = None, show: int = 8) -> dict:
    """How Mercato sits against Transfermarkt on regulars (not a target, a sanity check)."""
    r = v[regulars(v) & ~v["low_confidence"]].copy()
    ratio = r["p50"] / r["tm_value"]
    far = (ratio > 1.5) | (ratio < 0.5)
    out = {"regulars": len(r),
           "rank_agreement": round(float(spearmanr(r["p50"], r["tm_value"])[0]), 3),
           "median_gap": f"{ratio.median() - 1:+.0%}",
           "beyond_50pct": f"{far.mean():.0%}",
           "by_line": {}}
    for ln in ("ATT", "MID", "DEF", "GK"):
        m = r["line"] == ln
        if m.sum() >= 10:
            out["by_line"][ln] = {
                "rank_agreement": round(float(spearmanr(r.loc[m, "p50"], r.loc[m, "tm_value"])[0]), 3),
                "median_gap": f"{ratio[m].median() - 1:+.0%}", "beyond_50pct": f"{far[m].mean():.0%}"}
    if names is not None:
        r["name"] = r["player_id"].map(names)
        r["ratio"] = ratio
        out["far_above_tm"] = [f"{x.name} €{x.p50 / 1e6:.0f}m vs €{x.tm_value / 1e6:.0f}m"
                               for x in r[far].sort_values("ratio", ascending=False)
                               .head(show).itertuples()]
        out["far_below_tm"] = [f"{x.name} €{x.p50 / 1e6:.0f}m vs €{x.tm_value / 1e6:.0f}m"
                               for x in r[far].sort_values("ratio").head(show).itertuples()]
    return out


def _last_value(val: pd.DataFrame, lo, hi) -> pd.DataFrame:
    s = val[(val["date"] > lo) & (val["date"] <= hi)]
    return s.sort_values("date").groupby("player_id").tail(1)


def backtest_rows(d: Data, horizon_days: int = 365):
    """Players at a top-5 club a year before the newest valuations, with their Transfermarkt
    value then (tm_value) and a year later (tm_later)."""
    val = d.valuations.copy()
    val["date"] = pd.to_datetime(val["date"])
    end = val["date"].max()
    t0 = end - pd.Timedelta(days=horizon_days)
    then = _last_value(val, t0 - pd.Timedelta(days=200), t0)
    later = _last_value(val, t0 + pd.Timedelta(days=90), end)
    club_col = "current_club_id" if "current_club_id" in then else None
    league = dict(zip(d.clubs["club_id"], d.clubs["domestic_competition_id"]))
    clubs = (then[club_col] if club_col else
             then["player_id"].map(dict(zip(d.players["player_id"], d.players["current_club_id"]))))
    rows = pd.DataFrame({"player_id": then["player_id"].astype(int).to_numpy(), "as_of": t0,
                         "club_id": clubs.to_numpy(),
                         "tm_value": pd.to_numeric(then["market_value_in_eur"],
                                                   errors="coerce").to_numpy()})
    rows["league"] = rows["club_id"].map(league).fillna("other")
    rows = rows[rows["league"].isin(TOP5)]
    rows = rows.merge(_player_info(d), on="player_id", how="inner")
    rows = rows.dropna(subset=["birth", "group"]).reset_index(drop=True)
    later = later.set_index("player_id")["market_value_in_eur"]
    return rows, later, t0


def backtest(d: Data, weights: dict | None = None, scale: str = "rank_line",
             additive: bool = True, method: str = "projection", prefs: str = "fees",
             horizon_days: int = 365, min_test_moves: int = 200,
             names: dict | None = None, prep: dict | None = None) -> dict:
    """Value everyone as of a year ago with only what was known then (fees before then,
    stats before then, Transfermarkt's scale then), and see how Transfermarkt moved.
    Both the gap and the move are compared within the same line and age band, since young
    players rise anyway and that isn't skill."""
    rows, later, t0 = backtest_rows(d, horizon_days)
    prep = prep or prepare(d, rows, t0, min_test_moves)
    v = fit(prep, weights, scale, additive, method, prefs, verbose=False)["values"]
    v["tm_later"] = v["player_id"].map(later)
    v = v[regulars(v) & ~v["low_confidence"] & (v["tm_later"] > 0)].copy()
    v["gap0"] = np.log(v["p50"] / v["tm_value"])
    v["move"] = np.log(v["tm_later"] / v["tm_value"])
    band = (v["age"] // 2 * 2).clip(18, 34)
    for c in ("gap0", "move"):
        v[f"{c}_r"] = v[c] - v.groupby([v["line"], band])[c].transform("mean")
    v["q"] = pd.qcut(v["gap0_r"].rank(method="first"), 5, labels=False)
    by_q = v.groupby("q")["move_r"].mean()
    top = v[v["q"] == 4]
    out = {"as_of": str(t0.date()), "players": len(v),
           "rank_corr_gap_vs_move": round(float(spearmanr(v["gap0_r"], v["move_r"])[0]), 3),
           "move_by_gap_quintile": [f"{np.exp(x) - 1:+.1%}" for x in by_q],
           "top_quintile_beat_peers": f"{(top['move_r'] > 0).mean():.0%}",
           "top_minus_bottom": f"{np.exp(by_q.iloc[-1] - by_q.iloc[0]) - 1:+.1%}"}
    if names is not None:
        pick = top[top["tm_value"] >= 5e6].sort_values("gap0_r", ascending=False).head(10)
        out["most_undervalued_then"] = [
            f"{names.get(x.player_id, x.player_id)}: Mercato €{x.p50 / 1e6:.0f}m, TM "
            f"€{x.tm_value / 1e6:.0f}m then, €{x.tm_later / 1e6:.0f}m now"
            for x in pick.itertuples()]
    return out


# ---------- save ----------

def save(result: dict, values: pd.DataFrame) -> dict:
    from psycopg.types.json import Jsonb

    from mercato.db import connect

    out = MODEL_DIR / VERSION
    out.mkdir(parents=True, exist_ok=True)
    ref = result["ref"]
    if result["method"] == "curve":
        result["model"].save_model(out / "curve.json")
        method = {"age_rule": result["age_rule"]}
    else:
        method = {"beta": result["beta"], "delta": result["delta"],
                  "ols": list(map(float, result["ols"])),
                  "paths": {ln: {k: list(map(float, c)) for k, c in by.items()}
                            for ln, by in result["paths"]["coefs"].items()},
                  "dropout": result["paths"]["dropout"]}
    (out / "meta.json").write_text(json.dumps({
        "method": result["method"], **method,
        "weights": ref["weights"], "band": result["band"],
        "scale": result["scale_mode"], "cap": result["cap"], "metrics": result["metrics"],
        "reference": {"components": {c: {g: list(map(float, a)) for g, a in by.items()}
                                     for c, by in ref["components"].items()},
                      "score": {g: list(map(float, a)) for g, a in ref["score"].items()}},
    }, indent=2, default=str))

    with connect() as conn, conn.cursor() as cur:
        cur.execute("insert into model_runs (version, test_mdape) values (%s, %s) returning id",
                    (VERSION, result["metrics"]["fee_mdape_test"]))
        run_id = cur.fetchone()[0]
        cur.execute("select tm_player_id, mercato_id from id_map where tm_player_id is not null")
        mid = dict(cur.fetchall())
        rows = [(run_id, mid[int(r.player_id)], float(r.p10), float(r.p50), float(r.p90),
                 Jsonb(r.drivers), bool(r.capped), bool(r.low_confidence), float(r.rating),
                 Jsonb(r.components))
                for r in values.itertuples() if int(r.player_id) in mid]
        with conn.pipeline():
            cur.executemany("insert into valuations (model_run_id, mercato_id, p10_eur, p50_eur, "
                            "p90_eur, shap, capped, low_confidence, rating, components) "
                            "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", rows)
        conn.commit()
    return {"model_run_id": run_id, "valuations": len(rows), "saved_to": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="fit and print only")
    ap.add_argument("--backtest", action="store_true", help="also run the Transfermarkt backtest")
    args = ap.parse_args()
    d = load()
    names = dict(zip(d.players["player_id"], d.players["name"]))
    result = fit(prepare(d, current_rows(d)))
    v = result["values"]
    print(json.dumps(check(v, names), indent=2, ensure_ascii=False))
    report(d, v)
    if args.backtest:
        if d.valuations is None:
            print("backtest needs data/tm/player_valuations.csv")
        else:
            print(json.dumps(backtest(d, names=names), indent=2, ensure_ascii=False))
    if not args.dry:
        print(save(result, v))


if __name__ == "__main__":
    main()
