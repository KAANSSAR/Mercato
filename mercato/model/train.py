"""Valuation model v1.2: XGBoost median on inflation-adjusted log transfer fees, with a
conformal P10 to P90 band (where 80% of real fees landed around the model on held-out
data), an age decline beyond the ages the data supports, and a cap at the highest
inflation-adjusted fee ever paid.

    uv run python -m mercato.model.train --dry   # train, test, print, write nothing
    uv run python -m mercato.model.train         # also save models and write valuations

Real past fees train the model. Mercato sets the value. Transfermarkt is only the
benchmark: its market value is never a feature, only compared against at the end.

Time split, as it will be used: train on older seasons, tune on the next, score once on
the newest complete season.

v1 limitations, disclosed: no contract length (the dataset only has current contracts),
no shots, tackles or passing (those arrive from API-Football with the Pro backfill), and
fees exist only for players who moved, so valuing everyone else is extrapolation.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from mercato.features.elo import run as run_elo
from mercato.features.gfr_anchor import anchored
from mercato.model.features import (
    GROUP_LINE,
    LINES,
    NUMERIC_EURO,
    add_percentiles,
    additive_platform,
    apply_buyer_scenario,
    buyer_reference,
    design_matrix,
    finish,
    latest_elo,
    market_features,
    money_table,
    monotone_constraints,
    prepare_appearances,
    raw_features,
)
from mercato.model.features import club_games as club_games_table
from mercato.model.fee_index import build_index, deflate, season_of
from mercato.sources.tm_datasets import (
    DATA_DIR,
    TOP5,
    current_top5_player_mask,
    position_group,
    read,
)

VERSION = "v1.5"
BAND = (0.10, 0.90)  # P10 to P90
DENSE_QUANTILE = 0.98  # above this share of real fees, training data is too thin to trust
MIN_AGE_SUPPORT = 30  # transfers needed at or above an age to trust the model there
# Past that age, value falls at least this much a year. The fee data alone shows only
# about 5% because of selection: only good older players get sold for a fee at all.
MIN_AGE_DECLINE = 0.15
MIN_TRAIN_FEE = 500_000
FIRST_TRAIN_SEASON = 2015
MODEL_DIR = Path("data/models")
# Regularised so the few huge fees don't drag the top end (v1.2 put Morgan Rogers at €268m)
PARAMS = {"max_depth": 2, "learning_rate": 0.03, "min_child_weight": 25, "subsample": 0.8,
          "colsample_bytree": 0.8, "reg_lambda": 5.0, "gamma": 0.05, "tree_method": "hist"}


@dataclass
class Data:
    players: pd.DataFrame
    clubs: pd.DataFrame
    transfers: pd.DataFrame
    apps: pd.DataFrame
    team_elo: pd.DataFrame
    valuations: pd.DataFrame | None = None  # Transfermarkt value history, for the backtest
    _prepared: tuple | None = None

    _money: pd.DataFrame | None = None

    def prepared(self):
        """Appearances with opponent Elo, club game dates, club Elo history. Built once."""
        if self._prepared is None:
            self._prepared = (prepare_appearances(self.apps, self.team_elo),
                              club_games_table(self.team_elo), latest_elo(self.team_elo))
        return self._prepared

    def money(self) -> pd.DataFrame:
        if self._money is None:
            self._money = money_table(self.transfers)
        return self._money


def load(data_dir: Path = DATA_DIR) -> Data:
    games = read("games", data_dir)
    clubs = read("clubs", data_dir)
    team_elo, _ = anchored(run_elo(games), clubs, verbose=True)
    try:
        valuations = read("player_valuations", data_dir)
    except FileNotFoundError:
        valuations = None
    return Data(players=read("players", data_dir), clubs=clubs,
                transfers=read("transfers", data_dir), apps=read("appearances", data_dir),
                team_elo=team_elo, valuations=valuations)


def _player_info(d: Data) -> pd.DataFrame:
    p = d.players[["player_id", "date_of_birth", "sub_position", "position"]].copy()
    p["group"] = [position_group(s, q) for s, q in zip(p["sub_position"], p["position"])]
    p["line"] = p["position"].map(LINES).fillna(p["group"].map(GROUP_LINE))
    return p.rename(columns={"date_of_birth": "birth"})[["player_id", "birth", "group", "line"]]


def training_rows(d: Data) -> pd.DataFrame:
    t = d.transfers.copy()
    t["fee"] = pd.to_numeric(t["transfer_fee"], errors="coerce")
    t["as_of"] = pd.to_datetime(t["transfer_date"])
    t["season"] = season_of(t["as_of"])
    top5_ids = set(d.clubs.loc[d.clubs["domestic_competition_id"].isin(TOP5), "club_id"])
    t = t[(t["fee"] >= MIN_TRAIN_FEE) & t["to_club_id"].isin(top5_ids)
          & (t["season"] >= FIRST_TRAIN_SEASON)]
    league = dict(zip(d.clubs["club_id"], d.clubs["domestic_competition_id"]))
    rows = pd.DataFrame({
        "player_id": t["player_id"].astype(int), "as_of": t["as_of"],
        "club_id": t["from_club_id"], "league": t["from_club_id"].map(league).fillna("other"),
        "buyer_id": t["to_club_id"], "fee": t["fee"], "season": t["season"],
        "tm_value": pd.to_numeric(t.get("market_value_in_eur"), errors="coerce"),
    })
    rows = rows.merge(_player_info(d), on="player_id", how="inner")
    return rows.dropna(subset=["birth", "group"]).reset_index(drop=True)


def features_for(rows: pd.DataFrame, d: Data, state: dict | None = None):
    """All features. state carries what training learned (position averages, percentile
    references, the typical buyer) so prediction uses the same yardsticks."""
    state = state or {}
    f, gm = finish(raw_features(rows, *d.prepared()), state or None)
    f, pct = add_percentiles(f, state.get("_pct"))
    f = market_features(f, d.money(), d.prepared()[2], state.get("_buyer_ref"))
    return f, {**gm, "_pct": pct, "_buyer_ref": state.get("_buyer_ref")}


def fit_median(X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None,
               rounds: int | None = None, params: dict | None = None,
               additive: bool = False) -> tuple[xgb.XGBRegressor, int]:
    """Median (P50) model on log fees."""
    extra = {"interaction_constraints": additive_platform(list(X.columns))} if additive else {}
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
                         n_estimators=rounds or 3000,
                         monotone_constraints=monotone_constraints(list(X.columns)),
                         early_stopping_rounds=100 if X_val is not None else None,
                         **(params or PARAMS), **extra)
    if X_val is not None:
        m.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        return m, int(m.best_iteration) + 1
    m.fit(X, y, verbose=False)
    return m, rounds or 3000


def residual_band(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Split conformal band in log space: where real fees landed around the median
    prediction on held-out data. Never crosses, calibrated by construction."""
    r = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.quantile(r, BAND[0])), float(np.quantile(r, BAND[1]))


def age_decline(f: pd.DataFrame) -> tuple[float, float]:
    """Last age with enough transfers to learn from, and the yearly decline in log fee from
    29 up to that age. Beyond it the model is flat for lack of data, so we apply the decline."""
    ages = f["age"].astype(float)
    supported = [a for a in range(28, 41) if (ages >= a).sum() >= MIN_AGE_SUPPORT]
    support = max(supported) if supported else 30
    band = (ages >= 29) & (ages < support + 1)
    by_age = f.loc[band].assign(a=ages[band].astype(int)).groupby("a")["y"].median()
    slope = np.polyfit(by_age.index, by_age.values, 1)[0] if len(by_age) >= 3 else -0.15
    return float(support), float(min(slope, np.log(1 - MIN_AGE_DECLINE)))


def predict(model, X: pd.DataFrame, band: tuple[float, float], ages=None,
            age_rule: tuple[float, float] | None = None, cap: float | None = None,
            dense_max: float | None = None) -> pd.DataFrame:
    """P50 from the model, P10 and P90 from the conformal band, then the age decline past
    the supported ages and the cap at the highest inflation-adjusted fee ever paid
    (v1 put Lamine Yamal at €602m)."""
    log_p50 = model.predict(X)
    if ages is not None and age_rule is not None:
        support, slope = age_rule
        log_p50 = log_p50 + slope * np.clip(np.asarray(ages, dtype=float) - support, 0, None)
    out = pd.DataFrame({"p10": np.exp(log_p50 + band[0]), "p50": np.exp(log_p50),
                        "p90": np.exp(log_p50 + band[1])}, index=X.index)
    out["capped"] = False
    if cap is not None:
        out["capped"] = out["p50"] > cap
        out[["p10", "p50", "p90"]] = out[["p10", "p50", "p90"]].clip(upper=cap)
    # beyond the dense part of the training data the model is extrapolating: show the
    # range and a label, keep the player out of the under/overvalued rankings
    out["low_confidence"] = out["capped"]
    if dense_max is not None:
        out["low_confidence"] |= out["p50"] > dense_max
    return out


def drivers(model, X: pd.DataFrame, p50: pd.Series, top: int = 6) -> list[dict]:
    """TreeSHAP in log space, grouped, as € effects: how much lower P50 would be without it."""
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    c = pd.DataFrame(contrib, columns=X.columns, index=X.index)
    grouped = pd.DataFrame(index=X.index)
    grouped["age"] = c[["age_under_peak", "age_over_peak"]].sum(axis=1)
    grouped["position"] = c[[x for x in X.columns if x.startswith("pos_")]].sum(axis=1)
    grouped["league"] = c[[x for x in X.columns if x.startswith("league_")]].sum(axis=1)
    for col in X.columns:
        if col not in ("age_under_peak", "age_over_peak") and not col.startswith(
                ("pos_", "league_")):
            grouped[col] = c[col]
    out = []
    for i in X.index:
        phi = grouped.loc[i]
        eur = (p50.loc[i] * (1 - np.exp(-phi))).round(-4)
        keep = eur.abs().sort_values(ascending=False).head(top).index
        out.append({k: float(eur[k]) for k in keep})
    return out


def mdape(pred, actual) -> float:
    return float(np.median(np.abs(np.asarray(pred) / np.asarray(actual) - 1)))


def prepare(d: Data, min_test_moves: int = 200) -> dict:
    """Training set, fee index, time split and features: computed once, reused by variants."""
    rows = training_rows(d)
    index = build_index(rows[["season", "fee"]])
    counts = rows["season"].value_counts()
    # newest season with enough moves to score on (the latest window may still be filling)
    test_season = int(max(s for s, n in counts.items() if n >= min_test_moves))
    val_season, level = test_season - 1, test_season
    rows["fee_adj"] = deflate(rows["fee"], rows["season"], index, level)
    rows["y"] = np.log(rows["fee_adj"])
    tr, va, te = (rows["season"] <= val_season - 1), (rows["season"] == val_season), \
                 (rows["season"] == test_season)
    f_tr, gm = features_for(rows[tr], d)
    f_va, _ = features_for(rows[va], d, gm)
    f_te, _ = features_for(rows[te], d, gm)
    # plausible-buyer references from the three most recent seasons available at each stage,
    # never including the rows being scored
    def recent(f):
        return f[f["year"] >= f["year"].max() - 2]
    trva = pd.concat([f_tr, f_va], ignore_index=True)
    gm["_buyer_ref"] = buyer_reference(recent(pd.concat([trva, f_te], ignore_index=True)))
    return {"rows": rows, "ref_val": buyer_reference(recent(f_tr)),
            "ref_test": buyer_reference(recent(trva)), "index": index, "level": level,
            "test_season": test_season,
            "f_tr": f_tr, "f_va": f_va, "f_te": f_te, "gm": gm}


def fit(prep: dict, params: dict | None = None, numeric: list[str] | None = NUMERIC_EURO,
        additive: bool = True, verbose: bool = True) -> dict:
    """Defaults are v1.5 (variant C in model.compare): Europe as one feature, depth-2
    trees, talent features interact while platform features only add."""
    f_tr, f_va, f_te = prep["f_tr"], prep["f_va"], prep["f_te"]
    index, level = prep["index"], prep["level"]
    X_tr, X_va, X_te = (design_matrix(f, numeric) for f in (f_tr, f_va, f_te))
    kw = {"params": params, "additive": additive}

    # tune rounds on the validation season, and take the band from its residuals with the
    # real buyer hidden: valuations never know the buyer, so neither does the band
    m_tr, best = fit_median(X_tr, f_tr["y"].to_numpy(), X_va, f_va["y"].to_numpy(), **kw)
    X_va_scn = design_matrix(apply_buyer_scenario(f_va, prep["ref_val"]), numeric)
    band = residual_band(f_va["y"].to_numpy(), m_tr.predict(X_va_scn))
    trva = pd.concat([f_tr, f_va], ignore_index=True)
    rule = age_decline(trva)
    m_eval, _ = fit_median(design_matrix(trva, numeric), trva["y"].to_numpy(), rounds=best, **kw)
    p_te = predict(m_eval, X_te, band, f_te["age"], rule)
    X_te_scn = design_matrix(apply_buyer_scenario(f_te, prep["ref_test"]), numeric)
    p_scn = predict(m_eval, X_te_scn, band, f_te["age"], rule)
    actual = f_te["fee_adj"].to_numpy()
    tm_ok = f_te["tm_value"].notna() & (f_te["tm_value"] > 0)
    tm_adj = deflate(f_te["tm_value"], f_te["season"], index, level)
    metrics = {
        "train": len(f_tr), "val": len(f_va), "test": len(f_te),
        "test_season": prep["test_season"], "rounds": best,
        # headline: buyer hidden, as for every valuation on the site
        "test_mdape": mdape(p_scn["p50"], actual),
        "test_coverage_p10_p90": float(((actual >= p_scn["p10"]) & (actual <= p_scn["p90"])).mean()),
        # for reference: how well the fee is explained once the buyer is known
        "test_mdape_known_buyer": mdape(p_te["p50"], actual),
        "band_multipliers": [round(float(np.exp(b)), 2) for b in band],
        "age_support": rule[0], "age_decline_per_year": round(float(np.exp(rule[1]) - 1), 3),
        "tm_value_mdape_same_moves": mdape(tm_adj[tm_ok], f_te.loc[tm_ok, "fee_adj"]),
        "naive_median_mdape": mdape(np.full(len(actual), np.median(f_tr["fee_adj"])), actual),
        "mdape_by_line": {ln: round(mdape(p_scn.loc[m, "p50"], actual[m.to_numpy()]), 3)
                          for ln in ("ATT", "MID", "DEF", "GK")
                          if (m := f_te["line"] == ln).sum() >= 10},
        "fee_index": {int(k): round(float(v)) for k, v in index.items()},
    }

    allf = pd.concat([f_tr, f_va, f_te], ignore_index=True)
    final, _ = fit_median(design_matrix(allf, numeric), allf["y"].to_numpy(), rounds=best, **kw)
    cap = float(prep["rows"]["fee_adj"].max())
    dense_max = float(prep["rows"]["fee_adj"].quantile(DENSE_QUANTILE))
    metrics["cap_eur"] = round(cap)
    metrics["low_confidence_above_eur"] = round(dense_max)
    if verbose:
        print(json.dumps({k: v for k, v in metrics.items() if k != "fee_index"}, indent=2))
    return {"model": final, "group_means": prep["gm"], "metrics": metrics, "level": level,
            "columns": list(X_tr.columns), "cap": cap, "dense_max": dense_max,
            "band": band, "age_rule": rule,
            "numeric": numeric, "params": params, "additive": additive}


def train(d: Data, verbose: bool = True, min_test_moves: int = 200, **variant) -> dict:
    return fit(prepare(d, min_test_moves), verbose=verbose, **variant)


def current_rows(d: Data) -> pd.DataFrame:
    as_of = pd.to_datetime(d.apps["date"]).max() + pd.Timedelta(days=1)
    p = d.players[current_top5_player_mask(d.players, d.clubs)]
    league = dict(zip(d.clubs["club_id"], d.clubs["domestic_competition_id"]))
    rows = pd.DataFrame({
        "player_id": p["player_id"].astype(int), "as_of": as_of,
        "club_id": p["current_club_id"], "league": p["current_club_id"].map(league).fillna("other"),
        "tm_value": pd.to_numeric(p["market_value_in_eur"], errors="coerce"),
    })
    rows = rows.merge(_player_info(d), on="player_id", how="inner")
    return rows.dropna(subset=["birth", "group"]).reset_index(drop=True)


def current_features(d: Data, result: dict) -> pd.DataFrame:
    f, _ = features_for(current_rows(d), d, result["group_means"])
    f["year"] = result["level"]  # today's market level
    return f


def value_current(d: Data, result: dict, f: pd.DataFrame | None = None) -> pd.DataFrame:
    f = current_features(d, result) if f is None else f
    X = design_matrix(f, result.get("numeric"))[result["columns"]]
    p = predict(result["model"], X, result["band"], f["age"], result["age_rule"],
                cap=result["cap"], dense_max=result.get("dense_max"))
    out = pd.concat([f[["player_id", "minutes_1y", "minutes_best", "tm_value", "group",
                        "line", "age"]], p], axis=1)
    out["drivers"] = drivers(result["model"], X, p["p50"])
    out["gap"] = out["p50"] / out["tm_value"] - 1
    return out


def save(result: dict, values: pd.DataFrame) -> dict:
    from psycopg.types.json import Jsonb

    from mercato.db import connect

    out = MODEL_DIR / VERSION
    out.mkdir(parents=True, exist_ok=True)
    result["model"].save_model(out / "p50.json")
    gm = {k: v for k, v in result["group_means"].items() if k not in ("_pct", "_buyer_ref")}
    gm["_buyer_ref"] = result["group_means"]["_buyer_ref"].to_dict("list")
    gm["_pct"] = {src: {g: list(map(float, arr)) for g, arr in by.items()}
                  for src, by in result["group_means"]["_pct"].items()}
    (out / "meta.json").write_text(json.dumps(
        {"columns": result["columns"], "group_means": gm,
         "level": result["level"], "cap": result["cap"], "dense_max": result["dense_max"],
         "numeric": result["numeric"], "additive": result["additive"], "band": result["band"],
         "age_rule": result["age_rule"], "metrics": result["metrics"]},
        indent=2, default=str))

    with connect() as conn, conn.cursor() as cur:
        cur.execute("insert into model_runs (version, test_mdape) values (%s, %s) returning id",
                    (VERSION, result["metrics"]["test_mdape"]))
        run_id = cur.fetchone()[0]
        cur.execute("select tm_player_id, mercato_id from id_map where tm_player_id is not null")
        mid = dict(cur.fetchall())
        rows = [(run_id, mid[int(r.player_id)], float(r.p10), float(r.p50), float(r.p90),
                 Jsonb(r.drivers), bool(r.capped), bool(r.low_confidence))
                for r in values.itertuples() if int(r.player_id) in mid]
        with conn.pipeline():
            cur.executemany("insert into valuations (model_run_id, mercato_id, p10_eur, p50_eur, "
                            "p90_eur, shap, capped, low_confidence) "
                            "values (%s, %s, %s, %s, %s, %s, %s, %s)", rows)
        conn.commit()
    return {"model_run_id": run_id, "valuations": len(rows), "saved_to": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="train and print only")
    args = ap.parse_args()
    d = load()
    result = train(d)
    idx = result["metrics"]["fee_index"]
    print("fee index (median comparable move, €m):",
          ", ".join(f"{k}: {v / 1e6:.1f}" for k, v in idx.items()))
    values = value_current(d, result)
    report(d, values)
    if not args.dry:
        print(save(result, values))


def report(d: Data, values: pd.DataFrame) -> None:
    names = dict(zip(d.players["player_id"], d.players["name"]))
    low = values[values["low_confidence"]].assign(name=lambda x: x["player_id"].map(names))
    print(f"\nLow confidence (outside the model's data, kept out of rankings): {len(low)}: "
          + ", ".join(low.sort_values("p50", ascending=False)["name"].head(12)))
    regulars = values[(values["minutes_best"] >= 1500) & (values["tm_value"] >= 5e6)
                      & ~values["low_confidence"]].copy()
    regulars["name"] = regulars["player_id"].map(names)
    cols = ["name", "group", "age", "p10", "p50", "p90", "tm_value", "gap"]
    fmt = {"age": "{:.0f}".format, "gap": "{:+.0%}".format,
           **{c: (lambda v: f"€{v / 1e6:.1f}m") for c in ["p10", "p50", "p90", "tm_value"]}}
    print("\nMost undervalued by Transfermarkt (regulars, TM value €5m+), by line:")
    for ln, title in [("ATT", "Attack"), ("MID", "Midfield"), ("DEF", "Defence"), ("GK", "GK")]:
        sub = regulars[regulars["line"] == ln].sort_values("gap", ascending=False)
        print(f"\n{title}")
        print(sub[cols].head(5).to_string(index=False, formatters=fmt))
    print("\nHighest valuations:")
    top = values.assign(name=values["player_id"].map(names)).sort_values("p50", ascending=False)
    print(top[cols].head(10).to_string(index=False, formatters=fmt))
    print("\nMost overvalued:")
    print(regulars.sort_values("gap")[cols].head(10).to_string(index=False, formatters=fmt))


if __name__ == "__main__":
    main()
