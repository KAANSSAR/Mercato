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
from mercato.model.features import club_games as club_games_table
from mercato.model.features import (
    design_matrix,
    finish,
    latest_elo,
    monotone_constraints,
    prepare_appearances,
    raw_features,
)
from mercato.model.fee_index import build_index, deflate, season_of
from mercato.sources.tm_datasets import (
    DATA_DIR,
    TOP5,
    current_top5_player_mask,
    position_group,
    read,
)

VERSION = "v1.2"
BAND = (0.10, 0.90)  # P10 to P90
MIN_AGE_SUPPORT = 30  # transfers needed at or above an age to trust the model there
MIN_TRAIN_FEE = 500_000
FIRST_TRAIN_SEASON = 2015
MODEL_DIR = Path("data/models")
PARAMS = {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.8,
          "colsample_bytree": 0.8, "reg_lambda": 1.0, "tree_method": "hist"}


@dataclass
class Data:
    players: pd.DataFrame
    clubs: pd.DataFrame
    transfers: pd.DataFrame
    apps: pd.DataFrame
    team_elo: pd.DataFrame
    _prepared: tuple | None = None

    def prepared(self):
        """Appearances with opponent Elo, club game dates, club Elo history. Built once."""
        if self._prepared is None:
            self._prepared = (prepare_appearances(self.apps, self.team_elo),
                              club_games_table(self.team_elo), latest_elo(self.team_elo))
        return self._prepared


def load(data_dir: Path = DATA_DIR) -> Data:
    games = read("games", data_dir)
    return Data(players=read("players", data_dir), clubs=read("clubs", data_dir),
                transfers=read("transfers", data_dir), apps=read("appearances", data_dir),
                team_elo=run_elo(games))


def _player_info(d: Data) -> pd.DataFrame:
    p = d.players[["player_id", "date_of_birth", "sub_position", "position"]].copy()
    p["group"] = [position_group(s, q) for s, q in zip(p["sub_position"], p["position"])]
    return p.rename(columns={"date_of_birth": "birth"})[["player_id", "birth", "group"]]


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
        "fee": t["fee"], "season": t["season"],
        "tm_value": pd.to_numeric(t.get("market_value_in_eur"), errors="coerce"),
    })
    rows = rows.merge(_player_info(d), on="player_id", how="inner")
    return rows.dropna(subset=["birth", "group"]).reset_index(drop=True)


def features_for(rows: pd.DataFrame, d: Data, group_means: dict | None = None):
    return finish(raw_features(rows, *d.prepared()), group_means)


def fit_median(X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None,
               rounds: int | None = None) -> tuple[xgb.XGBRegressor, int]:
    """Median (P50) model on log fees."""
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
                         n_estimators=rounds or 3000,
                         monotone_constraints=monotone_constraints(list(X.columns)),
                         early_stopping_rounds=100 if X_val is not None else None, **PARAMS)
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
    return float(support), float(min(slope, -0.05))


def predict(model, X: pd.DataFrame, band: tuple[float, float], ages=None,
            age_rule: tuple[float, float] | None = None, cap: float | None = None) -> pd.DataFrame:
    """P50 from the model, P10 and P90 from the conformal band, then the age decline past
    the supported ages and the cap at the highest inflation-adjusted fee ever paid
    (v1 put Lamine Yamal at €602m)."""
    log_p50 = model.predict(X)
    if ages is not None and age_rule is not None:
        support, slope = age_rule
        log_p50 = log_p50 + slope * np.clip(np.asarray(ages, dtype=float) - support, 0, None)
    out = pd.DataFrame({"p10": np.exp(log_p50 + band[0]), "p50": np.exp(log_p50),
                        "p90": np.exp(log_p50 + band[1])}, index=X.index)
    if cap is not None:
        out["capped"] = out["p50"] > cap
        out[["p10", "p50", "p90"]] = out[["p10", "p50", "p90"]].clip(upper=cap)
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


def train(d: Data, verbose: bool = True, min_test_moves: int = 200) -> dict:
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
    X_tr, X_va, X_te = design_matrix(f_tr), design_matrix(f_va), design_matrix(f_te)

    # tune rounds on the validation season, and take the band from its residuals
    m_tr, best = fit_median(X_tr, f_tr["y"].to_numpy(), X_va, f_va["y"].to_numpy())
    band = residual_band(f_va["y"].to_numpy(), m_tr.predict(X_va))
    trva = pd.concat([f_tr, f_va], ignore_index=True)
    rule = age_decline(trva)
    m_eval, _ = fit_median(design_matrix(trva), trva["y"].to_numpy(), rounds=best)
    p_te = predict(m_eval, X_te, band, f_te["age"], rule)
    actual = f_te["fee_adj"].to_numpy()
    tm_ok = f_te["tm_value"].notna() & (f_te["tm_value"] > 0)
    tm_adj = deflate(f_te["tm_value"], f_te["season"], index, level)
    metrics = {
        "train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
        "test_season": test_season, "rounds": best,
        "test_mdape": mdape(p_te["p50"], actual),
        "test_coverage_p10_p90": float(((actual >= p_te["p10"]) & (actual <= p_te["p90"])).mean()),
        "band_multipliers": [round(float(np.exp(b)), 2) for b in band],
        "age_support": rule[0], "age_decline_per_year": round(float(np.exp(rule[1]) - 1), 3),
        "tm_value_mdape_same_moves": mdape(tm_adj[tm_ok], f_te.loc[tm_ok, "fee_adj"]),
        "naive_median_mdape": mdape(np.full(len(actual), np.median(f_tr["fee_adj"])), actual),
        "fee_index": {int(k): round(float(v)) for k, v in index.items()},
    }

    allf = pd.concat([f_tr, f_va, f_te], ignore_index=True)
    final, _ = fit_median(design_matrix(allf), allf["y"].to_numpy(), rounds=best)
    cap = float(rows["fee_adj"].max())
    metrics["cap_eur"] = round(cap)
    if verbose:
        print(json.dumps({k: v for k, v in metrics.items() if k != "fee_index"}, indent=2))
    return {"model": final, "group_means": gm, "metrics": metrics, "level": level,
            "columns": list(X_tr.columns), "cap": cap, "band": band, "age_rule": rule}


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


def value_current(d: Data, result: dict) -> pd.DataFrame:
    rows = current_rows(d)
    f, _ = features_for(rows, d, result["group_means"])
    f["year"] = result["level"]  # today's market level
    X = design_matrix(f)[result["columns"]]
    p = predict(result["model"], X, result["band"], f["age"], result["age_rule"],
                cap=result["cap"])
    out = pd.concat([f[["player_id", "minutes_1y", "minutes_best", "tm_value", "group",
                        "age"]], p], axis=1)
    out["drivers"] = drivers(result["model"], X, p["p50"])
    out["gap"] = out["p50"] / out["tm_value"] - 1
    return out


def save(result: dict, values: pd.DataFrame) -> dict:
    from psycopg.types.json import Jsonb

    from mercato.db import connect

    out = MODEL_DIR / VERSION
    out.mkdir(parents=True, exist_ok=True)
    result["model"].save_model(out / "p50.json")
    (out / "meta.json").write_text(json.dumps(
        {"columns": result["columns"], "group_means": result["group_means"],
         "level": result["level"], "cap": result["cap"], "band": result["band"],
         "age_rule": result["age_rule"], "metrics": result["metrics"]},
        indent=2, default=str))

    with connect() as conn, conn.cursor() as cur:
        cur.execute("insert into model_runs (version, test_mdape) values (%s, %s) returning id",
                    (VERSION, result["metrics"]["test_mdape"]))
        run_id = cur.fetchone()[0]
        cur.execute("select tm_player_id, mercato_id from id_map where tm_player_id is not null")
        mid = dict(cur.fetchall())
        rows = [(run_id, mid[int(r.player_id)], float(r.p10), float(r.p50), float(r.p90),
                 Jsonb(r.drivers)) for r in values.itertuples() if int(r.player_id) in mid]
        with conn.pipeline():
            cur.executemany("insert into valuations (model_run_id, mercato_id, p10_eur, p50_eur, "
                            "p90_eur, shap) values (%s, %s, %s, %s, %s, %s)", rows)
        conn.commit()
    return {"model_run_id": run_id, "valuations": len(rows), "saved_to": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="train and print only")
    args = ap.parse_args()
    d = load()
    result = train(d)
    values = value_current(d, result)
    report(d, values)
    if not args.dry:
        print(save(result, values))


def report(d: Data, values: pd.DataFrame) -> None:
    names = dict(zip(d.players["player_id"], d.players["name"]))
    regulars = values[(values["minutes_best"] >= 1500) & (values["tm_value"] >= 5e6)].copy()
    regulars["name"] = regulars["player_id"].map(names)
    cols = ["name", "group", "age", "p10", "p50", "p90", "tm_value", "gap"]
    fmt = {"age": "{:.0f}".format, "gap": "{:+.0%}".format,
           **{c: (lambda v: f"€{v / 1e6:.1f}m") for c in ["p10", "p50", "p90", "tm_value"]}}
    print("\nMost undervalued by Transfermarkt (regulars, TM value €5m+):")
    print(regulars.sort_values("gap", ascending=False)[cols].head(15).to_string(
        index=False, formatters=fmt))
    print("\nMost overvalued:")
    print(regulars.sort_values("gap")[cols].head(10).to_string(index=False, formatters=fmt))


if __name__ == "__main__":
    main()
