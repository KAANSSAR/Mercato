"""Projected performance: how a player's rating is likely to move over the next six years.

Every summer since 2016 we take a snapshot of every top-5 regular (900+ minutes in the
season): his rating over the last three seasons (score3) and over that season alone
(score1). Then, per line and per horizon k = 1 to 6 years, one regression:

    score1 in year t+k  ~  score3 in year t, age, age squared

That single fit captures three things at once: good seasons partly fade (regression to
the mean), young players improve and older ones decline (age), and the regression slope
says how much of today's level is real. A player who is no longer a top-5 regular k
years later (retired, released, gone to a smaller league, out injured all season) counts
as a weak season for his line, so survivors alone don't make old age look kind.

The value index is the next six seasons of projected performance, discounted:

    V = sum over k = 0..6 of  delta^k * exp(beta * projected score in year k)

beta: how much an elite season is worth next to a good one. delta: how much later years
count next to now. Both are fitted on real fees (the only thing fees decide); everything
else is on-pitch data. No season counts past age 37.

Replaces the age curve learned from fees, where value fell off a cliff after 26 because
older players get sold cheaply on expiring contracts, not because they play worse
(v2.2 put Mbappé at €80m), and where young players carried a potential premium the
stats can't see (Hinshelwood €180m).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mercato.model.features import GROUP_LINE
from mercato.sources.tm_datasets import TOP5

ONE_SEASON = (1.0, 0.0, 0.0)
HORIZON = 6
FIRST_SNAPSHOT = 2016
SNAP_MINUTES = 900
DROPOUT_Q = 0.10   # a player who drops out counts as this quantile of his line's seasons
MAX_AGE = 37
MIN_PAIRS = 60
LINES = ("ATT", "MID", "DEF", "GK")
BETAS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
DELTAS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def snapshot_rows(apps: pd.DataFrame, clubs: pd.DataFrame, info: pd.DataFrame,
                  cutoff) -> pd.DataFrame:
    """Top-5 regulars on 1 July of each year up to the cutoff."""
    a = apps[["player_id", "player_club_id", "date", "minutes_played"]].copy()
    a["date"] = pd.to_datetime(a["date"])
    league = dict(zip(clubs["club_id"], clubs["domestic_competition_id"]))
    cutoff = pd.Timestamp(cutoff)
    out = []
    for year in range(FIRST_SNAPSHOT, cutoff.year + 1):
        as_of = pd.Timestamp(f"{year}-07-01")
        if as_of > cutoff:
            break
        w = a[(a["date"] < as_of) & (a["date"] >= as_of - pd.Timedelta(days=365))]
        mins = w.groupby("player_id")["minutes_played"].sum()
        last = w.sort_values("date").groupby("player_id").tail(1).set_index("player_id")
        reg = mins[mins >= SNAP_MINUTES].index
        r = pd.DataFrame({"player_id": reg, "as_of": as_of, "year": year,
                          "club_id": last.loc[reg, "player_club_id"].to_numpy()})
        r["league"] = r["club_id"].map(league)
        out.append(r[r["league"].isin(TOP5)])
    if not out:
        return pd.DataFrame(columns=["player_id", "as_of", "year", "club_id", "league"])
    rows = pd.concat(out, ignore_index=True).merge(info, on="player_id", how="inner")
    return rows.dropna(subset=["birth", "group"]).reset_index(drop=True)


def _x(score, age) -> np.ndarray:
    a = np.asarray(age, dtype=float) - 27.0
    return np.column_stack([np.ones(len(a)), np.asarray(score, dtype=float), a, a * a])


def fit_paths(snaps: pd.DataFrame) -> dict:
    """snaps: player_id, year, line, age, score3, score1. Returns coefficients per line and
    horizon, and the dropout level per line."""
    s1 = snaps.set_index(["player_id", "year"])["score1"]
    s1 = s1[~s1.index.duplicated()]
    last_year = int(snaps["year"].max())
    coefs, drop = {}, {}
    for ln in LINES:
        part = snaps[snaps["line"] == ln]
        if part.empty:
            continue
        drop[ln] = float(np.quantile(part["score1"], DROPOUT_Q))
        prev = np.array([0.0, 1.0, 0.0, 0.0])  # fallback: no change
        coefs[ln] = {}
        for k in range(1, HORIZON + 1):
            p = part[part["year"] + k <= last_year]
            if len(p) >= MIN_PAIRS:
                idx = pd.MultiIndex.from_arrays([p["player_id"], p["year"] + k])
                y = s1.reindex(idx).fillna(drop[ln]).to_numpy()
                prev = np.linalg.lstsq(_x(p["score3"], p["age"]), y, rcond=None)[0]
            coefs[ln][k] = prev
    return {"coefs": coefs, "dropout": drop}


def project(f: pd.DataFrame, paths: dict) -> np.ndarray:
    """Projected score for years 0 (now) to HORIZON; NaN past MAX_AGE."""
    out = np.full((len(f), HORIZON + 1), np.nan)
    out[:, 0] = f["score"].to_numpy(float)
    line = f["group"].map(GROUP_LINE).to_numpy()
    age = f["age"].to_numpy(float)
    for ln, by_k in paths["coefs"].items():
        m = line == ln
        if not m.any():
            continue
        X = _x(out[m, 0], age[m])
        for k, c in by_k.items():
            out[m, k] = X @ c
    ages = age[:, None] + np.arange(HORIZON + 1)[None, :]
    out[ages > MAX_AGE] = np.nan
    return out


def log_index(path: np.ndarray, beta: float, delta: float) -> np.ndarray:
    disc = delta ** np.arange(path.shape[1])
    terms = np.where(np.isnan(path), 0.0, disc * np.exp(beta * np.nan_to_num(path)))
    return np.log(np.maximum(terms.sum(axis=1), 1e-12))


def fit_preferences(path: np.ndarray, line: np.ndarray, y: np.ndarray) -> tuple:
    """beta and delta that best order real (inflation-adjusted) fees within each line."""
    best, score = (1.0, 0.8), -np.inf
    for b in BETAS:
        for dl in DELTAS:
            li = log_index(path, b, dl)
            rho = [(spearmanr(li[m], y[m])[0], m.sum()) for ln in LINES
                   if (m := line == ln).sum() >= 20]
            s = sum(r * n for r, n in rho) / max(sum(n for _, n in rho), 1)
            if s > score:
                best, score = (b, dl), s
    return best[0], best[1], float(score)
