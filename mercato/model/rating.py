"""Mercato performance rating: how well a player plays for his position, 0 to 100.

On-pitch evidence only, over the last three years weighted 1, 0.75, 0.5 towards the recent:

- output: goals and assists per 90, each weighted by the opponent's Elo, shrunk towards
  the position average for small samples (from model.features.finish)
- impact: his team's results above what Elo expected, in the games he played, per 90
- defence: goals his team conceded below what Elo expected, in the games he played
- minutes: share of his club's minutes, best season and last season
- opposition: the average Elo of the opponents he played against (a Ligue 2 hat-trick
  isn't a Champions League one)
- team level: the average Elo of the teams he played for, minutes weighted. Until
  tackles, passing and saves arrive from API-Football, being a regular in a strong team
  is the best evidence we have for defenders and keepers. It counts only through minutes
  he actually played, never as a platform premium for the club he's at.

Each component is measured against regulars in his own position (strikers against
strikers): how many typical spreads above or below the typical regular, capped at four so
no single stat can run away with it. v2.0 used percentiles against everyone at a top-5
club, bench included, so every good player sat at 97 to 100 and the top couldn't be told
apart. Components are combined with weights per line; the combined score is ranked
against regulars in his position for the 0 to 100 rating shown on the site.

Fixed, published weights rather than fitted ones, so the rating means the same thing on
every page and can't quietly learn the market's habits.
"""

import itertools

import numpy as np
import pandas as pd

from mercato.features.elo import EloParams
from mercato.model.features import GROUP_LINE, GROUPS, RECENCY, YEARS

COMPONENTS = ["output", "impact", "defence", "minutes", "opposition", "team_level"]
WEIGHTS = {
    "ATT": {"output": 0.40, "impact": 0.10, "defence": 0.00, "minutes": 0.20,
            "opposition": 0.15, "team_level": 0.15},
    "MID": {"output": 0.25, "impact": 0.15, "defence": 0.05, "minutes": 0.20,
            "opposition": 0.15, "team_level": 0.20},
    "DEF": {"output": 0.05, "impact": 0.15, "defence": 0.20, "minutes": 0.20,
            "opposition": 0.15, "team_level": 0.25},
    "GK": {"output": 0.00, "impact": 0.10, "defence": 0.35, "minutes": 0.20,
           "opposition": 0.10, "team_level": 0.25},
}
# goals and assists in the output component, by line
OUTPUT_MIX = {"ATT": (1.0, 0.75), "MID": (1.0, 1.0), "DEF": (1.0, 1.0), "GK": (0.0, 0.0)}
K_GAMES = 15.0  # shrinkage for impact and defence, in full games
K_LEVEL = 5.0   # shrinkage for opposition and team level, in full games
ELO_PRIOR = 1500.0
Z_CAP = 4.0     # a component can lift a player at most four spreads above typical
REF_MINUTES = 900  # the yardstick is regulars: players with this many minutes in a season


def team_games(team_elo: pd.DataFrame, hfa: float = EloParams().hfa,
               bins: int = 20) -> pd.DataFrame:
    """One row per club per game with goals against and the goals against Elo expected.
    Expected goals against comes from the data: the average conceded at each Elo gap."""
    t = team_elo.copy()
    gap = t["elo_pre"] - t["opp_elo_pre"] + np.where(t["is_home"], hfa, -hfa)
    t["gap"] = gap
    edges = np.unique(np.quantile(gap, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, gap, side="right") - 1, 0, len(edges) - 2)
    centres = pd.Series(gap).groupby(which).mean().to_numpy()
    means = pd.Series(t["goals_against"].to_numpy(dtype=float)).groupby(which).mean().to_numpy()
    t["exp_ga"] = np.interp(gap, centres, means)
    t["date"] = pd.to_datetime(t["date"]).astype("datetime64[ns]")
    return t[["game_id", "tm_club_id", "result", "expected", "goals_against", "exp_ga",
              "elo_pre", "opp_elo_pre"]]


def game_features(rows: pd.DataFrame, apps: pd.DataFrame, tg: pd.DataFrame,
                  recency: tuple = RECENCY) -> pd.DataFrame:
    """Impact, defence, opposition and team level over the three years before each as_of."""
    a = apps[["game_id", "player_id", "player_club_id", "date", "minutes_played"]].merge(
        tg.rename(columns={"tm_club_id": "player_club_id"}), on=["game_id", "player_club_id"],
        how="inner")
    a = a[a["minutes_played"].fillna(0) > 0]
    a["date"] = pd.to_datetime(a["date"]).astype("datetime64[ns]")
    a = a.sort_values(["player_id", "date"])
    cols = {"impact": (a["result"] - a["expected"]).to_numpy(),
            "defence": (a["exp_ga"] - a["goals_against"]).to_numpy(),
            "opposition": a["opp_elo_pre"].to_numpy(), "team_level": a["elo_pre"].to_numpy()}
    dates, mins = a["date"].to_numpy(), a["minutes_played"].to_numpy(dtype=float) / 90
    spans = {}
    pids = a["player_id"].to_numpy()
    if len(pids):
        cut = np.flatnonzero(np.r_[True, pids[1:] != pids[:-1], True])
        spans = {pids[s]: (s, e) for s, e in itertools.pairwise(cut)}
    year = np.timedelta64(365, "D")
    out = np.zeros((len(rows), 5))
    for i, (pid, as_of) in enumerate(zip(rows["player_id"], pd.to_datetime(rows["as_of"]))):
        s, e = spans.get(pid, (0, 0))
        t = np.datetime64(as_of, "ns")
        d = dates[s:e]
        lo, hi = np.searchsorted(d, t - YEARS * year), np.searchsorted(d, t)
        if hi <= lo:
            continue
        k = ((t - d[lo:hi]) // year).astype(int).clip(0, YEARS - 1)
        w = np.asarray(recency)[k] * mins[s + lo:s + hi]
        out[i, 0] = w.sum()
        for j, c in enumerate(COMPONENTS[1:3] + COMPONENTS[4:], start=1):
            out[i, j] = (w * cols[c][s + lo:s + hi]).sum()
    n = out[:, 0]
    return pd.DataFrame({
        "games_w": n,
        "impact": out[:, 1] / (n + K_GAMES),
        "defence": out[:, 2] / (n + K_GAMES),
        "opposition": (out[:, 3] + K_LEVEL * ELO_PRIOR) / (n + K_LEVEL),
        "team_level": (out[:, 4] + K_LEVEL * ELO_PRIOR) / (n + K_LEVEL),
    }, index=rows.index)


def components(f: pd.DataFrame) -> pd.DataFrame:
    """Raw component values (higher is better) next to the game features."""
    f = f.copy()
    line = f["group"].map(GROUP_LINE)
    g = line.map(lambda ln: OUTPUT_MIX[ln][0])
    a = line.map(lambda ln: OUTPUT_MIX[ln][1])
    f["output"] = g * f["g90"] + a * f["a90"]
    f["minutes"] = 0.5 * f["share_best"] + 0.5 * f["share_1y"]
    return f


def _pct(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of x within the sorted reference."""
    if not len(ref):
        return np.full(len(x), 0.5)
    lo, hi = np.searchsorted(ref, x, side="left"), np.searchsorted(ref, x, side="right")
    return (lo + hi) / 2 / len(ref)


def _centre_scale(x: np.ndarray) -> tuple[float, float]:
    """Robust centre and spread (median and MAD), so a few outliers don't set the yardstick."""
    if not len(x):
        return 0.0, 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad <= 0:
        mad = float(np.std(x)) or 1.0
    return med, mad


def _scores(f: pd.DataFrame, comp_ref: dict, weights: dict) -> pd.DataFrame:
    z = pd.DataFrame(index=f.index)
    for c in COMPONENTS:
        vals = np.zeros(len(f))
        for grp in GROUPS:
            m = (f["group"] == grp).to_numpy()
            if m.any() and grp in comp_ref[c]:
                med, mad = comp_ref[c][grp]
                vals[m] = np.clip((f.loc[m, c].to_numpy(float) - med) / mad, -Z_CAP, Z_CAP)
        z[f"z_{c}"] = vals
    line = f["group"].map(GROUP_LINE)
    score = np.zeros(len(f))
    for ln, w in weights.items():
        m = (line == ln).to_numpy()
        score[m] = sum(w[c] * z.loc[m, f"z_{c}"].to_numpy() for c in COMPONENTS)
    z["score"] = score
    return z


def rate(f: pd.DataFrame, ref: dict | None = None,
         weights: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Adds z_<component>, score (continuous, feeds the value curve) and rating (0 to 100
    within position). ref holds the reference pool's distributions, reused later."""
    weights = weights or WEIGHTS
    f = components(f)
    if ref is None:
        pool = f[f["minutes_best"] >= REF_MINUTES]
        if len(pool) < 50:
            pool = f
        comp_ref = {c: {g: _centre_scale(pool.loc[pool["group"] == g, c].to_numpy(float))
                        for g in GROUPS if (pool["group"] == g).any()} for c in COMPONENTS}
        z = _scores(pool, comp_ref, weights)
        score_ref = {g: np.sort(z.loc[(pool["group"] == g).to_numpy(), "score"].to_numpy())
                     for g in GROUPS}
        ref = {"components": comp_ref, "score": score_ref, "weights": weights}
    z = _scores(f, ref["components"], ref["weights"])
    f = pd.concat([f, z], axis=1)
    rating = np.zeros(len(f))
    for g in GROUPS:
        m = (f["group"] == g).to_numpy()
        if m.any():
            rating[m] = 100 * _pct(ref["score"].get(g, np.array([])), f.loc[m, "score"].to_numpy())
    f["rating"] = rating.round(1)
    return f, ref
