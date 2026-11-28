"""Player features at a point in time, from Transfermarkt appearances and Mercato Elo.

For each (player, as_of date) we look back one year (and the year before that):

- minutes, appearances, minutes share of his club's games
- goals, assists and goal involvements per 90, each goal or assist weighted by the
  opponent's Elo, then shrunk towards his position's average for small samples
- Champions League and Europa League minutes over two years (proven at the top level)
- top-5 league minutes, his club's Elo, the league he plays in
- age, age squared, position group

Nothing here uses market values: they are the benchmark, never an input.
"""

import numpy as np
import pandas as pd

from mercato.features.adjust import ALPHA, ELO_BASE

K_MINUTES = 900.0  # shrinkage strength
TOP5 = ["GB1", "ES1", "IT1", "L1", "FR1"]
GROUPS = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]
LEAGUES = TOP5 + ["other"]

NUMERIC = ["age", "age2", "minutes_1y", "minutes_prev", "apps_1y", "minutes_share",
           "g90", "a90", "ga90", "ucl_minutes_2y", "uel_minutes_2y", "top5_minutes_1y",
           "club_elo", "year"]
MONOTONE_UP = {"minutes_1y", "minutes_share", "g90", "a90", "ga90", "ucl_minutes_2y",
               "club_elo", "top5_minutes_1y"}


def prepare_appearances(apps: pd.DataFrame, team_elo: pd.DataFrame) -> pd.DataFrame:
    """Attach opponent Elo and a per-appearance weight."""
    a = apps[["game_id", "player_id", "player_club_id", "date", "competition_id",
              "goals", "assists", "minutes_played"]].copy()
    a["date"] = pd.to_datetime(a["date"])
    opp = team_elo[["game_id", "tm_club_id", "opp_elo_pre"]].rename(
        columns={"tm_club_id": "player_club_id"})
    a = a.merge(opp, on=["game_id", "player_club_id"], how="left")
    a["w"] = (a["opp_elo_pre"].fillna(ELO_BASE) / ELO_BASE) ** ALPHA
    a["goals_w"] = a["goals"].fillna(0) * a["w"]
    a["assists_w"] = a["assists"].fillna(0) * a["w"]
    a["minutes_played"] = a["minutes_played"].fillna(0)
    return a.sort_values(["player_id", "date"]).reset_index(drop=True)


def club_games(team_elo: pd.DataFrame) -> pd.DataFrame:
    g = team_elo[["tm_club_id", "date"]].copy()
    g["date"] = pd.to_datetime(g["date"])
    return g.sort_values(["tm_club_id", "date"]).reset_index(drop=True)


def latest_elo(team_elo: pd.DataFrame) -> pd.DataFrame:
    e = team_elo[["tm_club_id", "date", "elo_post"]].copy()
    e["date"] = pd.to_datetime(e["date"])
    return e.sort_values("date").reset_index(drop=True)


def raw_features(rows: pd.DataFrame, apps: pd.DataFrame, games: pd.DataFrame,
                 elo: pd.DataFrame) -> pd.DataFrame:
    """rows: player_id, as_of, club_id, league, birth, group. Returns rows plus raw stats."""
    by_player = {pid: g for pid, g in apps.groupby("player_id", sort=False)}
    by_club = {cid: g["date"].to_numpy() for cid, g in games.groupby("tm_club_id", sort=False)}
    out = []
    for r in rows.itertuples(index=False):
        t = pd.Timestamp(r.as_of)
        y1, y2 = t - pd.Timedelta(days=365), t - pd.Timedelta(days=730)
        a = by_player.get(r.player_id)
        rec = {"minutes_1y": 0.0, "minutes_prev": 0.0, "apps_1y": 0, "goals_w": 0.0,
               "assists_w": 0.0, "ucl_minutes_2y": 0.0, "uel_minutes_2y": 0.0,
               "top5_minutes_1y": 0.0, "club_games_1y": 0}
        if a is not None:
            last = a[(a["date"] < t) & (a["date"] >= y1)]
            prev = a[(a["date"] < y1) & (a["date"] >= y2)]
            two = a[(a["date"] < t) & (a["date"] >= y2)]
            rec.update(
                minutes_1y=float(last["minutes_played"].sum()),
                minutes_prev=float(prev["minutes_played"].sum()),
                apps_1y=len(last),
                goals_w=float(last["goals_w"].sum()),
                assists_w=float(last["assists_w"].sum()),
                ucl_minutes_2y=float(two.loc[two["competition_id"] == "CL", "minutes_played"].sum()),
                uel_minutes_2y=float(two.loc[two["competition_id"] == "EL", "minutes_played"].sum()),
                top5_minutes_1y=float(last.loc[last["competition_id"].isin(TOP5),
                                               "minutes_played"].sum()),
            )
            clubs = last["player_club_id"].unique()
            rec["club_games_1y"] = int(sum(
                ((d < t.to_datetime64()) & (d >= y1.to_datetime64())).sum()
                for d in (by_club.get(c, np.array([], dtype="datetime64[ns]")) for c in clubs)))
        out.append(rec)
    feats = pd.concat([rows.reset_index(drop=True), pd.DataFrame(out)], axis=1)

    # club Elo on the as_of date
    left = pd.DataFrame({"club_id": pd.to_numeric(feats["club_id"], errors="coerce").astype(float),
                         "as_of": pd.to_datetime(feats["as_of"]).astype("datetime64[ns]"),
                         "_i": np.arange(len(feats))})
    right = elo.rename(columns={"tm_club_id": "club_id", "date": "as_of"})
    right = right.assign(club_id=right["club_id"].astype(float),
                         as_of=pd.to_datetime(right["as_of"]).astype("datetime64[ns]"))
    ce = pd.merge_asof(left.sort_values("as_of"), right.sort_values("as_of"),
                       on="as_of", by="club_id", direction="backward")
    feats["club_elo"] = ce.sort_values("_i")["elo_post"].to_numpy()
    return feats


def finish(feats: pd.DataFrame, group_means: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Per 90 with shrinkage towards the position average, age, encodings.
    group_means from training are reused at prediction time."""
    f = feats.copy()
    mins = f["minutes_1y"].clip(lower=0)
    for k, src in [("g90", "goals_w"), ("a90", "assists_w")]:
        f[f"{k}_raw"] = np.where(mins > 0, f[src] / mins.where(mins > 0, 1) * 90, 0.0)
    f["ga90_raw"] = f["g90_raw"] + f["a90_raw"]
    if group_means is None:
        group_means = {}
        for k in ("g90", "a90", "ga90"):
            weighted = (f[f"{k}_raw"] * mins).groupby(f["group"]).sum()
            group_means[k] = (weighted / mins.groupby(f["group"]).sum().replace(0, np.nan)
                              ).fillna(0).to_dict()
    for k in ("g90", "a90", "ga90"):
        prior = f["group"].map(group_means[k]).fillna(0)
        f[k] = (mins * f[f"{k}_raw"] + K_MINUTES * prior) / (mins + K_MINUTES)

    f["minutes_share"] = np.where(f["club_games_1y"] > 0,
                                  f["minutes_1y"] / (90 * f["club_games_1y"].clip(lower=1)), 0)
    f["minutes_share"] = f["minutes_share"].clip(0, 1)
    birth = pd.to_datetime(f["birth"])
    f["age"] = (pd.to_datetime(f["as_of"]) - birth).dt.days / 365.25
    f["age2"] = f["age"] ** 2
    f["year"] = pd.to_datetime(f["as_of"]).map(lambda d: d.year if d.month >= 7 else d.year - 1)
    f["league"] = f["league"].where(f["league"].isin(TOP5), "other")
    return f, group_means


def design_matrix(f: pd.DataFrame) -> pd.DataFrame:
    X = f[NUMERIC].astype(float).copy()
    for g in GROUPS:
        X[f"pos_{g}"] = (f["group"] == g).astype(float)
    for lg in LEAGUES:
        X[f"league_{lg}"] = (f["league"] == lg).astype(float)
    return X


def monotone_constraints(columns: list[str]) -> tuple[int, ...]:
    return tuple(1 if c in MONOTONE_UP else 0 for c in columns)
