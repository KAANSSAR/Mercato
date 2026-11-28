"""Player features at a point in time, from Transfermarkt appearances and Mercato Elo.

For each (player, as_of date) we look at the last two years, so one injured season
doesn't erase a player (v1 read Rodri's ACL year as decline):

- level: the better of the last two years for minutes and minutes share
- recency: minutes in the last year, unconstrained, so genuine decline still shows
- goals, assists and goal involvements per 90 over both years, each weighted by the
  opponent's Elo, then shrunk towards his position's average for small samples
- Champions League and Europa League minutes over two years (proven at the top level)
- top-5 league minutes over two years, his club's Elo, the league he plays in
- age as years under and over 27, so value has to fall past the peak even where
  there are almost no fees to learn from (v1 valued a 38-year-old like a 33-year-old)
- position group

Nothing here uses market values: they are the benchmark, never an input.
"""

import numpy as np
import pandas as pd

from mercato.features.adjust import ALPHA, ELO_BASE

K_MINUTES = 900.0  # shrinkage strength
TOP5 = ["GB1", "ES1", "IT1", "L1", "FR1"]
GROUPS = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]
LEAGUES = TOP5 + ["other"]

PEAK_AGE = 27.0
NUMERIC = ["age_under_peak", "age_over_peak", "minutes_best", "minutes_1y", "share_best",
           "apps_1y", "g90", "a90", "ga90", "ucl_minutes_2y", "uel_minutes_2y",
           "top5_minutes_2y", "club_elo", "year"]
MONOTONE = {"minutes_best": 1, "share_best": 1, "g90": 1, "a90": 1, "ga90": 1,
            "ucl_minutes_2y": 1, "top5_minutes_2y": 1, "club_elo": 1,
            "age_over_peak": -1}


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
               "top5_minutes_2y": 0.0, "club_games_1y": 0, "club_games_prev": 0}
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
                top5_minutes_2y=float(two.loc[two["competition_id"].isin(TOP5),
                                              "minutes_played"].sum()),
            )
            for key, part, lo, hi in (("club_games_1y", last, y1, t), ("club_games_prev", prev, y2, y1)):
                rec[key] = int(sum(
                    ((d < hi.to_datetime64()) & (d >= lo.to_datetime64())).sum()
                    for d in (by_club.get(c, np.array([], dtype="datetime64[ns]"))
                              for c in part["player_club_id"].unique())))
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
    mins = (f["minutes_1y"] + f["minutes_prev"]).clip(lower=0)  # per 90 over both years
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

    share_1y = np.where(f["club_games_1y"] > 0,
                        f["minutes_1y"] / (90 * f["club_games_1y"].clip(lower=1)), 0)
    share_prev = np.where(f["club_games_prev"] > 0,
                          f["minutes_prev"] / (90 * f["club_games_prev"].clip(lower=1)), 0)
    f["share_best"] = np.clip(np.maximum(share_1y, share_prev), 0, 1)
    f["minutes_best"] = np.maximum(f["minutes_1y"], f["minutes_prev"])
    birth = pd.to_datetime(f["birth"])
    f["age"] = (pd.to_datetime(f["as_of"]) - birth).dt.days / 365.25
    f["age_under_peak"] = (PEAK_AGE - f["age"]).clip(lower=0)
    f["age_over_peak"] = (f["age"] - PEAK_AGE).clip(lower=0)
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
    return tuple(MONOTONE.get(c, 0) for c in columns)
