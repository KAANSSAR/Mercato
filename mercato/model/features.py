"""Player features at a point in time, from Transfermarkt appearances and Mercato Elo.

For each (player, as_of date) we look at the last three seasons, so injured seasons
don't erase a player (v1 read Rodri's ACL year as decline; two seasons wasn't enough
either, since he was in and out after it):

- level: the best of the last three years for share of his club's minutes
- recency: share of his club's minutes in the last year, unconstrained, so genuine
  decline still shows
- goals, assists and goal involvements per 90 over three years weighted 1, 0.75, 0.5
  towards the recent, each goal or assist weighted by the opponent's Elo, then shrunk
  towards his position's average for small samples
- share of a full Champions League and Europa League run over two years, capped at one
  (proven at the top level, without rewarding sheer volume)
- share of top-5 league minutes over two years, capped at one, his club's Elo, the league
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
YEARS = 3
RECENCY = (1.0, 0.75, 0.5)  # weights for last year, the year before, and the one before that
# Playing time is counted once, as shares, not as four stacked raw totals: v1.3 added
# best-season, top-5 and Champions League minutes on top of each other, so an ironman like
# Morgan Rogers hit the cap on volume alone while his output per 90 trailed Palmer's.
LEAGUE_SEASON_MINUTES = 38 * 90
EURO_RUN_MINUTES = 1200  # roughly one full Champions League or Europa League campaign
NUMERIC = ["age_under_peak", "age_over_peak", "share_best", "share_1y", "top5_share_2y",
           "ucl_share_2y", "uel_share_2y", "g90", "a90", "ga90", "club_elo", "year"]
MONOTONE = {"share_best": 1, "top5_share_2y": 1, "ucl_share_2y": 1, "euro_share": 1,
            "g90": 1, "a90": 1, "ga90": 1, "club_elo": 1, "age_over_peak": -1}
NUMERIC_EURO = [c for c in NUMERIC if c not in ("ucl_share_2y", "uel_share_2y")] + ["euro_share"]
# talent: age, output and position can interact; platform features each act on their own
TALENT = ["age_under_peak", "age_over_peak", "g90", "a90", "ga90"]


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
    empty = np.array([], dtype="datetime64[ns]")
    for r in rows.itertuples(index=False):
        t = pd.Timestamp(r.as_of)
        edges = [t - pd.Timedelta(days=365 * k) for k in range(YEARS + 1)]  # t, t-1y, t-2y, t-3y
        a = by_player.get(r.player_id)
        rec = {"apps_1y": 0, "ucl_minutes_2y": 0.0, "uel_minutes_2y": 0.0, "top5_minutes_2y": 0.0}
        for k in range(YEARS):
            rec |= {f"minutes_y{k}": 0.0, f"goals_y{k}": 0.0, f"assists_y{k}": 0.0,
                    f"club_games_y{k}": 0}
        if a is not None:
            two = a[(a["date"] < t) & (a["date"] >= edges[2])]
            rec |= {"ucl_minutes_2y": float(two.loc[two["competition_id"] == "CL",
                                                     "minutes_played"].sum()),
                    "uel_minutes_2y": float(two.loc[two["competition_id"] == "EL",
                                                     "minutes_played"].sum()),
                    "top5_minutes_2y": float(two.loc[two["competition_id"].isin(TOP5),
                                                      "minutes_played"].sum())}
            for k in range(YEARS):
                hi, lo = edges[k], edges[k + 1]
                part = a[(a["date"] < hi) & (a["date"] >= lo)]
                if k == 0:
                    rec["apps_1y"] = len(part)
                rec[f"minutes_y{k}"] = float(part["minutes_played"].sum())
                rec[f"goals_y{k}"] = float(part["goals_w"].sum())
                rec[f"assists_y{k}"] = float(part["assists_w"].sum())
                rec[f"club_games_y{k}"] = int(sum(
                    ((d < hi.to_datetime64()) & (d >= lo.to_datetime64())).sum()
                    for d in (by_club.get(c, empty) for c in part["player_club_id"].unique())))
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
    # recency-weighted output over three seasons: an injured year neither erases a player
    # nor lets an old peak dominate
    w = RECENCY
    mins = sum(w[k] * f[f"minutes_y{k}"] for k in range(YEARS)).clip(lower=0)
    goals = sum(w[k] * f[f"goals_y{k}"] for k in range(YEARS))
    assists = sum(w[k] * f[f"assists_y{k}"] for k in range(YEARS))
    safe = mins.where(mins > 0, 1)
    f["g90_raw"] = np.where(mins > 0, goals / safe * 90, 0.0)
    f["a90_raw"] = np.where(mins > 0, assists / safe * 90, 0.0)
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

    # level: the best of the last three seasons
    shares = [np.where(f[f"club_games_y{k}"] > 0,
                       f[f"minutes_y{k}"] / (90 * f[f"club_games_y{k}"].clip(lower=1)), 0)
              for k in range(YEARS)]
    f["share_best"] = np.clip(np.maximum.reduce(shares), 0, 1)  # availability at his best
    f["share_1y"] = np.clip(shares[0], 0, 1)                    # recency, unconstrained
    f["minutes_best"] = np.maximum.reduce([f[f"minutes_y{k}"] for k in range(YEARS)])
    f["minutes_1y"] = f["minutes_y0"]
    f["top5_share_2y"] = (f["top5_minutes_2y"] / (2 * LEAGUE_SEASON_MINUTES)).clip(0, 1)
    f["ucl_share_2y"] = (f["ucl_minutes_2y"] / EURO_RUN_MINUTES).clip(0, 1)
    f["uel_share_2y"] = (f["uel_minutes_2y"] / EURO_RUN_MINUTES).clip(0, 1)
    # Europe as one signal: a Europa League run counts half a Champions League run, capped,
    # so a player can't collect both rewards (Rogers: UCL in 2024/25, UEL in 2025/26)
    f["euro_share"] = (f["ucl_share_2y"] + 0.5 * f["uel_share_2y"]).clip(0, 1)
    birth = pd.to_datetime(f["birth"])
    f["age"] = (pd.to_datetime(f["as_of"]) - birth).dt.days / 365.25
    f["age_under_peak"] = (PEAK_AGE - f["age"]).clip(lower=0)
    f["age_over_peak"] = (f["age"] - PEAK_AGE).clip(lower=0)
    f["year"] = pd.to_datetime(f["as_of"]).map(lambda d: d.year if d.month >= 7 else d.year - 1)
    f["league"] = f["league"].where(f["league"].isin(TOP5), "other")
    return f, group_means


def design_matrix(f: pd.DataFrame, numeric: list[str] | None = None) -> pd.DataFrame:
    X = f[numeric or NUMERIC].astype(float).copy()
    for g in GROUPS:
        X[f"pos_{g}"] = (f["group"] == g).astype(float)
    for lg in LEAGUES:
        X[f"league_{lg}"] = (f["league"] == lg).astype(float)
    return X


def monotone_constraints(columns: list[str]) -> tuple[int, ...]:
    return tuple(MONOTONE.get(c, 0) for c in columns)


def additive_platform(columns: list[str]) -> list[list[str]]:
    """Interaction constraints: talent features and positions may interact; every other
    feature (playing share, Europe, league, club Elo, season) only adds on its own."""
    talent = [c for c in columns if c in TALENT or c.startswith("pos_")]
    return [talent] + [[c] for c in columns if c not in talent]
