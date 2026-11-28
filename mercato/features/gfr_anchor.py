"""Anchor Mercato's Elo to Global Football Rankings' league levels.

GFR rates every club and league on one scale, with country and continent layers on top
of club Elo. Our Elo is linked across countries only through European games, so leagues
that rarely meet drift (v2.5: Denmark third, the Premier League fifth). GFR only publishes
today's numbers, so:

1. Scale: within each league, how many of our Elo points one GFR point is worth (slope of
   our Elo on theirs after taking out each league's average, on matched clubs).
2. League levels: each league's average is moved to where GFR puts it, keeping the top-5
   average where it was. The same shift applies to every game in that league's history,
   by the league each club was playing in at the time (promotion and relegation
   included). Order within a league stays our Elo, which has match-by-match history.

Fed by mercato.sources.gfr, which keeps data/gfr/leagues.csv and clubs.csv current.
"""

from pathlib import Path

import numpy as np
import pandas as pd

GFR_DIR = Path("data/gfr")
TOP5 = ["GB1", "ES1", "IT1", "L1", "FR1"]
MIN_CLUBS = 30


def club_league(team_elo: pd.DataFrame, leagues: set) -> pd.DataFrame:
    """The league each club was in on each game date: its latest league game so far
    (or its first one, for games before any league game)."""
    t = team_elo[["game_id", "date", "tm_club_id"]].copy()
    t["date"] = pd.to_datetime(t["date"]).astype("datetime64[ns]")
    lg = team_elo.loc[team_elo["competition_id"].isin(leagues),
                      ["tm_club_id", "date", "competition_id"]].copy()
    lg["date"] = pd.to_datetime(lg["date"]).astype("datetime64[ns]")
    lg = lg.rename(columns={"competition_id": "league"}).sort_values("date")
    t = t.reset_index().sort_values("date")
    back = pd.merge_asof(t, lg, on="date", by="tm_club_id", direction="backward")
    fwd = pd.merge_asof(t, lg, on="date", by="tm_club_id", direction="forward")
    back["league"] = back["league"].fillna(fwd["league"])
    return back.set_index("index").sort_index()["league"]


def current_league_avg(team_elo: pd.DataFrame, leagues: set) -> pd.Series:
    t = team_elo[team_elo["competition_id"].isin(leagues)].copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t[t["date"] >= t["date"].max() - pd.Timedelta(days=200)]
    last = t.sort_values(["date", "game_id"]).groupby("tm_club_id").tail(1)
    return last.groupby("competition_id")["elo_post"].mean()


def fit_offsets(team_elo: pd.DataFrame, gfr_clubs: pd.DataFrame,
                gfr_leagues: pd.DataFrame, leagues: set) -> dict:
    """League offsets in Elo points, and the scale used."""
    t = team_elo[team_elo["competition_id"].isin(leagues)].copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t[t["date"] >= t["date"].max() - pd.Timedelta(days=200)]
    last = t.sort_values(["date", "game_id"]).groupby("tm_club_id").tail(1)
    m = last.merge(gfr_clubs.dropna(subset=["tm_club_id"]).astype({"tm_club_id": int}),
                   on="tm_club_id")
    if len(m) < MIN_CLUBS:
        return {"offsets": {}, "scale": None, "clubs": len(m)}
    x = m["rating"] - m.groupby("competition_id")["rating"].transform("mean")
    y = m["elo_post"] - m.groupby("competition_id")["elo_post"].transform("mean")
    scale = float((x * y).sum() / (x * x).sum())
    ours = current_league_avg(team_elo, leagues)
    theirs = gfr_leagues.dropna(subset=["tm_competition_id"]).set_index(
        "tm_competition_id")["rating"]
    both = [lg for lg in ours.index if lg in theirs.index]
    raw = {lg: scale * theirs[lg] - ours[lg] for lg in both}
    top = [raw[lg] for lg in TOP5 if lg in raw]
    c = float(np.mean(top)) if top else 0.0
    return {"offsets": {lg: round(v - c, 1) for lg, v in raw.items()},
            "scale": round(scale, 2), "clubs": len(m)}


def anchor(team_elo: pd.DataFrame, offsets: dict, leagues: set) -> pd.DataFrame:
    """Shift every rating by its club's league offset at the time."""
    if not offsets:
        return team_elo
    t = team_elo.copy()
    lg = club_league(t, leagues)
    own = lg.map(offsets).fillna(0.0).to_numpy()
    key = dict(zip(zip(t["game_id"], t["tm_club_id"]), own))
    opp = np.array([key.get((g, o), 0.0) for g, o in zip(t["game_id"],
                                                          t["opponent_tm_club_id"])])
    t["elo_pre"] = t["elo_pre"] + own
    t["elo_post"] = t["elo_post"] + own
    t["opp_elo_pre"] = t["opp_elo_pre"] + opp
    return t


def load_gfr(data_dir: Path = GFR_DIR):
    try:
        return (pd.read_csv(data_dir / "leagues.csv"), pd.read_csv(data_dir / "clubs.csv"))
    except FileNotFoundError:
        return None, None


def anchored(team_elo: pd.DataFrame, clubs: pd.DataFrame, verbose: bool = False,
             data_dir: Path = GFR_DIR) -> tuple[pd.DataFrame, dict]:
    gl, gc = load_gfr(data_dir)
    if gl is None:
        if verbose:
            print("(no data/gfr snapshot: Elo not anchored; run mercato.sources.gfr)")
        return team_elo, {}
    leagues = set(clubs["domestic_competition_id"].dropna())
    fit = fit_offsets(team_elo, gc, gl, leagues)
    if verbose:
        print(f"GFR anchor: {fit['clubs']} clubs, 1 GFR point = {fit['scale']} Elo, "
              f"league shifts {fit['offsets']}")
    return anchor(team_elo, fit["offsets"], leagues), fit
