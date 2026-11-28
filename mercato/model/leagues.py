"""League check: after the match difficulty adjustment, is there a league effect left?

    uv run python -m mercato.model.leagues

For every attacker or attacking midfielder who played 1,500+ league minutes in one season
and 1,500+ in the next in a different league, compare goals and assists per 90 before
and after. The typical change for players who stayed in the same league (age, luck
fading) is taken off first. A ridge fit then gives each league a factor, so thin leagues
stay near 1. Shown twice: on raw goals and assists, and after the match difficulty
adjustment. If the adjusted factors sit near 1, the adjustment already handles leagues.
"""

import numpy as np
import pandas as pd

from mercato.model.features import GROUP_LINE
from mercato.model.fee_index import season_of
from mercato.sources.tm_datasets import TOP5

MIN_MINUTES = 1500
RIDGE = 20.0
RATE_FLOOR = 0.05  # per 90, so a goalless season doesn't blow up the log


def season_table(apps: pd.DataFrame, clubs: pd.DataFrame) -> pd.DataFrame:
    """Domestic league seasons per player: minutes, raw and adjusted goals and assists."""
    leagues = set(clubs["domestic_competition_id"].dropna())
    a = apps[apps["competition_id"].isin(leagues)].copy()
    a["season"] = season_of(a["date"])
    g = a.groupby(["player_id", "season", "competition_id"]).agg(
        minutes=("minutes_played", "sum"), goals=("goals", "sum"),
        assists=("assists", "sum"), goals_adj=("goals_w", "sum"),
        assists_adj=("assists_w", "sum")).reset_index()
    n90 = g["minutes"] / 90
    g["ga90"] = (g["goals"] + g["assists"]) / n90
    g["ga90_adj"] = (g["goals_adj"] + g["assists_adj"]) / n90
    return g.rename(columns={"competition_id": "league"})


def _pairs(st: pd.DataFrame, col: str) -> pd.DataFrame:
    s = st[st["minutes"] >= MIN_MINUTES].sort_values("minutes", ascending=False)
    s = s.drop_duplicates(["player_id", "season"]).set_index(["player_id", "season"])
    nxt = s.copy()
    nxt.index = pd.MultiIndex.from_arrays([nxt.index.get_level_values(0),
                                           nxt.index.get_level_values(1) - 1])
    j = s.merge(nxt, left_index=True, right_index=True, suffixes=("", "_next"))
    j["y"] = np.log(j[f"{col}_next"] + RATE_FLOOR) - np.log(j[col] + RATE_FLOOR)
    return j.reset_index()


def factors(st: pd.DataFrame, col: str) -> pd.DataFrame:
    p = _pairs(st, col)
    stay = p[p["league"] == p["league_next"]]
    move = p[p["league"] != p["league_next"]].copy()
    move["y"] -= stay["y"].mean()
    leagues = sorted(set(move["league"]) | set(move["league_next"]))
    k = {lg: i for i, lg in enumerate(leagues)}
    X = np.zeros((len(move), len(leagues)))
    X[np.arange(len(move)), move["league_next"].map(k)] += 1
    X[np.arange(len(move)), move["league"].map(k)] -= 1
    # output falls moving into a hard league: its factor (what a goal there is worth) is
    # the negative of the change
    beta = np.linalg.solve(X.T @ X + RIDGE * np.eye(len(leagues)), X.T @ (-move["y"]))
    top5 = [k[lg] for lg in TOP5 if lg in k]
    beta -= beta[top5].mean() if top5 else 0
    moves_in = move["league_next"].value_counts()
    moves_out = move["league"].value_counts()
    return pd.DataFrame({"league": leagues, "factor": np.exp(beta).round(2),
                         "moves_in": [int(moves_in.get(lg, 0)) for lg in leagues],
                         "moves_out": [int(moves_out.get(lg, 0)) for lg in leagues]})


def attackers(st: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    from mercato.sources.tm_datasets import position_group
    grp = {pid: position_group(s, q) for pid, s, q in
           zip(players["player_id"], players["sub_position"], players["position"])}
    keep = {pid for pid, g in grp.items() if g in ("ST", "W", "AM")
            and GROUP_LINE.get(g) in ("ATT", "MID")}
    return st[st["player_id"].isin(keep)]


def league_strength(team_elo: pd.DataFrame, clubs: pd.DataFrame) -> pd.DataFrame:
    """Average current Elo of each league's clubs (the Global Football Rankings and Opta
    approach to league strength), from Mercato's own Elo."""
    from mercato.features.elo import latest
    last = latest(team_elo)
    last = last[pd.to_datetime(last["date"]) >= pd.to_datetime(last["date"]).max()
                - pd.Timedelta(days=200)]
    lg = dict(zip(clubs["club_id"], clubs["domestic_competition_id"]))
    last["league"] = last["tm_club_id"].map(lg)
    out = last.dropna(subset=["league"]).groupby("league")["elo"].agg(["mean", "max", "count"])
    return out.round(0).sort_values("mean", ascending=False).rename(
        columns={"mean": "avg_elo", "max": "best_club", "count": "clubs"})


def main():
    from mercato.model.train import load
    d = load()
    apps, _, _ = d.prepared()
    st = attackers(season_table(apps, d.clubs), d.players)
    raw = factors(st, "ga90").rename(columns={"factor": "raw"})
    adj = factors(st, "ga90_adj")[["league", "factor"]].rename(columns={"factor": "adjusted"})
    out = raw.merge(adj, on="league")[["league", "raw", "adjusted", "moves_in", "moves_out"]]
    print("What a goal or assist in each league is worth next to the top-5 average")
    print("(raw: before the match difficulty adjustment; adjusted: after it)\n")
    print(out.sort_values("raw", ascending=False).to_string(index=False))
    print("\nLeague strength, average club Elo now (compare with globalfootballrankings.com):\n")
    print(league_strength(d.team_elo, d.clubs).to_string())


if __name__ == "__main__":
    main()
