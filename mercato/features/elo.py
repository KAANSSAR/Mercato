"""Mercato's own club Elo, computed from every Transfermarkt match result.

Replaces ClubElo, whose public CSV feed closed in late 2026. Same family of method:

    expected = 1 / (1 + 10 ^ (-(R_home + HFA - R_away) / 400))
    change   = K * weight * margin * (result - expected)

- HFA: home advantage in Elo points.
- margin: bigger wins move ratings more (1 for a one-goal game, 1.5 for two, (11 + gd) / 8 above).
- weight: league and European games count fully, domestic cups less.
- New clubs in a league start at a low percentile of that league, so promoted sides
  don't arrive rated as mid-table. Clubs returning after a long gap (relegation) get the
  same treatment if they've fallen below it.
- Ratings are zero-sum per game.

Parameters are tunable; brier() scores predictions so K and HFA can be fitted.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED = ["game_id", "date", "competition_id", "competition_type",
            "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"]


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    hfa: float = 65.0
    init: float = 1500.0          # first-ever club when a league has no history yet
    cup_init: float = 1400.0      # clubs only ever seen in cups (leagues outside the dataset)
    cup_weight: float = 0.75      # domestic cups
    new_club_pct: float = 0.2     # newcomers start at this percentile of their league
    gap_days: int = 300           # absence long enough to count as relegated
    active_days: int = 400        # clubs seen within this window form the league pool


def expected(diff: float) -> float:
    """Expected score for the side ahead by `diff` Elo points."""
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def margin(gd: int) -> float:
    gd = abs(int(gd))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def prepare(games: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in games.columns]
    if missing:
        raise ValueError(f"games is missing columns: {missing}")
    g = games.dropna(subset=["home_club_id", "away_club_id",
                             "home_club_goals", "away_club_goals", "date"]).copy()
    g["date"] = pd.to_datetime(g["date"])
    for c in ["home_club_id", "away_club_id", "home_club_goals", "away_club_goals"]:
        g[c] = g[c].astype(int)
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


def run(games: pd.DataFrame, p: EloParams = EloParams()) -> pd.DataFrame:
    """Rate every game in date order. Returns one row per club per game."""
    g = prepare(games)
    rating: dict[int, float] = {}
    last_seen: dict[int, pd.Timestamp] = {}
    league_of: dict[int, str] = {}
    out = []

    def league_seed(comp: str, date: pd.Timestamp) -> float | None:
        pool = [rating[c] for c, lg in league_of.items()
                if lg == comp and (date - last_seen[c]).days <= p.active_days]
        return float(np.quantile(pool, p.new_club_pct)) if len(pool) >= 5 else None

    for r in g.itertuples(index=False):
        is_league = r.competition_type == "domestic_league"
        for club in (r.home_club_id, r.away_club_id):
            if club not in rating:
                seed = league_seed(r.competition_id, r.date) if is_league else None
                rating[club] = seed if seed is not None else (p.init if is_league else p.cup_init)
            elif (is_league and (r.date - last_seen[club]).days > p.gap_days
                  and league_of.get(club) != r.competition_id):
                seed = league_seed(r.competition_id, r.date)  # back from a lower division
                if seed is not None:
                    rating[club] = min(rating[club], seed)
            last_seen[club] = r.date
            if is_league:
                league_of[club] = r.competition_id

        h, a = rating[r.home_club_id], rating[r.away_club_id]
        e = expected(h + p.hfa - a)
        gd = r.home_club_goals - r.away_club_goals
        s = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        w = p.cup_weight if r.competition_type == "domestic_cup" else 1.0
        delta = p.k * w * margin(gd) * (s - e)
        rating[r.home_club_id] = h + delta
        rating[r.away_club_id] = a - delta

        base = {"game_id": int(r.game_id), "date": r.date.date(),
                "competition_id": r.competition_id}
        out.append(base | {"tm_club_id": r.home_club_id, "opponent_tm_club_id": r.away_club_id,
                           "is_home": True, "elo_pre": h, "opp_elo_pre": a,
                           "elo_post": h + delta, "expected": e, "result": s})
        out.append(base | {"tm_club_id": r.away_club_id, "opponent_tm_club_id": r.home_club_id,
                           "is_home": False, "elo_pre": a, "opp_elo_pre": h,
                           "elo_post": a - delta, "expected": 1 - e, "result": 1 - s})
    return pd.DataFrame(out)


def brier(rated: pd.DataFrame, since: str = "2014-07-01") -> float:
    """Mean squared error of the home side's expected score. Lower is better."""
    home = rated[rated["is_home"] & (pd.to_datetime(rated["date"]) >= since)]
    return float(((home["result"] - home["expected"]) ** 2).mean())


def latest(rated: pd.DataFrame) -> pd.DataFrame:
    """Most recent rating per club."""
    last = rated.sort_values(["date", "game_id"]).groupby("tm_club_id").tail(1)
    return last[["tm_club_id", "date", "elo_post"]].rename(columns={"elo_post": "elo"})
