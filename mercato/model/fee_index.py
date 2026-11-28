"""Football's own inflation: a yearly price level for comparable top-5 moves.

    index[year] = median fee of paid moves into a top-5 club that season (fee above €1m)
    fee_today   = fee * index[now] / index[year]

General inflation is a footnote next to this: Neymar's €222m move in 2017 reset the
market, and COVID dented it in 2020 and 2021.
"""

import numpy as np
import pandas as pd

MIN_FEE = 1_000_000


def season_of(date) -> pd.Series | int:
    """Transfer windows: July to June. A move on 2019-08-01 or 2020-01-15 is season 2019."""
    d = pd.to_datetime(date)
    if isinstance(d, pd.Timestamp):
        return d.year if d.month >= 7 else d.year - 1
    return np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1)


def build_index(moves: pd.DataFrame, min_count: int = 30) -> pd.Series:
    """moves: columns season, fee. Returns index by season, smoothed where thin.
    Seasons with fewer than min_count qualifying moves borrow the previous season's level."""
    m = moves[moves["fee"] >= MIN_FEE]
    by = m.groupby("season")["fee"].agg(["median", "count"]).sort_index()
    idx = by["median"].copy()
    for s in by.index:
        if by.loc[s, "count"] < min_count and (s - 1) in idx.index:
            idx.loc[s] = idx.loc[s - 1]
    return idx


def deflate(fee: pd.Series, season: pd.Series, index: pd.Series, to_season: int) -> pd.Series:
    """Express fees at to_season's price level."""
    level = season.map(index)
    return fee * index.loc[to_season] / level
