"""Compute Mercato Elo from data/tm/games.csv and write it to team_elo.

    uv run python -m mercato.jobs.elo           # compute, print sanity checks, write
    uv run python -m mercato.jobs.elo --dry     # compute and print only
"""

import sys

import pandas as pd

from mercato.features.elo import EloParams, brier, latest, run
from mercato.sources.tm_datasets import DATA_DIR, read


def main(dry: bool = False) -> dict:
    games = read("games", DATA_DIR)
    clubs = read("clubs", DATA_DIR)
    p = EloParams()
    rated = run(games, p)

    names = dict(zip(clubs["club_id"], clubs["name"]))
    top = latest(rated).sort_values("elo", ascending=False).head(15)
    print(f"games rated: {len(rated) // 2:,}   brier since 2014/15: {brier(rated):.4f}")
    print("top 15 now:")
    for r in top.itertuples(index=False):
        print(f"  {r.elo:7.1f}  {names.get(r.tm_club_id, r.tm_club_id)}  (last game {r.date})")

    if dry:
        return {"rows": len(rated)}

    from mercato.db import connect, job_run, upsert

    rated = rated.drop(columns=["goals_for", "goals_against"])  # in games already
    rows = rated.astype(object).where(pd.notna(rated), None).to_dict("records")
    with connect() as conn, job_run(conn, "elo") as stats:
        stats["rows_written"] = upsert(conn, "team_elo", rows, ["game_id", "tm_club_id"],
                                       update=["elo_pre", "opp_elo_pre", "elo_post",
                                               "expected", "result"])
        return dict(stats)


if __name__ == "__main__":
    print(main(dry="--dry" in sys.argv))
