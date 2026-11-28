"""Why a player is valued where he is: raw stats, each component against his position,
the projected path and the value.

    uv run python -m mercato.model.explain "Wouter Burger" "Declan Rice"

Fits the current model first (a few minutes). Writes nothing.
"""

import sys

import numpy as np
import pandas as pd

from mercato.model import value as v2
from mercato.model.leagues import season_table
from mercato.model.projection import project
from mercato.model.rating import COMPONENTS, rate
from mercato.model.train import current_rows, load

RAW = ["g90", "a90", "share_best", "share_1y", "minutes_best", "games_w", "impact",
       "defence", "opposition", "team_level"]


def main(names: list[str]):
    d = load()
    by_name = dict(zip(d.players["name"], d.players["player_id"]))
    prep = v2.prepare(d, current_rows(d))
    result = v2.fit(prep, verbose=False)
    pool, _ = rate(prep["pool"], result["ref"])
    values = result["values"].set_index("player_id")
    paths = project(pool, result["paths"])
    seasons = season_table(d.prepared()[0], d.clubs)
    for name in names:
        pid = by_name.get(name)
        idx = pool.index[pool["player_id"] == pid]
        if pid is None or idx.empty:
            print(f"\n{name}: not found among current top-5 players")
            continue
        i = idx[0]
        r, v = pool.loc[i], values.loc[pid]
        ref = result["ref"]["components"]
        print(f"\n{name}  ({r['group']}, age {r['age']:.1f})")
        print(f"  value €{v['p50'] / 1e6:.1f}m  (range €{v['p10'] / 1e6:.1f}m to "
              f"€{v['p90'] / 1e6:.1f}m), Transfermarkt €{(r['tm_value'] or 0) / 1e6:.1f}m")
        print(f"  rating {r['rating']:.1f}, score {r['score']:+.2f}")
        print("  raw: " + ", ".join(f"{c} {r[c]:.3g}" for c in RAW))
        w = result["weights"][v["line"]]
        print("  component   value   typical   spread      z   weight  adds")
        for c in COMPONENTS:
            med, mad = ref[c].get(r["group"], (np.nan, np.nan))
            z = r[f"z_{c}"]
            print(f"  {c:<11} {r[c]:>7.3g} {med:>9.3g} {mad:>8.3g} {z:>+6.2f} {w[c]:>8.2f} "
                  f"{w[c] * z:>+5.2f}")
        path = paths[pool.index.get_loc(i)]
        print("  projected score by season: "
              + ", ".join("-" if np.isnan(x) else f"{x:+.2f}" for x in path[:10]))
        print(f"  drivers: {v['drivers']}")
        st = seasons[seasons["player_id"] == pid].sort_values("season").tail(6)
        if len(st):
            print("  league seasons: season league minutes goals assists  per90  adjusted")
            for x in st.itertuples():
                print(f"    {x.season}/{(x.season + 1) % 100:02d} {x.league:<5} {x.minutes:>7.0f} "
                      f"{x.goals:>5.0f} {x.assists:>7.0f} {x.ga90:>6.2f} {x.ga90_adj:>9.2f}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main(sys.argv[1:] or ["Wouter Burger"])
