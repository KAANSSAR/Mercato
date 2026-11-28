"""Train several model variants on the same features and compare them side by side.

    uv run python -m mercato.model.compare

Prints accuracy, range calibration and how many current players hit the cap, then the
P50 for a fixed set of reference players under each variant. Writes nothing.
"""

import pandas as pd

from mercato.model.features import NUMERIC_EURO
from mercato.model.train import PARAMS, current_features, fit, load, prepare, value_current

REFERENCE = ["Morgan Rogers", "Cole Palmer", "Florian Wirtz", "Bukayo Saka", "Erling Haaland",
             "Lamine Yamal", "Kylian Mbappé", "Hugo Ekitiké", "Ryan Gravenberch", "Declan Rice"]

VARIANTS = {
    "A v1.4": {},
    "B euro+depth2": {"numeric": NUMERIC_EURO, "params": PARAMS | {"max_depth": 2}},
    "C B+additive": {"numeric": NUMERIC_EURO, "params": PARAMS | {"max_depth": 2},
                     "additive": True},
    "D B+leaf50": {"numeric": NUMERIC_EURO,
                   "params": PARAMS | {"max_depth": 2, "min_child_weight": 50}},
}


def unique_names(v: pd.DataFrame) -> pd.DataFrame:
    """Names repeat across players (two Rodris, several Danilos): keep the most valuable."""
    return (v.sort_values("tm_value", ascending=False, na_position="last")
             .drop_duplicates("name").set_index("name"))


def main():
    d = load()
    prep = prepare(d)
    names = dict(zip(d.players["player_id"], d.players["name"]))
    feats = None
    summary, refs = {}, {}
    for name, variant in VARIANTS.items():
        print(f"training {name} ...", flush=True)
        r = fit(prep, verbose=False, **variant)
        if feats is None:
            feats = current_features(d, r)
        v = value_current(d, r, feats.copy())
        v["name"] = v["player_id"].map(names)
        regulars = v[(v["minutes_best"] >= 1500) & (v["tm_value"] >= 5e6)]
        m = r["metrics"]
        summary[name] = {
            "mdape": f"{m['test_mdape']:.1%}", "coverage": f"{m['test_coverage_p10_p90']:.0%}",
            "band": f"x{m['band_multipliers'][0]} to x{m['band_multipliers'][1]}",
            "at_cap": int(v["capped"].sum()),
            "median_gap_vs_tm": f"{(regulars['p50'] / regulars['tm_value']).median() - 1:+.0%}",
        }
        by_name = unique_names(v)
        refs[name] = by_name.reindex(REFERENCE)["p50"].map(
            lambda x: f"€{x / 1e6:.0f}m" if pd.notna(x) else "-")
    refs["TM value"] = by_name.reindex(REFERENCE)["tm_value"].map(
        lambda x: f"€{x / 1e6:.0f}m" if pd.notna(x) else "-")
    pd.set_option("display.width", 200)
    print(f"\ntest season {prep['test_season']}, naive median error "
          f"{r['metrics']['naive_median_mdape']:.1%}, Transfermarkt on the same moves "
          f"{r['metrics']['tm_value_mdape_same_moves']:.1%}\n")
    print(pd.DataFrame(summary).to_string())
    print()
    print(pd.DataFrame(refs).to_string())


if __name__ == "__main__":
    main()
