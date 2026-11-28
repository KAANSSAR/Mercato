"""Train several model variants on the same features and compare them side by side.

    uv run python -m mercato.model.compare

Prints accuracy, range calibration and how many current players hit the cap, then the
P50 for a fixed set of reference players under each variant. Writes nothing.
"""

import pandas as pd

from mercato.model.features import MARKET, NUMERIC, NUMERIC_EURO, PCTS
from mercato.model.train import PARAMS, current_features, fit, load, prepare, value_current

REFERENCE = ["Morgan Rogers", "Cole Palmer", "Florian Wirtz", "Bukayo Saka", "Erling Haaland",
             "Lamine Yamal", "Kylian Mbappé", "Hugo Ekitiké", "Ryan Gravenberch", "Declan Rice"]

VARIANTS = {
    "v1.5": {"numeric": NUMERIC_EURO},
    "+position pct": {"numeric": NUMERIC_EURO + PCTS},
    "+buyer/seller": {"numeric": NUMERIC_EURO + MARKET},
    "+both": {"numeric": NUMERIC_EURO + PCTS + MARKET},
}
OLD_VARIANTS = {  # structure comparison from v1.5, kept for reference
    "A v1.4": {"numeric": NUMERIC, "params": PARAMS | {"max_depth": 3}, "additive": False},
    "B euro+depth2": {"numeric": NUMERIC_EURO, "params": PARAMS, "additive": False},
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
            "mdape (buyer hidden)": f"{m['test_mdape']:.1%}",
            "mdape (buyer known)": f"{m['test_mdape_known_buyer']:.1%}",
            "coverage": f"{m['test_coverage_p10_p90']:.0%}",
            **{f"mdape {ln}": f"{v:.1%}" for ln, v in m["mdape_by_line"].items()},
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
