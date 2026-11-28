"""Compare model versions side by side on the same data.

    uv run python -m mercato.model.compare              # v1.5 against the v2 variants
    uv run python -m mercato.model.compare --backtest   # plus the Transfermarkt backtest

For each: rank agreement with Transfermarkt, the typical gap, the share of regulars more
than 50% away, then the value for a fixed set of reference players. Writes nothing.
"""

import argparse
import json

import pandas as pd

from mercato.model import value as v2
from mercato.model.rating import WEIGHTS
from mercato.model.train import current_features, current_rows, fit, load, value_current
from mercato.model.train import prepare as prepare_v1

REFERENCE = ["Morgan Rogers", "Cole Palmer", "Florian Wirtz", "Bukayo Saka", "Erling Haaland",
             "Lamine Yamal", "Kylian Mbappé", "Hugo Ekitiké", "Ryan Gravenberch", "Declan Rice",
             "David Raya", "Joan García", "Harry Kane", "Viktor Gyökeres", "Wouter Burger",
             "Juan Iglesias", "Virgil van Dijk", "Thibaut Courtois", "Bruno Fernandes",
             "Raphinha"]


def without(component: str) -> dict:
    """The weights with one component removed and the rest scaled back up to one."""
    out = {}
    for ln, w in WEIGHTS.items():
        rest = {c: x for c, x in w.items() if c != component}
        total = sum(rest.values())
        out[ln] = {**{c: x / total for c, x in rest.items()}, component: 0.0}
    return out


V2_VARIANTS = {
    "v2.8": {},
    "v2.8 fee prefs": {"prefs": "fees"},
}


def unique_names(v: pd.DataFrame) -> pd.DataFrame:
    """Names repeat across players (two Rodris, several Danilos): keep the most valuable."""
    return (v.sort_values("tm_value", ascending=False, na_position="last")
             .drop_duplicates("name").set_index("name"))


def eur(x) -> str:
    return f"€{x / 1e6:.0f}m" if pd.notna(x) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    args = ap.parse_args()
    d = load()
    names = dict(zip(d.players["player_id"], d.players["name"]))
    summary, refs, ratings = {}, {}, None

    print("training v1.5 ...", flush=True)
    r1 = fit(prepare_v1(d), verbose=False)
    values = {"v1.5": value_current(d, r1, current_features(d, r1))}
    prep = v2.prepare(d, current_rows(d))
    print("league factors:", prep["league_factors"])
    for name, kw in V2_VARIANTS.items():
        print(f"training {name} ...", flush=True)
        r = v2.fit(prep, verbose=False, **kw)
        values[name] = r["values"]
        if kw.get("method", "projection") == "projection":
            print("  ", {k: r["metrics"][k] for k in ("beta", "delta", "repl", "prefs_rank_corr",
                                                      "fee_rank_corr_test", "fee_mdape_test")})

    for name, v in values.items():
        v = v.assign(name=v["player_id"].map(names))
        c = v2.check(v)
        summary[name] = {"rank agreement": c["rank_agreement"], "median gap": c["median_gap"],
                         "beyond ±50%": c["beyond_50pct"],
                         **{f"{ln} rank / gap": f"{x['rank_agreement']} / {x['median_gap']}"
                            for ln, x in c["by_line"].items()}}
        by_name = unique_names(v).reindex(REFERENCE)
        refs[name] = by_name["p50"].map(eur)
        if name == "v2.8":
            ratings = by_name[["rating", "age"]].round(1)
    refs["TM value"] = by_name["tm_value"].map(eur)
    refs["rating"], refs["age"] = ratings["rating"], ratings["age"]
    pd.set_option("display.width", 200)
    print()
    print(pd.DataFrame(summary).to_string())
    print()
    print(pd.DataFrame(refs).to_string())

    if args.backtest:
        rows, _, t0 = v2.backtest_rows(d)
        print("\npreparing backtest ...", flush=True)
        bprep = v2.prepare(d, rows, t0)
        for name, kw in V2_VARIANTS.items():
            print(f"\nbacktest {name}:")
            print(json.dumps(v2.backtest(d, names=names, prep=bprep, **kw), indent=2,
                             ensure_ascii=False))


if __name__ == "__main__":
    main()
