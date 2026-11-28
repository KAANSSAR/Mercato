"""dcaribou/transfermarkt-datasets loader (CC0).

Download the CSVs from Kaggle (davidcariboo/player-scores) into data/tm/.
Files used: players, clubs, transfers, player_valuations (.csv or .csv.gz).

    uv run python -m mercato.sources.tm_datasets report   # coverage, no DB
    uv run python -m mercato.sources.tm_datasets load     # write to Postgres
"""

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/tm")
TOP5 = {"GB1", "ES1", "IT1", "L1", "FR1"}
MIN_SEASON = 2015  # earliest season we train on

# Transfermarkt sub_position -> Mercato position group
POSITION_MAP = {
    "Goalkeeper": "GK",
    "Centre-Back": "CB",
    "Left-Back": "FB",
    "Right-Back": "FB",
    "Defensive Midfield": "DM",
    "Central Midfield": "CM",
    "Left Midfield": "W",
    "Right Midfield": "W",
    "Attacking Midfield": "AM",
    "Left Winger": "W",
    "Right Winger": "W",
    "Second Striker": "ST",
    "Centre-Forward": "ST",
}
# Fallback when sub_position is missing
BROAD_MAP = {"Goalkeeper": "GK", "Defender": "CB", "Midfield": "CM", "Attack": "ST"}


def read(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    for ext in (".csv", ".csv.gz"):
        p = data_dir / f"{name}{ext}"
        if p.exists():
            return pd.read_csv(p, low_memory=False)
    raise FileNotFoundError(f"{name}.csv not found in {data_dir}")


def _read_optional(name: str, data_dir: Path) -> pd.DataFrame | None:
    try:
        return read(name, data_dir)
    except FileNotFoundError:
        return None


def position_group(sub_position, position) -> str | None:
    if isinstance(sub_position, str) and sub_position in POSITION_MAP:
        return POSITION_MAP[sub_position]
    if isinstance(position, str):
        return BROAD_MAP.get(position)
    return None


def top5_clubs(clubs: pd.DataFrame) -> pd.DataFrame:
    """Every club that has played in a top-5 league (relegated ones included)."""
    return clubs[clubs["domestic_competition_id"].isin(TOP5)]


def latest_season(clubs: pd.DataFrame) -> int:
    return int(clubs["last_season"].max())


def current_top5_club_ids(clubs: pd.DataFrame) -> set[int]:
    """Clubs in a top-5 league in the latest season of the dataset (about 96)."""
    t5 = top5_clubs(clubs)
    return set(t5.loc[t5["last_season"] == latest_season(clubs), "club_id"])


def current_top5_player_mask(players: pd.DataFrame, clubs: pd.DataFrame) -> pd.Series:
    """Players registered at a current top-5 club this season (retired players excluded)."""
    return (players["current_club_id"].isin(current_top5_club_ids(clubs))
            & (players["last_season"] >= latest_season(clubs)))


def relevant_players(players: pd.DataFrame, transfers: pd.DataFrame,
                     club_ids: set[int]) -> pd.DataFrame:
    """Players we keep:
    - top-5 players since MIN_SEASON, and anyone who moved to or from a top-5 club
    - anyone active in the last two seasons in any league the dataset covers, so summer
      signings from the Eredivisie, Liga Portugal, Super Lig and so on are already loaded
    """
    current = (players["current_club_id"].isin(club_ids)
               & (players["last_season"] >= MIN_SEASON))
    t = transfers[pd.to_datetime(transfers["transfer_date"]).dt.year >= MIN_SEASON]
    movers = set(t.loc[t["from_club_id"].isin(club_ids) | t["to_club_id"].isin(club_ids),
                       "player_id"])
    active = players["last_season"] >= players["last_season"].max() - 1
    return players[current | active | players["player_id"].isin(movers)].copy()


def paid_moves(transfers: pd.DataFrame, club_ids: set[int]) -> pd.DataFrame:
    """Training-grade moves: real fee > 0, buyer in a top-5 league, since MIN_SEASON."""
    t = transfers.copy()
    t["transfer_date"] = pd.to_datetime(t["transfer_date"])
    fee = pd.to_numeric(t["transfer_fee"], errors="coerce")
    return t[(fee > 0) & t["to_club_id"].isin(club_ids)
             & (t["transfer_date"].dt.year >= MIN_SEASON)]


def coverage_report(data_dir: Path = DATA_DIR) -> dict:
    players, clubs, transfers = (read(n, data_dir) for n in ("players", "clubs", "transfers"))
    club_ids = set(top5_clubs(clubs)["club_id"])
    rel = relevant_players(players, transfers, club_ids)
    rel["group"] = [position_group(s, p) for s, p in zip(rel["sub_position"], rel["position"])]
    current = rel[current_top5_player_mask(rel, clubs)]
    moves = paid_moves(transfers, club_ids)
    return {
        "latest_season": latest_season(clubs),
        "top5_clubs_ever": len(club_ids),
        "current_top5_clubs": len(current_top5_club_ids(clubs)),
        "relevant_players": len(rel),
        "current_top5_players": len(current),
        "position_group_coverage": round(rel["group"].notna().mean(), 3),
        "contract_coverage_current": round(current["contract_expiration_date"].notna().mean(), 3),
        "dob_coverage": round(rel["date_of_birth"].notna().mean(), 3),
        "paid_moves_into_top5": len(moves),
        "paid_moves_by_year": moves["transfer_date"].dt.year.value_counts().sort_index().to_dict(),
        "groups": rel["group"].value_counts().to_dict(),
    }


def load(data_dir: Path = DATA_DIR) -> dict:
    from mercato.db import connect, job_run, upsert

    players, clubs, transfers, vals = (
        read(n, data_dir) for n in ("players", "clubs", "transfers", "player_valuations")
    )
    t5 = top5_clubs(clubs)
    club_ids = set(t5["club_id"])
    rel = relevant_players(players, transfers, club_ids)

    with connect() as conn, job_run(conn, "tm_datasets_load") as stats:
        # Clubs: every club in any game (European opponents included), any club a relevant
        # player moved between, and clubs.csv last so its names win
        tr = transfers[transfers["player_id"].isin(rel["player_id"])]
        names = {}
        games = _read_optional("games", data_dir)
        if games is not None and {"home_club_name", "away_club_name"} <= set(games.columns):
            names.update(dict(zip(games["home_club_id"], games["home_club_name"])))
            names.update(dict(zip(games["away_club_id"], games["away_club_name"])))
        names.update(dict(zip(tr["from_club_id"], tr["from_club_name"])))
        names.update(dict(zip(tr["to_club_id"], tr["to_club_name"])))
        names.update(dict(zip(clubs["club_id"], clubs["name"])))
        comp = dict(zip(clubs["club_id"], clubs["domestic_competition_id"]))
        cur_clubs = current_top5_club_ids(clubs)
        club_rows = [
            {"tm_club_id": int(cid), "name": str(n), "tm_competition": _s(comp.get(cid)),
             "current_top5": cid in cur_clubs}
            for cid, n in names.items() if pd.notna(cid) and pd.notna(n)
        ]
        stats["rows_written"] += upsert(conn, "clubs", club_rows, ["tm_club_id"],
                                        update=["name", "tm_competition", "current_top5"])

        nations = sorted(rel["country_of_citizenship"].dropna().unique())
        stats["rows_written"] += upsert(conn, "nations", [{"name": n} for n in nations], ["name"])

        with conn.cursor() as cur:
            cur.execute("select tm_club_id, id from clubs where tm_club_id is not null")
            club_map = dict(cur.fetchall())
            cur.execute("select name, id from nations")
            nation_map = dict(cur.fetchall())
            cur.execute("select tm_player_id, mercato_id from id_map where tm_player_id is not null")
            pid_map = dict(cur.fetchall())

        # Players: reserve ids for new ones in one call, then upsert in bulk
        new = [int(p) for p in rel["player_id"] if int(p) not in pid_map]
        if new:
            with conn.cursor() as cur:
                cur.execute("select nextval('players_mercato_id_seq') "
                            "from generate_series(1, %s)", (len(new),))
                pid_map.update(zip(new, (row[0] for row in cur.fetchall())))
        is_current = dict(zip(rel["player_id"], current_top5_player_mask(rel, clubs)))
        p_rows = [
            {
                "mercato_id": pid_map[int(r.player_id)],
                "name": str(r.name),
                "birth_date": _d(r.date_of_birth),
                "nation_id": nation_map.get(r.country_of_citizenship),
                "club_id": club_map.get(r.current_club_id),
                "position_group": position_group(r.sub_position, r.position),
                "sub_position": _s(r.sub_position),
                "contract_until": _d(r.contract_expiration_date),
                "foot": _s(r.foot),
                "height_cm": _i(r.height_in_cm),
                "current_top5": bool(is_current[r.player_id]),
            }
            for r in rel.itertuples(index=False)
        ]
        stats["rows_written"] += upsert(conn, "players", p_rows, ["mercato_id"],
                                        update=[c for c in p_rows[0] if c != "mercato_id"]
                                        if p_rows else None)
        m_rows = [{"mercato_id": pid_map[p], "tm_player_id": p, "match_method": "source"}
                  for p in new]
        upsert(conn, "id_map", m_rows, ["mercato_id"])

        # Transfers (all, fee null when free or undisclosed; training filters later)
        t_rows = []
        for r in tr.itertuples(index=False):
            fee = pd.to_numeric(r.transfer_fee, errors="coerce")
            t_rows.append({
                "mercato_id": pid_map[int(r.player_id)],
                "transfer_date": _d(r.transfer_date),
                "from_club_id": club_map.get(r.from_club_id),
                "to_club_id": club_map.get(r.to_club_id),
                "fee_eur": float(fee) if pd.notna(fee) and fee > 0 else None,
                "source": "tm_datasets",
                "source_key": f"tm:{r.player_id}:{r.transfer_date}:{r.from_club_id}:{r.to_club_id}",
            })
        stats["rows_written"] += upsert(conn, "transfers", t_rows, ["source_key"],
                                        update=["fee_eur"])

        # Market value history (benchmark only, never a training target)
        v = vals[vals["player_id"].isin(pid_map) & (pd.to_datetime(vals["date"]).dt.year
                                                     >= MIN_SEASON)]
        v_rows = [
            {"mercato_id": pid_map[int(r.player_id)], "as_of": _d(r.date),
             "value_eur": float(r.market_value_in_eur), "source": "dataset"}
            for r in v.itertuples(index=False) if pd.notna(r.market_value_in_eur)
        ]
        stats["rows_written"] += upsert(conn, "tm_values", v_rows,
                                        ["mercato_id", "as_of", "source"], update=["value_eur"])
        return dict(stats)


def _s(x):
    return x if isinstance(x, str) else None


def _i(x):
    return int(x) if pd.notna(x) else None


def _d(x):
    return pd.to_datetime(x).date() if pd.notna(x) else None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "report":
        for k, val in coverage_report().items():
            print(f"{k}: {val}")
    elif cmd == "load":
        print(load())
    else:
        sys.exit("usage: tm_datasets [report|load]")
