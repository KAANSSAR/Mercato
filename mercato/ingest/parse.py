"""Pure parsing of API-Football responses into flat rows. No database, no network."""

import datetime as dt

FINISHED = {"FT", "AET", "PEN"}


def _num(v):
    """'55%' -> 55.0, '1.23' -> 1.23, None -> None, 7 -> 7."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return v
    s = str(v).strip().rstrip("%")
    try:
        f = float(s)
    except ValueError:
        return None
    return int(f) if f.is_integer() and "." not in s else f


def parse_fixture(f: dict) -> dict:
    fx, lg, teams, goals = f["fixture"], f["league"], f["teams"], f.get("goals") or {}
    return {
        "api_football_id": fx["id"],
        "kickoff": dt.datetime.fromisoformat(fx["date"]),
        "status": fx["status"]["short"],
        "league_id": lg["id"],
        "season": lg["season"],
        "round": lg.get("round"),
        "home": {"id": teams["home"]["id"], "name": teams["home"]["name"]},
        "away": {"id": teams["away"]["id"], "name": teams["away"]["name"]},
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
    }


def flatten_player_stats(s: dict) -> dict:
    """statistics[0] block -> {'shots_total': 3, 'passes_key': 2, ...} (games block excluded)."""
    out = {}
    for group, vals in s.items():
        if group == "games":
            continue
        if isinstance(vals, dict):
            for k, v in vals.items():
                out[f"{group}_{k}"] = _num(v)
        else:
            out[group] = _num(vals)
    return out


def parse_players(resp: list[dict]) -> list[dict]:
    """fixtures/players -> one row per player who played (minutes > 0)."""
    rows = []
    for team in resp:
        for p in team.get("players", []):
            stats = (p.get("statistics") or [{}])[0]
            games = stats.get("games") or {}
            minutes = _num(games.get("minutes"))
            if not minutes:
                continue  # unused substitute
            rows.append({
                "team_api_id": team["team"]["id"],
                "team_name": team["team"]["name"],
                "api_player_id": p["player"]["id"],
                "name": p["player"]["name"],
                "minutes": int(minutes),
                "position": games.get("position"),
                "rating": _num(games.get("rating")),
                "is_sub": bool(games.get("substitute")),
                "stats": flatten_player_stats(stats),
            })
    return rows


def parse_team_stats(resp: list[dict]) -> list[dict]:
    """fixtures/statistics -> one row per team with a type->value dict."""
    rows = []
    for team in resp:
        stats = {}
        for item in team.get("statistics", []):
            key = item["type"].lower().replace(" ", "_").replace("%", "pct")
            stats[key] = _num(item.get("value"))
        rows.append({
            "team_api_id": team["team"]["id"],
            "possession": stats.get("ball_possession"),
            "xg": stats.get("expected_goals"),
            "stats": stats,
        })
    return rows


def league_table(results: list[dict]) -> list[dict]:
    """Results [{home, away, home_goals, away_goals}] -> table sorted by points, goal
    difference, goals scored. Head-to-head tie-breaks (Spain, Italy) are not applied."""
    t: dict = {}
    for r in results:
        for side, gf, ga in ((r["home"], r["home_goals"], r["away_goals"]),
                             (r["away"], r["away_goals"], r["home_goals"])):
            row = t.setdefault(side, {"club": side, "played": 0, "won": 0, "drawn": 0,
                                      "lost": 0, "goals_for": 0, "goals_against": 0})
            row["played"] += 1
            row["goals_for"] += gf
            row["goals_against"] += ga
            row["won" if gf > ga else "drawn" if gf == ga else "lost"] += 1
    rows = list(t.values())
    for r in rows:
        r["points"] = 3 * r["won"] + r["drawn"]
        r["goal_diff"] = r["goals_for"] - r["goals_against"]
    rows.sort(key=lambda r: (-r["points"], -r["goal_diff"], -r["goals_for"], str(r["club"])))
    for i, r in enumerate(rows, 1):
        r["position"] = i
    return rows
