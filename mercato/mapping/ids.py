"""Map API-Football club and player ids onto Mercato's Transfermarkt-based records.

    uv run python -m mercato.mapping.ids clubs     # ~20 requests
    uv run python -m mercato.mapping.ids players   # 1 request per club, resumable

Every API response is cached under data/api_cache, so a run that stops at the daily
limit picks up where it left off tomorrow. Uncertain matches go to data/review/*.csv.
Confirmed fixes go in db/overrides/*.csv (committed), which always win.
"""

import csv
import datetime as dt
import sys
from pathlib import Path

from rapidfuzz import fuzz, process

from mercato.config import COMPETITIONS, FREE_SEASONS
from mercato.mapping.match import Match, assign
from mercato.mapping.names import (
    CLUB_NOISE,
    club_similarity,
    player_key,
    player_similarity,
    position_compatible,
    tokens,
)
from mercato.sources.api_football import ApiFootball, BudgetExceeded

LEAGUE_TM = {"premier_league": "GB1", "laliga": "ES1", "serie_a": "IT1",
             "bundesliga": "L1", "ligue_1": "FR1"}
OVERRIDES = Path("db/overrides")
REVIEW_DIR = Path("data/review")


def read_overrides(name: str) -> dict[int, int]:
    f = OVERRIDES / f"{name}.csv"
    if not f.exists():
        return {}
    with f.open() as fh:
        return {int(r["tm_id"]): int(r["api_football_id"]) for r in csv.DictReader(fh)
                if r.get("tm_id") and r.get("api_football_id")}


def write_review(name: str, matches: list[Match], fields: list[str]) -> Path:
    """Writes <name>.csv (plausible matches to check, best first) and <name>_unmatched.csv."""
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    header = ["status", "score", *fields, "api_football_id", "api_name", "api_extra"]
    files = {"review": REVIEW_DIR / f"{name}.csv",
             "unmatched": REVIEW_DIR / f"{name}_unmatched.csv"}
    for status, f in files.items():
        with f.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for m in sorted(matches, key=lambda m: -m.score):
                if m.status != status:
                    continue
                r = m.right or {}
                w.writerow([m.status, round(m.score, 1), *[m.left.get(x) for x in fields],
                            r.get("id"), r.get("name"), r.get("extra")])
    return files["review"]


# ---------- clubs ----------

def club_score(tm: dict, api: dict) -> float:
    return club_similarity(tm["name"], api["name"])


def map_clubs(conn, api: ApiFootball) -> dict:
    with conn.cursor() as cur:
        cur.execute("select id, tm_club_id, name, tm_competition from clubs "
                    "where current_top5 order by tm_competition, name")
        tm = [dict(zip(["id", "tm_id", "name", "league"], r)) for r in cur.fetchall()]

    season = max(FREE_SEASONS)
    matches: list[Match] = []
    for key, tm_code in LEAGUE_TM.items():
        api_teams = [{"id": t["team"]["id"], "name": t["team"]["name"],
                      "extra": t["team"].get("code")}
                     for t in api.cached("teams", league=COMPETITIONS[key], season=season)["response"]]
        matches += assign([c for c in tm if c["league"] == tm_code], api_teams, club_score)

    # Clubs promoted since that season: search by name, 1 request each
    for m in matches:
        if m.status == "auto":
            continue
        q = " ".join(w for w in tokens(m.left["name"])
                     if w not in CLUB_NOISE and len(w) >= 3)[:30]
        if len(q) < 3:
            continue
        found = [{"id": t["team"]["id"], "name": t["team"]["name"],
                  "extra": t["team"].get("country")}
                 for t in api.cached("teams", search=q)["response"] if not t["team"]["national"]]
        best = assign([m.left], found, club_score)[0]
        if best.right and best.score > m.score:
            m.right, m.score, m.status = best.right, best.score, best.status

    overrides = read_overrides("clubs")
    taken = set(overrides.values())
    final = {}
    for m in matches:
        if m.left["tm_id"] in overrides:
            final[m.left["id"]] = overrides[m.left["tm_id"]]
        elif m.status == "auto" and m.right["id"] not in taken:
            final[m.left["id"]] = m.right["id"]
            taken.add(m.right["id"])
    with conn.cursor() as cur:
        cur.execute("update clubs set api_football_id = null where api_football_id is not null")
        for club_id, api_id in final.items():
            cur.execute("update clubs set api_football_id = %s where id = %s", (api_id, club_id))
        n = len(final)
    conn.commit()
    open_items = [m for m in matches if m.status != "auto" and m.left["tm_id"] not in overrides]
    review = write_review("clubs", open_items, ["tm_id", "name", "league"])
    return {"clubs": len(tm), "mapped": n, "to_review": len(open_items),
            "review_file": str(review), "requests": api.requests_made}


# ---------- players ----------

def player_score(tm: dict, api: dict, same_club: bool = False) -> float:
    s = player_similarity(api["name"], tm["name"])
    gap = None
    if tm.get("age") is not None and api.get("age") is not None:
        gap = abs(tm["age"] - api["age"])
    if same_club and gap is not None and gap <= 1:
        # nicknames break initials ('T. Douvikas' = Anastasios, 'V. Castellanos' = Taty),
        # so in one squad an identical surname and age is enough
        a_last, t_last = player_key(api["name"]).split()[-1:], player_key(tm["name"]).split()[-1:]
        if a_last and a_last == t_last and len(a_last[0]) >= 4:
            s = max(s, 82.0)
    if gap is not None:
        s += 8 if gap <= 1 else -25 if gap >= 3 else -5
    if not position_compatible(api.get("position"), tm.get("group")):
        s -= 10
    return s


def age_on(birth: dt.date | None, today: dt.date) -> int | None:
    if not birth:
        return None
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))


WITHIN_CLUB_AUTO = 80.0  # squad-by-squad pairing plus age makes a lower bar safe
GLOBAL_AUTO = 92.0       # stricter bar when searching every player, not just one squad
GLOBAL_REVIEW = 80.0     # cross-club suggestions below this were all wrong on real data


def global_candidates(api_player: dict, pool: list[dict], keys: list[str], limit: int = 30):
    """Name shortlist. Wide, because common surnames ('N. Jackson') tie with dozens of others
    and age and position only separate them afterwards."""
    hits = process.extract(player_key(api_player["name"]), keys, scorer=fuzz.token_set_ratio,
                           limit=limit, score_cutoff=70)
    return [pool[i] for _, _, i in hits]


def map_players(conn, api: ApiFootball) -> dict:
    today = dt.date.today()
    with conn.cursor() as cur:
        cur.execute("select id, api_football_id, name from clubs "
                    "where current_top5 and api_football_id is not null order by id")
        clubs = cur.fetchall()
        cur.execute("""select p.mercato_id, i.tm_player_id, p.name, p.birth_date,
                              p.position_group, p.club_id, p.current_top5, c.name
                       from players p join id_map i using (mercato_id)
                       left join clubs c on c.id = p.club_id""")
        everyone = [dict(zip(["mercato_id", "tm_id", "name", "birth", "group", "club_id",
                              "current", "club"], r)) for r in cur.fetchall()]
    for p in everyone:
        p["age"] = age_on(p["birth"], today)
    by_club: dict[int, list[dict]] = {}
    for p in everyone:
        if p["current"]:
            by_club.setdefault(p["club_id"], []).append(p)

    # Pass 1: within each club's squad
    results: list[tuple[dict, dict | None, float, str, int]] = []  # api, tm, score, status, club
    done = stopped = 0
    for club_id, api_club, club_name in clubs:
        try:
            body = api.cached("players/squads", team=api_club)
        except BudgetExceeded:
            stopped = len(clubs) - done
            break
        done += 1
        squad = body["response"][0]["players"] if body["response"] else []
        api_players = [{"id": x["id"], "name": x["name"], "age": x.get("age"),
                        "position": x.get("position"), "extra": club_name} for x in squad]
        for m in assign(api_players, by_club.get(club_id, []),
                        lambda a, t: player_score(t, a, same_club=True),
                        auto=WITHIN_CLUB_AUTO):
            results.append((m.left, m.right, m.score, m.status, club_id))

    # Pass 2: unmatched squad players against everyone (summer moves the dataset hasn't caught)
    taken = {tm["mercato_id"] for _, tm, _, st, _ in results if tm and st == "auto"}
    pool = [p for p in everyone if p["mercato_id"] not in taken]
    keys = [player_key(p["name"]) for p in pool]
    for i, (a, tm, sc, st, club_id) in enumerate(results):
        if st == "auto":
            continue
        scored = sorted(((player_score(c, a), c) for c in global_candidates(a, pool, keys)),
                        key=lambda x: -x[0])
        if not scored:
            continue
        best_sc, best = scored[0]
        clear = len(scored) == 1 or best_sc - scored[1][0] >= 5
        if best_sc >= GLOBAL_AUTO and clear and best["mercato_id"] not in taken:
            results[i] = (a, best, best_sc, "auto", club_id)
            taken.add(best["mercato_id"])
        elif best_sc > sc and best_sc >= GLOBAL_REVIEW:
            results[i] = (a, best, best_sc, "review", club_id)

    # Overrides win; drop any auto match that collides with one
    overrides = read_overrides("players")
    tm_to_mid = {p["tm_id"]: p["mercato_id"] for p in everyone}
    api_overridden = set(overrides.values())
    final: dict[int, tuple[int, float, str, int | None]] = {}  # mercato_id -> api, score, how, club
    for a, tm, sc, st, club_id in results:
        if st == "auto" and tm and a["id"] not in api_overridden:
            final[tm["mercato_id"]] = (a["id"], sc, "auto", club_id)
    api_club_of = {a["id"]: club_id for a, _, _, _, club_id in results}
    for tm_id, api_id in overrides.items():
        if tm_id in tm_to_mid:
            final[tm_to_mid[tm_id]] = (api_id, 100.0, "manual", api_club_of.get(api_id))

    id_rows = [(api_id, how, min(sc, 100.0), mid) for mid, (api_id, sc, how, _) in final.items()]
    # API squads are more current than the dataset
    club_rows = [(club_id, mid) for mid, (_, _, _, club_id) in final.items() if club_id]
    with conn.cursor() as cur:
        cur.execute("update id_map set api_football_id = null where api_football_id is not null")
        with conn.pipeline():
            cur.executemany("update id_map set api_football_id = %s, match_method = %s, "
                            "match_score = %s where mercato_id = %s", id_rows)
            cur.executemany("update players set club_id = %s, current_top5 = true "
                            "where mercato_id = %s", club_rows)
    conn.commit()

    review = write_review("players", [
        Match(tm or {}, {"id": a["id"], "name": a["name"],
                         "extra": f"{a['extra']}, {a.get('position')}, age {a.get('age')}"}, sc, st)
        for a, tm, sc, st, _ in results], ["tm_id", "name", "age", "group", "club"])
    return {"clubs_done": done, "clubs_left": stopped, "api_players": len(results),
            "mapped": len(final), "to_review": sum(st == "review" for _, _, _, st, _ in results),
            "unmatched": sum(st == "unmatched" for _, _, _, st, _ in results),
            "review_file": str(review), "requests": api.requests_made}


if __name__ == "__main__":
    from mercato.db import connect, job_run

    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what not in ("clubs", "players"):
        sys.exit("usage: python -m mercato.mapping.ids [clubs|players]")
    api = ApiFootball()
    with connect() as conn, job_run(conn, f"map_{what}") as stats:
        result = map_clubs(conn, api) if what == "clubs" else map_players(conn, api)
        stats["requests"] = api.requests_made
    print(result)
