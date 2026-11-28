"""Global Football Rankings (globalfootballrankings.com): daily league and club ratings.

    uv run python -m mercato.sources.gfr --dry   # scrape, match, compare, print only
    uv run python -m mercato.sources.gfr         # also write a snapshot to Postgres

Reads the two public ranking pages as a visitor sees them (rendered in headless Chromium,
two page loads a run, once a day), identifying itself. Their robots.txt allows all
crawlers. Ratings are 0 to 100 (50 is the world average). Credit GFR wherever they appear.

They only publish current ratings, so history (every match date since 2015) still comes
from Mercato's own Elo: mercato.features.gfr_anchor moves each league's level to GFR's,
across all history, and keeps our match-by-match order within the league. Each run also
writes data/gfr/leagues.csv and clubs.csv for the model, and the daily snapshots build a
GFR history of their own from the first one.
"""

import argparse
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from mercato.mapping.names import club_key, club_similarity

BASE = "https://globalfootballrankings.com/rankings"
PAGES = {"leagues": f"{BASE}/leagues/men", "clubs": f"{BASE}/teams/men"}
USER_AGENT = ("Mozilla/5.0 (compatible; MercatoBot/1.0; "
              "+https://mercato.kanishka-sarkar.com)")
MIN_ROWS = {"leagues": 50, "clubs": 500}
CLUB_MATCH = 85
CLUB_CLEAR = (75, 20)  # or at least 75 and 20 clear of the next club in that country
OVERRIDES = Path("db/overrides/gfr_clubs.csv")  # country, slug, tm_club_id

# Transfermarkt league id -> GFR country slug. The top flight is the country's
# highest-rated league on GFR.
COUNTRY = {"GB1": "england", "ES1": "spain", "IT1": "italy", "L1": "germany",
           "FR1": "france", "PO1": "portugal", "NL1": "netherlands", "BE1": "belgium",
           "TR1": "turkiye", "SC1": "scotland", "GR1": "greece", "DK1": "denmark",
           "RU1": "russia", "UKR1": "ukraine"}

# Runs in the page: each ranking link, and the text of its row ("1 Arsenal 97.2")
EXTRACT = r"""
(kind) => {
  const re = new RegExp('/rankings/' + kind + '/men/[^/]+/[^/?#]+$');
  return [...document.querySelectorAll('main a')].filter(a => re.test(a.href)).map(a => {
    let el = a;
    for (let i = 0; i < 6 && el.parentElement; i++) {
      el = el.parentElement;
      const t = el.innerText.replace(/\s+/g, ' ').trim();
      if (/^\d+ .+ \d+(\.\d+)?$/.test(t)) return {href: a.href, text: t};
    }
    return {href: a.href, text: null};
  });
}
"""
ROW = re.compile(r"^(\d+) (.+) (\d+(?:\.\d+)?)$")
UPDATED = re.compile(r"Updated ([A-Z][a-z]{2} \d{1,2}, \d{4})")


def scrape(timeout_ms: int = 60_000) -> dict:
    """Render both pages; returns {"leagues": [...], "clubs": [...], "updated": date}."""
    from playwright.sync_api import sync_playwright

    out = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=USER_AGENT)
        for kind, url in PAGES.items():
            # not "networkidle": their ads and analytics never go quiet
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            path = "leagues" if kind == "leagues" else "teams"
            page.wait_for_function(
                f"document.querySelectorAll('main a[href*=\"/rankings/{path}/men/\"]').length"
                f" >= {MIN_ROWS[kind]}", timeout=timeout_ms)
            out[kind] = page.evaluate(EXTRACT, path)
            if kind == "leagues":
                out["updated"] = parse_updated(page.inner_text("main"))
        browser.close()
    return out


def parse_updated(text: str) -> date:
    m = UPDATED.search(text or "")
    return datetime.strptime(m.group(1), "%b %d, %Y").date() if m else date.today()


def parse_rows(raw: list[dict]) -> pd.DataFrame:
    """Links and row texts into rank, name, rating, country and slug."""
    rows = []
    for r in raw:
        m = ROW.match(r.get("text") or "")
        if not m:
            continue
        country, slug = r["href"].rstrip("/").split("/")[-2:]
        rows.append({"rank": int(m.group(1)), "name": m.group(2),
                     "rating": float(m.group(3)), "country": country, "slug": slug})
    df = pd.DataFrame(rows, columns=["rank", "name", "rating", "country", "slug"])
    return df.drop_duplicates(["country", "slug"]).sort_values("rank").reset_index(drop=True)


def map_leagues(leagues: pd.DataFrame) -> pd.DataFrame:
    """Attach our league id to each country's top flight."""
    top = leagues.sort_values("rating", ascending=False).drop_duplicates("country")
    by_country = {c: lg for lg, c in COUNTRY.items()}
    leagues = leagues.copy()
    leagues["tm_competition_id"] = None
    for i in top.index:
        leagues.loc[i, "tm_competition_id"] = by_country.get(top.loc[i, "country"])
    return leagues


FILLER = {"football", "futebol", "futbol", "sporting", "association", "sad", "spa", "sa"}


def _initials(name: str) -> str:
    return "".join(t[0] for t in club_key(name).split() if t not in FILLER)


def _score(gfr_name: str, tm_name: str) -> float:
    """Name similarity, or a short name that is the other's initials (PSG)."""
    s = club_similarity(gfr_name, tm_name)
    short = club_key(gfr_name).replace(" ", "")
    if 2 <= len(short) <= 4 and short == _initials(tm_name):
        s = max(s, 95.0)
    return s


def map_clubs(clubs: pd.DataFrame, tm_clubs: pd.DataFrame,
              overrides: pd.DataFrame | None = None) -> pd.DataFrame:
    """Match each GFR club to a Transfermarkt club in the same country by name."""
    tm = tm_clubs[["club_id", "name", "domestic_competition_id"]].copy()
    tm["country"] = tm["domestic_competition_id"].map(COUNTRY)
    tm = tm.dropna(subset=["country"])
    by_country = {c: g for c, g in tm.groupby("country")}
    ids = []
    for r in clubs.itertuples(index=False):
        cand = by_country.get(r.country)
        best, score, second = None, 0.0, 0.0
        if cand is not None:
            for cid, name in zip(cand["club_id"], cand["name"]):
                s = _score(r.name, name)
                if s > score:
                    best, score, second = cid, s, score
                elif s > second:
                    second = s
        ok = score >= CLUB_MATCH or (score >= CLUB_CLEAR[0] and score - second >= CLUB_CLEAR[1])
        ids.append(int(best) if best is not None and ok else None)
    clubs = clubs.copy()
    clubs["tm_club_id"] = pd.array(ids, dtype="Int64")
    if overrides is not None and len(overrides):
        o = overrides.set_index(["country", "slug"])["tm_club_id"]
        key = list(zip(clubs["country"], clubs["slug"]))
        clubs["tm_club_id"] = [o.get(k, v) for k, v in zip(key, clubs["tm_club_id"])]
        clubs["tm_club_id"] = pd.array(clubs["tm_club_id"], dtype="Int64")
    return clubs


def compare(leagues: pd.DataFrame, clubs: pd.DataFrame, data_dir=None) -> None:
    """Side by side with Mercato's own Elo, if the Transfermarkt data is on this machine."""
    try:
        from mercato.features.elo import run
        from mercato.model.leagues import league_strength
        from mercato.sources.tm_datasets import DATA_DIR, read
        tm_clubs = read("clubs", data_dir or DATA_DIR)
        elo = run(read("games", data_dir or DATA_DIR))
    except FileNotFoundError:
        print("(Transfermarkt data not found: skipping the comparison with Mercato Elo)")
        return
    ours = league_strength(elo, tm_clubs)
    lg = leagues.dropna(subset=["tm_competition_id"]).set_index("tm_competition_id")
    side = ours.join(lg[["name", "rating", "rank"]], how="inner")
    print("\nLeagues: Mercato average Elo vs GFR rating")
    print(side.sort_values("avg_elo", ascending=False).to_string())
    from mercato.features.elo import latest
    last = latest(elo).rename(columns={"tm_club_id": "tm_club_id", "elo": "mercato_elo"})
    m = clubs.dropna(subset=["tm_club_id"]).merge(last, on="tm_club_id")
    if len(m) > 10:
        rho = m["rating"].corr(m["mercato_elo"], method="spearman")
        print(f"\nClubs matched: {len(m)}; rank agreement GFR vs Mercato Elo: {rho:.3f}")
        m["gap"] = m["rating"].rank(pct=True) - m["mercato_elo"].rank(pct=True)
        cols = ["name", "country", "rating", "mercato_elo"]
        print("GFR rates higher than we do:")
        print(m.sort_values("gap", ascending=False)[cols].head(8).to_string(index=False))
        print("We rate higher than GFR:")
        print(m.sort_values("gap")[cols].head(8).to_string(index=False))


def save(updated: date, leagues: pd.DataFrame, clubs: pd.DataFrame) -> dict:
    from mercato.db import connect, job_run, upsert

    def recs(df, extra):
        out = df.assign(snapshot=updated).astype(object)
        return out.where(pd.notna(out), None)[["snapshot", "country", "slug", "name", "rank",
                                               "rating", extra]].to_dict("records")
    with connect() as conn, job_run(conn, "gfr") as stats:
        stats["requests"] = 2
        n = upsert(conn, "gfr_league_ratings", recs(leagues, "tm_competition_id"),
                   ["snapshot", "country", "slug"],
                   update=["name", "rank", "rating", "tm_competition_id"])
        n += upsert(conn, "gfr_club_ratings", recs(clubs, "tm_club_id"),
                    ["snapshot", "country", "slug"],
                    update=["name", "rank", "rating", "tm_club_id"])
        stats["rows_written"] = n
        return dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    raw = scrape()
    leagues = map_leagues(parse_rows(raw["leagues"]))
    clubs = parse_rows(raw["clubs"])
    for kind, df in (("leagues", leagues), ("clubs", clubs)):
        if len(df) < MIN_ROWS[kind]:  # the page changed: fail loudly, write nothing
            raise RuntimeError(f"only {len(df)} {kind} parsed; the page layout may have changed")
    try:
        from mercato.sources.tm_datasets import DATA_DIR, read
        tm_clubs = read("clubs", DATA_DIR)
    except FileNotFoundError:
        tm_clubs = None
    if tm_clubs is not None:
        overrides = pd.read_csv(OVERRIDES) if OVERRIDES.exists() else None
        clubs = map_clubs(clubs, tm_clubs, overrides)
    else:
        clubs["tm_club_id"] = pd.array([None] * len(clubs), dtype="Int64")
    print(f"GFR updated {raw['updated']}: {len(leagues)} leagues, {len(clubs)} clubs, "
          f"{int(clubs['tm_club_id'].notna().sum())} matched to Transfermarkt clubs")
    print(leagues.head(15).to_string(index=False))
    in_scope = clubs[clubs["country"].isin(COUNTRY.values())]
    miss = in_scope[in_scope["tm_club_id"].isna()]
    if tm_clubs is not None and len(miss):
        print(f"\nUnmatched clubs in our countries ({len(miss)}), top 15 by rating:")
        print(miss.head(15)[["rank", "name", "country", "slug", "rating"]].to_string(index=False))
    # local copy for the model (mercato.features.gfr_anchor), kept current on every run
    from mercato.features.gfr_anchor import GFR_DIR
    GFR_DIR.mkdir(parents=True, exist_ok=True)
    leagues.assign(snapshot=raw["updated"]).to_csv(GFR_DIR / "leagues.csv", index=False)
    clubs.assign(snapshot=raw["updated"]).to_csv(GFR_DIR / "clubs.csv", index=False)
    compare(leagues, clubs)
    if not args.dry:
        print(save(raw["updated"], leagues, clubs))


if __name__ == "__main__":
    main()
