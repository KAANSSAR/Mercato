"""ClubElo loader. Free, no key, plain CSV.

    api.clubelo.com/YYYY-MM-DD   every rated club on that date
    api.clubelo.com/ManCity      one club's full history (name without spaces)

Both return: Rank, Club, Country, Level, Elo, From, To

    uv run python -m mercato.sources.clubelo        # full load, ~10 min the first time
"""

import datetime as dt
import io
import time

import httpx
import pandas as pd

from mercato.config import CURRENT_SEASON, FIRST_SEASON

BASE = "http://api.clubelo.com"
TOP5_COUNTRIES = {"ENG", "ESP", "ITA", "GER", "FRA"}
HISTORY_FROM = dt.date(FIRST_SEASON - 1, 7, 1)  # a season of lead-in before training starts


def parse(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df = df.rename(columns={"Club": "clubelo_name", "Country": "country", "Level": "level",
                            "Elo": "elo", "From": "valid_from", "To": "valid_to"})
    df["valid_from"] = pd.to_datetime(df["valid_from"]).dt.date
    df["valid_to"] = pd.to_datetime(df["valid_to"]).dt.date
    return df[["clubelo_name", "country", "level", "elo", "valid_from", "valid_to"]]


def url_name(club: str) -> str:
    return club.replace(" ", "")


def clubs_to_track(snapshots: list[pd.DataFrame]) -> set[str]:
    """Every top-flight European club (possible UCL/UEL opponents) plus top-5 second tiers
    (relegated and promoted sides)."""
    names = set()
    for s in snapshots:
        keep = (s["level"] == 1) | (s["country"].isin(TOP5_COUNTRIES) & (s["level"] == 2))
        names |= set(s.loc[keep, "clubelo_name"])
    return names


def elo_on(history: pd.DataFrame, club: str, date: dt.date) -> float | None:
    """Rating of a club on a date, from elo_history rows."""
    h = history[(history["clubelo_name"] == club)
                & (history["valid_from"] <= date) & (history["valid_to"] >= date)]
    return float(h["elo"].iloc[0]) if len(h) else None


class ClubElo:
    def __init__(self, transport: httpx.BaseTransport | None = None, pause: float = 0.3):
        self.client = httpx.Client(base_url=BASE, timeout=30, transport=transport)
        self.pause = pause

    def _get(self, path: str) -> pd.DataFrame:
        resp = self.client.get(f"/{path}")
        resp.raise_for_status()
        time.sleep(self.pause)
        return parse(resp.text)

    def snapshot(self, date: dt.date) -> pd.DataFrame:
        return self._get(date.isoformat())

    def history(self, club: str) -> pd.DataFrame:
        h = self._get(url_name(club))
        return h[h["valid_to"] >= HISTORY_FROM]


def load() -> dict:
    from mercato.db import connect, job_run, upsert

    api = ClubElo()
    dates = [dt.date(y, 9, 1) for y in range(FIRST_SEASON, CURRENT_SEASON + 1)]
    dates.append(dt.date.today())
    snapshots = [api.snapshot(d) for d in dates]
    clubs = sorted(clubs_to_track(snapshots))
    print(f"{len(clubs)} clubs to fetch")

    with connect() as conn, job_run(conn, "clubelo_load") as stats:
        stats["requests"] = len(dates)
        failed = []
        for i, club in enumerate(clubs, 1):
            try:
                h = api.history(club)
            except httpx.HTTPError as e:
                failed.append((club, str(e)))
                continue
            stats["requests"] += 1
            rows = h.astype(object).where(h.notna(), None).to_dict("records")
            stats["rows_written"] += upsert(conn, "elo_history", rows,
                                            ["clubelo_name", "valid_from"],
                                            update=["elo", "valid_to", "level", "country"])
            conn.commit()
            if i % 50 == 0:
                print(f"{i}/{len(clubs)}")
        if failed:
            print(f"failed: {failed[:10]}{' ...' if len(failed) > 10 else ''}")
        return dict(stats) | {"clubs": len(clubs), "failed": len(failed)}


if __name__ == "__main__":
    print(load())
