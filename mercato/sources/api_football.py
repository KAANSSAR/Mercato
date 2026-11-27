"""API-Football client with a hard daily request budget.

Free tier: 100 requests a day, 10 a minute, resets at midnight UTC.
"""

import json
import time
from pathlib import Path

import httpx

from mercato.config import settings

BASE_URL = "https://v3.football.api-sports.io"
CACHE_DIR = Path("data/api_cache")


class BudgetExceeded(RuntimeError):
    pass


class PlanRestricted(RuntimeError):
    """The current plan can't reach this season or endpoint."""


class ApiFootball:
    def __init__(self, key: str | None = None, budget: int | None = None,
                 transport: httpx.BaseTransport | None = None):
        self.client = httpx.Client(
            base_url=BASE_URL,
            transport=transport,
            headers={"x-apisports-key": key or settings.api_football_key},
            timeout=30,
        )
        self.budget = budget if budget is not None else settings.daily_request_budget
        self.requests_made = 0
        self.remaining_today: int | None = None
        self._last_call = 0.0
        self.min_interval = 6.1  # seconds, keeps under 10 a minute

    def get(self, path: str, **params) -> dict:
        if self.requests_made >= self.budget:
            raise BudgetExceeded(f"Run budget of {self.budget} requests used")
        if self.remaining_today is not None and self.remaining_today <= 0:
            raise BudgetExceeded("API reports 0 requests left today")

        # Stay under 10 a minute
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

        resp = self.client.get(f"/{path}", params=params)
        self._last_call = time.monotonic()
        self.requests_made += 1

        left = resp.headers.get("x-ratelimit-requests-remaining")
        if left is not None:
            self.remaining_today = int(left)

        resp.raise_for_status()
        body = resp.json()
        errors = body.get("errors")
        if errors:
            if isinstance(errors, dict) and "plan" in errors:
                raise PlanRestricted(errors["plan"])
            if isinstance(errors, dict) and "requests" in errors:  # daily limit reached
                self.remaining_today = 0
                raise BudgetExceeded(errors["requests"])
            raise RuntimeError(f"API-Football error on {path}: {errors}")
        return body

    def cached(self, path: str, refresh: bool = False, **params) -> dict:
        """GET with an on-disk cache, so interrupted jobs resume without re-spending requests."""
        key = "_".join(f"{k}-{v}" for k, v in sorted(params.items())) or "all"
        file = CACHE_DIR / path.replace("/", "_") / f"{key}.json"
        if file.exists() and not refresh:
            return json.loads(file.read_text())
        body = self.get(path, **params)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(body))
        return body

    # Endpoints we use
    def fixtures_by_date(self, date: str) -> list[dict]:
        return self.get("fixtures", date=date)["response"]

    def fixture_players(self, fixture_id: int) -> list[dict]:
        return self.get("fixtures/players", fixture=fixture_id)["response"]

    def fixture_statistics(self, fixture_id: int) -> list[dict]:
        return self.get("fixtures/statistics", fixture=fixture_id)["response"]

    def standings(self, league: int, season: int) -> list[dict]:
        return self.get("standings", league=league, season=season)["response"]

    def status(self) -> dict:
        """Account status. Does not count against the daily quota."""
        resp = self.client.get("/status")
        resp.raise_for_status()
        return resp.json()["response"]
