"""API-Football client with a hard daily request budget.

Free tier: 100 requests a day, 10 a minute, resets at midnight UTC.
"""

import time

import httpx

from mercato.config import settings

BASE_URL = "https://v3.football.api-sports.io"


class BudgetExceeded(RuntimeError):
    pass


class ApiFootball:
    def __init__(self, key: str | None = None, budget: int | None = None):
        self.client = httpx.Client(
            base_url=BASE_URL,
            headers={"x-apisports-key": key or settings.api_football_key},
            timeout=30,
        )
        self.budget = budget if budget is not None else settings.daily_request_budget
        self.requests_made = 0
        self.remaining_today: int | None = None
        self._last_call = 0.0

    def get(self, path: str, **params) -> dict:
        if self.requests_made >= self.budget:
            raise BudgetExceeded(f"Run budget of {self.budget} requests used")
        if self.remaining_today is not None and self.remaining_today <= 0:
            raise BudgetExceeded("API reports 0 requests left today")

        # Stay under 10 a minute
        wait = 6.1 - (time.monotonic() - self._last_call)
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
        if body.get("errors"):
            raise RuntimeError(f"API-Football error on {path}: {body['errors']}")
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
        return self.client.get("/status").json()
