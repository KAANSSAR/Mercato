from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_football_key: str = ""
    database_url: str = ""
    daily_request_budget: int = 95
    # Seconds between API calls. Free plan: 10 a minute, so 6.1. Pro allows far more,
    # so set API_MIN_INTERVAL=0.25 in .env for the backfill month.
    api_min_interval: float = 6.1


settings = Settings()

# Seasons are labelled by start year: 2026 = 2026/27
CURRENT_SEASON = 2026
FIRST_SEASON = 2015
# Free plan: whole-season queries only for these. Fixtures and match stats by date work
# for yesterday to tomorrow on any season, which covers live updates (verified 27 Nov 2026).
FREE_SEASONS = (2022, 2023, 2024)


def api_seasons(plan: str) -> tuple[int, ...]:
    """Seasons API-Football will serve on this plan."""
    if plan.lower() == "free":
        return FREE_SEASONS
    return tuple(range(FIRST_SEASON, CURRENT_SEASON + 1))


# API-Football league IDs (verified 27 Nov 2026)
COMPETITIONS = {
    "premier_league": 39,
    "laliga": 140,
    "serie_a": 135,
    "bundesliga": 78,
    "ligue_1": 61,
    "champions_league": 2,
    "europa_league": 3,
}
