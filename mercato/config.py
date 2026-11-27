from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_football_key: str = ""
    database_url: str = ""
    daily_request_budget: int = 95


settings = Settings()

# Seasons are labelled by start year: 2026 = 2026/27
CURRENT_SEASON = 2026
FIRST_SEASON = 2015
# Free plan only reaches these (verified 27 Nov 2026)
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
