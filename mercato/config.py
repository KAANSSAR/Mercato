from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_football_key: str = ""
    database_url: str = ""
    daily_request_budget: int = 95


settings = Settings()

# API-Football league IDs. Verify on day one with the /leagues endpoint.
COMPETITIONS = {
    "premier_league": 39,
    "laliga": 140,
    "serie_a": 135,
    "bundesliga": 78,
    "ligue_1": 61,
    "champions_league": 2,
    "europa_league": 3,
}
