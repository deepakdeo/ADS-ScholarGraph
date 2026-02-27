"""Application configuration loaded from environment variables."""

from functools import lru_cache
from importlib import import_module
from typing import Any

SettingsConfigDict: Any = dict

try:
    pydantic_settings = import_module("pydantic_settings")
    BaseSettings = pydantic_settings.BaseSettings
    SettingsConfigDict = pydantic_settings.SettingsConfigDict
    _USING_PYDANTIC_SETTINGS = True
except ModuleNotFoundError:
    from pydantic.v1 import BaseSettings

    _USING_PYDANTIC_SETTINGS = False


class Settings(BaseSettings):  # type: ignore[misc, valid-type]
    """Runtime settings for ADS and Neo4j connectivity."""

    if _USING_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"

    ADS_API_TOKEN: str | None = None
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "change_me"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
