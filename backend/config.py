"""Environment-driven settings with automatic provider detection.

The whole app is designed to run offline. When the relevant environment
variables are present, the corresponding real provider is enabled instead.
"""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Providers (empty => offline fallback used)
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Google Gemini (alternative provider)
    gemini_api_key: str = ""
    gemini_embed_model: str = "text-embedding-004"
    gemini_chat_model: str = "gemini-2.0-flash"
    gemini_embed_dim: int = 768

    database_url: str = ""

    # Retrieval tuning
    top_k: int = 4
    semantic_cache_threshold: float = 0.95

    # Local fallback embedding dimensionality
    local_embed_dim: int = 384

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def pgvector_enabled(self) -> bool:
        return bool(self.database_url)


@lru_cache
def get_settings() -> Settings:
    # Allow process env to win over .env file for container deployments.
    return Settings(_env_file=os.getenv("ENV_FILE", ".env"))
