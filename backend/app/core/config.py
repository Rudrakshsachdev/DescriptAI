"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized environment configuration for Descript AI Backend."""

    # Project Information
    PROJECT_NAME: str = "Descript AI API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DESCRIPTION: str = "Production FastAPI backend for AI-powered product description generation."

    # Environment
    ENVIRONMENT: Literal["development", "staging", "production", "testing"] = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v: Any) -> bool:
        """Safely parses boolean debug flags, ignoring arbitrary OS strings like 'release'."""
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on", "debug")
        return bool(v)

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]

    # Model & Inference Settings
    MODEL_CHECKPOINT_PATH: str = "checkpoints/sft/best_checkpoint"
    MODEL_FALLBACK_ID: str = "google/flan-t5-small"
    DEVICE: Literal["auto", "cuda", "cpu"] = "auto"
    USE_HALF_PRECISION: bool = True

    # Generation Defaults (Beam-2 Recommended from Step 9 Audit)
    DEFAULT_NUM_BEAMS: int = 2
    DEFAULT_MAX_NEW_TOKENS: int = 128
    DEFAULT_MIN_NEW_TOKENS: int = 10
    DEFAULT_NO_REPEAT_NGRAM_SIZE: int = 3
    DEFAULT_LENGTH_PENALTY: float = 1.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @property
    def resolved_checkpoint_path(self) -> Path:
        """Returns the absolute or verified Path to model checkpoint."""
        return Path(self.MODEL_CHECKPOINT_PATH)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns cached application settings instance."""
    return Settings()
