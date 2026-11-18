"""Tests for application configuration."""

from pathlib import Path
from backend.app.core.config import Settings, get_settings


def test_default_settings():
    """Verify default settings instantiation and properties."""
    settings = Settings()
    assert settings.PROJECT_NAME == "Descript AI API"
    assert settings.VERSION == "1.0.0"
    assert settings.DEFAULT_NUM_BEAMS == 2
    assert settings.DEFAULT_MAX_NEW_TOKENS == 128
    assert settings.USE_HALF_PRECISION is True
    assert isinstance(settings.resolved_checkpoint_path, Path)


def test_settings_singleton_caching():
    """Verify get_settings returns the cached singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_custom_environment_settings(monkeypatch):
    """Verify settings override via environment variables."""
    monkeypatch.setenv("DEFAULT_NUM_BEAMS", "4")
    monkeypatch.setenv("DEVICE", "cpu")
    custom_settings = Settings()
    assert custom_settings.DEFAULT_NUM_BEAMS == 4
    assert custom_settings.DEVICE == "cpu"


def test_reliability_environment_settings(monkeypatch):
    monkeypatch.setenv("INFERENCE_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("INFERENCE_CAPACITY_WAIT_SECONDS", "0.5")
    monkeypatch.setenv("MAX_CONCURRENT_INFERENCE_REQUESTS", "2")

    custom_settings = Settings()
    assert custom_settings.INFERENCE_TIMEOUT_SECONDS == 12.0
    assert custom_settings.INFERENCE_CAPACITY_WAIT_SECONDS == 0.5
    assert custom_settings.MAX_CONCURRENT_INFERENCE_REQUESTS == 2
