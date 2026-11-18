"""Tests for FastAPI app lifespan and exception handlers."""

from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

from backend.app.core.exceptions import (
    InferenceTimeoutException,
    ModelInferenceException,
    ModelNotReadyException,
    ServiceOverloadedException,
)
from backend.app.main import create_app


def test_app_creation():
    """Verify application factory configuration."""
    app = create_app()
    assert app.title == "Descript AI API"
    assert app.version == "1.0.0"


def test_model_not_ready_exception_handler():
    """Verify custom exception handler for ModelNotReadyException."""
    app = create_app()

    @app.get("/test-not-ready")
    async def trigger_not_ready():
        raise ModelNotReadyException("Predictor is unmounted.")

    app.state.predictor = MagicMock()
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/test-not-ready")
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "model_not_ready"
        assert "Predictor is unmounted" in data["message"]


def test_model_inference_exception_handler():
    """Verify custom exception handler for ModelInferenceException."""
    app = create_app()

    @app.get("/test-inference-error")
    async def trigger_inference_error():
        raise ModelInferenceException("Generation timeout.")

    app.state.predictor = MagicMock()
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/test-inference-error")
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "inference_error"
        assert "Generation timeout" in data["message"]


def test_inference_timeout_exception_handler():
        app = create_app()

        @app.get("/test-inference-timeout")
        async def trigger_timeout():
            raise InferenceTimeoutException("Generation timed out.")

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/test-inference-timeout")
            assert response.status_code == 504
            assert response.json()["error"] == "inference_timeout"


def test_service_overload_exception_handler():
        app = create_app()

        @app.get("/test-overload")
        async def trigger_overload():
            raise ServiceOverloadedException("Service overloaded.")

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/test-overload")
            assert response.status_code == 503
            assert response.json()["error"] == "service_overloaded"


def test_lifespan_startup_failure():
    """Verify graceful startup degradation when predictor initialization fails."""
    app = create_app()

    with patch("backend.app.main.DescriptionPredictor", side_effect=FileNotFoundError("Checkpoint missing")):
        with TestClient(app, raise_server_exceptions=False) as client:
            # Liveness should still work
            res_live = client.get("/health")
            assert res_live.status_code == 200

            # Readiness should report 503 with the startup error
            res_ready = client.get("/health/ready")
            assert res_ready.status_code == 503
            data = res_ready.json()
            assert data["status"] == "not_ready"
            assert "Checkpoint missing" in data["startup_error"]
