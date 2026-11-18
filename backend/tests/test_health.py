"""Tests for health and readiness endpoints."""

from starlette.testclient import TestClient
from backend.app.main import create_app


def test_liveness_endpoint(client: TestClient):
    """Verify GET /health returns 200 with healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data


def test_readiness_endpoint_ready(client: TestClient):
    """Verify GET /health/ready returns 200 when model is loaded."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert "device" in data
    assert data["startup_error"] is None


def test_readiness_endpoint_not_ready():
    """Verify GET /health/ready returns 503 when model is unavailable."""
    app = create_app()
    app.state.startup_error = "Model weights failed to mount."

    with TestClient(app, raise_server_exceptions=False) as unready_client:
        response = unready_client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["model_loaded"] is False
        assert "Model weights failed to mount" in data["startup_error"]
