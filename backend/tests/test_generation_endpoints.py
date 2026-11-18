"""Integration tests for Generation and Model Info REST API endpoints."""

from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from backend.app.core.exceptions import ModelInferenceException, ModelNotReadyException
from backend.app.main import create_app
from backend.app.services.generation import GenerationService


def test_generate_endpoint_success(client: TestClient):
    """Verify POST /api/v1/generate returns 200 with structured description payload."""
    payload = {
        "product_name": "Hydrating Face Cream",
        "category": "Beauty",
        "brand": "DermaPure",
        "features": ["Contains Hyaluronic Acid", "Paraben-Free"],
        "specifications": {"Volume": "50ml", "Skin Type": "Sensitive"},
        "audience": "Dry skin sufferers",
        "generation_config": {
            "num_beams": 2,
            "max_new_tokens": 64,
            "min_new_tokens": 10,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.0,
            "do_sample": False,
        },
    }

    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "data" in data
    res_data = data["data"]
    assert "generated_description" in res_data
    assert len(res_data["generated_description"]) > 0
    assert res_data["model"] == "google/flan-t5-small"
    assert res_data["generation_strategy"] == "beam_search_2"
    assert res_data["output_tokens"] == 32
    assert res_data["latency_ms"] == 185.5
    assert "input_prompt" in res_data


def test_generate_endpoint_minimal_valid_payload(client: TestClient):
    """Verify POST /api/v1/generate functions correctly with only product_name."""
    payload = {"product_name": "Organic Green Tea"}
    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["generated_description"] != ""


def test_generate_endpoint_missing_product_name(client: TestClient):
    """Verify POST /api/v1/generate returns 422 when required product_name is absent."""
    payload = {"category": "Electronics"}
    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 422


def test_generate_endpoint_empty_product_name(client: TestClient):
    """Verify POST /api/v1/generate returns 422 for blank or whitespace product_name."""
    response_empty = client.post("/api/v1/generate", json={"product_name": ""})
    assert response_empty.status_code == 422

    response_spaces = client.post("/api/v1/generate", json={"product_name": "     "})
    assert response_spaces.status_code == 422

    response_short = client.post("/api/v1/generate", json={"product_name": "x"})
    assert response_short.status_code == 422


def test_generate_endpoint_invalid_config_parameters(client: TestClient):
    """Verify POST /api/v1/generate returns 422 when generation_config violates constraints."""
    # num_beams out of bounds (> 5)
    res_beams = client.post("/api/v1/generate", json={
        "product_name": "Test Product",
        "generation_config": {"num_beams": 10},
    })
    assert res_beams.status_code == 422

    # max_new_tokens out of bounds (< 10)
    res_tokens = client.post("/api/v1/generate", json={
        "product_name": "Test Product",
        "generation_config": {"max_new_tokens": 5},
    })
    assert res_tokens.status_code == 422

    # temperature out of bounds (> 2.0)
    res_temp = client.post("/api/v1/generate", json={
        "product_name": "Test Product",
        "generation_config": {"temperature": 3.5},
    })
    assert res_temp.status_code == 422


def test_generate_endpoint_model_unready():
    """Verify POST /api/v1/generate returns 503 when model is not ready."""
    app = create_app()
    app.state.predictor = None
    app.state.generation_service = None
    app.state.is_ready = False
    app.state.startup_error = "Model unmounted"

    with TestClient(app, raise_server_exceptions=False) as client_unready:
        response = client_unready.post("/api/v1/generate", json={"product_name": "Test Product"})
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "model_not_ready"


def test_generate_endpoint_inference_failure(client: TestClient, mock_predictor: MagicMock):
    """Verify POST /api/v1/generate returns 500 when predictor raises runtime error."""
    mock_predictor.predict.side_effect = RuntimeError("CUDA device out of memory")

    response = client.post("/api/v1/generate", json={"product_name": "Test Product"})
    assert response.status_code == 500
    data = response.json()
    assert data["error"] == "inference_error"
    assert "generating the product description" in data["message"]

    # Reset side effect
    mock_predictor.predict.side_effect = None


def test_model_info_endpoint_success(client: TestClient):
    """Verify GET /api/v1/model/info returns 200 and sanitized metadata without absolute paths."""
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    info = data["data"]
    assert info["model_id"] == "google/flan-t5-small"
    assert "Transformer" in info["architecture"]
    assert info["device"] == "cuda:0"
    assert info["total_parameters"] == 76961152
    assert "beam_search" in info["supported_strategies"]
    assert info["is_ready"] is True

    # Security check: Ensure raw absolute Windows drive paths are never exposed
    assert "C:" not in info["checkpoint"]
    assert "C:/" not in info["checkpoint"]
    assert "C:\\" not in info["checkpoint"]
    assert "checkpoints/" in info["checkpoint"]


def test_model_info_endpoint_unready():
    """Verify GET /api/v1/model/info returns 503 when model is unready."""
    app = create_app()
    app.state.predictor = None
    app.state.generation_service = None
    app.state.is_ready = False
    app.state.startup_error = "Model unmounted"

    with TestClient(app, raise_server_exceptions=False) as client_unready:
        response = client_unready.get("/api/v1/model/info")
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "model_not_ready"


def test_openapi_documentation(client: TestClient):
    """Verify OpenAPI schema registers generation and model endpoints under tags."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    paths = schema.get("paths", {})
    assert "/api/v1/generate" in paths
    assert "/api/v1/model/info" in paths
    assert "/health" in paths
    assert "/health/ready" in paths

    # Verify tags
    tags = [t["name"] for t in schema.get("tags", [])]
    assert "Generation" in tags
    assert "Model" in tags
    assert "Health" in tags
