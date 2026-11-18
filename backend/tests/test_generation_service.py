"""Tests for GenerationService."""

from unittest.mock import MagicMock
import pytest

from backend.app.core.exceptions import ModelInferenceException, ModelNotReadyException
from backend.app.schemas.generation import GenerationConfigSchema, ProductDescriptionRequest
from backend.app.services.generation import GenerationService
from descra.inference.predictor import PredictionOutput


def test_generation_service_success(mock_predictor, mock_prediction_output):
    """Verify successful generation dispatch and output preservation."""
    service = GenerationService(predictor=mock_predictor)
    assert service.is_ready is True

    result = service.generate(
        product_name="Hydrating Facial Serum",
        features=["Hyaluronic Acid", "Niacinamide"],
        category="Skincare",
        brand="GlowLab",
        num_beams=2,
    )

    assert isinstance(result, PredictionOutput)
    assert result.generated_description == mock_prediction_output.generated_description
    assert result.latency_ms == mock_prediction_output.latency_ms
    mock_predictor.predict.assert_called_once()


def test_generation_service_not_ready():
    """Verify ModelNotReadyException when predictor is None."""
    service = GenerationService(predictor=None)
    assert service.is_ready is False

    with pytest.raises(ModelNotReadyException) as exc_info:
        service.generate(product_name="Test Product")
    assert "not loaded or ready" in str(exc_info.value)


def test_generation_service_inference_failure(mock_predictor):
    """Verify ModelInferenceException wrapping on model failure."""
    mock_predictor.predict.side_effect = RuntimeError("CUDA out of memory simulation")
    service = GenerationService(predictor=mock_predictor)

    with pytest.raises(ModelInferenceException) as exc_info:
        service.generate(product_name="Test Product")
    assert "generating the product description" in str(exc_info.value)
    mock_predictor.predict.side_effect = None


def test_generation_service_from_request(mock_predictor, mock_prediction_output):
    """Verify generate_from_request integrates specifications and audience into features."""
    service = GenerationService(predictor=mock_predictor)
    req = ProductDescriptionRequest(
        product_name="Smart Fitness Watch",
        category="Wearables",
        brand="FitTech",
        features=["Heart Rate Monitor", "GPS"],
        specifications={"Battery Life": "7 Days", "Water Resistance": "50m"},
        audience="Athletes and runners",
        generation_config=GenerationConfigSchema(num_beams=3, max_new_tokens=90),
    )

    result = service.generate_from_request(req)
    assert isinstance(result, PredictionOutput)

    # Check arguments passed to predictor.predict
    call_kwargs = mock_predictor.predict.call_args.kwargs
    assert call_kwargs["product_name"] == "Smart Fitness Watch"
    assert call_kwargs["category"] == "Wearables"
    assert call_kwargs["brand"] == "FitTech"
    assert "Heart Rate Monitor" in call_kwargs["features"]
    assert "Battery Life: 7 Days" in call_kwargs["features"]
    assert "Target Audience: Athletes and runners" in call_kwargs["features"]
    assert call_kwargs["config"].num_beams == 3
    assert call_kwargs["config"].max_new_tokens == 90


def test_generation_service_get_model_info(mock_predictor):
    """Verify get_model_info returns sanitized ModelInfoData."""
    service = GenerationService(predictor=mock_predictor)
    info = service.get_model_info()

    assert info.model_id == "google/flan-t5-small"
    assert "Transformer" in info.architecture
    assert info.total_parameters == 76961152
    assert info.is_ready is True
    assert "checkpoints/" in info.checkpoint
    assert "C:" not in info.checkpoint


def test_generation_service_get_model_info_unready():
    """Verify get_model_info raises ModelNotReadyException when predictor is None."""
    service = GenerationService(predictor=None)
    with pytest.raises(ModelNotReadyException):
        service.get_model_info()
