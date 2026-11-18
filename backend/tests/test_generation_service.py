"""Tests for GenerationService."""

from unittest.mock import MagicMock
import pytest

from backend.app.core.exceptions import ModelInferenceException, ModelNotReadyException
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
