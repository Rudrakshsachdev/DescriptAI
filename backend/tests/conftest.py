"""Pytest fixtures for backend test suite."""

import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

# Ensure src and root are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
src_dir = root_dir / "src"
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Ensure test environment variables
os.environ["ENVIRONMENT"] = "testing"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["MODEL_CHECKPOINT_PATH"] = "checkpoints/sft/best_checkpoint"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from backend.app.core.config import Settings, get_settings
from backend.app.main import create_app
from backend.app.services.generation import GenerationService
from descra.inference.predictor import PredictionOutput


@pytest.fixture
def mock_prediction_output():
    """Provides a realistic PredictionOutput fixture."""
    return PredictionOutput(
        generated_description="A premium hydrating face cream formulated with organic aloe and hyaluronic acid.",
        input_prompt="Generate a clear and engaging product description. Product Name: Hydrating Cream",
        output_tokens=32,
        latency_ms=185.5,
        device="cuda:0",
        strategy="beam_search_2",
        metadata={"num_beams": 2, "input_token_count": 28},
    )


@pytest.fixture
def mock_predictor(mock_prediction_output):
    """Provides a mocked DescriptionPredictor instance."""
    predictor = MagicMock()
    predictor.device = "cuda:0"
    predictor.checkpoint_path = "checkpoints/sft/best_checkpoint"
    predictor.model = MagicMock()
    predictor.tokenizer = MagicMock()
    predictor.predict.return_value = mock_prediction_output
    return predictor


@pytest.fixture
def client(mock_predictor):
    """Provides a TestClient with mocked predictor in lifespan."""
    with patch("backend.app.main.DescriptionPredictor", return_value=mock_predictor):
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client
