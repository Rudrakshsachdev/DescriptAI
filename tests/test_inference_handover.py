"""Tests for Step 10 Model Handover & Production Inference Engine."""

from pathlib import Path
import pytest
import torch

from descra.inference.predictor import DescriptionPredictor, GenerationConfig, PredictionOutput


def test_predictor_prompt_construction():
    """Verify deterministic prompt construction."""
    prompt = DescriptionPredictor.construct_prompt(
        product_name="Hydrating Facial Serum",
        features=["Hyaluronic Acid", "Fragrance-Free", "30ml"],
        category="Skin Care",
        brand="GlowLab",
    )
    assert "Product Name: Hydrating Facial Serum" in prompt
    assert "Category: Skin Care" in prompt
    assert "Brand: GlowLab" in prompt
    assert "Features: - Hyaluronic Acid - Fragrance-Free - 30ml" in prompt


def test_predictor_loads_and_generates():
    """Verify that the fine-tuned checkpoint loads independently and generates descriptions."""
    ckpt_path = Path("checkpoints/sft/best_checkpoint")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint not present on disk.")

    predictor = DescriptionPredictor(checkpoint_path=ckpt_path)
    assert predictor.model is not None
    assert predictor.tokenizer is not None

    # Test single prediction
    output = predictor.predict(
        product_name="Organic Argan Oil Hair Treatment",
        features=["Cold-pressed", "Nourishing", "100ml"],
    )

    assert isinstance(output, PredictionOutput)
    assert len(output.generated_description) > 0
    assert output.output_tokens > 0
    assert output.latency_ms > 0
    assert output.strategy == "beam_search_2"


def test_predictor_custom_generation_config():
    """Verify greedy generation override."""
    ckpt_path = Path("checkpoints/sft/best_checkpoint")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint not present on disk.")

    predictor = DescriptionPredictor(checkpoint_path=ckpt_path)
    greedy_cfg = GenerationConfig(num_beams=1, max_new_tokens=32)

    output = predictor.predict(
        product_name="Aloe Vera Soothing Gel",
        features=["99% Pure Aloe", "Hydrating"],
        config=greedy_cfg,
    )

    assert output.strategy == "greedy"
    assert len(output.generated_description) > 0
