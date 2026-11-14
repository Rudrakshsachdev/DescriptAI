"""Unit tests for baseline model architecture, prompt formatting, and inference pipeline."""

import pytest
import torch

from descra.models.generation import (
    DEFAULT_GENERATION_CONFIG,
    GREEDY_GENERATION_CONFIG,
    construct_prompt,
)
from descra.models.loader import get_optimal_device


def test_get_optimal_device():
    # Test explicit CPU
    dev_cpu = get_optimal_device("cpu")
    assert dev_cpu.type == "cpu"

    # Test auto selection
    dev_auto = get_optimal_device("auto")
    if torch.cuda.is_available():
        assert dev_auto.type == "cuda"
    else:
        assert dev_auto.type == "cpu"


def test_construct_prompt_complete():
    p_name = "Facial Moisturizer"
    category = "Skincare"
    features = ["Hydrating hyaluronic acid", "24 hour moisture"]
    specs = {"Volume": "50ml", "Skin Type": "Sensitive"}

    prompt = construct_prompt(p_name, category, features, specs)

    assert "Product Name: Facial Moisturizer" in prompt
    assert "Category: Skincare" in prompt
    assert "Features:" in prompt
    assert "- Hydrating hyaluronic acid" in prompt
    assert "Specifications:" in prompt
    assert "- Volume: 50ml" in prompt
    assert "Write a concise product description." in prompt


def test_construct_prompt_missing_optional_fields():
    # Only product name provided
    p_name = "Basic Lip Gloss"
    prompt = construct_prompt(p_name)

    assert "Product Name: Basic Lip Gloss" in prompt
    assert "Category:" not in prompt
    assert "Features:" not in prompt
    assert "Specifications:" not in prompt
    assert "Write a concise product description." in prompt


def test_construct_prompt_general_category_omitted():
    # General / All Beauty categories should be omitted to reduce prompt noise
    prompt = construct_prompt("Shampoo", category="All Beauty")
    assert "Category: All Beauty" not in prompt


def test_generation_config_structures():
    assert DEFAULT_GENERATION_CONFIG["num_beams"] == 4
    assert DEFAULT_GENERATION_CONFIG["no_repeat_ngram_size"] == 3
    assert DEFAULT_GENERATION_CONFIG["max_new_tokens"] == 128

    assert GREEDY_GENERATION_CONFIG["num_beams"] == 1
    assert GREEDY_GENERATION_CONFIG["do_sample"] is False

