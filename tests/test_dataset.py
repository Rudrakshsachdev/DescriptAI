"""Unit tests for dataset loading, prompt construction, and PyTorch dataset interface."""

import json
import os
import pytest
from descra.data.dataset import (
    ProductDescriptionDataset,
    format_model_input,
    format_model_target,
)


def test_format_model_input():
    p_name = "Sony WH-1000XM5"
    cat = "Electronics"
    feats = ["Noise cancelling", "30-hour battery life"]
    specs = {"Weight": "250g", "Bluetooth": "5.2"}

    prompt = format_model_input(p_name, cat, feats, specs)
    assert "Product Name: Sony WH-1000XM5" in prompt
    assert "Category: Electronics" in prompt
    assert "- Noise cancelling" in prompt
    assert "- 30-hour battery life" in prompt
    assert "- Weight: 250g" in prompt
    assert "Product Description:" in prompt


def test_product_description_dataset_loads(tmp_path):
    jsonl_file = tmp_path / "test_data.jsonl"
    records = [
        {
            "product_id": f"P_{i}",
            "product_name": f"Product {i}",
            "category": "Electronics",
            "features": [f"Feature {i}"],
            "specifications": {"spec": f"val_{i}"},
            "reference_description": f"Description for product {i}.",
        }
        for i in range(10)
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ds = ProductDescriptionDataset(str(jsonl_file))
    assert len(ds) == 10
    item = ds[0]
    assert item["product_id"] == "P_0"
    assert "Product Name: Product 0" in item["input_text"]
    assert "Description for product 0." in item["target_text"]


def test_product_description_dataset_with_tokenizer(tmp_path):
    from transformers import AutoTokenizer

    jsonl_file = tmp_path / "test_data.jsonl"
    record = {
        "product_id": "P_001",
        "product_name": "Wireless Mouse",
        "category": "Electronics",
        "features": ["2.4 GHz wireless", "Ergonomic design"],
        "specifications": {"DPI": "1600"},
        "reference_description": "Ergonomic wireless mouse with 1600 DPI sensor.",
    }
    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Load FLAN-T5-small tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    ds = ProductDescriptionDataset(str(jsonl_file), tokenizer=tokenizer, max_input_length=64, max_target_length=32)

    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape[0] == 64
    assert item["labels"].shape[0] == 32
    # Verify label padding replaced with -100
    assert -100 in item["labels"]

