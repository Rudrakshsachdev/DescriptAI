"""Automated test suite for Step 9: Held-Out Test Evaluation & Statistical Verification."""

import json
from pathlib import Path
import pytest

from descra.evaluation.failure_analysis import aggregate_failure_statistics, classify_generation_failure
from descra.evaluation.statistical import compute_paired_bootstrap_ci
from descra.evaluation.test_integrity import compute_file_sha256, verify_test_dataset


def test_test_dataset_integrity():
    """Verifies that the canonical held-out test split passes all 10 integrity checks."""
    report = verify_test_dataset(
        test_path="data/processed/test.jsonl",
        train_path="data/processed/train.jsonl",
        val_path="data/processed/validation.jsonl",
        output_report_path="reports/evaluation/test_dataset_integrity.json",
    )

    assert report["total_valid_records"] == 480
    assert report["unique_product_ids_count"] == 480
    assert report["all_product_ids_unique"] is True
    assert report["malformed_lines_count"] == 0
    assert report["missing_references_count"] == 0
    assert report["missing_titles_count"] == 0
    assert report["train_set_overlap_count"] == 0
    assert report["val_set_overlap_count"] == 0
    assert report["test_split_isolated_from_training"] is True
    assert report["overall_integrity_passed"] is True


def test_paired_bootstrap_ci():
    """Tests the paired bootstrap confidence interval calculation for correctness and determinism."""
    base_scores = [0.2, 0.3, 0.4, 0.5, 0.6]
    ft_scores = [0.3, 0.4, 0.5, 0.6, 0.7]

    res1 = compute_paired_bootstrap_ci(base_scores, ft_scores, n_bootstrap=100, seed=42)
    res2 = compute_paired_bootstrap_ci(base_scores, ft_scores, n_bootstrap=100, seed=42)

    assert res1["mean_difference"] == pytest.approx(0.1, abs=1e-4)
    assert res1["ci_lower"] == pytest.approx(res2["ci_lower"], abs=1e-5)
    assert res1["ci_upper"] == pytest.approx(res2["ci_upper"], abs=1e-5)
    assert res1["statistically_conclusive"] is True
    assert res1["ci_lower"] > 0.0


def test_failure_classification():
    """Tests failure classification rules for empty outputs, brand collapse, and repetitions."""
    # 1. Empty Output
    f_empty = classify_generation_failure("P1", "Product", "prompt", "")
    assert any(f["category"] == "EMPTY_OUTPUT" for f in f_empty)

    # 2. Brand Only
    f_brand = classify_generation_failure("P2", "Brand Name", "prompt", "Brand Name")
    assert any(f["category"] == "BRAND_OR_EXTREMELY_SHORT" for f in f_brand)

    # 3. Repetitive Loop
    rep_text = "great item for face great item for face great item for face great item for face"
    f_rep = classify_generation_failure("P3", "Product", "prompt", rep_text)
    assert any(f["category"] == "REPETITIVE_OUTPUT" for f in f_rep)

    # 4. Aggregator
    agg = aggregate_failure_statistics([{"failures": f_empty}, {"failures": f_brand}], 2)
    assert agg["total_evaluations"] == 2
    assert agg["total_failures_detected"] == 2
    assert "EMPTY_OUTPUT" in agg["category_counts"]
