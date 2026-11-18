"""Tests for Step 10 Evaluation Audit & Data Verification."""

import json
from pathlib import Path
import pytest


def test_held_out_evaluation_dataset_completeness():
    """Verify that test_predictions.jsonl has 2,880 valid rows covering all 480 test records."""
    pred_path = Path("data/evaluation/test_predictions.jsonl")
    test_path = Path("data/processed/test.jsonl")

    if not pred_path.exists() or not test_path.exists():
        pytest.skip("Evaluation files missing.")

    # Read canonical test IDs
    test_ids = set()
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_ids.add(json.loads(line)["product_id"])

    assert len(test_ids) == 480

    # Audit predictions
    pred_keys = set()
    counts = {
        "baseline": {"greedy": 0, "beam_search_2": 0, "beam_search_4": 0},
        "finetuned": {"greedy": 0, "beam_search_2": 0, "beam_search_4": 0},
    }

    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            m = r["model_type"]
            s = r["strategy"]
            pid = r["product_id"]

            key = f"{m}_{s}_{pid}"
            assert key not in pred_keys, f"Duplicate prediction key: {key}"
            pred_keys.add(key)

            assert pid in test_ids, f"Unknown product_id: {pid}"
            counts[m][s] += 1

    assert len(pred_keys) == 2880
    for m in ["baseline", "finetuned"]:
        for s in ["greedy", "beam_search_2", "beam_search_4"]:
            assert counts[m][s] == 480, f"Expected 480 for {m} {s}, got {counts[m][s]}"


def test_evaluation_summary_matches_schema():
    """Verify evaluation_summary.json schema and statistical consistency."""
    summary_path = Path("data/evaluation/evaluation_summary.json")
    if not summary_path.exists():
        pytest.skip("Summary missing.")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert "metadata" in summary
    assert "results_summary" in summary
    assert "statistical_analysis" in summary
    assert summary["metadata"]["test_record_count"] == 480

    # Verify statistical confidence intervals enclose mean difference
    for s in ["greedy", "beam_search_2", "beam_search_4"]:
        for metric in ["rouge1_f1", "rouge2_f1", "rougeL_f1", "factual_score", "latency_ms"]:
            st = summary["statistical_analysis"][s][metric]
            lo = st["ci_lower"]
            hi = st["ci_upper"]
            diff = st["mean_difference"]
            assert lo <= diff <= hi, f"CI bounds invalid for {s} {metric}: {lo} <= {diff} <= {hi}"
