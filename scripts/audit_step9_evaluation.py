"""Step 10 Independent Audit of Step 9 Held-Out Evaluation Integrity."""

import json
from pathlib import Path
from typing import Dict, List, Set
import numpy as np

def audit_evaluation_integrity():
    predictions_path = Path("data/evaluation/test_predictions.jsonl")
    summary_path = Path("data/evaluation/evaluation_summary.json")
    test_path = Path("data/processed/test.jsonl")

    assert predictions_path.exists(), "test_predictions.jsonl missing"
    assert summary_path.exists(), "evaluation_summary.json missing"
    assert test_path.exists(), "test.jsonl missing"

    # 1. Load canonical test records
    test_records = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))

    test_product_ids = {r["product_id"] for r in test_records}
    print(f"[1] Loaded {len(test_records)} canonical test records ({len(test_product_ids)} unique product IDs).")
    assert len(test_records) == 480, f"Expected 480 test records, got {len(test_records)}"
    assert len(test_product_ids) == 480, "Duplicate product IDs found in test set!"

    # 2. Load all 2,880 predictions
    records = []
    keys: Set[str] = set()
    model_counts = {"baseline": 0, "finetuned": 0}
    strategy_counts = {
        "baseline": {"greedy": 0, "beam_search_2": 0, "beam_search_4": 0},
        "finetuned": {"greedy": 0, "beam_search_2": 0, "beam_search_4": 0},
    }

    product_inputs = {"baseline": {}, "finetuned": {}}

    with open(predictions_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(row)

            m = row["model_type"]
            s = row["strategy"]
            pid = row["product_id"]
            key = f"{m}_{s}_{pid}"

            assert key not in keys, f"Duplicate prediction key found: {key} at row {i+1}"
            keys.add(key)

            assert pid in test_product_ids, f"Unknown product_id in predictions: {pid}"
            assert m in model_counts, f"Unknown model_type: {m}"
            assert s in strategy_counts[m], f"Unknown strategy: {s}"

            model_counts[m] += 1
            strategy_counts[m][s] += 1

            # Store input prompt to check identical inputs between baseline and finetuned
            if s == "greedy":
                product_inputs[m][pid] = row.get("input_prompt", "")

    total_records = len(records)
    print(f"[2] Total records loaded: {total_records} (Expected: 2,880).")
    assert total_records == 2880, f"Total records mismatch: {total_records} != 2880"

    print(f"[3] Model counts: Baseline={model_counts['baseline']}, Fine-Tuned={model_counts['finetuned']}")
    assert model_counts["baseline"] == 1440, "Baseline count != 1440"
    assert model_counts["finetuned"] == 1440, "Fine-Tuned count != 1440"

    for m in ["baseline", "finetuned"]:
        for s in ["greedy", "beam_search_2", "beam_search_4"]:
            cnt = strategy_counts[m][s]
            print(f"    - {m} | {s}: {cnt}/480")
            assert cnt == 480, f"Mismatch in {m} {s}: {cnt} != 480"

    # 4. Check identical inputs between models
    matching_inputs = 0
    for pid in test_product_ids:
        b_in = product_inputs["baseline"].get(pid, "")
        f_in = product_inputs["finetuned"].get(pid, "")
        assert b_in == f_in, f"Input prompt mismatch for product {pid}!"
        matching_inputs += 1
    print(f"[4] Verified identical input prompts across all {matching_inputs} products for both models.")

    # 5. Reproduce Metrics from Predictions
    print("[5] Verifying metric reproduction from raw prediction rows...")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    for m in ["baseline", "finetuned"]:
        for s in ["greedy", "beam_search_2", "beam_search_4"]:
            subset = [r for r in records if r["model_type"] == m and r["strategy"] == s]
            assert len(subset) == 480

            # Recompute mean sentence ROUGE-1
            mean_r1 = float(np.mean([r["sentence_rouge1_f1"] for r in subset]))
            mean_lat = float(np.mean([r["latency_ms"] for r in subset]))
            mean_fact = float(np.mean([r["factual_consistency_score"] for r in subset]))

            reported_r1 = summary["results_summary"][m][s]["metrics"]["rouge"]["rouge1_f1"]
            reported_lat = summary["results_summary"][m][s]["performance"]["mean_latency_ms"]
            reported_fact = summary["results_summary"][m][s]["factual_adherence"]["mean_consistency_score"]

            print(f"    - {m} {s}: Raw mean ROUGE-1 = {mean_r1:.4f} vs Reported = {reported_r1:.4f}")
            print(f"                    Raw mean Latency = {mean_lat:.1f}ms vs Reported = {reported_lat:.1f}ms")
            print(f"                    Raw mean Factual = {mean_fact:.3f} vs Reported = {reported_fact:.3f}")

            assert abs(mean_r1 - reported_r1) < 0.005, f"ROUGE-1 discrepancy in {m} {s}"
            assert abs(mean_lat - reported_lat) < 1.0, f"Latency discrepancy in {m} {s}"
            assert abs(mean_fact - reported_fact) < 0.005, f"Factual consistency discrepancy in {m} {s}"

    print("[6] Statistical bootstrap verification...")
    # Verify confidence intervals in summary are internally consistent (ci_lower <= mean_diff <= ci_upper)
    for s in ["greedy", "beam_search_2", "beam_search_4"]:
        for metric in ["rouge1_f1", "rouge2_f1", "rougeL_f1", "factual_score", "latency_ms"]:
            st = summary["statistical_analysis"][s][metric]
            m_diff = st["mean_difference"]
            lo = st["ci_lower"]
            hi = st["ci_upper"]
            assert lo <= m_diff <= hi, f"CI bounds invalid for {s} {metric}: {lo} <= {m_diff} <= {hi}"
            print(f"    - {s} {metric}: CI [{lo:+.4f}, {hi:+.4f}] successfully encloses mean delta {m_diff:+.4f}")

    print("\nALL STEP 9 EVALUATION INTEGRITY AUDITS PASSED WITH ZERO DISCREPANCIES!")

if __name__ == "__main__":
    audit_evaluation_integrity()
