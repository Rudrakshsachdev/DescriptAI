"""Audit and independent verification script for Descra EDA and factual alignment.

Audits dataset integrity, factual alignment discrepancies, token distributions,
cross-split leakage, and generates audit reports in reports/audit/.
"""

import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Set, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

# Ensure src in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from descra.data.dataset import format_model_input
from descra.data.preprocessing import clean_text, verify_factual_alignment
from descra.data.splitting import detect_data_leakage
from descra.utils.logging import get_logger

logger = get_logger("descra.audit")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def compute_jaccard_similarity(text1: str, text2: str) -> float:
    t1 = set(clean_text(text1).lower().split())
    t2 = set(clean_text(text2).lower().split())
    union = t1.union(t2)
    if not union:
        return 0.0
    return len(t1.intersection(t2)) / len(union)

def run_audit(
    data_dir: str = "data/processed",
    reports_dir: str = "reports/audit",
    tokenizer_name: str = "google/flan-t5-small",
) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("DESCRA — FACTUAL ALIGNMENT AUDIT & EDA VERIFICATION")
    logger.info("=" * 60)

    out_path = Path(reports_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load all splits
    train_recs = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val_recs = load_jsonl(os.path.join(data_dir, "validation.jsonl"))
    test_recs = load_jsonl(os.path.join(data_dir, "test.jsonl"))
    review_recs = load_jsonl(os.path.join(data_dir, "needs_review.jsonl"))

    all_accepted = train_recs + val_recs + test_recs
    total_accepted = len(all_accepted)
    total_review = len(review_recs)

    # 2. Dataset Integrity Audit
    train_asins = {r["product_id"] for r in train_recs}
    val_asins = {r["product_id"] for r in val_recs}
    test_asins = {r["product_id"] for r in test_recs}
    all_asins = [r["product_id"] for r in all_accepted]
    unique_asins = set(all_asins)

    train_descs = {clean_text(r["reference_description"]).lower() for r in train_recs}
    val_descs = {clean_text(r["reference_description"]).lower() for r in val_recs}
    test_descs = {clean_text(r["reference_description"]).lower() for r in test_recs}
    all_descs = [clean_text(r["reference_description"]).lower() for r in all_accepted]
    unique_descs = set(all_descs)

    asin_overlap_tv = len(train_asins.intersection(val_asins))
    asin_overlap_tt = len(train_asins.intersection(test_asins))
    asin_overlap_vt = len(val_asins.intersection(test_asins))

    desc_overlap_tv = len(train_descs.intersection(val_descs))
    desc_overlap_tt = len(train_descs.intersection(test_descs))
    desc_overlap_vt = len(val_descs.intersection(test_descs))

    leakage_result = detect_data_leakage(train_recs, val_recs, test_recs)

    integrity_audit = {
        "total_accepted_records": total_accepted,
        "total_review_records": total_review,
        "split_counts": {
            "train": len(train_recs),
            "validation": len(val_recs),
            "test": len(test_recs),
        },
        "unique_product_ids": len(unique_asins),
        "product_id_uniqueness_rate": round(len(unique_asins) / total_accepted, 4),
        "unique_descriptions": len(unique_descs),
        "description_uniqueness_rate": round(len(unique_descs) / total_accepted, 4),
        "product_id_leakage": {
            "train_val": asin_overlap_tv,
            "train_test": asin_overlap_tt,
            "val_test": asin_overlap_vt,
        },
        "description_leakage": {
            "train_val": desc_overlap_tv,
            "train_test": desc_overlap_tt,
            "val_test": desc_overlap_vt,
        },
        "zero_leakage_verified": not leakage_result["has_leakage"],
        "missing_values": {
            "product_id": sum(1 for r in all_accepted if not r.get("product_id")),
            "product_name": sum(1 for r in all_accepted if not r.get("product_name")),
            "features": sum(1 for r in all_accepted if not r.get("features")),
            "reference_description": sum(1 for r in all_accepted if not r.get("reference_description")),
        }
    }

    # 3. Input Features and Specifications Statistics
    feat_counts = [len(r.get("features", [])) for r in all_accepted]
    spec_counts = [len(r.get("specifications", {})) for r in all_accepted]
    has_specs = [1 if len(r.get("specifications", {})) > 0 else 0 for r in all_accepted]

    input_stats = {
        "bullet_points": {
            "mean": round(float(np.mean(feat_counts)), 2),
            "std": round(float(np.std(feat_counts)), 2),
            "median": int(np.median(feat_counts)),
            "min": int(np.min(feat_counts)),
            "max": int(np.max(feat_counts)),
        },
        "specifications": {
            "records_with_specs": int(sum(has_specs)),
            "pct_with_specs": round(float(np.mean(has_specs) * 100), 2),
            "mean_specs_count": round(float(np.mean(spec_counts)), 2),
            "median_specs_count": int(np.median(spec_counts)),
            "max_specs_count": int(np.max(spec_counts)),
        }
    }

    # 4. Target Description Statistics
    desc_words = [len(str(r.get("reference_description", "")).split()) for r in all_accepted]
    desc_chars = [len(str(r.get("reference_description", ""))) for r in all_accepted]

    desc_stats = {
        "word_count": {
            "mean": round(float(np.mean(desc_words)), 2),
            "std": round(float(np.std(desc_words)), 2),
            "median": int(np.median(desc_words)),
            "min": int(np.min(desc_words)),
            "max": int(np.max(desc_words)),
            "p75": int(np.percentile(desc_words, 75)),
            "p90": int(np.percentile(desc_words, 90)),
            "p95": int(np.percentile(desc_words, 95)),
            "p99": int(np.percentile(desc_words, 99)),
        },
        "char_count": {
            "mean": round(float(np.mean(desc_chars)), 2),
            "std": round(float(np.std(desc_chars)), 2),
            "median": int(np.median(desc_chars)),
            "min": int(np.min(desc_chars)),
            "max": int(np.max(desc_chars)),
            "p95": int(np.percentile(desc_chars, 95)),
        }
    }

    # 5. Factual Alignment Investigation
    # Check stored vs recomputed values
    stored_statuses = [r.get("alignment_status") for r in all_accepted]
    stored_scores = [r.get("alignment_score", 0.0) for r in all_accepted]

    recomputed_statuses = []
    recomputed_scores = []
    jaccard_scores = []

    for r in all_accepted:
        status, score = verify_factual_alignment(
            product_name=r.get("product_name", ""),
            features=r.get("features", []),
            specifications=r.get("specifications", {}),
            reference_description=r.get("reference_description", "")
        )
        recomputed_statuses.append(status)
        recomputed_scores.append(score)

        # Pure Jaccard similarity (without numerical penalty or stopword stripping)
        source_txt = f"{r.get('product_name', '')} {' '.join(r.get('features', []))} {' '.join(f'{k} {v}' for k, v in r.get('specifications', {}).items())}"
        jacc = compute_jaccard_similarity(source_txt, r.get("reference_description", ""))
        jaccard_scores.append(jacc)

    from collections import Counter
    stored_counter = Counter(stored_statuses)
    recomp_counter = Counter(recomputed_statuses)

    alignment_audit = {
        "stored_in_dataset_jsonl": {
            "ALIGNED": stored_counter.get("ALIGNED", 0),
            "PARTIALLY_ALIGNED": stored_counter.get("PARTIALLY_ALIGNED", 0),
            "NEEDS_REVIEW": stored_counter.get("NEEDS_REVIEW", 0),
            "INSUFFICIENT_INPUT": stored_counter.get("INSUFFICIENT_INPUT", 0),
            "total": len(stored_statuses)
        },
        "recomputed_with_current_pipeline": {
            "ALIGNED": recomp_counter.get("ALIGNED", 0),
            "PARTIALLY_ALIGNED": recomp_counter.get("PARTIALLY_ALIGNED", 0),
            "NEEDS_REVIEW": recomp_counter.get("NEEDS_REVIEW", 0),
            "INSUFFICIENT_INPUT": recomp_counter.get("INSUFFICIENT_INPUT", 0),
            "total": len(recomputed_statuses)
        },
        "score_distribution": {
            "mean": round(float(np.mean(stored_scores)), 3),
            "median": round(float(np.median(stored_scores)), 3),
            "std": round(float(np.std(stored_scores)), 3),
            "min": round(float(np.min(stored_scores)), 3),
            "max": round(float(np.max(stored_scores)), 3),
            "p25": round(float(np.percentile(stored_scores, 25)), 3),
            "p75": round(float(np.percentile(stored_scores, 75)), 3),
            "p90": round(float(np.percentile(stored_scores, 90)), 3),
        },
        "pure_jaccard_similarity": {
            "mean": round(float(np.mean(jaccard_scores)), 3),
            "median": round(float(np.median(jaccard_scores)), 3),
            "std": round(float(np.std(jaccard_scores)), 3),
            "min": round(float(np.min(jaccard_scores)), 3),
            "max": round(float(np.max(jaccard_scores)), 3),
            "count_ge_0_20": int(sum(1 for j in jaccard_scores if j >= 0.20)),
            "count_ge_0_35": int(sum(1 for j in jaccard_scores if j >= 0.35)),
            "count_ge_0_50": int(sum(1 for j in jaccard_scores if j >= 0.50)),
        },
        "hypothetical_threshold_analysis": {
            "if_aligned_threshold_was_0_50": int(sum(1 for s in stored_scores if s >= 0.50)),
            "if_aligned_threshold_was_0_63": int(sum(1 for s in stored_scores if s >= 0.63)),
            "canonical_threshold_0_35": int(sum(1 for s in stored_scores if s >= 0.35)),
        }
    }

    # 6. FLAN-T5 Tokenization Verification
    logger.info("Verifying FLAN-T5 subword tokenization...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size

    inp_lens = []
    tgt_lens = []

    for r in all_accepted:
        prompt = format_model_input(
            product_name=r.get("product_name", ""),
            category=r.get("category", "General"),
            features=r.get("features", []),
            specifications=r.get("specifications", {})
        )
        target = r.get("reference_description", "")

        inp_tokens = len(tokenizer.encode(prompt, truncation=False))
        tgt_tokens = len(tokenizer.encode(target, truncation=False))

        inp_lens.append(inp_tokens)
        tgt_lens.append(tgt_tokens)

    inp_arr = np.array(inp_lens)
    tgt_arr = np.array(tgt_lens)

    token_audit = {
        "tokenizer_name": tokenizer_name,
        "vocabulary_size": vocab_size,
        "input_prompt_tokens": {
            "mean": round(float(np.mean(inp_arr)), 2),
            "std": round(float(np.std(inp_arr)), 2),
            "median": int(np.median(inp_arr)),
            "min": int(np.min(inp_arr)),
            "max": int(np.max(inp_arr)),
            "p50": int(np.percentile(inp_arr, 50)),
            "p75": int(np.percentile(inp_arr, 75)),
            "p90": int(np.percentile(inp_arr, 90)),
            "p95": int(np.percentile(inp_arr, 95)),
            "p99": int(np.percentile(inp_arr, 99)),
            "pct_exceeding_256": round(float(np.mean(inp_arr > 256) * 100), 2),
            "pct_exceeding_512": round(float(np.mean(inp_arr > 512) * 100), 2),
        },
        "target_description_tokens": {
            "mean": round(float(np.mean(tgt_arr)), 2),
            "std": round(float(np.std(tgt_arr)), 2),
            "median": int(np.median(tgt_arr)),
            "min": int(np.min(tgt_arr)),
            "max": int(np.max(tgt_arr)),
            "p50": int(np.percentile(tgt_arr, 50)),
            "p75": int(np.percentile(tgt_arr, 75)),
            "p90": int(np.percentile(tgt_arr, 90)),
            "p95": int(np.percentile(tgt_arr, 95)),
            "p99": int(np.percentile(tgt_arr, 99)),
            "pct_exceeding_128": round(float(np.mean(tgt_arr > 128) * 100), 2),
            "pct_exceeding_256": round(float(np.mean(tgt_arr > 256) * 100), 2),
        }
    }

    # 7. Split Distribution Comparison
    def split_metrics(records):
        fw = [len(r.get("features", [])) for r in records]
        dw = [len(str(r.get("reference_description", "")).split()) for r in records]
        al = [r.get("alignment_score", 0.0) for r in records]
        return {
            "count": len(records),
            "feature_mean": round(float(np.mean(fw)), 2),
            "desc_words_mean": round(float(np.mean(dw)), 2),
            "desc_words_median": int(np.median(dw)),
            "alignment_mean": round(float(np.mean(al)), 3),
            "pct_aligned": round(float(sum(1 for r in records if r.get("alignment_status") == "ALIGNED") / len(records) * 100), 1)
        }

    split_comparison = {
        "train": split_metrics(train_recs),
        "validation": split_metrics(val_recs),
        "test": split_metrics(test_recs),
    }

    full_audit_results = {
        "dataset_integrity": integrity_audit,
        "input_statistics": input_stats,
        "target_description_statistics": desc_stats,
        "factual_alignment_audit": alignment_audit,
        "tokenization_audit": token_audit,
        "split_comparison": split_comparison,
    }

    # Export ALIGNMENT_COMPARISON.json
    comp_json_path = out_path / "ALIGNMENT_COMPARISON.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(full_audit_results, f, indent=2)
    logger.info(f"Audit results JSON saved to: {comp_json_path}")

    return full_audit_results

if __name__ == "__main__":
    results = run_audit()
    print("Audit executed successfully.")
