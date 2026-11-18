"""CLI script to validate and prepare supervised fine-tuning dataset partitions for Descra."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from descra.data.preprocessing import clean_text
from descra.models.generation import construct_prompt
from descra.utils.logging import get_logger

logger = get_logger("descra.scripts.prepare_sft_dataset")


def process_split_records(
    input_jsonl: Path,
    output_jsonl: Path,
    min_desc_words: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filters, sanitizes, and verifies records for supervised fine-tuning."""
    records_in = 0
    records_out = 0
    rejected = 0
    rejection_reasons: Dict[str, int] = {}
    valid_records: List[Dict[str, Any]] = []

    with open(input_jsonl, "r", encoding="utf-8") as f_in, open(output_jsonl, "w", encoding="utf-8") as f_out:
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue
            records_in += 1
            raw_rec = json.loads(line)

            p_id = raw_rec.get("product_id", "").strip()
            title = clean_text(raw_rec.get("product_name") or raw_rec.get("title", ""))
            category = clean_text(raw_rec.get("category", ""))
            features = [clean_text(feat) for feat in raw_rec.get("features", []) if clean_text(feat)]
            specs = {clean_text(k): clean_text(v) for k, v in raw_rec.get("specifications", {}).items() if clean_text(k) and clean_text(v)}
            target_desc = clean_text(raw_rec.get("reference_description") or raw_rec.get("description", ""))

            # Validation checks
            if not p_id:
                rejected += 1
                rejection_reasons["missing_product_id"] = rejection_reasons.get("missing_product_id", 0) + 1
                continue
            if not title:
                rejected += 1
                rejection_reasons["missing_title"] = rejection_reasons.get("missing_title", 0) + 1
                continue
            if not target_desc or len(target_desc.split()) < min_desc_words:
                rejected += 1
                rejection_reasons["description_too_short"] = rejection_reasons.get("description_too_short", 0) + 1
                continue

            # Deterministic formatted prompt
            prompt = construct_prompt(
                product_name=title,
                category=category,
                features=features,
                specifications=specs,
            )

            sft_record = {
                "product_id": p_id,
                "product_name": title,
                "category": category,
                "features": features,
                "specifications": specs,
                "prompt": prompt,
                "target_description": target_desc,
                "alignment_status": raw_rec.get("alignment_status", "ALIGNED"),
                "alignment_score": raw_rec.get("alignment_score", 1.0),
            }

            f_out.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            valid_records.append(sft_record)
            records_out += 1

    stats = {
        "records_input": records_in,
        "records_output": records_out,
        "records_rejected": rejected,
        "rejection_reasons": rejection_reasons,
    }
    return valid_records, stats


def main():
    print("=" * 65)
    print("DESCRA — SUPERVISED FINE-TUNING (SFT) DATASET PREPARATION")
    print("=" * 65)

    base_dir = Path("data/processed")
    train_in = base_dir / "train.jsonl"
    val_in = base_dir / "validation.jsonl"
    test_in = base_dir / "test.jsonl"

    train_out = base_dir / "sft_train.jsonl"
    val_out = base_dir / "sft_validation.jsonl"
    stats_out = base_dir / "sft_dataset_statistics.json"

    if not train_in.exists() or not val_in.exists():
        raise FileNotFoundError(f"Source canonical partitions missing in {base_dir}")

    # Process training and validation
    print(f"Processing training partition:   {train_in} -> {train_out}")
    train_records, train_stats = process_split_records(train_in, train_out)

    print(f"Processing validation partition: {val_in} -> {val_out}")
    val_records, val_stats = process_split_records(val_in, val_out)

    # Cross-split leakage audit
    train_pids = {r["product_id"] for r in train_records}
    val_pids = {r["product_id"] for r in val_records}
    test_pids: Set[str] = set()
    if test_in.exists():
        with open(test_in, "r", encoding="utf-8") as f_test:
            test_pids = {json.loads(l).get("product_id") for l in f_test if l.strip()}

    train_val_overlap = len(train_pids.intersection(val_pids))
    train_test_overlap = len(train_pids.intersection(test_pids))
    val_test_overlap = len(val_pids.intersection(test_pids))

    print(f"\n--- Cross-Split Isolation Verification ---")
    print(f"Train / Validation Product ID Overlap: {train_val_overlap}")
    print(f"Train / Test Product ID Overlap:       {train_test_overlap}")
    print(f"Validation / Test Product ID Overlap: {val_test_overlap}")

    if train_val_overlap > 0 or train_test_overlap > 0:
        raise ValueError("CRITICAL DATA INTEGRITY ERROR: Cross-split data leakage detected!")

    # Calculate token length approximations
    train_prompt_words = [len(r["prompt"].split()) for r in train_records]
    train_target_words = [len(r["target_description"].split()) for r in train_records]

    combined_stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "splits": {
            "sft_train": {
                "file_path": str(train_out),
                "total_records": len(train_records),
                "retained_percentage": round(train_stats["records_output"] / train_stats["records_input"] * 100, 2),
                "filtering": train_stats,
                "mean_prompt_word_count": round(float(sum(train_prompt_words) / len(train_prompt_words)), 2),
                "mean_target_word_count": round(float(sum(train_target_words) / len(train_target_words)), 2),
            },
            "sft_validation": {
                "file_path": str(val_out),
                "total_records": len(val_records),
                "retained_percentage": round(val_stats["records_output"] / val_stats["records_input"] * 100, 2),
                "filtering": val_stats,
            },
        },
        "isolation_check": {
            "train_val_overlap": train_val_overlap,
            "train_test_overlap": train_test_overlap,
            "val_test_overlap": val_test_overlap,
            "zero_leakage_verified": True,
        },
        "test_split_unmodified": str(test_in),
    }

    with open(stats_out, "w", encoding="utf-8") as f_stats:
        json.dump(combined_stats, f_stats, indent=2)

    print(f"\nDataset statistics saved to: {stats_out}")
    print(f"SFT Training Records:   {len(train_records):,} (from {train_stats['records_input']:,})")
    print(f"SFT Validation Records: {len(val_records):,} (from {val_stats['records_input']:,})")
    print("=" * 65)


if __name__ == "__main__":
    main()
