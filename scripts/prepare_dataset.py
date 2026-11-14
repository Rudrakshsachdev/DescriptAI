"""CLI pipeline to preprocess, validate factual alignment, split, and audit dataset for Descra."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from descra.data.preprocessing import (
    generate_preprocessing_markdown_report,
    preprocess_dataset,
)
from descra.data.splitting import (
    detect_data_leakage,
    save_split_jsonl,
    split_canonical_records,
)
from descra.utils.logging import get_logger

logger = get_logger("descra.scripts.prepare_dataset")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and split Descra dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to data configuration YAML.",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default="data/raw/meta_All_Beauty.parquet",
        help="Path to raw source file.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=10000,
        help="Target size for accepted canonical records (default: 10,000).",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("DESCRA — DATASET PREPROCESSING & SPLITTING PIPELINE")
    logger.info("=" * 60)

    # 1. Preprocess & Align
    accepted_records, review_records, stats = preprocess_dataset(
        raw_parquet_path=args.raw_path,
        config=config,
        max_records=args.target_size,
    )

    # 2. Split
    train_ratio = config.get("train_ratio", 0.80)
    val_ratio = config.get("val_ratio", 0.10)
    test_ratio = config.get("test_ratio", 0.10)
    seed = config.get("random_seed", 42)

    train_recs, val_recs, test_recs = split_canonical_records(
        records=accepted_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # 3. Data Leakage Detection
    leakage_report = detect_data_leakage(train_recs, val_recs, test_recs)
    assert not leakage_report["has_leakage"], "CRITICAL: Cross-split data leakage detected!"

    # 4. Save Processed Splits
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    save_split_jsonl(train_recs, str(processed_dir / "train.jsonl"))
    save_split_jsonl(val_recs, str(processed_dir / "validation.jsonl"))
    save_split_jsonl(test_recs, str(processed_dir / "test.jsonl"))
    save_split_jsonl(review_records, str(processed_dir / "needs_review.jsonl"))

    # 5. Generate Reports
    reports_dir = Path("reports/dataset")
    reports_dir.mkdir(parents=True, exist_ok=True)

    md_report = generate_preprocessing_markdown_report(stats)
    with open(reports_dir / "preprocessing_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    with open(reports_dir / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

    # 6. Generate Dataset Manifest
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Load source manifest if available
    source_meta = {}
    source_manifest_path = manifest_dir / "source_manifest.json"
    if source_manifest_path.exists():
        with open(source_manifest_path, "r", encoding="utf-8") as f:
            source_meta = json.load(f)

    dataset_manifest = {
        "dataset_name": "Descra Product Description Dataset",
        "dataset_version": "v1.0",
        "source_name": source_meta.get("source_name", "UCSD McAuley-Lab Amazon Product Metadata"),
        "source_version": source_meta.get("dataset_version", "Amazon-Reviews-2023"),
        "license": source_meta.get("license_type", "Research & Non-Commercial Educational Use"),
        "attribution": source_meta.get("attribution", ""),
        "raw_source_file": args.raw_path,
        "raw_sha256": source_meta.get("sha256_checksum", ""),
        "preprocessing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": seed,
        "total_raw_records": stats.total_raw_records,
        "accepted_canonical_records": len(accepted_records),
        "split_counts": {
            "train": len(train_recs),
            "validation": len(val_recs),
            "test": len(test_recs),
            "needs_review": len(review_records),
        },
        "split_ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": test_ratio,
        },
        "leakage_verification": leakage_report,
        "filtering_configuration": config.get("filters", {}),
        "data_quality_metrics": {
            "avg_input_features": stats.avg_input_features,
            "avg_description_words": stats.avg_description_words,
            "alignment_breakdown": stats.alignment_breakdown,
        },
    }

    with open(manifest_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("DATASET PREPROCESSING & SPLITTING COMPLETE")
    print("=" * 60)
    print(f"Total Raw Evaluated:        {stats.total_raw_records:,}")
    print(f"Accepted Canonical Records: {len(accepted_records):,}")
    print(f"Training Split (80%):       {len(train_recs):,} records -> data/processed/train.jsonl")
    print(f"Validation Split (10%):     {len(val_recs):,} records -> data/processed/validation.jsonl")
    print(f"Testing Split (10%):        {len(test_recs):,} records -> data/processed/test.jsonl")
    print(f"Review Split:               {len(review_records):,} records -> data/processed/needs_review.jsonl")
    print(f"Data Leakage Audit:         {leakage_report['leakage_audit_status']}")
    print(f"Manifest Generated:         data/manifests/dataset_manifest.json")
    print(f"Report Generated:           reports/dataset/preprocessing_report.md")
    print("=" * 60)


if __name__ == "__main__":
    main()

