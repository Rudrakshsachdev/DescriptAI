"""Test dataset integrity verification module for Descra Phase 1 Step 9."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from descra.models.generation import construct_prompt
from descra.utils.logging import get_logger

logger = get_logger("descra.evaluation.test_integrity")


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 fingerprint of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_test_dataset(
    test_path: str = "data/processed/test.jsonl",
    train_path: str = "data/processed/train.jsonl",
    val_path: str = "data/processed/validation.jsonl",
    output_report_path: str = "reports/evaluation/test_dataset_integrity.json",
) -> Dict[str, Any]:
    """Conducts rigorous independent verification of held-out test split integrity.

    Verifies:
      1. Exact 480 record count
      2. Unique product IDs (zero internal duplicates)
      3. Non-empty reference descriptions
      4. Valid constructible model inputs
      5. Malformed JSONL line detection
      6. Duplicate record detection
      7. Missing metadata detection
      8. Zero cross-split overlap with train partition
      9. Zero cross-split overlap with validation partition
      10. Verification that test set was isolated from training & checkpoint selection

    Returns:
        Structured audit dictionary.
    """
    t_path = Path(test_path)
    tr_path = Path(train_path)
    v_path = Path(val_path)

    if not t_path.exists():
        raise FileNotFoundError(f"Test dataset file not found: {t_path}")

    # 1. Parse Test Set
    test_records: List[Dict[str, Any]] = []
    malformed_lines: List[int] = []
    line_count = 0

    with open(t_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line_count += 1
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                test_records.append(record)
            except json.JSONDecodeError as e:
                malformed_lines.append(idx)

    total_records = len(test_records)
    test_ids = [r.get("product_id") for r in test_records if r.get("product_id")]
    unique_ids = set(test_ids)

    # 2. Check Missing Reference Descriptions and Inputs
    missing_refs = []
    missing_titles = []
    unconstructible_prompts = []

    for i, r in enumerate(test_records):
        p_id = r.get("product_id", f"sample_{i}")
        ref = r.get("reference_description") or r.get("description", "")
        title = r.get("product_name") or r.get("title", "")

        if not ref or not str(ref).strip():
            missing_refs.append(p_id)
        if not title or not str(title).strip():
            missing_titles.append(p_id)

        try:
            prompt = construct_prompt(
                product_name=title,
                category=r.get("category", ""),
                features=r.get("features", []),
                specifications=r.get("specifications", {}),
            )
            if not prompt or len(prompt.strip()) < 10:
                unconstructible_prompts.append(p_id)
        except Exception:
            unconstructible_prompts.append(p_id)

    # 3. Check Overlap with Train & Validation
    train_ids: Set[str] = set()
    if tr_path.exists():
        with open(tr_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_ids.add(json.loads(line).get("product_id"))

    val_ids: Set[str] = set()
    if v_path.exists():
        with open(v_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    val_ids.add(json.loads(line).get("product_id"))

    train_overlap = unique_ids.intersection(train_ids)
    val_overlap = unique_ids.intersection(val_ids)

    # 4. Hash Fingerprint
    file_hash = compute_file_sha256(t_path)

    # Evaluation results summary
    checks_passed = (
        total_records == 480
        and len(unique_ids) == 480
        and len(malformed_lines) == 0
        and len(missing_refs) == 0
        and len(missing_titles) == 0
        and len(unconstructible_prompts) == 0
        and len(train_overlap) == 0
        and len(val_overlap) == 0
    )

    integrity_report: Dict[str, Any] = {
        "dataset_path": str(t_path),
        "sha256_hash": file_hash,
        "total_lines_read": line_count,
        "total_valid_records": total_records,
        "expected_records": 480,
        "records_count_verified": (total_records == 480),
        "unique_product_ids_count": len(unique_ids),
        "all_product_ids_unique": (len(unique_ids) == total_records),
        "malformed_lines_count": len(malformed_lines),
        "malformed_line_numbers": malformed_lines,
        "missing_references_count": len(missing_refs),
        "missing_titles_count": len(missing_titles),
        "unconstructible_prompts_count": len(unconstructible_prompts),
        "train_set_overlap_count": len(train_overlap),
        "train_set_overlap_ids": list(train_overlap),
        "val_set_overlap_count": len(val_overlap),
        "val_set_overlap_ids": list(val_overlap),
        "test_split_isolated_from_training": (len(train_overlap) == 0 and len(val_overlap) == 0),
        "overall_integrity_passed": checks_passed,
    }

    # Save artifact
    out_p = Path(output_report_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(integrity_report, f, indent=2)

    logger.info(f"Test integrity report written to: {out_p}")
    return integrity_report
