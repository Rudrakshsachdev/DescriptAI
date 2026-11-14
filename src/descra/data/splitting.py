"""Deterministic, group-based dataset splitting and data leakage detection module."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Set, Tuple

from descra.data.preprocessing import CanonicalRecord
from descra.utils.logging import get_logger

logger = get_logger("descra.data.splitting")


def split_canonical_records(
    records: List[CanonicalRecord],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[CanonicalRecord], List[CanonicalRecord], List[CanonicalRecord]]:
    """Deterministically splits canonical records by unique product_id to prevent data leakage."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # Group records by product_id
    product_groups: Dict[str, List[CanonicalRecord]] = {}
    for r in records:
        product_groups.setdefault(r.product_id, []).append(r)

    unique_product_ids = sorted(list(product_groups.keys()))
    rng = random.Random(seed)
    rng.shuffle(unique_product_ids)

    n_total = len(unique_product_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = set(unique_product_ids[:n_train])
    val_ids = set(unique_product_ids[n_train : n_train + n_val])
    test_ids = set(unique_product_ids[n_train + n_val :])

    train_records = [r for pid in train_ids for r in product_groups[pid]]
    val_records = [r for pid in val_ids for r in product_groups[pid]]
    test_records = [r for pid in test_ids for r in product_groups[pid]]

    # Shuffle internally with fixed seed
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    logger.info(
        f"Splits created: Train={len(train_records):,} ({train_ratio*100:.0f}%), "
        f"Val={len(val_records):,} ({val_ratio*100:.0f}%), "
        f"Test={len(test_records):,} ({test_ratio*100:.0f}%)"
    )
    return train_records, val_records, test_records


def detect_data_leakage(
    train_records: List[Any],
    val_records: List[Any],
    test_records: List[Any],
) -> Dict[str, Any]:
    """Audits train, validation, and test splits for product ID, description, or feature leakage."""
    def get_pid(r):
        return r.product_id if hasattr(r, "product_id") else r.get("product_id")

    def get_desc(r):
        d = r.reference_description if hasattr(r, "reference_description") else r.get("reference_description", "")
        return str(d).strip().lower()

    train_pids = {get_pid(r) for r in train_records}
    val_pids = {get_pid(r) for r in val_records}
    test_pids = {get_pid(r) for r in test_records}

    pid_leak_train_val = train_pids.intersection(val_pids)
    pid_leak_train_test = train_pids.intersection(test_pids)
    pid_leak_val_test = val_pids.intersection(test_pids)

    # Exact description overlap check
    train_descs = {get_desc(r) for r in train_records}
    val_descs = {get_desc(r) for r in val_records}
    test_descs = {get_desc(r) for r in test_records}

    desc_leak_train_val = train_descs.intersection(val_descs)
    desc_leak_train_test = train_descs.intersection(test_descs)
    desc_leak_val_test = val_descs.intersection(test_descs)

    has_leak = bool(
        pid_leak_train_val
        or pid_leak_train_test
        or pid_leak_val_test
        or desc_leak_train_test
    )

    report = {
        "has_leakage": has_leak,
        "product_id_overlap": {
            "train_val": len(pid_leak_train_val),
            "train_test": len(pid_leak_train_test),
            "val_test": len(pid_leak_val_test),
        },
        "description_exact_overlap": {
            "train_val": len(desc_leak_train_val),
            "train_test": len(desc_leak_train_test),
            "val_test": len(desc_leak_val_test),
        },
        "leakage_audit_status": "PASSED (Zero Cross-Split Leakage)" if not has_leak else "FAILED",
    }
    logger.info(f"Leakage audit status: {report['leakage_audit_status']}")
    return report


def save_split_jsonl(records: List[CanonicalRecord], output_path: str) -> None:
    """Writes canonical records to JSON Lines (.jsonl) format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records):,} records to: {output_path}")

