"""Unit tests for deterministic splitting and cross-split leakage detection."""

import pytest
from descra.data.preprocessing import CanonicalRecord
from descra.data.splitting import detect_data_leakage, split_canonical_records


def _make_record(idx: int, desc: str) -> CanonicalRecord:
    return CanonicalRecord(
        product_id=f"PROD_{idx:04d}",
        product_name=f"Product Name {idx}",
        category="Electronics",
        features=[f"Feature A for product {idx}", f"Feature B for product {idx}"],
        specifications={"Model": f"V{idx}"},
        reference_description=desc,
        alignment_status="ALIGNED",
        alignment_score=0.9,
    )


def test_split_canonical_records_deterministic():
    records = [_make_record(i, f"Unique reference description for product {i}") for i in range(100)]

    t1, v1, te1 = split_canonical_records(records, 0.8, 0.1, 0.1, seed=42)
    t2, v2, te2 = split_canonical_records(records, 0.8, 0.1, 0.1, seed=42)

    assert len(t1) == 80
    assert len(v1) == 10
    assert len(te1) == 10

    # Determinism
    assert [r.product_id for r in t1] == [r.product_id for r in t2]
    assert [r.product_id for r in te1] == [r.product_id for r in te2]


def test_detect_data_leakage_clean():
    records = [_make_record(i, f"Unique reference description for product {i}") for i in range(50)]
    train, val, test = split_canonical_records(records, 0.8, 0.1, 0.1, seed=42)

    audit = detect_data_leakage(train, val, test)
    assert audit["has_leakage"] is False
    assert audit["product_id_overlap"]["train_test"] == 0
    assert audit["description_exact_overlap"]["train_test"] == 0


def test_detect_data_leakage_flags_leak():
    train = [_make_record(1, "Shared description between train and test")]
    val = [_make_record(2, "Independent validation description")]
    test = [_make_record(3, "Shared description between train and test")]  # Leaked description!

    audit = detect_data_leakage(train, val, test)
    assert audit["has_leakage"] is True
    assert audit["description_exact_overlap"]["train_test"] == 1

