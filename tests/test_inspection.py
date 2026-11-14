"""Unit tests for raw schema inspection utilities."""

import os
import pytest
from descra.data.inspection import (
    RawSchemaInspection,
    generate_inspection_markdown_report,
    inspect_raw_parquet,
)


def test_inspect_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        inspect_raw_parquet("data/raw/non_existent_file.parquet")


def test_inspect_real_raw_parquet():
    raw_path = "data/raw/meta_All_Beauty.parquet"
    if not os.path.exists(raw_path):
        pytest.skip(f"{raw_path} not downloaded yet.")

    insp = inspect_raw_parquet(raw_path, sample_size=100)
    assert isinstance(insp, RawSchemaInspection)
    assert insp.total_records > 0
    assert len(insp.column_names) > 0
    assert "title" in insp.column_names
    assert "features" in insp.column_names
    assert "description" in insp.column_names

    md = generate_inspection_markdown_report(insp)
    assert "# DESCЯA — RAW DATASET SCHEMA INSPECTION" in md

