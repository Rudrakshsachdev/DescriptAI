"""Unit tests for dataset acquisition logic, hashing, and disk validation."""

import os
import pytest
from descra.data.acquisition import compute_sha256, validate_disk_space


def test_validate_disk_space_success(tmp_path):
    # Should not raise for small required space
    validate_disk_space(str(tmp_path), required_mb=1.0)


def test_validate_disk_space_failure(tmp_path):
    # Requesting an astronomical amount of disk space should raise OSError
    with pytest.raises(OSError, match="Insufficient disk space"):
        validate_disk_space(str(tmp_path), required_mb=999999999.0)


def test_compute_sha256(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Descra Seq2Seq Transformer Test", encoding="utf-8")
    hash_val = compute_sha256(str(test_file))
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64  # SHA256 hex digest length

