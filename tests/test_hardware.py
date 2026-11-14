"""Tests for hardware inspection and capability profiling."""

import pytest
from descra.utils.hardware import HardwareProfile, inspect_hardware


def test_hardware_inspection_structure():
    profile = inspect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.os_name != ""
    assert profile.python_version != ""
    assert profile.total_ram_gb > 0
    assert profile.available_ram_gb > 0
    assert profile.cpu_physical_cores >= 1
    assert profile.cpu_logical_cores >= 1
    assert profile.disk_total_gb > 0
    assert profile.disk_free_gb > 0
    assert profile.recommended_model in ["google/flan-t5-small", "google/flan-t5-base"]
    assert profile.recommended_batch_size in [2, 4, 8]
    assert profile.recommended_precision in ["fp16", "fp32"]

