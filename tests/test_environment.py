"""Automated tests for environment isolation, PyTorch, Transformers, and CUDA acceleration."""

import sys
import pytest


def test_python_environment_isolation():
    """Verify that Python is running inside the isolated .venv environment."""
    assert sys.prefix != sys.base_prefix, "Tests must run inside an isolated virtual environment!"
    assert sys.version_info.major == 3
    assert sys.version_info.minor in [10, 11], f"Supported Python version is 3.10 or 3.11, got {sys.version}"


def test_core_dependencies_importable():
    """Verify that all core NLP, Deep Learning, and evaluation packages are present."""
    import torch
    import transformers
    import datasets
    import accelerate
    import tokenizers
    import sentencepiece
    import pandas
    import sklearn
    import sacrebleu
    import rouge_score
    import bert_score
    import evaluate

    assert torch.__version__ != ""
    assert transformers.__version__ != ""
    assert datasets.__version__ != ""


def test_tensor_computation_and_device():
    """Verify tensor allocation, matrix multiplication, and finite results on CPU or CUDA."""
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Allocate tensors
    a = torch.randn(128, 128, device=device)
    b = torch.randn(128, 128, device=device)

    # Matrix multiplication
    c = torch.matmul(a, b)

    if device.type == "cuda":
        torch.cuda.synchronize()

    assert c.device.type == device.type
    assert c.shape == (128, 128)
    assert bool(torch.isfinite(c).all().item()) is True


def test_cuda_details_if_available():
    """If CUDA is present, verify device properties, memory allocation, and VRAM reporting."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is not available on this host environment.")

    dev_name = torch.cuda.get_device_name(0)
    assert "NVIDIA" in dev_name or "GeForce" in dev_name or "RTX" in dev_name
    props = torch.cuda.get_device_properties(0)
    assert props.total_memory > 1024 * 1024 * 1024  # At least 1 GB VRAM

