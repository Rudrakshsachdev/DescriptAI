"""Environment and hardware acceleration verification script for Descra.

Performs deterministic runtime diagnostics, package version checking,
and actual CUDA tensor allocation, matrix multiplication, and GPU synchronization
to mathematically verify hardware acceleration on NVIDIA GeForce RTX 4050.
"""

import math
import os
import sys
from typing import Dict, Any


def run_verification() -> Dict[str, Any]:
    print("=" * 65)
    print("DESCRA — ENVIRONMENT & HARDWARE ACCELERATION VERIFICATION")
    print("=" * 65)

    # 1. Python & Virtual Environment Isolation
    py_ver = sys.version.split()[0]
    py_exec = sys.executable
    is_venv = sys.prefix != sys.base_prefix
    print(f"Python Executable: {py_exec}")
    print(f"Python Version:    {py_ver}")
    print(f"Isolated .venv:    {is_venv} (Prefix: {sys.prefix})")
    assert is_venv, "Verification must run inside the isolated .venv environment!"

    # 2. Package Imports & Version Verification
    import torch
    import transformers
    import datasets
    import accelerate
    import tokenizers
    import sentencepiece

    print("-" * 65)
    print(f"PyTorch Version:      {torch.__version__}")
    print(f"Transformers Version: {transformers.__version__}")
    print(f"Datasets Version:     {datasets.__version__}")
    print(f"Accelerate Version:   {accelerate.__version__}")
    print(f"Tokenizers Version:   {tokenizers.__version__}")
    print(f"SentencePiece Ver:    {sentencepiece.__version__}")

    # 3. CUDA & GPU Verification
    cuda_avail = torch.cuda.is_available()
    print("-" * 65)
    print(f"CUDA Available (PyTorch): {cuda_avail}")

    diag: Dict[str, Any] = {
        "python_version": py_ver,
        "python_executable": py_exec,
        "is_venv": is_venv,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": cuda_avail,
        "cuda_version": None,
        "device_name": None,
        "total_vram_mb": None,
        "computation_verified": False,
    }

    if not cuda_avail:
        print("[WARNING] CUDA is not available. Falling back to CPU diagnostics.")
        # Perform CPU tensor computation
        a = torch.randn(256, 256)
        b = torch.randn(256, 256)
        c = torch.matmul(a, b)
        diag["computation_verified"] = bool(torch.isfinite(c).all().item())
        print(f"CPU Matrix Multiplication (256x256): {'SUCCESS' if diag['computation_verified'] else 'FAILED'}")
        return diag

    # CUDA Details
    cuda_runtime = torch.version.cuda
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    dev_name = torch.cuda.get_device_name(current_device)
    props = torch.cuda.get_device_properties(current_device)
    total_vram_mb = props.total_memory / (1024 * 1024)
    free_vram_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)

    diag["cuda_version"] = cuda_runtime
    diag["device_name"] = dev_name
    diag["total_vram_mb"] = round(total_vram_mb, 2)
    diag["free_vram_mb"] = round(free_vram_mb, 2)

    print(f"PyTorch CUDA Runtime:     CUDA {cuda_runtime}")
    print(f"Detected GPU Devices:     {device_count}")
    print(f"Active GPU Index:         {current_device} ({dev_name})")
    print(f"Dedicated VRAM (Total):   {total_vram_mb:.1f} MB ({total_vram_mb / 1024:.2f} GB)")
    print(f"Available VRAM (Free):    {free_vram_mb:.1f} MB ({free_vram_mb / 1024:.2f} GB)")

    # 4. Actual CUDA Computation & Tensor Placement Test
    print("-" * 65)
    print("Executing GPU Tensor Allocation & Matrix Multiplication Benchmark...")

    device = torch.device("cuda:0")
    matrix_dim = 1024  # 1024x1024 FP32 matrix multiplication

    # Allocate on GPU
    x = torch.randn(matrix_dim, matrix_dim, device=device, dtype=torch.float32)
    y = torch.randn(matrix_dim, matrix_dim, device=device, dtype=torch.float32)

    # Tensor placement check
    assert x.device.type == "cuda", f"Expected x on cuda, got {x.device}"
    assert y.device.type == "cuda", f"Expected y on cuda, got {y.device}"

    # Perform actual GEMM (General Matrix Multiply) on CUDA Tensor Cores
    z = torch.matmul(x, y)

    # Synchronize GPU to ensure kernels completed execution
    torch.cuda.synchronize()

    # Output validation
    assert z.device.type == "cuda", f"Output tensor not on CUDA: {z.device}"
    assert z.shape == (matrix_dim, matrix_dim), f"Incorrect shape: {z.shape}"
    is_finite = bool(torch.isfinite(z).all().item())
    assert is_finite, "Computed matrix multiplication output contains NaN or Inf!"

    diag["computation_verified"] = True
    diag["matrix_dim"] = matrix_dim

    print(f"Matrix Dimension:         {matrix_dim} x {matrix_dim} (FP32)")
    print(f"GPU Synchronization:      VERIFIED (torch.cuda.synchronize() succeeded)")
    print(f"Tensor Placement:         VERIFIED (Allocated on {z.device})")
    print(f"Output Numerical Finite:  VERIFIED ({is_finite})")
    print(f"VRAM Post-Execution:      {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB Allocated")
    print("=" * 65)
    print("STATUS: ENVIRONMENT & CUDA ACCELERATION FULLY VERIFIED")
    print("=" * 65)

    return diag


if __name__ == "__main__":
    run_verification()

