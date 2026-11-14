"""GPU performance and latency benchmarking script for baseline FLAN-T5."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np
import torch

# Ensure offline mode and paths
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from descra.models.loader import load_model_and_tokenizer
from descra.utils.logging import get_logger

logger = get_logger("descra.scripts.benchmark_baseline")


def benchmark_gpu_performance(
    model_name: str = "google/flan-t5-small",
    output_dir: str = "reports/baseline",
    device_str: str = "auto",
) -> Dict[str, Any]:
    print("=" * 60)
    print("DESCRA — GPU LATENCY & VRAM PERFORMANCE BENCHMARK")
    print("=" * 60)

    # 1. Benchmark Cold Load
    start_load = time.perf_counter()
    model, tokenizer, device = load_model_and_tokenizer(
        model_name=model_name,
        device=device_str,
        use_fp16=True,
    )
    load_time_sec = round(time.perf_counter() - start_load, 3)
    print(f"Model Cold Loading Time: {load_time_sec} s on {device}")

    is_cuda = device.type == "cuda"
    if is_cuda:
        gpu_name = torch.cuda.get_device_name(device)
        total_vram_mb = round(torch.cuda.get_device_properties(device).total_memory / (1024**2), 1)
        print(f"Target GPU:              {gpu_name} ({total_vram_mb} MB VRAM)")
    else:
        gpu_name = "CPU"
        total_vram_mb = 0.0

    # 2. Warm-up
    print("Executing GPU kernel warm-up passes...")
    dummy_input = tokenizer("Warm up prompt for Transformer generation.", return_tensors="pt").to(device)
    with torch.inference_mode():
        for _ in range(3):
            model.generate(**dummy_input, max_new_tokens=16)
    if is_cuda:
        torch.cuda.synchronize()

    # 3. Test Matrix: Batch Sizes x Sequence Lengths
    batch_sizes = [1, 4, 8]
    sequence_lengths = [128, 256, 512]
    num_iterations = 5

    benchmark_matrix: List[Dict[str, Any]] = []

    print("\nRunning benchmark matrix across batch sizes and sequence lengths...")
    for b_size in batch_sizes:
        for seq_len in sequence_lengths:
            # Generate synthetic prompt of target token length
            raw_text = "Product bullet point feature with high quality specifications and performance details. " * 30
            tokens = tokenizer(raw_text, truncation=True, max_length=seq_len, return_tensors="pt")["input_ids"]
            # Repeat to batch size
            batch_input_ids = tokens.repeat(b_size, 1).to(device)
            batch_attention_mask = torch.ones_like(batch_input_ids).to(device)

            if is_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()

            latencies = []
            tokens_generated = []

            for _ in range(num_iterations):
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        max_new_tokens=64,
                        num_beams=4,
                        early_stopping=True,
                    )

                if is_cuda:
                    torch.cuda.synchronize()
                t_elapsed = (time.perf_counter() - t0) * 1000  # ms

                latencies.append(t_elapsed)
                tokens_generated.append(output_ids.shape[1] * b_size)

            if is_cuda:
                peak_alloc_mb = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
                peak_reserved_mb = round(torch.cuda.max_memory_reserved(device) / (1024**2), 2)
            else:
                peak_alloc_mb = 0.0
                peak_reserved_mb = 0.0

            mean_lat = round(float(np.mean(latencies)), 2)
            median_lat = round(float(np.median(latencies)), 2)
            p95_lat = round(float(np.percentile(latencies, 95)), 2)
            total_tokens = sum(tokens_generated)
            total_time_s = sum(latencies) / 1000
            throughput = round(total_tokens / total_time_s, 2) if total_time_s > 0 else 0.0

            entry = {
                "batch_size": b_size,
                "input_seq_length": seq_len,
                "max_new_tokens": 64,
                "num_beams": 4,
                "mean_latency_ms": mean_lat,
                "median_latency_ms": median_lat,
                "p95_latency_ms": p95_lat,
                "throughput_tokens_per_sec": throughput,
                "peak_vram_allocated_mb": peak_alloc_mb,
                "peak_vram_reserved_mb": peak_reserved_mb,
            }
            benchmark_matrix.append(entry)
            print(
                f"  Batch {b_size:>2} | Seq {seq_len:>3} | Latency: {mean_lat:>7.2f} ms | "
                f"Throughput: {throughput:>7.2f} tok/s | Peak VRAM: {peak_alloc_mb:>6.1f} MB"
            )

    results = {
        "hardware": {
            "device": str(device),
            "gpu_name": gpu_name,
            "total_vram_mb": total_vram_mb,
            "cuda_available": is_cuda,
        },
        "model_loading": {
            "model_name": model_name,
            "cold_load_time_seconds": load_time_sec,
            "precision": "float16" if (is_cuda) else "float32",
        },
        "benchmark_matrix": benchmark_matrix,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "performance_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Benchmark results exported to: {json_path}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    benchmark_gpu_performance()

