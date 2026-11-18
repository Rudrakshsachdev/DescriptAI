"""Lightweight service-level benchmark for the Descript AI backend."""

import argparse
import asyncio
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import psutil
import torch

from backend.app.core.config import Settings
from backend.app.core.exceptions import ServiceOverloadedException
from backend.app.services.generation import GenerationService
from descra.inference.predictor import DescriptionPredictor


SAMPLE_INPUTS = [
    ("Hydrating Face Cream", ["Hyaluronic acid", "Fragrance-free"], "Beauty", "DermaPure"),
    ("Wireless Noise-Cancelling Headphones", ["Bluetooth 5.3", "30-hour battery"], "Electronics", "Acousta"),
    ("Insulated Stainless Steel Bottle", ["750ml capacity", "Leak-proof"], "Home and Kitchen", "TrailPro"),
]


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile_value))
    return ordered[index]


async def run_requests(
    service: GenerationService,
    count: int,
    concurrency: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    failures = 0

    async def execute(index: int) -> None:
        nonlocal failures
        async with semaphore:
            product_name, features, category, brand = SAMPLE_INPUTS[index % len(SAMPLE_INPUTS)]
            started = time.perf_counter()
            try:
                await service.generate_async(
                    product_name=product_name,
                    features=features,
                    category=category,
                    brand=brand,
                )
            except Exception:
                failures += 1
            else:
                latencies.append((time.perf_counter() - started) * 1000.0)

    started = time.perf_counter()
    await asyncio.gather(*(execute(index) for index in range(count)))
    elapsed = time.perf_counter() - started

    return {
        "requests": count,
        "successful": len(latencies),
        "failed": failures,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_rps": round(len(latencies) / elapsed, 3) if elapsed else 0.0,
        "average_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "median_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95_ms": round(percentile(latencies, 0.95), 2),
    }


async def probe_overload(service: GenerationService) -> dict[str, Any]:
    """Runs a small two-request overload probe against the real service limiter."""
    overloaded = 0

    async def execute() -> None:
        nonlocal overloaded
        product_name, features, category, brand = SAMPLE_INPUTS[0]
        try:
            await service.generate_async(
                product_name=product_name,
                features=features,
                category=category,
                brand=brand,
            )
        except ServiceOverloadedException:
            overloaded += 1

    await asyncio.gather(*(execute() for _ in range(2)))
    return {
        "requests": 2,
        "overloaded": overloaded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the Descript AI generation service.")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--checkpoint", default="checkpoints/sft/best_checkpoint")
    args = parser.parse_args()

    settings = Settings(MODEL_CHECKPOINT_PATH=args.checkpoint)
    checkpoint = Path(settings.MODEL_CHECKPOINT_PATH)
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_bytes": psutil.virtual_memory().total,
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint": str(checkpoint),
    }

    if not checkpoint.exists():
        print({
            "status": "not_run",
            "reason": "Model checkpoint is not mounted or available.",
            "environment": environment,
        })
        return 2

    initialization_started = time.perf_counter()
    predictor = DescriptionPredictor(
        checkpoint_path=settings.MODEL_CHECKPOINT_PATH,
        device=None if settings.DEVICE == "auto" else settings.DEVICE,
        use_half_precision=settings.USE_HALF_PRECISION,
    )
    model_initialization_ms = (time.perf_counter() - initialization_started) * 1000.0
    service = GenerationService(predictor=predictor)

    sequential = asyncio.run(run_requests(service, args.requests, concurrency=1))
    concurrent = asyncio.run(
        run_requests(service, args.requests, concurrency=max(1, args.concurrency))
    )
    overload = asyncio.run(probe_overload(service))

    gpu_memory: dict[str, int] | None = None
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }

    print({
        "status": "completed",
        "environment": environment,
        "model_initialization_ms": round(model_initialization_ms, 2),
        "sequential": sequential,
        "controlled_concurrent": concurrent,
        "overload": overload,
        "cpu_memory_bytes": psutil.Process().memory_info().rss,
        "gpu_memory": gpu_memory,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
