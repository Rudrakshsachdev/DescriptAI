# Step 3 Performance Benchmark Report

**Date:** 2026-11-18  
**Phase:** Phase 2 — FastAPI Backend & REST Serving  
**Milestone:** Step 3 — Reliability, Performance & Deployment Readiness  
**Status:** EXECUTED & VERIFIED ON HARDWARE  

---

## 1. Executive Summary

This report documents the actual empirical performance measurements of the **Descript AI** backend service using the standalone benchmark harness `scripts/benchmark_backend.py`. Inference was benchmarked against the real fine-tuned checkpoint at `checkpoints/sft/best_checkpoint/` running on local GPU hardware.

---

## 2. Benchmark Environment

| Parameter | Measured Specification |
|---|---|
| **Operating System** | Windows 11 (AMD64, Build 10.0.26300) |
| **Python Version** | Python 3.10.11 (virtual environment: `.venv`) |
| **PyTorch Runtime** | PyTorch 2.6.0+cu124 (CUDA 12.4 runtime) |
| **Hardware Accelerator** | NVIDIA GeForce RTX 4050 Laptop GPU (6,140 MB VRAM) |
| **Host CPU** | 16 logical cores (AMD / Intel x86_64) |
| **Host System Memory** | 16.0 GB RAM (16,793,001,984 bytes) |
| **Model Weights** | `checkpoints/sft/best_checkpoint` (307.8 MB `model.safetensors`, 76.96M parameters) |
| **Precision** | BFloat16 / Half Precision (`USE_HALF_PRECISION=True`) |

---

## 3. Empirical Performance Measurements

The benchmark utility executed sequential requests, controlled concurrent requests, and overload rejection probes through `GenerationService` and `InferenceConcurrencyLimiter`:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_backend.py --requests 3 --concurrency 1
```

### 3.1 Initialization Latency
- **Model Loading & Tokenizer Mount:** **2,930.00 ms** (~2.93 seconds)
- **Checkpoint Load Memory Footprint:** 172.27 MB allocated GPU VRAM; 182.45 MB reserved VRAM.

### 3.2 Sequential Inference Workload (3 Requests, Concurrency=1)
| Metric | Measured Value |
|---|---:|
| **Total Requests** | 3 |
| **Successful Requests** | 3 (100%) |
| **Failed Requests** | 0 (0%) |
| **Total Elapsed Time** | 3.591 seconds |
| **Throughput** | **0.835 req/sec** |
| **Average Latency** | 1,196.79 ms |
| **Median Latency** | **557.21 ms** |
| **P95 Latency** | 2,545.32 ms |

*Note: The first sequential request incurs initial CUDA kernel compilation and warm-up latency (~2.4s), after which subsequent requests complete in ~550 ms.*

### 3.3 Controlled Concurrent Workload (3 Requests, Concurrency=1 Bounded)
| Metric | Measured Value |
|---|---:|
| **Total Requests** | 3 |
| **Successful Requests** | 3 (100%) |
| **Failed Requests** | 0 (0%) |
| **Total Elapsed Time** | 1.975 seconds |
| **Throughput** | **1.519 req/sec** |
| **Average Latency** | **658.16 ms** |
| **Median Latency** | **552.51 ms** |
| **P95 Latency** | **969.21 ms** |

*Note: Across warm sequential runs, single-request generation consistently completes in ~550 ms (P95 < 1 second).*

### 3.4 Capacity-Overload Behavior Probe
- **Requests Dispatched:** 2 simultaneous requests with `MAX_CONCURRENT_INFERENCE_REQUESTS=1` and `INFERENCE_CAPACITY_WAIT_SECONDS=0.25`.
- **Accepted Request:** 1 request acquired semaphore slot immediately.
- **Overloaded Request:** 1 request exceeded 250 ms capacity wait and raised `ServiceOverloadedException` (HTTP 503).
- **Result:** Capacity bounds strictly prevent multiple concurrent GPU kernel launches, protecting the 6 GB VRAM from exhaustion.

### 3.5 Resource Utilization
- **Host Process Memory (RSS):** 1,672 MB (~1.55 GB)
- **GPU Memory Allocated:** **172.27 MB**
- **GPU Memory Reserved:** **182.45 MB**
- **Peak VRAM Headroom:** > 5.8 GB remaining on the 6 GB RTX 4050 GPU.

---

## 4. Methodology & Reproducibility

1. **Isolation:** Inference runs via `asyncio.to_thread` preventing event-loop stall.
2. **Deterministic Inputs:** Three realistic e-commerce products (Cosmetics, Consumer Electronics, Home Goods) rotated across requests.
3. **Reproducibility Command:**
   ```powershell
   .\.venv\Scripts\python.exe scripts/benchmark_backend.py --requests 3 --concurrency 1
   ```
