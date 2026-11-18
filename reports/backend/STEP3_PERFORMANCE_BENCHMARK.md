# Step 3 Performance Benchmark Report

## Status

No model benchmark was executed in this environment. The repository has no fine-tuned
checkpoint at `checkpoints/sft/best_checkpoint`, and Docker is unavailable. Therefore,
this report does not claim startup, latency, throughput, GPU memory, or throughput
results.

## Measurement Utility

The repository includes `scripts/benchmark_backend.py`. It benchmarks the existing
`GenerationService` and `DescriptionPredictor` service boundary, not a simulated HTTP
response.

Example:

```powershell
python scripts/benchmark_backend.py --requests 3 --concurrency 1
```

The utility reports:

- sequential and controlled-concurrent request measurements
- success and failure counts
- elapsed time and throughput
- average, median, and P95 latency
- CPU process memory
- service overload response count
- CUDA memory when CUDA is available
- platform, Python, PyTorch, and checkpoint availability

## Planned Scenarios

1. Single sequential request
2. Small sequential batch
3. Controlled concurrent requests
4. Capacity-overload behavior
5. CPU execution fallback when a CUDA device is unavailable

## Documented Environment Blockers

| Item | Observed state |
|---|---|
| Model checkpoint | Missing; `checkpoints/sft/best_checkpoint` does not exist |
| Docker | `docker` command unavailable |
| Project virtual environment | Python 3.10.11, but FastAPI/Starlette dependencies are not installed |
| PyTorch runtime | CPU-only PyTorch 2.14.1 in the available check |
| Baseline Phase 1 tests | 45 passed, 5 failed, 3 skipped; failures include unavailable Hugging Face resources |

## Measurement Boundaries

The final benchmark should be run with the same checkpoint and environment for all
scenarios. Latency must include service-level inference but exclude the model download
phase. Startup and model initialization should be recorded separately.
