# Step 3 Deployment Readiness Report

## Implementation Summary

Step 3 adds process-local inference concurrency control, request timeout and overload
responses, asynchronous dispatch for the existing synchronous predictor, container
assets, a benchmark utility, and reliability tests.

## Concurrency Architecture

The `InferenceConcurrencyLimiter` wraps the existing `DescriptionPredictor` with:

- a bounded asyncio semaphore
- `asyncio.to_thread` so synchronous PyTorch inference does not block the event loop
- a short capacity-acquisition wait
- a separate inference timeout
- task completion callbacks that release capacity only after underlying inference ends

The default maximum concurrency is 1, appropriate for the 6 GB laptop GPU. The
application reuses the same predictor and `GenerationService` instance per process.

## Timeout and Overload Behavior

| Condition | HTTP status | Error code |
|---|---:|---|
| Model unavailable | 503 | `model_not_ready` |
| Capacity unavailable | 503 | `service_overloaded` |
| Inference timeout | 504 | `inference_timeout` |
| Inference exception | 500 | `inference_error` |

Timeout responses indicate only the request deadline; a timed-out synchronous inference
task may continue in the thread pool until it completes, and capacity remains occupied
during that period.

## Resource Management

The model is initialized once during FastAPI lifespan startup and retained as
application state. Shutdown waits briefly for active inference tasks before predictor
cleanup. The CUDA cache is cleared only during shutdown. CPU deployment is supported
through `DEVICE=cpu`.

## Rate Limiting and Caching

No Redis or distributed cache is introduced. The current use case does not justify
cross-process caching, because caching generation results could change with model,
checkpoint, decoding configuration, or input state. No distributed rate limiter is
claimed or implemented.

## Docker Architecture

- CPU-only PyTorch is used in the base image.
- The Docker image health check tests `/health`, while `/health/ready` remains available
  for explicit readiness probes.
- GPU use remains optional and requires a compatible runtime, base image strategy, and
  environment override.
- The fine-tuned checkpoint is mounted read-only at `/app/checkpoints`.
- Compose exposes port 8000 and uses health checks.
- No Redis, database, reverse proxy, or other optional service is included.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_CHECKPOINT_PATH` | `checkpoints/sft/best_checkpoint` | Predictor checkpoint |
| `DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `USE_HALF_PRECISION` | `true` | Half precision on CUDA |
| `INFERENCE_TIMEOUT_SECONDS` | `30` | Underlying request deadline |
| `INFERENCE_CAPACITY_WAIT_SECONDS` | `0.25` | Capacity acquisition wait |
| `MAX_CONCURRENT_INFERENCE_REQUESTS` | `1` | Process-local concurrent inference limit |

## Local Execution

Dependencies are listed in `requirements.txt`, but this environment was not modified
or dependency-installed.

```powershell
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

## Docker Execution

Place or mount the fine-tuned checkpoint under `./checkpoints/sft/best_checkpoint`,
then run:

```powershell
docker compose up --build
```

This Docker validation could not be performed because Docker is unavailable.

## Health Checks

- `GET /health` checks process liveness.
- `GET /health/ready` checks model readiness and startup status.

## Validation Blockers

- FastAPI/Starlette are absent from the current `.venv`.
- The fine-tuned model checkpoint is absent.
- Docker is unavailable.
- A real GPU benchmark was not run.
- No benchmark or test result has been fabricated.

## Final Code Review Verification

- Changed Python files compile successfully with `python -m compileall`.
- `git diff --check` reports no whitespace errors.
- Dependency-free standalone checks confirmed that timed-out and cancelled inference
  remain capacity-held until underlying work completes.
- FastAPI pytest collection remains blocked by missing Starlette/Pydantic Settings
  dependencies in the existing virtual environment.
- Docker build and container startup remain unvalidated because Docker is unavailable.
