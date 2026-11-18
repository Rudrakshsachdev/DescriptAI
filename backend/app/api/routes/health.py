"""Health and readiness monitoring endpoints."""

from fastapi import APIRouter, Request, Response, status

from backend.app.core.config import get_settings
from backend.app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Probe",
    description="Returns HTTP 200 if the FastAPI application process is alive and responding.",
)
async def liveness() -> HealthResponse:
    """Liveness probe for orchestration and load balancers."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={
        200: {"description": "Application and ML model are fully initialized and ready."},
        503: {"description": "Application is running but ML model is not yet ready."},
    },
    summary="Readiness Probe",
    description="Returns HTTP 200 if the ML model is initialized; otherwise returns HTTP 503.",
)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    """Readiness probe checking ML model initialization status."""
    settings = get_settings()
    predictor = getattr(request.app.state, "predictor", None)
    is_ready = getattr(request.app.state, "is_ready", False)
    startup_error = getattr(request.app.state, "startup_error", None)

    if not is_ready or predictor is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            model_loaded=False,
            model_path=settings.MODEL_CHECKPOINT_PATH,
            device="none",
            half_precision=settings.USE_HALF_PRECISION,
            startup_error=startup_error or "Model has not finished initializing.",
        )

    return ReadinessResponse(
        status="ready",
        model_loaded=True,
        model_path=str(getattr(predictor, "checkpoint_path", settings.MODEL_CHECKPOINT_PATH)),
        device=str(getattr(predictor, "device", "unknown")),
        half_precision=settings.USE_HALF_PRECISION,
        startup_error=None,
    )
