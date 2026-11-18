"""Main FastAPI Application Entrypoint for Project Descra."""

from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import AsyncGenerator

# Ensure 'src' is in sys.path so 'descra' is importable across all run modes
src_path = str(Path(__file__).resolve().parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch

from backend.app.api.routes.generation import router as generation_router
from backend.app.api.routes.health import router as health_router
from backend.app.core.config import get_settings
from backend.app.core.exceptions import (
    DescriptAIException,
    InferenceTimeoutException,
    ModelInferenceException,
    ModelNotReadyException,
    ServiceOverloadedException,
)
from backend.app.core.logging import get_logger, setup_logging
from backend.app.services.generation import GenerationService
from descra.inference.predictor import DescriptionPredictor

setup_logging()
logger = get_logger("backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI application lifespan managing ML model initialization and shutdown."""
    settings = get_settings()
    logger.info("Starting up %s v%s [Env: %s]", settings.PROJECT_NAME, settings.VERSION, settings.ENVIRONMENT)

    # 1. Initialize ML Predictor (Loaded once per process)
    if getattr(app.state, "predictor", None) is not None:
        logger.info("Using pre-configured ModelPredictor from app.state.")
        app.state.is_ready = True
        app.state.startup_error = None
        if not getattr(app.state, "generation_service", None):
            app.state.generation_service = GenerationService(app.state.predictor)
    elif getattr(app.state, "startup_error", None) is not None:
        logger.info("Preserving pre-configured startup error: %s", app.state.startup_error)
        app.state.predictor = None
        app.state.generation_service = None
        app.state.is_ready = False
    else:
        try:
            device_arg = None if settings.DEVICE == "auto" else settings.DEVICE
            logger.info(
                "Loading ModelPredictor from checkpoint: '%s' (device=%s, half_precision=%s)...",
                settings.MODEL_CHECKPOINT_PATH,
                settings.DEVICE,
                settings.USE_HALF_PRECISION,
            )

            predictor = DescriptionPredictor(
                checkpoint_path=settings.MODEL_CHECKPOINT_PATH,
                device=device_arg,
                use_half_precision=settings.USE_HALF_PRECISION,
            )

            app.state.predictor = predictor
            app.state.generation_service = GenerationService(predictor)
            app.state.is_ready = True
            app.state.startup_error = None
            logger.info(
                "ModelPredictor initialized successfully on device: %s. Model is READY for serving.",
                predictor.device,
            )
        except Exception as exc:
            logger.error("Failed to initialize ModelPredictor during startup: %s", exc, exc_info=True)
            app.state.predictor = None
            app.state.generation_service = None
            app.state.is_ready = False
            app.state.startup_error = str(exc)

    yield

    # 2. Shutdown & Resource Cleanup
    logger.info("Executing graceful shutdown for %s...", settings.PROJECT_NAME)
    generation_service = getattr(app.state, "generation_service", None)
    if generation_service is not None:
        inference_idle = await generation_service.wait_for_inference_idle(timeout=5.0)
        if not inference_idle:
            logger.warning("Shutdown continuing while inference work is still active.")
    if hasattr(app.state, "predictor") and app.state.predictor is not None:
        del app.state.predictor
    if hasattr(app.state, "generation_service") and app.state.generation_service is not None:
        del app.state.generation_service
    app.state.is_ready = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Shutdown cleanup complete.")


def create_app() -> FastAPI:
    """Application factory creating and configuring the FastAPI instance."""
    settings = get_settings()

    tags_metadata = [
        {
            "name": "Health",
            "description": "Liveness and readiness probes for orchestrators and load balancers.",
        },
        {
            "name": "Generation",
            "description": "Product description generation powered by fine-tuned FLAN-T5 Seq2Seq model.",
        },
        {
            "name": "Model",
            "description": "Operational metadata, model architecture, and serving configurations.",
        },
    ]

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        openapi_tags=tags_metadata,
        lifespan=lifespan,
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    )

    # Add CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    app.include_router(health_router)
    app.include_router(generation_router, prefix=settings.API_V1_STR)

    # Exception Handlers
    @app.exception_handler(ModelNotReadyException)
    async def model_not_ready_handler(request: Request, exc: ModelNotReadyException) -> JSONResponse:
        logger.warning("ModelNotReadyException: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "model_not_ready", "message": exc.message},
        )

    @app.exception_handler(ModelInferenceException)
    async def model_inference_handler(request: Request, exc: ModelInferenceException) -> JSONResponse:
        logger.error("ModelInferenceException: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "inference_error", "message": exc.message},
        )

    @app.exception_handler(InferenceTimeoutException)
    async def inference_timeout_handler(
        request: Request,
        exc: InferenceTimeoutException,
    ) -> JSONResponse:
        logger.warning("Inference timeout: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={"error": "inference_timeout", "message": exc.message},
        )

    @app.exception_handler(ServiceOverloadedException)
    async def service_overloaded_handler(
        request: Request,
        exc: ServiceOverloadedException,
    ) -> JSONResponse:
        logger.warning("Inference overload: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "service_overloaded", "message": exc.message},
        )

    @app.exception_handler(DescriptAIException)
    async def generic_domain_handler(request: Request, exc: DescriptAIException) -> JSONResponse:
        logger.warning("DescriptAIException: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "bad_request", "message": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        logger.warning("Request validation failed: %s", exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "message": "The request contains invalid parameters.",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unexpected application error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "message": "An unexpected server error occurred.",
            },
        )

    return app


app = create_app()
