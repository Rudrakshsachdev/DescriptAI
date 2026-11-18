"""Health and readiness monitoring schemas."""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness probe response model."""

    status: str = Field(default="healthy", description="Application liveness status.")
    version: str = Field(..., description="API semantic version.")
    environment: str = Field(..., description="Operating environment.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Current UTC timestamp.",
    )


class ReadinessResponse(BaseModel):
    """Readiness probe response model indicating model availability."""

    status: str = Field(..., description="Readiness status ('ready' or 'not_ready').")
    model_loaded: bool = Field(..., description="Whether the inference predictor is loaded.")
    model_path: str = Field(..., description="Active checkpoint path or model identifier.")
    device: str = Field(..., description="Active compute device (e.g., 'cuda:0', 'cpu').")
    half_precision: bool = Field(..., description="Whether BF16/FP16 is enabled.")
    startup_error: Optional[str] = Field(default=None, description="Startup error message if model failed to load.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Current UTC timestamp.",
    )
