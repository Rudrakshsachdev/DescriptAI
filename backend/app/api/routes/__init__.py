"""API routes package."""

from backend.app.api.routes.generation import router as generation_router
from backend.app.api.routes.health import router as health_router

__all__ = ["health_router", "generation_router"]
