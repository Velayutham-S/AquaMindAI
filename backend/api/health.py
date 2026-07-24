"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe used by the frontend and by deployment platforms."""
    return HealthResponse()
