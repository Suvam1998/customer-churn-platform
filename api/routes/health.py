"""Health / readiness endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from src import __version__
from src.config import get_config

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health() -> HealthResponse:
    """Return service liveness. No model or DB dependency (Phase 1)."""
    cfg = get_config()
    return HealthResponse(
        status="ok",
        service=cfg.get("api.title", "Churn & Retention Platform API"),
        version=__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
