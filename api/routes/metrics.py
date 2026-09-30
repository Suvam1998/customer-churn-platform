"""Dashboard + model metrics and monitoring/drift endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.schemas import DashboardMetrics, DriftResponse, ModelMetrics
from api.services.model_service import ModelNotReadyError, get_service

router = APIRouter(tags=["metrics"])


def _svc():
    try:
        return get_service()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/dashboard/metrics", response_model=DashboardMetrics)
def dashboard_metrics() -> DashboardMetrics:
    return DashboardMetrics(**_svc().dashboard_metrics())


@router.get("/model/metrics", response_model=ModelMetrics)
def model_metrics() -> ModelMetrics:
    return ModelMetrics(**_svc().model_metrics())


@router.get("/monitoring/drift", response_model=DriftResponse)
def monitoring_drift() -> DriftResponse:
    """Real feature + prediction drift (reference=train vs current=test).
    RETRAINING_REQUIRED is a signal only — no model is auto-deployed."""
    return DriftResponse(**_svc().drift())
