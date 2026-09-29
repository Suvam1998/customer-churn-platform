"""Prediction + retrain endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.schemas import (
    PredictionResponse,
    PredictRequest,
    RetrainResponse,
)
from api.services.model_service import ModelNotReadyError, get_service

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict(req: PredictRequest) -> PredictionResponse:
    if not req.customer_id and not req.customer:
        raise HTTPException(status_code=422, detail="Provide customer_id or customer features")
    try:
        svc = get_service()
        if req.customer_id:
            result = svc.predict_existing(req.customer_id)
        else:
            result = svc.predict_raw(req.customer.model_dump())
        return PredictionResponse(**result)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"customer_id {req.customer_id} not found")
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/retrain", response_model=RetrainResponse, status_code=202)
def retrain() -> RetrainResponse:
    """Acknowledge a retrain request. Actual retraining is an offline pipeline
    (Phase 43); models are NOT auto-promoted without validation."""
    return RetrainResponse(
        status="accepted",
        note="Retraining is an offline MLOps pipeline (scripts/tune + calibrate). "
             "Models are not auto-promoted; promotion requires validation (Phase 42/43).",
    )
