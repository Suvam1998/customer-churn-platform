"""Customer view endpoints: profile, explanation, recommendation, high-risk."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.schemas.schemas import (
    CustomerResponse,
    ExplanationResponse,
    HighRiskItem,
    RecommendationResponse,
)
from api.services.model_service import ModelNotReadyError, get_service

router = APIRouter(tags=["customers"])


def _svc():
    try:
        return get_service()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/customers/high-risk", response_model=list[HighRiskItem])
def high_risk(
    min_level: str = Query("HIGH", pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$"),
    limit: int = Query(50, ge=1, le=500),
) -> list[HighRiskItem]:
    return [HighRiskItem(**x) for x in _svc().high_risk(min_level=min_level, limit=limit)]


@router.get("/customer/{customer_id}", response_model=CustomerResponse)
def get_customer(customer_id: str) -> CustomerResponse:
    try:
        return CustomerResponse(**_svc().get_customer(customer_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id} not found")


@router.get("/customer/{customer_id}/explanation", response_model=ExplanationResponse)
def explanation(customer_id: str) -> ExplanationResponse:
    try:
        return ExplanationResponse(**_svc().explain(customer_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id} not found")


@router.get("/customer/{customer_id}/recommendation", response_model=RecommendationResponse)
def recommendation(customer_id: str) -> RecommendationResponse:
    try:
        return RecommendationResponse(**_svc().recommend(customer_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id} not found")
