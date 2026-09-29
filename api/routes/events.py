"""Event ingestion endpoint (SIMULATED real-time events).

Phase 15 provides synchronous event handling: validate the event, apply a
documented heuristic adjustment to the customer's risk, and return the updated
risk. The full streaming pipeline (feature recompute + model re-inference via
Kafka) is Phase 16/17.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.schemas import EventRequest, EventResponse
from api.services.model_service import ModelNotReadyError, get_service
from src.streaming.risk_update import RiskState

router = APIRouter(tags=["events"])

# Per-service RiskState so successive events on a customer accumulate.
_state: RiskState | None = None


def _get_state() -> RiskState:
    global _state
    svc = get_service()
    if _state is None or _state.service is not svc:
        _state = RiskState(svc)
    return _state


@router.post("/event", response_model=EventResponse)
def ingest_event(event: EventRequest) -> EventResponse:
    try:
        state = _get_state()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if event.customer_id not in state.service.table.index:
        raise HTTPException(status_code=404, detail=f"customer_id {event.customer_id} not found")

    upd = state.apply({
        "customer_id": event.customer_id,
        "event_type": event.event_type,
        "timestamp": event.timestamp,
        "value": event.value,
    })

    return EventResponse(
        accepted=True,
        is_simulated=True,
        customer_id=upd.customer_id,
        event_type=upd.event_type,
        previous_probability=upd.previous_probability,
        new_probability=upd.new_probability,
        risk_change=upd.risk_change,
        risk_level=upd.new_risk_level,
        recommended_action=upd.recommended_action,
        note="SIMULATED event: risk adjusted by a documented heuristic delta. "
             "Full Kafka streaming is Phase 17.",
    )
