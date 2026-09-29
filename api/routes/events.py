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
from src.risk import classify_risk

router = APIRouter(tags=["events"])

# Documented heuristic deltas on churn probability per simulated event type.
# These are illustrative adjustments for the demo; the model itself is not
# retrained per event. Kept small and bounded.
_EVENT_DELTAS = {
    "payment_failed": +0.15,
    "complaint": +0.12,
    "support_ticket": +0.05,
    "cancellation_attempt": +0.30,
    "inactivity": +0.08,
    "plan_downgrade": +0.06,
    "login": -0.03,
    "purchase": -0.05,
    "plan_upgrade": -0.10,
}


@router.post("/event", response_model=EventResponse)
def ingest_event(event: EventRequest) -> EventResponse:
    try:
        svc = get_service()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if event.customer_id not in svc.table.index:
        raise HTTPException(status_code=404, detail=f"customer_id {event.customer_id} not found")

    row = svc.table.loc[event.customer_id]
    prev = float(row["churn_probability"])
    delta = _EVENT_DELTAS.get(event.event_type, 0.0)
    new_prob = min(1.0, max(0.0, prev + delta))
    new_level = classify_risk(new_prob)

    # Recommendation for the updated risk state.
    updated = row.to_dict()
    updated.update({"churn_probability": new_prob, "risk_level": new_level})
    from src.retention.recommendations import make_decision

    action = make_decision(updated, cfg=svc.cfg).recommended_action

    return EventResponse(
        accepted=True,
        is_simulated=True,
        customer_id=event.customer_id,
        event_type=event.event_type,
        previous_probability=round(prev, 4),
        new_probability=round(new_prob, 4),
        risk_change=round(new_prob - prev, 4),
        risk_level=new_level,
        recommended_action=action,
        note="SIMULATED event: risk adjusted by a documented heuristic delta. "
             "Full streaming re-inference is Phase 16/17.",
    )
