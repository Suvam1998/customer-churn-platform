"""Phase 16 tests: event generator + risk-update flow (simulated)."""
from __future__ import annotations

from datetime import datetime

import pytest

from src.config import get_config
from src.streaming.event_generator import (
    EVENT_TYPES,
    generate_events,
    load_customer_ids,
)
from src.streaming.risk_update import EVENT_DELTAS, adjusted_probability


def test_events_use_real_customer_ids():
    ids = set(load_customer_ids())
    events = generate_events(50, seed=1)
    assert len(events) == 50
    for e in events:
        assert e["customer_id"] in ids          # never invented
        assert e["event_type"] in EVENT_TYPES
        assert e["is_simulated"] is True
        datetime.fromisoformat(e["timestamp"])  # valid ISO timestamp


def test_events_reproducible_with_seed():
    a = generate_events(20, seed=42)
    b = generate_events(20, seed=42)
    assert [(e["customer_id"], e["event_type"]) for e in a] == \
        [(e["customer_id"], e["event_type"]) for e in b]


def test_adjusted_probability_direction_and_clip():
    assert adjusted_probability(0.5, "payment_failed") == pytest.approx(0.65)
    assert adjusted_probability(0.5, "plan_upgrade") == pytest.approx(0.40)
    # Clipping to [0, 1].
    assert adjusted_probability(0.95, "cancellation_attempt") == 1.0
    assert adjusted_probability(0.01, "plan_upgrade") == 0.0
    # Unknown event -> no change.
    assert adjusted_probability(0.3, "unknown") == pytest.approx(0.3)


def test_all_event_types_have_deltas():
    for etype in EVENT_TYPES:
        assert etype in EVENT_DELTAS


def _model_ready() -> bool:
    return (get_config().resolve_path("paths.models") / "production_model.joblib").exists()


@pytest.mark.skipif(not _model_ready(), reason="run Phase 9 first")
def test_riskstate_apply_and_accumulate():
    from api.services.model_service import ChurnService
    from src.streaming.risk_update import RiskState

    svc = ChurnService()
    cid = svc.table.index[0]
    state = RiskState(svc)

    base = state.current(cid)
    u1 = state.apply({"customer_id": cid, "event_type": "payment_failed"})
    assert u1.previous_probability == pytest.approx(round(base, 4))
    assert u1.new_probability >= u1.previous_probability
    assert u1.is_simulated is True
    assert u1.recommended_action

    # Second event accumulates on the updated state.
    u2 = state.apply({"customer_id": cid, "event_type": "payment_failed"})
    assert u2.previous_probability == pytest.approx(u1.new_probability)
