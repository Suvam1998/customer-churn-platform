"""Event consumer — the streaming risk-update pipeline.

Consumes ``customer-events``, applies the risk-update flow, publishes results to
``customer-risk-updates``, and optionally persists to the database. Backend-
agnostic (Kafka or local).
"""
from __future__ import annotations

import logging
from datetime import datetime

from src.config import Config, get_config
from src.streaming.broker import get_broker
from src.streaming.risk_update import RiskState, RiskUpdate

logger = logging.getLogger("churn.consumer")


class EventConsumer:
    def __init__(self, service, broker=None, cfg: Config | None = None) -> None:
        self.cfg = cfg or get_config()
        self.broker = broker or get_broker(self.cfg)
        self.service = service
        self.state = RiskState(service, self.cfg)
        self.events_topic = self.cfg.get("streaming.topic_events", "customer-events")
        self.risk_topic = self.cfg.get("streaming.topic_risk_updates", "customer-risk-updates")

    def _persist(self, session, event_value: dict, upd: RiskUpdate) -> None:
        from src.db.models import CustomerEvent, RiskScore

        ts = event_value.get("timestamp")
        session.add(CustomerEvent(
            customer_id=upd.customer_id, event_type=upd.event_type,
            event_timestamp=datetime.fromisoformat(ts) if ts else None,
            value=event_value.get("value", 1.0), is_simulated=True,
            payload={"delta_applied": upd.risk_change}))
        session.add(RiskScore(
            customer_id=upd.customer_id,
            previous_probability=upd.previous_probability,
            new_probability=upd.new_probability, risk_change=upd.risk_change,
            risk_level=upd.new_risk_level, revenue_at_risk=upd.revenue_at_risk))

    def process_available(self, from_offset: int = 0, persist: bool = False) -> list[RiskUpdate]:
        """Consume all currently-available events, publish risk updates, return them."""
        session = None
        if persist:
            try:
                from src.db.session import get_session, init_db
                init_db()
                session = get_session()
            except Exception as exc:  # pragma: no cover
                logger.warning("DB unavailable; not persisting: %s", exc)
                session = None

        updates: list[RiskUpdate] = []
        for msg in self.broker.consume(self.events_topic, from_offset=from_offset):
            ev = msg.value
            if ev.get("customer_id") not in self.state.prob:
                logger.warning("skipping event for unknown customer %s", ev.get("customer_id"))
                continue
            upd = self.state.apply(ev)
            self.broker.produce(self.risk_topic, value=upd.to_dict(), key=upd.customer_id)
            updates.append(upd)
            if session is not None:
                self._persist(session, ev, upd)

        if session is not None:
            session.commit()
        logger.info("processed %d events -> %s", len(updates), self.risk_topic)
        return updates

    def latest_risk_updates(self, n: int = 20) -> list[dict]:
        """Return the most recent risk updates from the risk topic."""
        msgs = list(self.broker.consume(self.risk_topic))
        return [m.value for m in msgs[-n:]]
