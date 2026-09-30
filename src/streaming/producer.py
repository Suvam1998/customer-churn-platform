"""Event producer — publishes simulated events to the ``customer-events`` topic.

Backend-agnostic (Kafka or local). Events reference real customer IDs and are
flagged simulated (see event_generator).
"""
from __future__ import annotations

import logging

from src.config import Config, get_config
from src.streaming.broker import get_broker
from src.streaming.event_generator import generate_events

logger = logging.getLogger("churn.producer")


class EventProducer:
    def __init__(self, broker=None, cfg: Config | None = None) -> None:
        self.cfg = cfg or get_config()
        self.broker = broker or get_broker(self.cfg)
        self.topic = self.cfg.get("streaming.topic_events", "customer-events")

    def send(self, event: dict) -> None:
        self.broker.produce(self.topic, value=event, key=event.get("customer_id"))

    def send_many(self, events: list[dict]) -> int:
        for e in events:
            self.send(e)
        logger.info("produced %d events to %s", len(events), self.topic)
        return len(events)

    def generate_and_send(self, n: int, seed: int | None = None) -> list[dict]:
        events = generate_events(n, seed=seed, cfg=self.cfg)
        self.send_many(events)
        return events
