"""Simulated real-time event generator.

Events reference REAL customer IDs from the IBM Telco dataset (never invented
IDs). Each event is clearly a SIMULATION — the source dataset is historical, so
these stand in for a live event stream.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from src.config import Config, get_config
from src.ingestion.load_data import load_raw
from src.validation.schema import ID_COLUMN

# The 9 simulated event types with sampling weights (payments/logins common,
# cancellation attempts rare).
EVENT_TYPES = [
    "login", "purchase", "payment_failed", "support_ticket", "complaint",
    "plan_upgrade", "plan_downgrade", "cancellation_attempt", "inactivity",
]
EVENT_WEIGHTS = [0.30, 0.15, 0.12, 0.10, 0.06, 0.05, 0.05, 0.02, 0.15]


@lru_cache(maxsize=1)
def load_customer_ids(cfg: Config | None = None) -> tuple[str, ...]:
    cfg = cfg or get_config()
    return tuple(load_raw(cfg=cfg)[ID_COLUMN].tolist())


def generate_event(
    customer_ids: tuple[str, ...],
    rng: random.Random,
    when: datetime | None = None,
) -> dict:
    cid = rng.choice(customer_ids)
    etype = rng.choices(EVENT_TYPES, weights=EVENT_WEIGHTS, k=1)[0]
    ts = (when or datetime.now(timezone.utc)).isoformat()
    return {
        "customer_id": cid,
        "event_type": etype,
        "timestamp": ts,
        "value": 1.0,
        "is_simulated": True,
    }


def generate_events(
    n: int,
    seed: int | None = None,
    cfg: Config | None = None,
    spread_seconds: int = 3600,
) -> list[dict]:
    """Generate ``n`` simulated events over a recent time window."""
    cfg = cfg or get_config()
    rng = random.Random(seed if seed is not None else cfg.get("project.random_seed", 42))
    ids = load_customer_ids(cfg)
    now = datetime.now(timezone.utc)
    events = []
    for i in range(n):
        when = now - timedelta(seconds=rng.randint(0, spread_seconds))
        events.append(generate_event(ids, rng, when))
    events.sort(key=lambda e: e["timestamp"])
    return events


def event_stream(seed: int | None = None, cfg: Config | None = None):
    """Infinite generator of simulated events (for continuous local streaming)."""
    cfg = cfg or get_config()
    rng = random.Random(seed)
    ids = load_customer_ids(cfg)
    while True:
        yield generate_event(ids, rng)
