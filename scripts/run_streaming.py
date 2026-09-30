"""Phase 17 entrypoint: end-to-end streaming pipeline.

Event Generator -> Producer -> [broker: customer-events] -> Consumer ->
risk update -> [broker: customer-risk-updates] -> DB.

Runs with the LOCAL broker by default (no Kafka needed). Set streaming.mode
to "kafka" in configs/config.yaml (or STREAMING_MODE=kafka) plus a running
broker to use Kafka.

Usage (repo root, venv active):
    python scripts/run_streaming.py --n 30
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402
from src.streaming.broker import get_broker  # noqa: E402
from src.streaming.consumer import EventConsumer  # noqa: E402
from src.streaming.producer import EventProducer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the streaming pipeline")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--no-db", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()

    from api.services.model_service import ChurnService

    broker = get_broker(cfg)
    events_topic = cfg.get("streaming.topic_events", "customer-events")
    risk_topic = cfg.get("streaming.topic_risk_updates", "customer-risk-updates")

    # Fresh run: purge local topics if supported.
    if hasattr(broker, "purge"):
        broker.purge(events_topic)
        broker.purge(risk_topic)

    service = ChurnService(cfg)
    producer = EventProducer(broker=broker, cfg=cfg)
    consumer = EventConsumer(service, broker=broker, cfg=cfg)

    produced = producer.generate_and_send(args.n, seed=args.seed)
    updates = consumer.process_available(persist=not args.no_db)

    print("=" * 96)
    print(f"STREAMING PIPELINE (backend={broker.backend})  — events are SIMULATED")
    print("=" * 96)
    print(f"Produced to '{events_topic}' : {len(produced)}")
    print(f"Risk updates to '{risk_topic}': {len(updates)}")
    print("-" * 96)
    print(f"{'customer':<12}{'event':<20}{'prev':>7}{'new':>7}{'chg':>8}{'level':>10}  action")
    print("-" * 96)
    for u in updates[:15]:
        print(f"{u.customer_id:<12}{u.event_type:<20}{u.previous_probability:>7}"
              f"{u.new_probability:>7}{u.risk_change:>+8}{u.new_risk_level:>10}  "
              f"{u.recommended_action}")
    if len(updates) > 15:
        print(f"... and {len(updates) - 15} more")
    print("-" * 96)
    risers = sorted(updates, key=lambda u: u.risk_change, reverse=True)[:3]
    print("Largest risk increases:")
    for u in risers:
        print(f"  {u.customer_id}: {u.previous_probability} -> {u.new_probability} "
              f"({u.risk_change:+}) via {u.event_type}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
