"""Phase 16 entrypoint: simulate real-time events and update customer risk.

Generates SIMULATED events over REAL customer IDs, applies the risk-update
flow (previous -> new probability, risk change, recommendation), and persists
customer_events + risk_scores to the database.

Usage (repo root, venv active):
    python scripts/run_event_simulation.py --n 30
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime  # noqa: E402

from src.config import get_config  # noqa: E402
from src.db.models import CustomerEvent, RiskScore  # noqa: E402
from src.db.session import get_session, init_db  # noqa: E402
from src.streaming.event_generator import generate_events  # noqa: E402
from src.streaming.risk_update import RiskState  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate real-time events")
    parser.add_argument("--n", type=int, default=30, help="number of events")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-db", action="store_true", help="skip DB persistence")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()

    from api.services.model_service import ChurnService

    service = ChurnService(cfg)
    state = RiskState(service, cfg)
    events = generate_events(args.n, seed=args.seed, cfg=cfg)

    persist = not args.no_db
    session = None
    if persist:
        try:
            init_db()
            session = get_session()
        except Exception as exc:  # pragma: no cover
            logging.warning("DB unavailable, continuing without persistence: %s", exc)
            persist = False

    updates = []
    for ev in events:
        upd = state.apply(ev)
        updates.append(upd)
        if persist:
            ts = datetime.fromisoformat(ev["timestamp"])
            session.add(CustomerEvent(
                customer_id=ev["customer_id"], event_type=ev["event_type"],
                event_timestamp=ts, value=ev["value"], is_simulated=True,
                payload={"delta_applied": upd.risk_change}))
            session.add(RiskScore(
                customer_id=upd.customer_id,
                previous_probability=upd.previous_probability,
                new_probability=upd.new_probability,
                risk_change=upd.risk_change, risk_level=upd.new_risk_level,
                revenue_at_risk=upd.revenue_at_risk))
    if persist:
        session.commit()

    # Report.
    print("=" * 96)
    print(f"SIMULATED REAL-TIME EVENTS ({len(updates)} events)  — data is SIMULATED")
    print("=" * 96)
    print(f"{'customer':<12}{'event':<20}{'prev':>7}{'new':>7}{'chg':>8}"
          f"{'level':>10}  action")
    print("-" * 96)
    for u in updates[:20]:
        print(f"{u.customer_id:<12}{u.event_type:<20}{u.previous_probability:>7}"
              f"{u.new_probability:>7}{u.risk_change:>+8}{u.new_risk_level:>10}  "
              f"{u.recommended_action}")
    if len(updates) > 20:
        print(f"... and {len(updates) - 20} more")
    print("-" * 96)
    risers = sorted(updates, key=lambda u: u.risk_change, reverse=True)[:3]
    print("Largest risk increases:")
    for u in risers:
        print(f"  {u.customer_id}: {u.previous_probability} -> {u.new_probability} "
              f"({u.risk_change:+}) via {u.event_type}")
    if persist:
        print(f"\nPersisted {len(updates)} events + risk scores to the database.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
