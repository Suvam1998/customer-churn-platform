"""Phase 14 entrypoint: create tables and seed the database.

Uses the SQLite fallback unless DATABASE_URL / POSTGRES_HOST is set, so it runs
with no PostgreSQL installed.

Usage (repo root, venv active):
    python scripts/init_db.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import func, select  # noqa: E402

from src.config import get_config  # noqa: E402
from src.db.models import (  # noqa: E402
    Customer,
    ModelVersion,
    Prediction,
    Recommendation,
    Subscription,
)
from src.db.session import get_database_url, get_session, init_db  # noqa: E402
from database.seed import (  # noqa: E402
    seed_customers,
    seed_model_version,
    seed_predictions_and_recommendations,
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()

    url = get_database_url(cfg)
    print(f"Database URL: {url.split('@')[-1] if '@' in url else url}")

    init_db(drop=True)
    n_cust = seed_customers(get_session(), cfg=cfg)
    seeded_mv = seed_model_version(get_session(), cfg=cfg)
    n_rec = seed_predictions_and_recommendations(get_session(), cfg=cfg)

    session = get_session()
    counts = {
        "customers": session.scalar(select(func.count()).select_from(Customer)),
        "subscriptions": session.scalar(select(func.count()).select_from(Subscription)),
        "predictions": session.scalar(select(func.count()).select_from(Prediction)),
        "recommendations": session.scalar(select(func.count()).select_from(Recommendation)),
        "model_versions": session.scalar(select(func.count()).select_from(ModelVersion)),
    }

    print("=" * 60)
    print("DATABASE INITIALISED & SEEDED")
    print("=" * 60)
    print(f"Customers seeded from real dataset : {n_cust:,}")
    print(f"Predictions/recommendations seeded : {n_rec:,} "
          f"({'from artifacts' if n_rec else 'no artifacts — run Phase 13 first'})")
    print(f"Model version seeded               : {seeded_mv}")
    print("-" * 60)
    print("Row counts:")
    for t, c in counts.items():
        print(f"  {t:<18} {c:,}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
