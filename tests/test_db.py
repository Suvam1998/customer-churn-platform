"""Phase 14 tests: ORM schema, session fallback, and seeding (in-memory SQLite)."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import (
    ALL_TABLES,
    Base,
    Customer,
    Prediction,
    Subscription,
)
from src.db.session import get_database_url, init_db


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    sm = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    with sm() as s:
        yield s


def test_all_tables_created(session):
    tables = set(Base.metadata.tables.keys())
    expected = {
        "customers", "subscriptions", "customer_events", "customer_features",
        "predictions", "risk_scores", "recommendations", "model_versions",
        "experiments", "monitoring_metrics",
    }
    assert expected <= tables
    assert len(ALL_TABLES) == 10


def test_insert_and_query_customer(session):
    session.add(Customer(customer_id="TEST-001", gender="Female", tenure=5))
    session.add(Subscription(customer_id="TEST-001", contract="Month-to-month",
                             monthly_charges=70.0, services={"TechSupport": "No"}))
    session.add(Prediction(customer_id="TEST-001", churn_probability=0.8,
                           risk_level="CRITICAL", model_version="v-test"))
    session.commit()

    cust = session.get(Customer, "TEST-001")
    assert cust.tenure == 5
    assert cust.subscription.contract == "Month-to-month"
    assert cust.subscription.services["TechSupport"] == "No"

    pred = session.scalar(select(Prediction).where(Prediction.customer_id == "TEST-001"))
    assert pred.churn_probability == 0.8
    assert pred.risk_level == "CRITICAL"


def test_seed_customers_limited(session):
    from database.seed import seed_customers

    n = seed_customers(session, limit=10)
    assert n == 10
    assert session.scalar(select(func.count()).select_from(Customer)) == 10
    assert session.scalar(select(func.count()).select_from(Subscription)) == 10


def test_database_url_sqlite_fallback(monkeypatch):
    # With no env vars, the URL falls back to SQLite.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    url = get_database_url()
    assert url.startswith("sqlite:///")


def test_database_url_prefers_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://u:p@host:5432/db")
    assert get_database_url() == "postgresql+psycopg2://u:p@host:5432/db"
