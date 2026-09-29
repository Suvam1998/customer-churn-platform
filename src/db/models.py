"""SQLAlchemy ORM models.

Portable across PostgreSQL (production) and SQLite (local fallback): uses the
generic ``JSON`` type (JSONB on Postgres) and avoids DB-specific column types.
Mirrors ``database/schema.sql``.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    gender: Mapped[str | None] = mapped_column(String(16))
    senior_citizen: Mapped[int | None] = mapped_column(Integer)
    partner: Mapped[str | None] = mapped_column(String(8))
    dependents: Mapped[str | None] = mapped_column(String(8))
    tenure: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    subscription: Mapped["Subscription"] = relationship(
        back_populates="customer", uselist=False, cascade="all, delete-orphan")
    events: Mapped[list["CustomerEvent"]] = relationship(
        back_populates="customer", cascade="all, delete-orphan")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), index=True)
    contract: Mapped[str | None] = mapped_column(String(32))
    payment_method: Mapped[str | None] = mapped_column(String(48))
    paperless_billing: Mapped[str | None] = mapped_column(String(8))
    monthly_charges: Mapped[float | None] = mapped_column(Float)
    total_charges: Mapped[float | None] = mapped_column(Float)
    internet_service: Mapped[str | None] = mapped_column(String(24))
    phone_service: Mapped[str | None] = mapped_column(String(8))
    services: Mapped[dict | None] = mapped_column(JSON)  # remaining service flags
    churn: Mapped[int | None] = mapped_column(Integer)   # historical label (0/1)

    customer: Mapped[Customer] = relationship(back_populates="subscription")


class CustomerEvent(Base):
    __tablename__ = "customer_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), index=True)
    event_type: Mapped[str] = mapped_column(String(32))
    event_timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    value: Mapped[float | None] = mapped_column(Float)
    payload: Mapped[dict | None] = mapped_column(JSON)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    customer: Mapped[Customer] = relationship(back_populates="events")


class CustomerFeature(Base):
    __tablename__ = "customer_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), index=True)
    features: Mapped[dict] = mapped_column(JSON)  # engineered feature snapshot
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), index=True)
    churn_probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str | None] = mapped_column(String(64))
    prediction_latency_ms: Mapped[float | None] = mapped_column(Float)
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class RiskScore(Base):
    __tablename__ = "risk_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), index=True)
    previous_probability: Mapped[float | None] = mapped_column(Float)
    new_probability: Mapped[float] = mapped_column(Float)
    risk_change: Mapped[float | None] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    revenue_at_risk: Mapped[float | None] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), index=True)
    recommended_action: Mapped[str] = mapped_column(String(48))
    urgency: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text)
    priority_score: Mapped[float | None] = mapped_column(Float)
    estimated_net_value: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    base_model: Mapped[str | None] = mapped_column(String(48))
    calibration: Mapped[str | None] = mapped_column(String(24))
    roc_auc: Mapped[float | None] = mapped_column(Float)
    pr_auc: Mapped[float | None] = mapped_column(Float)
    f1: Mapped[float | None] = mapped_column(Float)
    brier: Mapped[float | None] = mapped_column(Float)
    stage: Mapped[str] = mapped_column(String(16), default="development")
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_name: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str | None] = mapped_column(String(48))
    params: Mapped[dict | None] = mapped_column(JSON)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class MonitoringMetric(Base):
    __tablename__ = "monitoring_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metric_name: Mapped[str] = mapped_column(String(48), index=True)
    metric_value: Mapped[float | None] = mapped_column(Float)
    context: Mapped[dict | None] = mapped_column(JSON)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


ALL_TABLES = [
    Customer, Subscription, CustomerEvent, CustomerFeature, Prediction,
    RiskScore, Recommendation, ModelVersion, Experiment, MonitoringMetric,
]
