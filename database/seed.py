"""Seed the database from the real dataset + computed artifacts.

Customers/subscriptions come from the REAL IBM Telco data. Predictions,
risk scores, and recommendations come from the model artifacts produced in
earlier phases (if present). Everything is idempotent-ish: seeding clears the
relevant tables first.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from src.config import Config, get_config
from src.db.models import (
    Customer,
    ModelVersion,
    Prediction,
    Recommendation,
    RiskScore,
    Subscription,
)
from src.ingestion.load_data import load_raw
from src.preprocessing.preprocess import clean_raw
from src.risk import classify_risk

logger = logging.getLogger("churn.seed")

_SERVICE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def seed_customers(session: Session, limit: int | None = None, cfg: Config | None = None) -> int:
    """Seed customers + subscriptions from the real dataset. Returns count."""
    cfg = cfg or get_config()
    df = clean_raw(load_raw(cfg=cfg))
    if limit:
        df = df.head(limit)

    session.execute(delete(Subscription))
    session.execute(delete(Customer))

    for row in df.to_dict(orient="records"):
        session.add(Customer(
            customer_id=row["customerID"],
            gender=row.get("gender"),
            senior_citizen=int(row.get("SeniorCitizen", 0)),
            partner=row.get("Partner"),
            dependents=row.get("Dependents"),
            tenure=int(row.get("tenure", 0)),
        ))
        session.add(Subscription(
            customer_id=row["customerID"],
            contract=row.get("Contract"),
            payment_method=row.get("PaymentMethod"),
            paperless_billing=row.get("PaperlessBilling"),
            monthly_charges=float(row.get("MonthlyCharges", 0) or 0),
            total_charges=float(row.get("TotalCharges", 0) or 0),
            internet_service=row.get("InternetService"),
            phone_service=row.get("PhoneService"),
            services={c: row.get(c) for c in _SERVICE_COLS},
            churn=int(row.get("Churn", 0)),
        ))
    session.commit()
    return int(len(df))


def seed_model_version(session: Session, cfg: Config | None = None) -> bool:
    cfg = cfg or get_config()
    meta_path = cfg.resolve_path("paths.models") / "production_model_meta.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    vm = meta.get("val_metrics", {})
    session.execute(delete(ModelVersion))
    session.add(ModelVersion(
        model_version=meta.get("model_version", "unknown"),
        base_model=meta.get("base_model"),
        calibration=meta.get("calibration"),
        roc_auc=vm.get("roc_auc"), pr_auc=vm.get("pr_auc"),
        f1=vm.get("f1"), brier=vm.get("brier"),
        stage="production",
    ))
    session.commit()
    return True


def seed_predictions_and_recommendations(session: Session, cfg: Config | None = None) -> int:
    """Seed predictions/risk/recommendations from parquet artifacts if present."""
    cfg = cfg or get_config()
    features_dir = cfg.resolve_path("paths.data_features")
    rec_path = features_dir / "retention_recommendations.parquet"
    if not rec_path.exists():
        return 0

    model_version = "unknown"
    meta_path = cfg.resolve_path("paths.models") / "production_model_meta.json"
    if meta_path.exists():
        model_version = json.loads(meta_path.read_text(encoding="utf-8")).get(
            "model_version", "unknown")

    recs = pd.read_parquet(rec_path)
    session.execute(delete(Recommendation))
    session.execute(delete(RiskScore))
    session.execute(delete(Prediction))

    n = 0
    for r in recs.to_dict(orient="records"):
        cid = r["customer_id"]
        prob = float(r["churn_probability"])
        level = r.get("risk_level") or classify_risk(prob)
        session.add(Prediction(customer_id=cid, churn_probability=prob,
                               risk_level=level, model_version=model_version))
        session.add(RiskScore(customer_id=cid, previous_probability=None,
                              new_probability=prob, risk_change=None,
                              risk_level=level,
                              revenue_at_risk=float(r.get("revenue_at_risk", 0) or 0)))
        session.add(Recommendation(
            customer_id=cid,
            recommended_action=r.get("recommended_action", "no_intervention"),
            urgency=r.get("urgency", "LOW"),
            reason=r.get("reason"),
            priority_score=float(r.get("priority_score", 0) or 0),
            estimated_net_value=float(r.get("estimated_net_value", 0) or 0),
        ))
        n += 1
    session.commit()
    return n
