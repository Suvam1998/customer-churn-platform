"""Databricks Gold Layer Feature Store & Aggregation Module.

Responsible for:
1. Reading validated datasets from the Silver Delta tier.
2. Engineering comprehensive business, demographic, financial, and event-based features.
3. Building aggregated Customer 360 gold tables for analytical querying and ML consumption.
4. Writing to the Gold Delta tier (gold_customer_features_daily, gold_churn_features).
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Bootstrap path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd

from src.config import Config, get_config
from databricks.silver_cleaning import SilverCleaningPipeline

logger = logging.getLogger("churn.databricks.gold")


class GoldFeatureStorePipeline:
    """Builds the Gold Delta Tier Feature Store for ML training and Customer 360 analytics."""

    def __init__(self, cfg: Optional[Config] = None, delta_base_path: Optional[Path] = None):
        self.cfg = cfg or get_config()
        self.delta_base_path = delta_base_path or (self.cfg.root / "data" / "delta" / "gold")
        self.delta_base_path.mkdir(parents=True, exist_ok=True)
        self.features_gold_path = self.delta_base_path / "customer_features_daily"
        self.features_gold_path.mkdir(parents=True, exist_ok=True)
        self.silver = SilverCleaningPipeline(cfg=self.cfg)

    def build_gold_features(self, silver_customers: Optional[pd.DataFrame] = None, silver_events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Engineer advanced Gold tier features from Silver customers and events."""
        if silver_customers is None:
            silver_customers = self.silver.get_silver_customers()
        if silver_events is None:
            silver_events = self.silver.get_silver_events()

        df = silver_customers.copy()

        # 1. Tenure groupings
        df["tenure_cohort"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 72, 100],
            labels=["0-12m (New)", "12-24m (Early)", "24-48m (Established)", "48-72m (Loyal)", "72m+ (Champion)"]
        ).astype(str)

        # 2. Financial ratios
        safe_tenure = df["tenure"].replace(0, 1)
        df["avg_monthly_charges_history"] = df["TotalCharges"] / safe_tenure
        df["charge_discrepancy"] = df["MonthlyCharges"] - df["avg_monthly_charges_history"]
        df["charge_ratio"] = (df["MonthlyCharges"] / (df["avg_monthly_charges_history"] + 1e-6)).clip(0.1, 10.0)

        # 3. Service bundle scores
        security_services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
        df["security_bundle_score"] = 0
        for s in security_services:
            if s in df.columns:
                df["security_bundle_score"] += (df[s] == "Yes").astype(int)

        streaming_services = ["StreamingTV", "StreamingMovies"]
        df["streaming_bundle_score"] = 0
        for s in streaming_services:
            if s in df.columns:
                df["streaming_bundle_score"] += (df[s] == "Yes").astype(int)

        df["total_active_services"] = (
            (df.get("PhoneService", "No") == "Yes").astype(int) +
            (df.get("MultipleLines", "No") == "Yes").astype(int) +
            (df.get("InternetService", "No").isin(["DSL", "Fiber optic"])).astype(int) +
            df["security_bundle_score"] +
            df["streaming_bundle_score"]
        )

        # 4. Behavioral & Contract risk weights
        contract_risk_map = {"Month-to-month": 3, "One year": 2, "Two year": 1}
        df["contract_risk_index"] = df["Contract"].map(contract_risk_map).fillna(2).astype(int)

        df["is_high_risk_payment"] = (df["PaymentMethod"] == "Electronic check").astype(int)
        df["is_fiber_without_techsupport"] = (
            (df["InternetService"] == "Fiber optic") & (df["TechSupport"] != "Yes")
        ).astype(int)

        # 5. Simulated Event Aggregations (from Silver events if present)
        if not silver_events.empty and "customer_id" in silver_events.columns:
            evt_agg = silver_events.groupby("customer_id").agg(
                complaint_count=("event_type", lambda s: (s == "complaint").sum()),
                support_ticket_count=("event_type", lambda s: (s == "support_ticket").sum()),
                payment_failed_count=("event_type", lambda s: (s == "payment_failed").sum()),
                login_count=("event_type", lambda s: (s == "login").sum()),
                recent_events_count=("event_type", "count")
            ).reset_index()
            df = df.merge(evt_agg, left_on="customerID", right_on="customer_id", how="left")
            for col in ["complaint_count", "support_ticket_count", "payment_failed_count", "login_count", "recent_events_count"]:
                df[col] = df[col].fillna(0).astype(int)
        else:
            df["complaint_count"] = 0
            df["support_ticket_count"] = 0
            df["payment_failed_count"] = 0
            df["login_count"] = 0
            df["recent_events_count"] = 0

        # 6. Customer Lifetime Value (CLV) Baseline
        margin = self.cfg.get("clv.gross_margin", 0.30)
        expected_lifetime = self.cfg.get("clv.expected_lifetime_months", 24)
        df["estimated_clv"] = (df["MonthlyCharges"] * expected_lifetime * margin).round(2)

        # 7. Add Gold tier metadata
        now_iso = datetime.now(timezone.utc).isoformat()
        df["_gold_feature_timestamp"] = now_iso
        df["_gold_feature_version"] = "v2.0"

        # Save Gold Delta table
        target_file = self.features_gold_path / "gold_features.parquet"
        df.to_parquet(target_file, index=False)

        meta = {
            "table_name": "gold_customer_features_daily",
            "tier": "GOLD",
            "record_count": len(df),
            "feature_count": len(df.columns),
            "columns": list(df.columns),
            "computed_at": now_iso,
        }
        (self.features_gold_path / "_delta_log.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Computed %d Gold feature records at %s", len(df), self.features_gold_path)
        return df

    def get_gold_features(self) -> pd.DataFrame:
        target_file = self.features_gold_path / "gold_features.parquet"
        if target_file.exists():
            return pd.read_parquet(target_file)
        return self.build_gold_features()
