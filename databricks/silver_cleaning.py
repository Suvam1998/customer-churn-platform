"""Databricks Silver Layer Cleaning & Standardization Module.

Responsible for:
1. Reading raw tables from the Bronze Delta tier.
2. Cleaning, parsing, and standardizing data types (e.g., whitespace TotalCharges to float).
3. Schema validation, constraint checking, and deduplication.
4. Handling missing values and standardization of categorical fields.
5. Writing to the Silver Delta tier (silver_customers_clean, silver_events_clean).
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

# Bootstrap path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd

from src.config import Config, get_config
from databricks.bronze_ingestion import BronzeIngestionPipeline

logger = logging.getLogger("churn.databricks.silver")


class SilverCleaningPipeline:
    """Cleans, standardizes, and validates Bronze layer data into the Silver Delta tier."""

    def __init__(self, cfg: Optional[Config] = None, delta_base_path: Optional[Path] = None):
        self.cfg = cfg or get_config()
        self.delta_base_path = delta_base_path or (self.cfg.root / "data" / "delta" / "silver")
        self.delta_base_path.mkdir(parents=True, exist_ok=True)
        self.customers_silver_path = self.delta_base_path / "customers_clean"
        self.events_silver_path = self.delta_base_path / "events_clean"
        self.customers_silver_path.mkdir(parents=True, exist_ok=True)
        self.events_silver_path.mkdir(parents=True, exist_ok=True)
        self.bronze = BronzeIngestionPipeline(cfg=self.cfg)

    def clean_customers(self, bronze_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, dict]:
        """Clean, standardize, and validate customer records from Bronze."""
        if bronze_df is None:
            bronze_df = self.bronze.get_bronze_customers()

        df = bronze_df.copy()
        raw_count = len(df)

        # 1. Deduplication on customerID
        df = df.drop_duplicates(subset=["customerID"], keep="last")
        dedup_count = len(df)

        # 2. Type casting and null cleaning for TotalCharges
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
            df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0.0)
            df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"].clip(lower=1))

        # 3. Numeric conversions
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0).astype(float)
        df["TotalCharges"] = df["TotalCharges"].astype(float)

        # 4. Binary target mapping
        if "Churn" in df.columns:
            df["Churn_Numeric"] = df["Churn"].apply(lambda x: 1 if str(x).strip().lower() in ("yes", "1", "true") else 0)

        # 5. String standardization
        categorical_cols = [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"
        ]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # 6. Constraint checks
        valid_mask = (df["tenure"] >= 0) & (df["MonthlyCharges"] >= 0) & (df["TotalCharges"] >= 0)
        invalid_records = int((~valid_mask).sum())
        df = df[valid_mask].copy()

        # 7. Add Silver metadata
        now_iso = datetime.now(timezone.utc).isoformat()
        df["_silver_processed_at"] = now_iso
        df["_silver_quality_score"] = 1.0

        # Save to Silver Delta parquet
        target_file = self.customers_silver_path / "data.parquet"
        df.to_parquet(target_file, index=False)

        quality_report = {
            "table_name": "silver_customers_clean",
            "raw_records": raw_count,
            "deduplicated_records": dedup_count,
            "invalid_records_dropped": invalid_records,
            "clean_records": len(df),
            "processed_at": now_iso,
            "quality_status": "PASSED",
        }
        (self.customers_silver_path / "_delta_log.json").write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
        logger.info("Cleaned %d records into Silver Delta layer at %s", len(df), self.customers_silver_path)
        return df, quality_report

    def clean_events(self, bronze_events_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Clean and validate streaming events from Bronze."""
        if bronze_events_df is None:
            bronze_events_df = self.bronze.get_bronze_events()

        if bronze_events_df.empty:
            df = pd.DataFrame(columns=["customer_id", "event_type", "timestamp", "value", "is_simulated", "_silver_processed_at"])
        else:
            df = bronze_events_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
            df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
            df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(1.0)
            df["_silver_processed_at"] = datetime.now(timezone.utc).isoformat()

        target_file = self.events_silver_path / "events.parquet"
        df.to_parquet(target_file, index=False)

        meta = {
            "table_name": "silver_events_clean",
            "tier": "SILVER",
            "record_count": len(df),
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.events_silver_path / "_delta_log.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return df

    def get_silver_customers(self) -> pd.DataFrame:
        target_file = self.customers_silver_path / "data.parquet"
        if target_file.exists():
            return pd.read_parquet(target_file)
        df, _ = self.clean_customers()
        return df

    def get_silver_events(self) -> pd.DataFrame:
        target_file = self.events_silver_path / "events.parquet"
        if target_file.exists():
            return pd.read_parquet(target_file)
        return self.clean_events()
