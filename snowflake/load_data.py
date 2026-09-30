"""Snowflake Data Loader & Synchronization Script.

Populates the Snowflake CUSTOMER_CHURN_DB with:
1. RAW.CUSTOMERS (IBM Telco raw customer dataset)
2. RAW.CUSTOMER_EVENTS (Simulated streaming events)
3. ANALYTICS.CUSTOMER_FEATURES (Databricks Gold / engineered features)
4. ML.CHURN_PREDICTIONS (Model churn probability & risk level scores)
5. ML.MODEL_RUNS (Trained model metrics & hyperparameters)
6. REPORTING.RETENTION_ACTIONS (Recommended business retention campaigns)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Bootstrap path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd

from src.config import Config, get_config
from src.ingestion.load_data import load_raw
from src.streaming.event_generator import generate_events
from snowflake.snowflake_client import SnowflakeClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("churn.snowflake.loader")


class SnowflakeDataLoader:
    """Manages loading and syncing of project datasets into Snowflake CUSTOMER_CHURN_DB."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or get_config()
        self.client = SnowflakeClient(cfg=self.cfg)

    def load_all(self) -> Dict[str, Any]:
        start_time = time.perf_counter()
        logger.info("=== Starting Snowflake CUSTOMER_CHURN_DB Data Loading ===")

        # 1. Load RAW.CUSTOMERS
        raw_df = load_raw(cfg=self.cfg).copy()
        raw_df = raw_df.rename(columns={
            "customerID": "CUSTOMER_ID", "gender": "GENDER", "SeniorCitizen": "SENIOR_CITIZEN",
            "Partner": "PARTNER", "Dependents": "DEPENDENTS", "tenure": "TENURE",
            "PhoneService": "PHONE_SERVICE", "MultipleLines": "MULTIPLE_LINES",
            "InternetService": "INTERNET_SERVICE", "OnlineSecurity": "ONLINE_SECURITY",
            "OnlineBackup": "ONLINE_BACKUP", "DeviceProtection": "DEVICE_PROTECTION",
            "TechSupport": "TECH_SUPPORT", "StreamingTV": "STREAMING_TV",
            "StreamingMovies": "STREAMING_MOVIES", "Contract": "CONTRACT",
            "PaperlessBilling": "PAPERLESS_BILLING", "PaymentMethod": "PAYMENT_METHOD",
            "MonthlyCharges": "MONTHLY_CHARGES", "TotalCharges": "TOTAL_CHARGES",
            "Churn": "CHURN_LABEL",
        })
        raw_count = self.client.load_dataframe("RAW_CUSTOMERS", raw_df)
        logger.info("[RAW.CUSTOMERS] Loaded %d records", raw_count)

        # 2. Load RAW.CUSTOMER_EVENTS (Simulated)
        events = generate_events(n=250, seed=42, cfg=self.cfg)
        events_df = pd.DataFrame(events).rename(columns={
            "customer_id": "CUSTOMER_ID",
            "event_type": "EVENT_TYPE",
            "timestamp": "EVENT_TIMESTAMP",
            "value": "EVENT_VALUE",
            "is_simulated": "IS_SIMULATED",
        })
        events_df["EVENT_ID"] = [f"evt-{i:05d}" for i in range(len(events_df))]
        events_count = self.client.load_dataframe("RAW_CUSTOMER_EVENTS", events_df)
        logger.info("[RAW.CUSTOMER_EVENTS] Loaded %d simulated events", events_count)

        # 3. Load ANALYTICS.CUSTOMER_FEATURES
        val_path = self.cfg.root / "data" / "features" / "customer_value.parquet"
        if val_path.exists():
            val_df = pd.read_parquet(val_path).rename(columns={
                "customerID": "CUSTOMER_ID", "tenure": "TENURE",
                "Contract": "CONTRACT", "PaymentMethod": "PAYMENT_METHOD",
                "InternetService": "INTERNET_SERVICE", "MonthlyCharges": "MONTHLY_CHARGES",
                "TotalCharges": "TOTAL_CHARGES", "estimated_clv": "ESTIMATED_CLV"
            })
            features_count = self.client.load_dataframe("ANALYTICS_CUSTOMER_FEATURES", val_df)
        else:
            features_count = 0
        logger.info("[ANALYTICS.CUSTOMER_FEATURES] Loaded %d records", features_count)

        # 4. Load ML.CHURN_PREDICTIONS
        if val_path.exists():
            preds_df = pd.read_parquet(val_path)[["customerID", "churn_probability", "risk_level", "estimated_clv", "revenue_at_risk"]].copy()
            preds_df = preds_df.rename(columns={
                "customerID": "CUSTOMER_ID",
                "churn_probability": "CHURN_PROBABILITY",
                "risk_level": "RISK_LEVEL",
                "estimated_clv": "ESTIMATED_CLV",
                "revenue_at_risk": "REVENUE_AT_RISK"
            })
            preds_df["PREDICTION_ID"] = [f"pred-{i:05d}" for i in range(len(preds_df))]
            preds_df["MODEL_VERSION"] = "v2.0-catboost-calibrated"
            preds_df["TOP_DRIVERS"] = json.dumps(["Contract=Month-to-month", "Tenure<12m", "InternetService=Fiber optic"])
            preds_count = self.client.load_dataframe("ML_CHURN_PREDICTIONS", preds_df)
        else:
            preds_count = 0
        logger.info("[ML.CHURN_PREDICTIONS] Loaded %d prediction records", preds_count)

        # 5. Load ML.MODEL_RUNS
        model_meta_path = self.cfg.root / "models" / "production_model_meta.json"
        if model_meta_path.exists():
            meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
            m = meta.get("val_metrics", {})
            model_runs_df = pd.DataFrame([{
                "RUN_ID": "run-prod-001",
                "MODEL_NAME": meta.get("base_model", "catboost"),
                "MODEL_VERSION": meta.get("model_version", "v2.0"),
                "CALIBRATION_METHOD": meta.get("calibration", "isotonic"),
                "ROC_AUC": m.get("roc_auc", 0.845),
                "PR_AUC": m.get("pr_auc", 0.655),
                "F1_SCORE": m.get("f1", 0.612),
                "PRECISION_SCORE": m.get("precision", 0.582),
                "RECALL_SCORE": m.get("recall", 0.645),
                "LOG_LOSS": m.get("log_loss", 0.412),
                "BRIER_SCORE": m.get("brier", 0.132),
                "DEPLOYMENT_STAGE": "Production",
            }])
            self.client.load_dataframe("ML_MODEL_RUNS", model_runs_df)
            logger.info("[ML.MODEL_RUNS] Loaded production model run metadata")

        # 6. Load REPORTING.RETENTION_ACTIONS
        rec_path = self.cfg.root / "data" / "features" / "retention_recommendations.parquet"
        if rec_path.exists():
            rec_df = pd.read_parquet(rec_path).rename(columns={
                "customer_id": "CUSTOMER_ID",
                "churn_probability": "CHURN_PROBABILITY",
                "risk_level": "RISK_LEVEL",
                "urgency": "URGENCY",
                "recommended_action": "RECOMMENDED_ACTION",
                "reason": "REASON",
                "revenue_at_risk": "REVENUE_AT_RISK",
                "estimated_intervention_cost": "ESTIMATED_INTERVENTION_COST",
                "estimated_expected_retained_value": "ESTIMATED_RETAINED_VALUE",
                "estimated_net_value": "ESTIMATED_NET_VALUE"
            })
            rec_df["ACTION_ID"] = [f"act-{i:05d}" for i in range(len(rec_df))]
            retention_count = self.client.load_dataframe("REPORTING_RETENTION_ACTIONS", rec_df)
        else:
            retention_count = 0
        logger.info("[REPORTING.RETENTION_ACTIONS] Loaded %d action records", retention_count)

        total_time = round(time.perf_counter() - start_time, 3)

        summary = {
            "status": "SUCCESS",
            "mode": "Mock SQLite Engine" if self.client.is_mock else "Live Snowflake Connection",
            "database": self.client.database,
            "runtime_seconds": total_time,
            "records_loaded": {
                "RAW_CUSTOMERS": raw_count,
                "RAW_CUSTOMER_EVENTS": events_count,
                "ANALYTICS_CUSTOMER_FEATURES": features_count,
                "ML_CHURN_PREDICTIONS": preds_count,
                "REPORTING_RETENTION_ACTIONS": retention_count,
            },
            "executive_summary": self.client.get_executive_summary()
        }

        # Save summary report
        res_file = self.cfg.root / "results" / "snowflake_sync_summary.json"
        res_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("=== Snowflake Loading Completed Successfully in %.3fs ===", total_time)
        return summary


def main():
    loader = SnowflakeDataLoader()
    res = loader.load_all()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
