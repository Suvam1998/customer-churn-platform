"""Tests for Databricks Medallion Architecture (Bronze-Silver-Gold) pipeline."""
from __future__ import annotations

import pytest
import pandas as pd
from databricks.bronze_ingestion import BronzeIngestionPipeline
from databricks.silver_cleaning import SilverCleaningPipeline
from databricks.gold_feature_store import GoldFeatureStorePipeline
from databricks.pipeline_orchestrator import MedallionPipelineOrchestrator


def test_bronze_ingestion():
    bronze = BronzeIngestionPipeline()
    df = bronze.ingest_customers_raw()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "_bronze_ingested_at" in df.columns
    assert "_bronze_batch_id" in df.columns
    assert "_bronze_record_hash" in df.columns


def test_bronze_events_ingestion():
    bronze = BronzeIngestionPipeline()
    events = [
        {"customer_id": "7590-VHVEG", "event_type": "complaint", "timestamp": "2026-09-30T12:00:00Z", "value": 1.0, "is_simulated": True}
    ]
    df_evt = bronze.ingest_events_raw(events)
    assert len(df_evt) == 1
    assert "_bronze_ingested_at" in df_evt.columns


def test_silver_cleaning():
    silver = SilverCleaningPipeline()
    df, quality_report = silver.clean_customers()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert quality_report["quality_status"] == "PASSED"
    assert "_silver_processed_at" in df.columns
    assert "Churn_Numeric" in df.columns
    assert (df["tenure"] >= 0).all()
    assert (df["MonthlyCharges"] >= 0).all()
    assert (df["TotalCharges"] >= 0).all()


def test_gold_feature_store():
    gold = GoldFeatureStorePipeline()
    df = gold.build_gold_features()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "tenure_cohort" in df.columns
    assert "contract_risk_index" in df.columns
    assert "security_bundle_score" in df.columns
    assert "total_active_services" in df.columns
    assert "estimated_clv" in df.columns
    assert "_gold_feature_timestamp" in df.columns


def test_medallion_orchestrator():
    orchestrator = MedallionPipelineOrchestrator()
    summary = orchestrator.run_pipeline()
    assert summary["status"] == "SUCCESS"
    assert summary["bronze"]["records"] > 0
    assert summary["silver"]["records"] > 0
    assert summary["gold"]["features_count"] > 30
