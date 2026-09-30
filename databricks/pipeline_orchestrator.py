"""Databricks Medallion Architecture Pipeline Orchestrator.

Orchestrates the entire Bronze -> Silver -> Gold Delta Lake pipeline:
1. Bronze: Ingestion of raw customer data and streaming events with metadata tracking.
2. Silver: Type casting, data cleansing, constraint checking, deduplication.
3. Gold: Feature store generation, aggregations, customer 360 attributes.
4. Generates an execution summary report and quality audit logs.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Bootstrap path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from databricks.bronze_ingestion import BronzeIngestionPipeline
from databricks.silver_cleaning import SilverCleaningPipeline
from databricks.gold_feature_store import GoldFeatureStorePipeline
from src.config import Config, get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("churn.databricks.orchestrator")


class MedallionPipelineOrchestrator:
    """Executes and verifies the Databricks Bronze-Silver-Gold pipeline."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or get_config()
        self.bronze = BronzeIngestionPipeline(cfg=self.cfg)
        self.silver = SilverCleaningPipeline(cfg=self.cfg)
        self.gold = GoldFeatureStorePipeline(cfg=self.cfg)

    def run_pipeline(self) -> Dict[str, Any]:
        """Execute Bronze -> Silver -> Gold sequentially with quality checkpoints."""
        start_time = time.perf_counter()
        logger.info("=== Starting Databricks Medallion Architecture Execution ===")

        # Step 1: Bronze Ingestion
        t0 = time.perf_counter()
        bronze_df = self.bronze.ingest_customers_raw()
        bronze_time = round(time.perf_counter() - t0, 3)
        logger.info("[BRONZE COMPLETE] Ingested %d records in %.3fs", len(bronze_df), bronze_time)

        # Step 2: Silver Cleaning & Standardization
        t1 = time.perf_counter()
        silver_df, quality_report = self.silver.clean_customers(bronze_df)
        silver_time = round(time.perf_counter() - t1, 3)
        logger.info("[SILVER COMPLETE] Processed %d clean records in %.3fs (dropped %d invalid)",
                    len(silver_df), silver_time, quality_report.get("invalid_records_dropped", 0))

        # Step 3: Gold Feature Store Engineering
        t2 = time.perf_counter()
        gold_df = self.gold.build_gold_features(silver_df)
        gold_time = round(time.perf_counter() - t2, 3)
        logger.info("[GOLD COMPLETE] Engineered %d features across %d records in %.3fs",
                    len(gold_df.columns), len(gold_df), gold_time)

        total_time = round(time.perf_counter() - start_time, 3)

        summary = {
            "pipeline_name": "Databricks_Medallion_Bronze_Silver_Gold",
            "status": "SUCCESS",
            "total_runtime_seconds": total_time,
            "bronze": {
                "records": len(bronze_df),
                "columns": len(bronze_df.columns),
                "runtime_seconds": bronze_time,
                "location": str(self.bronze.customers_bronze_path),
            },
            "silver": {
                "records": len(silver_df),
                "columns": len(silver_df.columns),
                "runtime_seconds": silver_time,
                "quality_report": quality_report,
                "location": str(self.silver.customers_silver_path),
            },
            "gold": {
                "records": len(gold_df),
                "features_count": len(gold_df.columns),
                "runtime_seconds": gold_time,
                "location": str(self.gold.features_gold_path),
            }
        }

        # Save orchestrator run log
        log_path = self.cfg.root / "results" / "databricks_pipeline_summary.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        logger.info("=== Medallion Pipeline Execution Succeeded in %.3fs ===", total_time)
        return summary


def main():
    orchestrator = MedallionPipelineOrchestrator()
    summary = orchestrator.run_pipeline()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
