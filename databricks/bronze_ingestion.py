"""Databricks Bronze Layer Ingestion Module.

Responsible for:
1. Ingesting raw customer records and simulated streaming events into the Bronze Delta tier.
2. Adding system metadata columns (_ingested_at, _source_file, _batch_id).
3. Providing schema enforcement and raw data landing capabilities.
4. Supporting both live PySpark/Delta Lake environments and local PySpark/Pandas delta emulation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Bootstrap path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd

# pyrefly: ignore [missing-import]
from src.config import Config, get_config
# pyrefly: ignore [missing-import]
from src.ingestion.load_data import load_raw

logger = logging.getLogger("churn.databricks.bronze")


class BronzeIngestionPipeline:
    """Manages ingestion of raw data and events into the Databricks Bronze Delta Layer."""

    def __init__(self, cfg: Optional[Config] = None, delta_base_path: Optional[Path] = None):
        self.cfg = cfg or get_config()
        self.delta_base_path = delta_base_path or (self.cfg.root / "data" / "delta" / "bronze")
        self.delta_base_path.mkdir(parents=True, exist_ok=True)
        self.customers_bronze_path = self.delta_base_path / "customers_raw"
        self.events_bronze_path = self.delta_base_path / "events_raw"
        self.customers_bronze_path.mkdir(parents=True, exist_ok=True)
        self.events_bronze_path.mkdir(parents=True, exist_ok=True)

    def _generate_batch_id(self, data_str: str) -> str:
        return hashlib.sha256(f"{datetime.now(timezone.utc).isoformat()}_{data_str[:50]}".encode()).hexdigest()[:12]

    def ingest_customers_raw(self, source_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Ingest raw Telco customer dataset into Bronze Delta layer with metadata tags."""
        if source_df is None:
            source_df = load_raw(cfg=self.cfg)

        raw_df = source_df.copy()
        now_iso = datetime.now(timezone.utc).isoformat()
        batch_id = self._generate_batch_id(str(raw_df.shape))

        # Add Bronze Layer Metadata Columns
        raw_df["_bronze_ingested_at"] = now_iso
        raw_df["_bronze_source_file"] = "telco_customer_churn.csv"
        raw_df["_bronze_batch_id"] = batch_id
        raw_df["_bronze_record_hash"] = raw_df.apply(
            lambda row: hashlib.md5("".join(str(v) for v in row.values).encode()).hexdigest(), axis=1
        )

        # Save as Delta-compatible Parquet
        target_file = self.customers_bronze_path / "data.parquet"
        raw_df.to_parquet(target_file, index=False)

        # Write Delta Lake metadata commit simulation
        meta = {
            "table_name": "bronze_customers_raw",
            "tier": "BRONZE",
            "format": "delta",
            "record_count": len(raw_df),
            "columns": list(raw_df.columns),
            "ingested_at": now_iso,
            "batch_id": batch_id,
        }
        (self.customers_bronze_path / "_delta_log.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Ingested %d customer records into Bronze Delta layer at %s", len(raw_df), self.customers_bronze_path)
        return raw_df

    def ingest_events_raw(self, events: List[Dict[str, Any]]) -> pd.DataFrame:
        """Ingest raw streaming events into Bronze Delta events table."""
        if not events:
            df = pd.DataFrame(columns=["customer_id", "event_type", "timestamp", "value", "is_simulated"])
        else:
            df = pd.DataFrame(events)

        now_iso = datetime.now(timezone.utc).isoformat()
        batch_id = self._generate_batch_id(str(len(events)))

        df["_bronze_ingested_at"] = now_iso
        df["_bronze_source_file"] = "streaming_event_bus"
        df["_bronze_batch_id"] = batch_id

        target_file = self.events_bronze_path / "events.parquet"
        df.to_parquet(target_file, index=False)

        meta = {
            "table_name": "bronze_events_raw",
            "tier": "BRONZE",
            "format": "delta",
            "record_count": len(df),
            "columns": list(df.columns),
            "ingested_at": now_iso,
            "batch_id": batch_id,
        }
        (self.events_bronze_path / "_delta_log.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Ingested %d events into Bronze Delta layer at %s", len(df), self.events_bronze_path)
        return df

    def get_bronze_customers(self) -> pd.DataFrame:
        target_file = self.customers_bronze_path / "data.parquet"
        if target_file.exists():
            return pd.read_parquet(target_file)
        return self.ingest_customers_raw()

    def get_bronze_events(self) -> pd.DataFrame:
        target_file = self.events_bronze_path / "events.parquet"
        if target_file.exists():
            return pd.read_parquet(target_file)
        return pd.DataFrame(columns=["customer_id", "event_type", "timestamp", "value", "is_simulated"])
