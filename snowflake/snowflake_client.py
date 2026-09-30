"""Snowflake Integration Client.

Supports:
1. Live connection to Snowflake Data Cloud via `snowflake.connector` / `snowflake-sqlalchemy`.
2. Automatic Local/Mock Database engine fallback for offline demonstration and testing.
3. Loading dataframes into Snowflake tables (CUSTOMERS, CUSTOMER_EVENTS, CUSTOMER_FEATURES, CHURN_PREDICTIONS, RETENTION_ACTIONS, MODEL_RUNS).
4. Executing analytical views and reporting queries.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Bootstrap path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd

from src.config import Config, get_config

logger = logging.getLogger("churn.snowflake")


class SnowflakeClient:
    """Manages connection and querying for Snowflake CUSTOMER_CHURN_DB with mock fallback."""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or get_config()
        self.account = os.getenv("SNOWFLAKE_ACCOUNT", self.cfg.get("snowflake.account", ""))
        self.user = os.getenv("SNOWFLAKE_USER", self.cfg.get("snowflake.user", ""))
        self.password = os.getenv("SNOWFLAKE_PASSWORD", self.cfg.get("snowflake.password", ""))
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", self.cfg.get("snowflake.warehouse", "CHURN_WH"))
        self.database = os.getenv("SNOWFLAKE_DATABASE", self.cfg.get("snowflake.database", "CUSTOMER_CHURN_DB"))
        self.schema = os.getenv("SNOWFLAKE_SCHEMA", self.cfg.get("snowflake.schema", "REPORTING"))
        self.role = os.getenv("SNOWFLAKE_ROLE", self.cfg.get("snowflake.role", "CHURN_ADMIN_ROLE"))

        self.mock_db_path = self.cfg.root / "data" / "snowflake_mock.db"
        self._conn = None
        self._is_mock = not bool(self.account and self.user and self.password)

        if self._is_mock:
            logger.info("Snowflake credentials not configured in environment. Using Local Mock Engine at %s", self.mock_db_path)
            self._init_mock_db()
        else:
            logger.info("Initializing Snowflake Live Connection for account: %s, database: %s", self.account, self.database)

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    def _init_mock_db(self) -> None:
        """Initialize local SQLite database mirroring Snowflake CUSTOMER_CHURN_DB."""
        self.mock_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.mock_db_path) as conn:
            cur = conn.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS RAW_CUSTOMERS (
                    CUSTOMER_ID TEXT PRIMARY KEY,
                    GENDER TEXT,
                    SENIOR_CITIZEN INTEGER,
                    PARTNER TEXT,
                    DEPENDENTS TEXT,
                    TENURE INTEGER,
                    PHONE_SERVICE TEXT,
                    MULTIPLE_LINES TEXT,
                    INTERNET_SERVICE TEXT,
                    ONLINE_SECURITY TEXT,
                    ONLINE_BACKUP TEXT,
                    DEVICE_PROTECTION TEXT,
                    TECH_SUPPORT TEXT,
                    STREAMING_TV TEXT,
                    STREAMING_MOVIES TEXT,
                    CONTRACT TEXT,
                    PAPERLESS_BILLING TEXT,
                    PAYMENT_METHOD TEXT,
                    MONTHLY_CHARGES REAL,
                    TOTAL_CHARGES REAL,
                    CHURN_LABEL TEXT,
                    _INGESTED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS RAW_CUSTOMER_EVENTS (
                    EVENT_ID TEXT PRIMARY KEY,
                    CUSTOMER_ID TEXT NOT NULL,
                    EVENT_TYPE TEXT NOT NULL,
                    EVENT_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    EVENT_VALUE REAL DEFAULT 1.0,
                    EVENT_PAYLOAD TEXT,
                    IS_SIMULATED BOOLEAN DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS ANALYTICS_CUSTOMER_FEATURES (
                    CUSTOMER_ID TEXT PRIMARY KEY,
                    TENURE INTEGER,
                    TENURE_COHORT TEXT,
                    CONTRACT TEXT,
                    CONTRACT_RISK_INDEX INTEGER,
                    PAYMENT_METHOD TEXT,
                    IS_HIGH_RISK_PAYMENT INTEGER,
                    INTERNET_SERVICE TEXT,
                    IS_FIBER_WITHOUT_TECHSUPPORT INTEGER,
                    MONTHLY_CHARGES REAL,
                    TOTAL_CHARGES REAL,
                    AVG_MONTHLY_CHARGES_HISTORY REAL,
                    CHARGE_DISCREPANCY REAL,
                    CHARGE_RATIO REAL,
                    SECURITY_BUNDLE_SCORE INTEGER,
                    STREAMING_BUNDLE_SCORE INTEGER,
                    TOTAL_ACTIVE_SERVICES INTEGER,
                    COMPLAINT_COUNT INTEGER DEFAULT 0,
                    SUPPORT_TICKET_COUNT INTEGER DEFAULT 0,
                    PAYMENT_FAILED_COUNT INTEGER DEFAULT 0,
                    LOGIN_COUNT INTEGER DEFAULT 0,
                    RECENT_EVENTS_COUNT INTEGER DEFAULT 0,
                    ESTIMATED_CLV REAL
                );

                CREATE TABLE IF NOT EXISTS ML_CHURN_PREDICTIONS (
                    PREDICTION_ID TEXT PRIMARY KEY,
                    CUSTOMER_ID TEXT NOT NULL,
                    CHURN_PROBABILITY REAL NOT NULL,
                    RISK_LEVEL TEXT NOT NULL,
                    ESTIMATED_CLV REAL,
                    REVENUE_AT_RISK REAL,
                    TOP_DRIVERS TEXT,
                    MODEL_VERSION TEXT NOT NULL,
                    PREDICTION_LATENCY_MS REAL,
                    PREDICTION_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS ML_MODEL_RUNS (
                    RUN_ID TEXT PRIMARY KEY,
                    MODEL_NAME TEXT NOT NULL,
                    MODEL_VERSION TEXT NOT NULL,
                    CALIBRATION_METHOD TEXT,
                    ROC_AUC REAL,
                    PR_AUC REAL,
                    F1_SCORE REAL,
                    PRECISION_SCORE REAL,
                    RECALL_SCORE REAL,
                    LOG_LOSS REAL,
                    BRIER_SCORE REAL,
                    DEPLOYMENT_STAGE TEXT DEFAULT 'Production',
                    TRAINED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS REPORTING_RETENTION_ACTIONS (
                    ACTION_ID TEXT PRIMARY KEY,
                    CUSTOMER_ID TEXT NOT NULL,
                    CHURN_PROBABILITY REAL NOT NULL,
                    RISK_LEVEL TEXT NOT NULL,
                    URGENCY TEXT NOT NULL,
                    RECOMMENDED_ACTION TEXT NOT NULL,
                    REASON TEXT,
                    REVENUE_AT_RISK REAL,
                    ESTIMATED_INTERVENTION_COST REAL,
                    ESTIMATED_RETAINED_VALUE REAL,
                    ESTIMATED_NET_VALUE REAL,
                    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()

    def get_live_connection(self):
        """Open live connection to Snowflake."""
        if self._is_mock:
            return None
        try:
            import snowflake.connector
            return snowflake.connector.connect(
                account=self.account,
                user=self.user,
                password=self.password,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema,
                role=self.role,
            )
        except Exception as exc:
            logger.warning("Failed to connect to live Snowflake (%s). Falling back to mock engine.", exc)
            self._is_mock = True
            self._init_mock_db()
            return None

    def execute_query(self, query: str, params: Optional[Union[tuple, dict]] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results as a list of dicts."""
        if self._is_mock:
            # Clean snowflake specific keywords if any for sqlite
            clean_q = query.replace("CUSTOMER_CHURN_DB.RAW.CUSTOMERS", "RAW_CUSTOMERS") \
                           .replace("CUSTOMER_CHURN_DB.RAW.CUSTOMER_EVENTS", "RAW_CUSTOMER_EVENTS") \
                           .replace("CUSTOMER_CHURN_DB.ANALYTICS.CUSTOMER_FEATURES", "ANALYTICS_CUSTOMER_FEATURES") \
                           .replace("CUSTOMER_CHURN_DB.ML.CHURN_PREDICTIONS", "ML_CHURN_PREDICTIONS") \
                           .replace("CUSTOMER_CHURN_DB.ML.MODEL_RUNS", "ML_MODEL_RUNS") \
                           .replace("CUSTOMER_CHURN_DB.REPORTING.RETENTION_ACTIONS", "REPORTING_RETENTION_ACTIONS") \
                           .replace("TIMESTAMP_NTZ", "TIMESTAMP") \
                           .replace("NUMBER(38, 0)", "INTEGER") \
                           .replace("FLOAT", "REAL") \
                           .replace("VARIANT", "TEXT") \
                           .replace("UUID_STRING()", "'mock-uuid'") \
                           .replace("CURRENT_TIMESTAMP()", "CURRENT_TIMESTAMP")
            with sqlite3.connect(self.mock_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                try:
                    cur.execute(clean_q, params or ())
                    rows = cur.fetchall()
                    return [dict(r) for r in rows]
                except Exception as e:
                    logger.error("Error executing mock query: %s", e)
                    return []
        else:
            conn = self.get_live_connection()
            if conn is None:
                return self.execute_query(query, params)
            try:
                cur = conn.cursor()
                cur.execute(query, params)
                cols = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
                return [dict(zip(cols, row)) for row in rows]
            finally:
                conn.close()

    def query_df(self, query: str, params: Optional[Union[tuple, dict]] = None) -> pd.DataFrame:
        """Execute query and return as a Pandas DataFrame."""
        rows = self.execute_query(query, params)
        return pd.DataFrame(rows)

    def load_dataframe(self, table_name: str, df: pd.DataFrame, if_exists: str = "replace") -> int:
        """Load a Pandas DataFrame into Snowflake / Mock table."""
        if df.empty:
            return 0
        if self._is_mock:
            with sqlite3.connect(self.mock_db_path) as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
                return len(df)
        else:
            conn = self.get_live_connection()
            if conn is None:
                return self.load_dataframe(table_name, df, if_exists=if_exists)
            try:
                from snowflake.connector.pandas_tools import write_pandas
                success, nchunks, nrows, _ = write_pandas(
                    conn,
                    df,
                    table_name.upper(),
                    database=self.database,
                    schema=self.schema,
                    auto_create_table=True,
                    overwrite=(if_exists == "replace")
                )
                return nrows if success else 0
            finally:
                conn.close()

    def get_executive_summary(self) -> Dict[str, Any]:
        """Fetch executive churn summary metrics from Snowflake."""
        q = """
            SELECT
                COUNT(DISTINCT c.CUSTOMER_ID) AS TOTAL_CUSTOMERS,
                SUM(CASE WHEN p.RISK_LEVEL IN ('HIGH', 'CRITICAL') THEN 1 ELSE 0 END) AS AT_RISK_CUSTOMERS,
                ROUND(AVG(p.CHURN_PROBABILITY), 4) AS AVERAGE_CHURN_PROBABILITY,
                ROUND(SUM(p.REVENUE_AT_RISK), 2) AS TOTAL_REVENUE_AT_RISK,
                ROUND(SUM(r.ESTIMATED_NET_VALUE), 2) AS TOTAL_POTENTIAL_RECOVERY_VALUE
            FROM RAW_CUSTOMERS c
            LEFT JOIN ML_CHURN_PREDICTIONS p ON c.CUSTOMER_ID = p.CUSTOMER_ID
            LEFT JOIN REPORTING_RETENTION_ACTIONS r ON c.CUSTOMER_ID = r.CUSTOMER_ID;
        """
        rows = self.execute_query(q)
        return rows[0] if rows else {}
