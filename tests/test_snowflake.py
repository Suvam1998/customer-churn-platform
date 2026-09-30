"""Tests for Snowflake Data Cloud integration & Customer Churn DB schemas."""
from __future__ import annotations

import pytest
import pandas as pd
from snowflake.snowflake_client import SnowflakeClient
from snowflake.load_data import SnowflakeDataLoader


def test_snowflake_client_init():
    client = SnowflakeClient()
    assert client.database == "CUSTOMER_CHURN_DB"
    assert client.warehouse == "CHURN_WH"
    # Should work in mock or live mode
    assert isinstance(client.is_mock, bool)


def test_snowflake_data_loader():
    loader = SnowflakeDataLoader()
    summary = loader.load_all()
    assert summary["status"] == "SUCCESS"
    assert summary["records_loaded"]["RAW_CUSTOMERS"] > 0
    assert summary["records_loaded"]["RAW_CUSTOMER_EVENTS"] > 0
    assert summary["records_loaded"]["ANALYTICS_CUSTOMER_FEATURES"] > 0
    assert summary["records_loaded"]["ML_CHURN_PREDICTIONS"] > 0
    assert summary["records_loaded"]["REPORTING_RETENTION_ACTIONS"] > 0


def test_snowflake_queries():
    client = SnowflakeClient()
    exec_summary = client.get_executive_summary()
    assert isinstance(exec_summary, dict)
    assert exec_summary.get("TOTAL_CUSTOMERS", 0) > 0

    q_high_risk = """
        SELECT CUSTOMER_ID, CHURN_PROBABILITY, RISK_LEVEL, REVENUE_AT_RISK
        FROM ML_CHURN_PREDICTIONS
        WHERE RISK_LEVEL IN ('HIGH', 'CRITICAL')
        LIMIT 5;
    """
    df_high_risk = client.query_df(q_high_risk)
    assert isinstance(df_high_risk, pd.DataFrame)
    assert len(df_high_risk) > 0
