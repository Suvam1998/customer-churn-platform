"""Phase 12 tests: CLV, revenue-at-risk, priority scoring, RQ3 shift."""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import get_config
from src.retention.scoring import (
    compute_value_table,
    estimate_clv,
    prioritization_shift,
    revenue_at_risk,
    score_customers,
)
from src.validation.schema import ID_COLUMN


def test_estimate_clv_formula():
    # 100 * 24 * 0.30 = 720
    assert estimate_clv(100.0, 24, 0.30) == pytest.approx(720.0)


def test_revenue_at_risk_formula():
    assert revenue_at_risk(0.5, 720.0) == pytest.approx(360.0)


def test_compute_value_table_columns_and_consistency():
    cfg = get_config()
    df = pd.DataFrame({
        ID_COLUMN: ["a", "b", "c", "d"],
        "MonthlyCharges": [20.0, 50.0, 100.0, 80.0],
        "churn_probability": [0.05, 0.4, 0.9, 0.75],
    })
    vt = compute_value_table(df, cfg=cfg)
    lifetime = cfg.get("clv.expected_lifetime_months", 24)
    margin = cfg.get("clv.gross_margin", 0.30)

    # CLV and revenue-at-risk follow the documented formulas.
    assert vt.loc[0, "estimated_clv"] == pytest.approx(20 * lifetime * margin)
    assert vt.loc[2, "revenue_at_risk"] == pytest.approx(0.9 * 100 * lifetime * margin, rel=1e-3)
    # Priority ranking is by revenue at risk.
    assert vt.sort_values("priority_rank").iloc[0][ID_COLUMN] == "c"
    assert set(vt["risk_level"]).issubset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})


def test_prioritization_shift_differs_from_prob_ranking():
    # Construct a case where value reorders the top list.
    df = pd.DataFrame({
        ID_COLUMN: [f"c{i}" for i in range(10)],
        "MonthlyCharges": [10, 10, 10, 10, 10, 200, 200, 200, 200, 200],
        "churn_probability": [0.9, 0.85, 0.8, 0.75, 0.7, 0.4, 0.35, 0.3, 0.25, 0.2],
    })
    vt = compute_value_table(df)
    shift = prioritization_shift(vt, top_percent=0.3)  # top 3
    # High-value low-prob customers should enter the value-ranked top set.
    assert shift["swapped_in_by_value"] >= 1
    assert shift["overlap"] < shift["top_n"]


def _has_model():
    return (get_config().resolve_path("paths.models") / "production_model.joblib").exists()


@pytest.mark.skipif(not _has_model(), reason="run Phase 9 first")
def test_score_customers_real_data():
    vt = compute_value_table(score_customers())
    assert (vt["revenue_at_risk"] <= vt["estimated_clv"]).all()  # prob <= 1
    assert (vt["revenue_at_risk"] >= 0).all()
    assert vt["priority_flag"].sum() > 0
