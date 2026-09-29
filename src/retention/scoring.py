"""Customer value, revenue-at-risk, and priority scoring.

IMPORTANT: every monetary figure here is an ESTIMATE built on configurable
business assumptions (``configs/config.yaml``), NOT an observed financial
outcome. The IBM Telco dataset contains no revenue ledger or realised churn
losses.

Formulas (documented):
    CLV (estimate)      = MonthlyCharges * expected_lifetime_months * gross_margin
    revenue_at_risk     = churn_probability * CLV
    priority_score      = revenue_at_risk        (expected value at risk)

Ranking customers by revenue_at_risk (value-weighted) differs from ranking by
churn_probability alone — this is the basis of RQ3.
"""
from __future__ import annotations

import joblib
import pandas as pd

from src.config import Config, get_config
from src.explainability.shap_explainer import engineered_dataset
from src.features.feature_engineering import FEATURED_FEATURE_COLUMNS
from src.risk import classify_risk
from src.validation.schema import ID_COLUMN, TARGET_COLUMN


def estimate_clv(
    monthly_charges: float,
    expected_lifetime_months: float,
    gross_margin: float,
) -> float:
    """Estimated customer lifetime value (see module docstring)."""
    return float(monthly_charges) * float(expected_lifetime_months) * float(gross_margin)


def revenue_at_risk(churn_probability: float, customer_value: float) -> float:
    """Estimated revenue at risk = churn probability * estimated customer value."""
    return float(churn_probability) * float(customer_value)


def score_customers(cfg: Config | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load engineered customers, attach model churn probability."""
    cfg = cfg or get_config()
    df = df if df is not None else engineered_dataset(cfg)

    model_path = cfg.resolve_path("paths.models") / "production_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run scripts/calibrate_model.py (Phase 9) first."
        )
    model = joblib.load(model_path)
    out = df.copy()
    out["churn_probability"] = model.predict_proba(out[FEATURED_FEATURE_COLUMNS])[:, 1]
    return out


def compute_value_table(df_scored: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Add CLV, revenue_at_risk, risk_level, and priority ranking columns.

    ``df_scored`` must contain ``MonthlyCharges`` and ``churn_probability``.
    """
    cfg = cfg or get_config()
    lifetime = cfg.get("clv.expected_lifetime_months", 24)
    margin = cfg.get("clv.gross_margin", 0.30)

    out = df_scored.copy()
    out["estimated_clv"] = (out["MonthlyCharges"] * lifetime * margin).round(2)
    out["revenue_at_risk"] = (out["churn_probability"] * out["estimated_clv"]).round(2)
    out["risk_level"] = out["churn_probability"].apply(classify_risk)
    out["priority_score"] = out["revenue_at_risk"]  # value-weighted priority

    # Priority ranking + top-percentile flag.
    out["priority_rank"] = out["priority_score"].rank(ascending=False, method="first").astype(int)
    top_pct = cfg.get("retention.priority_top_percent", 0.10)
    threshold_rank = max(1, int(len(out) * top_pct))
    out["priority_flag"] = out["priority_rank"] <= threshold_rank
    return out


def value_columns() -> list[str]:
    return ["estimated_clv", "revenue_at_risk", "risk_level",
            "priority_score", "priority_rank", "priority_flag"]


def prioritization_shift(value_table: pd.DataFrame, top_percent: float = 0.10) -> dict:
    """RQ3: how differently are customers prioritised by revenue-at-risk vs.
    churn probability alone?"""
    n = len(value_table)
    top_n = max(1, int(n * top_percent))

    by_prob = set(
        value_table.sort_values("churn_probability", ascending=False)
        .head(top_n)[ID_COLUMN]
    )
    by_rar = set(
        value_table.sort_values("revenue_at_risk", ascending=False)
        .head(top_n)[ID_COLUMN]
    )
    overlap = by_prob & by_rar
    only_rar = by_rar - by_prob
    return {
        "top_percent": top_percent,
        "top_n": top_n,
        "overlap": len(overlap),
        "overlap_pct": round(len(overlap) / top_n * 100, 1),
        "swapped_in_by_value": len(only_rar),
        "swapped_pct": round(len(only_rar) / top_n * 100, 1),
    }
