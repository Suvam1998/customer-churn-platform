"""EDA analysis: churn-rate tables and numeric summaries (numbers, not plots).

Operates on the CLEANED frame (``Churn`` encoded 0/1, ``TotalCharges`` numeric).
All figures/reports downstream are built from these tables, so every reported
number traces back to the real dataset.
"""
from __future__ import annotations

import pandas as pd

from src.validation.schema import CATEGORICAL_COLUMNS, TARGET_COLUMN

# Human-friendly tenure bands (months).
TENURE_BINS = [-0.1, 12, 24, 36, 48, 60, 72]
TENURE_LABELS = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]


def overall_churn(df: pd.DataFrame) -> dict:
    """Overall churn counts and rate."""
    n = len(df)
    churned = int(df[TARGET_COLUMN].sum())
    return {
        "n_customers": n,
        "n_churned": churned,
        "n_retained": n - churned,
        "churn_rate": round(churned / n, 4),
    }


def churn_rate_by_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per-category churn rate + counts, sorted by churn rate descending."""
    grp = df.groupby(column, observed=True)[TARGET_COLUMN]
    out = pd.DataFrame({
        "n_customers": grp.size(),
        "n_churned": grp.sum(),
        "churn_rate": grp.mean().round(4),
    })
    return out.sort_values("churn_rate", ascending=False).reset_index()


def churn_rate_by_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["tenure_band"] = pd.cut(tmp["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)
    return churn_rate_by_category(tmp, "tenure_band").sort_values("tenure_band")


def churn_rate_by_monthly_charge_band(df: pd.DataFrame, n_bands: int = 5) -> pd.DataFrame:
    tmp = df.copy()
    tmp["charge_band"] = pd.qcut(tmp["MonthlyCharges"], q=n_bands, duplicates="drop")
    tmp["charge_band"] = tmp["charge_band"].astype(str)
    return churn_rate_by_category(tmp, "charge_band")


def numeric_summary_by_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median of numeric features split by churn outcome."""
    cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    agg = df.groupby(TARGET_COLUMN)[cols].agg(["mean", "median"]).round(2)
    return agg


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation among numeric features and the (0/1) target."""
    cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", TARGET_COLUMN]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    return numeric.corr().round(3)


def all_category_churn_rates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Churn-rate tables for every categorical feature."""
    return {col: churn_rate_by_category(df, col) for col in CATEGORICAL_COLUMNS}
