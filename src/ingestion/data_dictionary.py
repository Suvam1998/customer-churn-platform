"""Generate a Markdown data dictionary from the raw dataset + known semantics.

Column descriptions for the IBM Telco Customer Churn dataset are well
documented publicly; we pair those human descriptions with the actual dtypes
and example values observed in the downloaded file (so the doc reflects reality,
not assumptions).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config, get_config

# Human-readable descriptions for the standard Telco churn columns.
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "customerID": "Unique customer identifier.",
    "gender": "Customer gender (Male/Female).",
    "SeniorCitizen": "Whether the customer is a senior citizen (0/1).",
    "Partner": "Whether the customer has a partner (Yes/No).",
    "Dependents": "Whether the customer has dependents (Yes/No).",
    "tenure": "Number of months the customer has stayed with the company.",
    "PhoneService": "Whether the customer has phone service (Yes/No).",
    "MultipleLines": "Whether the customer has multiple lines (Yes/No/No phone service).",
    "InternetService": "Customer's internet service provider (DSL/Fiber optic/No).",
    "OnlineSecurity": "Whether the customer has online security (Yes/No/No internet service).",
    "OnlineBackup": "Whether the customer has online backup (Yes/No/No internet service).",
    "DeviceProtection": "Whether the customer has device protection (Yes/No/No internet service).",
    "TechSupport": "Whether the customer has tech support (Yes/No/No internet service).",
    "StreamingTV": "Whether the customer has streaming TV (Yes/No/No internet service).",
    "StreamingMovies": "Whether the customer has streaming movies (Yes/No/No internet service).",
    "Contract": "Contract term (Month-to-month/One year/Two year).",
    "PaperlessBilling": "Whether the customer has paperless billing (Yes/No).",
    "PaymentMethod": "Payment method (Electronic check/Mailed check/Bank transfer/Credit card).",
    "MonthlyCharges": "The amount charged to the customer monthly.",
    "TotalCharges": "The total amount charged to the customer (text in raw file; blanks when tenure==0).",
    "Churn": "TARGET — whether the customer churned (Yes/No).",
}


def _example_value(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return ""
    return str(non_null.iloc[0])


def build_data_dictionary(df: pd.DataFrame) -> str:
    """Return a Markdown data dictionary table for ``df``."""
    lines = [
        "# Data Dictionary — IBM Telco Customer Churn",
        "",
        "> Generated from the real downloaded dataset. Descriptions are the "
        "standard IBM Telco churn definitions; dtype and example come from the "
        "actual file.",
        "",
        f"- **Rows:** {len(df):,}",
        f"- **Columns:** {df.shape[1]}",
        f"- **Target column:** `Churn` (Yes = churned, No = retained)",
        "",
        "| # | Column | Dtype | Non-null | Example | Description |",
        "|--:|--------|-------|---------:|---------|-------------|",
    ]
    for i, col in enumerate(df.columns, start=1):
        dtype = str(df[col].dtype)
        non_null = int(df[col].notna().sum())
        example = _example_value(df[col]).replace("|", "\\|")
        desc = COLUMN_DESCRIPTIONS.get(col, "(no description available)")
        lines.append(f"| {i} | `{col}` | {dtype} | {non_null:,} | {example} | {desc} |")
    lines.append("")
    return "\n".join(lines)


def save_data_dictionary(df: pd.DataFrame, cfg: Config | None = None) -> Path:
    """Write the data dictionary to ``docs/data_dictionary.md``."""
    cfg = cfg or get_config()
    docs_dir = cfg.root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "data_dictionary.md"
    out.write_text(build_data_dictionary(df), encoding="utf-8")
    return out
