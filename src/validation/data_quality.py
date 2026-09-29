"""Data-quality validation for the raw Telco dataset.

Runs a battery of checks and returns a :class:`ValidationReport`. Nothing is
mutated or dropped here — the report documents issues so downstream cleaning
decisions are explicit and auditable.
"""
from __future__ import annotations

import logging

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.validation.schema import (
    ALLOWED_SENIOR_CITIZEN,
    ALLOWED_VALUES,
    ID_COLUMN,
    TARGET_COLUMN,
    ValidationReport,
    expected_columns,
)

logger = logging.getLogger("churn.validation")


def _totalcharges_blanks(df: pd.DataFrame) -> int:
    if "TotalCharges" not in df.columns:
        return 0
    col = df["TotalCharges"]
    if is_numeric_dtype(col):
        return int(col.isna().sum())
    return int((col.astype(str).str.strip() == "").sum())


def validate(df: pd.DataFrame) -> ValidationReport:
    """Validate ``df`` against the expected schema and quality rules."""
    report = ValidationReport()

    # 1. Required columns present.
    missing_cols = [c for c in expected_columns() if c not in df.columns]
    report.add(
        "required_columns_present",
        not missing_cols,
        "all present" if not missing_cols else f"missing: {missing_cols}",
    )

    # 2. Duplicate rows / IDs.
    dup_rows = int(df.duplicated().sum())
    report.add("no_duplicate_rows", dup_rows == 0, f"{dup_rows} duplicate rows")
    if ID_COLUMN in df.columns:
        dup_ids = int(df.duplicated(subset=[ID_COLUMN]).sum())
        report.add("unique_customer_ids", dup_ids == 0, f"{dup_ids} duplicate ids")

    # 3. Unexpected missing values (NaN) in non-TotalCharges columns.
    na_counts = df.isna().sum()
    unexpected_na = {
        c: int(na_counts[c])
        for c in df.columns
        if na_counts[c] > 0 and c != "TotalCharges"
    }
    report.add(
        "no_unexpected_missing",
        not unexpected_na,
        "none" if not unexpected_na else f"{unexpected_na}",
    )

    # 4. Target validity.
    if TARGET_COLUMN in df.columns:
        target_vals = set(df[TARGET_COLUMN].dropna().unique())
        allowed = ALLOWED_VALUES[TARGET_COLUMN]
        report.add(
            "target_values_valid",
            target_vals.issubset(allowed),
            f"observed={sorted(map(str, target_vals))}, allowed={sorted(allowed)}",
        )

    # 5. Categorical value validity.
    for col, allowed in ALLOWED_VALUES.items():
        if col == TARGET_COLUMN or col not in df.columns:
            continue
        observed = set(df[col].dropna().astype(str).unique())
        unexpected = observed - allowed
        report.add(
            f"values_valid[{col}]",
            not unexpected,
            "ok" if not unexpected else f"unexpected={sorted(unexpected)}",
        )

    # 6. SeniorCitizen domain.
    if "SeniorCitizen" in df.columns:
        sc_vals = set(pd.to_numeric(df["SeniorCitizen"], errors="coerce").dropna().unique())
        report.add(
            "senior_citizen_binary",
            sc_vals.issubset({float(v) for v in ALLOWED_SENIOR_CITIZEN}),
            f"observed={sorted(sc_vals)}",
        )

    # 7. TotalCharges blanks — reported as a KNOWN, non-fatal quirk.
    blanks = _totalcharges_blanks(df)
    report.add(
        "totalcharges_blanks_documented",
        True,  # informational: we expect a small number, handled in preprocessing
        f"{blanks} blank/NaN TotalCharges (converted & imputed in preprocessing, not dropped)",
    )

    # 8. Numeric ranges sanity (non-negative charges/tenure).
    for col in ["tenure", "MonthlyCharges"]:
        if col in df.columns and is_numeric_dtype(df[col]):
            neg = int((df[col] < 0).sum())
            report.add(f"non_negative[{col}]", neg == 0, f"{neg} negatives")

    return report


def print_report(report: ValidationReport) -> None:
    print("=" * 70)
    print("DATA QUALITY VALIDATION")
    print("=" * 70)
    for c in report.checks:
        mark = "PASS" if c.passed else "FAIL"
        print(f"[{mark}] {c.name:<32} {c.detail}")
    print("-" * 70)
    status = "PASSED" if report.passed else f"FAILED ({len(report.failures)} issue(s))"
    print(f"Overall: {status}")
    print("=" * 70)
