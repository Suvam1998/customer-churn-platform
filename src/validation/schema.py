"""Explicit schema for the IBM Telco Customer Churn dataset.

Centralises the column groups, expected dtypes, and allowed categorical values
so validation, preprocessing, and feature engineering all agree on one source
of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

# Raw numeric columns. TotalCharges is text in the raw file and becomes numeric
# after cleaning; it is listed here as its post-cleaning (intended) type.
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

# SeniorCitizen is a raw 0/1 integer flag — semantically categorical/binary.
BINARY_NUMERIC_COLUMNS = ["SeniorCitizen"]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURE_COLUMNS = (
    NUMERIC_COLUMNS + BINARY_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
)

# Allowed value sets for validation. Unexpected values are reported (not
# silently coerced or dropped).
_YES_NO = {"Yes", "No"}
_YES_NO_NOINT = {"Yes", "No", "No internet service"}

ALLOWED_VALUES: dict[str, set[str]] = {
    "gender": {"Male", "Female"},
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": _YES_NO_NOINT,
    "OnlineBackup": _YES_NO_NOINT,
    "DeviceProtection": _YES_NO_NOINT,
    "TechSupport": _YES_NO_NOINT,
    "StreamingTV": _YES_NO_NOINT,
    "StreamingMovies": _YES_NO_NOINT,
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    TARGET_COLUMN: _YES_NO,
}

ALLOWED_SENIOR_CITIZEN = {0, 1}


@dataclass
class SchemaCheck:
    """Result of one schema/quality assertion."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    """Aggregate of all schema/quality checks."""

    checks: list[SchemaCheck] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(SchemaCheck(name, passed, detail))

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[SchemaCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_failures": len(self.failures),
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


def expected_columns() -> list[str]:
    """All columns the raw dataset must contain."""
    return [ID_COLUMN, *ALL_FEATURE_COLUMNS, TARGET_COLUMN]
