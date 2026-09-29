"""Feature engineering — documented, leakage-safe, row-wise features.

Leakage safety: every engineered feature is a deterministic function of a
single customer's own attributes. No feature uses the target, and none uses a
statistic computed across other rows or the whole dataset (the only constants
are fixed domain values, e.g. "there are 9 add-on services"). Therefore these
features can be computed before the train/val/test split without leakage.

The risk indicators (payment/contract/tenure) encode well-established telecom
domain knowledge (electronic-check payers, month-to-month contracts, and
brand-new customers churn more). They are NOT thresholds fitted on this
dataset's labels, so they do not leak the target.

Each feature's rationale is recorded in FEATURE_DESCRIPTIONS and rendered into
docs/feature_dictionary.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)

# The nine add-on / core services counted for adoption scoring.
_SERVICE_FLAGS = [
    "has_phone",
    "has_multiple_lines",
    "has_internet",
    "has_online_security",
    "has_online_backup",
    "has_device_protection",
    "has_tech_support",
    "has_streaming_tv",
    "has_streaming_movies",
]
_N_SERVICES = len(_SERVICE_FLAGS)

# Engineered numeric features (includes 0/1 flags & risk indicators).
ENGINEERED_NUMERIC = [
    "average_charge_per_month",
    "total_services",
    "service_adoption_score",
    "is_month_to_month",
    "is_long_term_contract",
    "has_tech_support",
    "has_online_security",
    "has_online_backup",
    "has_device_protection",
    "has_streaming",
    "payment_risk_indicator",
    "contract_risk_indicator",
    "tenure_risk_indicator",
]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "average_charge_per_month": "TotalCharges / tenure (falls back to MonthlyCharges when tenure==0). Captures effective spend rate vs. headline monthly price.",
    "total_services": "Count of subscribed services among the 9 core/add-on services (0-9). Engagement/stickiness proxy.",
    "service_adoption_score": "total_services / 9 in [0,1]. Normalised engagement; higher adoption tends to reduce churn.",
    "is_month_to_month": "1 if Contract == 'Month-to-month'. The highest-churn contract type (EDA: 42.7%).",
    "is_long_term_contract": "1 if Contract in {'One year','Two year'}. Long-term commitment lowers churn.",
    "has_tech_support": "1 if TechSupport == 'Yes'. Support access is protective against churn.",
    "has_online_security": "1 if OnlineSecurity == 'Yes'. Security add-on correlates with retention.",
    "has_online_backup": "1 if OnlineBackup == 'Yes'.",
    "has_device_protection": "1 if DeviceProtection == 'Yes'.",
    "has_streaming": "1 if the customer streams TV or movies. Higher engagement signal.",
    "payment_risk_indicator": "1 if PaymentMethod == 'Electronic check' (domain: highest-churn payment method).",
    "contract_risk_indicator": "1 if month-to-month contract (domain risk flag).",
    "tenure_risk_indicator": "1 if tenure < 12 months (domain: new customers churn most).",
}

# Featured column sets used by the "engineered" experiments.
FEATURED_NUMERIC_FEATURES = NUMERIC_FEATURES + ENGINEERED_NUMERIC
FEATURED_CATEGORICAL_FEATURES = list(CATEGORICAL_FEATURES)
FEATURED_FEATURE_COLUMNS = FEATURED_NUMERIC_FEATURES + FEATURED_CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` augmented with engineered features (new columns only).

    Expects a CLEANED frame (``TotalCharges`` numeric). Pure & row-wise.
    """
    out = df.copy()

    # --- service adoption flags ---
    out["has_phone"] = (out["PhoneService"] == "Yes").astype(int)
    out["has_multiple_lines"] = (out["MultipleLines"] == "Yes").astype(int)
    out["has_internet"] = (out["InternetService"] != "No").astype(int)
    out["has_online_security"] = (out["OnlineSecurity"] == "Yes").astype(int)
    out["has_online_backup"] = (out["OnlineBackup"] == "Yes").astype(int)
    out["has_device_protection"] = (out["DeviceProtection"] == "Yes").astype(int)
    out["has_tech_support"] = (out["TechSupport"] == "Yes").astype(int)
    out["has_streaming_tv"] = (out["StreamingTV"] == "Yes").astype(int)
    out["has_streaming_movies"] = (out["StreamingMovies"] == "Yes").astype(int)
    out["has_streaming"] = (
        (out["has_streaming_tv"] == 1) | (out["has_streaming_movies"] == 1)
    ).astype(int)

    out["total_services"] = out[_SERVICE_FLAGS].sum(axis=1)
    out["service_adoption_score"] = (out["total_services"] / _N_SERVICES).round(4)

    # --- value features ---
    tenure = out["tenure"].astype(float)
    total = out["TotalCharges"].astype(float)
    monthly = out["MonthlyCharges"].astype(float)
    avg = np.where(tenure > 0, total / tenure.replace(0, np.nan), monthly)
    out["average_charge_per_month"] = pd.Series(avg, index=out.index).round(2)

    # --- contract / payment / tenure risk features ---
    out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    out["is_long_term_contract"] = out["Contract"].isin(["One year", "Two year"]).astype(int)
    out["contract_risk_indicator"] = out["is_month_to_month"]
    out["payment_risk_indicator"] = (out["PaymentMethod"] == "Electronic check").astype(int)
    out["tenure_risk_indicator"] = (out["tenure"] < 12).astype(int)

    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless sklearn transformer wrapping :func:`engineer_features`.

    Stateless => ``fit`` learns nothing, so placing it inside a Pipeline before
    the ColumnTransformer introduces no train/test leakage.
    """

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        return self

    def transform(self, X):
        return engineer_features(X)


def build_featured_preprocessor():
    """ColumnTransformer over base + engineered feature columns (unfitted)."""
    return build_preprocessor(
        numeric_features=FEATURED_NUMERIC_FEATURES,
        categorical_features=FEATURED_CATEGORICAL_FEATURES,
    )


def build_feature_dictionary() -> str:
    """Markdown documenting every engineered feature and its rationale."""
    lines = [
        "# Feature Dictionary — Engineered Features",
        "",
        "> Leakage-safe, row-wise features (no target, no cross-row statistics). "
        "See `src/features/feature_engineering.py`.",
        "",
        "| Feature | Group | Rationale |",
        "|---------|-------|-----------|",
    ]
    groups = {
        "average_charge_per_month": "value",
        "total_services": "engagement",
        "service_adoption_score": "engagement",
        "is_month_to_month": "contract",
        "is_long_term_contract": "contract",
        "has_tech_support": "service",
        "has_online_security": "service",
        "has_online_backup": "service",
        "has_device_protection": "service",
        "has_streaming": "engagement",
        "payment_risk_indicator": "risk",
        "contract_risk_indicator": "risk",
        "tenure_risk_indicator": "risk",
    }
    for feat, desc in FEATURE_DESCRIPTIONS.items():
        lines.append(f"| `{feat}` | {groups.get(feat, '-')} | {desc} |")
    lines.append("")
    return "\n".join(lines)
