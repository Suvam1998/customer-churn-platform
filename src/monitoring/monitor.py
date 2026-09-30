"""High-level monitoring: build drift reports from the platform's splits.

Defines the monitored feature set and convenience builders reused by the
monitoring script and the API's ``/monitoring/drift`` endpoint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config, get_config
from src.monitoring.drift import drift_report

# Representative monitored features.
MONITOR_NUMERIC = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "average_charge_per_month", "total_services", "service_adoption_score",
]
MONITOR_CATEGORICAL = [
    "Contract", "PaymentMethod", "InternetService", "TechSupport",
    "OnlineSecurity", "contract_risk_indicator", "payment_risk_indicator",
    "tenure_risk_indicator",
]


def build_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_probs=None,
    current_probs=None,
    scenario: str = "train-vs-test",
    cfg: Config | None = None,
) -> dict:
    cfg = cfg or get_config()
    return drift_report(
        reference, current, MONITOR_NUMERIC, MONITOR_CATEGORICAL,
        reference_probs=reference_probs, current_probs=current_probs,
        scenario=scenario, cfg=cfg,
    )


def inject_drift(current: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Return a copy of ``current`` with a CLEARLY-LABELLED synthetic covariate
    shift (older, higher-charge, fibre-heavy population) to demonstrate the
    detector firing. This is a SIMULATED scenario, not real data drift."""
    rng = np.random.default_rng(seed)
    shifted = current.copy()
    shifted["tenure"] = np.clip(shifted["tenure"] + rng.integers(20, 40, len(shifted)), 0, 72)
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 1.4
    if "Contract" in shifted.columns:
        # Push more customers to month-to-month.
        mask = rng.random(len(shifted)) < 0.5
        shifted.loc[mask, "Contract"] = "Month-to-month"
    if "PaymentMethod" in shifted.columns:
        mask = rng.random(len(shifted)) < 0.5
        shifted.loc[mask, "PaymentMethod"] = "Electronic check"
    return shifted
