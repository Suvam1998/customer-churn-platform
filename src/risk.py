"""Risk-level classification from churn probability.

Thresholds are read from ``configs/config.yaml`` (never hardcoded here) so the
business can retune risk bands without code changes.

Bands (lower-bound semantics) using the configured thresholds
``medium``, ``high``, ``critical``:

    LOW      : p < medium
    MEDIUM   : medium <= p < high
    HIGH     : high   <= p < critical
    CRITICAL : p >= critical

The ``low`` threshold is retained in config as an informational "needs no
attention" floor and is not required for the four-band mapping.
"""
from __future__ import annotations

from src.config import get_config

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def classify_risk(probability: float) -> str:
    """Map a churn probability in [0, 1] to a configured risk band."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")

    t = get_config().get("risk_thresholds", {})
    medium = t.get("medium", 0.50)
    high = t.get("high", 0.70)
    critical = t.get("critical", 0.85)

    if probability >= critical:
        return "CRITICAL"
    if probability >= high:
        return "HIGH"
    if probability >= medium:
        return "MEDIUM"
    return "LOW"
