"""Tests for churn-probability -> risk-band classification."""
from __future__ import annotations

import pytest

from src.risk import RISK_LEVELS, classify_risk


@pytest.mark.parametrize(
    "prob,expected",
    [
        (0.00, "LOW"),
        (0.29, "LOW"),
        (0.49, "LOW"),
        (0.50, "MEDIUM"),
        (0.69, "MEDIUM"),
        (0.70, "HIGH"),
        (0.84, "HIGH"),
        (0.85, "CRITICAL"),
        (1.00, "CRITICAL"),
    ],
)
def test_classify_risk_bands(prob, expected):
    assert classify_risk(prob) == expected


def test_all_bands_are_known():
    for p in (0.1, 0.55, 0.75, 0.95):
        assert classify_risk(p) in RISK_LEVELS


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        classify_risk(bad)
