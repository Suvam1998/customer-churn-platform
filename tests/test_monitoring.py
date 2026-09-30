"""Phase 20 tests: PSI drift metrics + report status logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import get_config
from src.monitoring.drift import (
    classify_psi,
    drift_report,
    prediction_drift,
    psi_categorical,
    psi_numeric,
)


def test_psi_zero_for_identical_numeric():
    s = pd.Series(np.random.default_rng(0).normal(size=1000))
    assert psi_numeric(s, s) == pytest.approx(0.0, abs=1e-6)


def test_psi_positive_for_shifted_numeric():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 2000))
    cur = pd.Series(rng.normal(2, 1, 2000))  # big shift
    assert psi_numeric(ref, cur) > 0.25


def test_psi_categorical_detects_shift():
    ref = pd.Series(["A"] * 800 + ["B"] * 200)
    cur = pd.Series(["A"] * 200 + ["B"] * 800)
    assert psi_categorical(ref, cur) > 0.25
    assert psi_categorical(ref, ref) == pytest.approx(0.0, abs=1e-6)


def test_classify_psi_bands():
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "moderate"
    assert classify_psi(0.40) == "significant"


def test_prediction_drift_flag():
    rng = np.random.default_rng(1)
    ref = rng.uniform(0, 0.3, 1000)
    cur = rng.uniform(0.6, 1.0, 1000)
    pd_ = prediction_drift(ref, cur)
    assert pd_["drifted"] is True


def test_drift_report_stable_and_retraining():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({
        "num": rng.normal(0, 1, 1000),
        "cat": rng.choice(["A", "B", "C"], 1000),
    })
    # Identical -> STABLE.
    stable = drift_report(ref, ref.copy(), ["num"], ["cat"])
    assert stable["status"] == "STABLE"
    assert stable["feature_drift"]["n_drifted"] == 0

    # Strong shift on all features + prediction drift -> RETRAINING_REQUIRED.
    cur = pd.DataFrame({
        "num": rng.normal(4, 1, 1000),
        "cat": rng.choice(["X", "Y"], 1000),
    })
    ref_p = rng.uniform(0, 0.3, 1000)
    cur_p = rng.uniform(0.6, 1.0, 1000)
    drifted = drift_report(ref, cur, ["num"], ["cat"],
                           reference_probs=ref_p, current_probs=cur_p)
    assert drifted["status"] == "RETRAINING_REQUIRED"


def _model_ready() -> bool:
    return (get_config().resolve_path("paths.models") / "production_model.joblib").exists()


@pytest.mark.skipif(not _model_ready(), reason="run Phase 9 first")
def test_real_drift_is_stable():
    """Train-vs-test on the static dataset should be STABLE (no real drift)."""
    from src.models.train import load_splits
    from src.monitoring.monitor import build_report

    splits = load_splits(include_engineered=True)
    r = build_report(splits.X_train, splits.X_test)
    assert r["status"] == "STABLE"
