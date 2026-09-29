"""Phase 9 tests: calibration selection + production artifact."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import get_config
from src.models.calibration import select_best_variant


def test_select_best_variant_picks_min_brier():
    variants = {
        "uncalibrated": {"metrics": {"brier": 0.150, "log_loss": 0.45}},
        "platt_sigmoid": {"metrics": {"brier": 0.140, "log_loss": 0.44}},
        "isotonic": {"metrics": {"brier": 0.140, "log_loss": 0.43}},
    }
    # Lowest Brier ties -> log loss breaks the tie (isotonic).
    assert select_best_variant(variants) == "isotonic"


def _models_dir():
    return get_config().resolve_path("paths.models")


def test_tuned_meta_and_estimator_buildable():
    meta = _models_dir() / "tuned_best_meta.json"
    if not meta.exists():
        pytest.skip("run Phase 8 (tune_models) first")
    from src.models.calibration import build_tuned_estimator

    name, pipe = build_tuned_estimator()
    assert isinstance(name, str)
    assert pipe.steps[-1][0] == "clf"  # unfitted pipeline


def test_production_model_loads_and_predicts():
    artifact = _models_dir() / "production_model.joblib"
    if not artifact.exists():
        pytest.skip("run Phase 9 (calibrate_model) first")
    import joblib

    from src.models.train import load_splits

    model = joblib.load(artifact)
    splits = load_splits(include_engineered=True)
    proba = model.predict_proba(splits.X_val)[:, 1]
    assert proba.shape[0] == len(splits.y_val)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert not np.isnan(proba).any()
