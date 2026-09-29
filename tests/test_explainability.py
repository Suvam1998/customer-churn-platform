"""Phase 10 tests: SHAP explainer correctness (nothing hardcoded)."""
from __future__ import annotations

import numpy as np
import pytest

from src.explainability.shap_explainer import (
    ChurnExplainer,
    engineered_dataset,
    get_customer_features,
)
from src.models.train import load_splits


@pytest.fixture(scope="module")
def explainer():
    return ChurnExplainer.from_tuned()


@pytest.fixture(scope="module")
def val_sample():
    splits = load_splits(include_engineered=True)
    return splits.X_val.iloc[:50]


def test_shap_values_shape(explainer, val_sample):
    sv = explainer.shap_values(val_sample)
    assert sv.shape == (len(val_sample), len(explainer.feature_names))


def test_shap_additivity_matches_probability(explainer, val_sample):
    """base + sum(shap) (log-odds) -> sigmoid ~= model predict_proba."""
    sv = explainer.shap_values(val_sample)
    base = np.ravel(explainer.explainer.expected_value)[-1]
    margin = base + sv.sum(axis=1)
    prob_from_shap = 1.0 / (1.0 + np.exp(-margin))
    prob_model = explainer.predict_proba(val_sample)
    np.testing.assert_allclose(prob_from_shap, prob_model, atol=1e-2)


def test_global_importance_ranked(explainer, val_sample):
    gi = explainer.global_importance(val_sample, top_n=10)
    assert len(gi) == 10
    # Sorted descending by mean |shap|.
    assert gi["mean_abs_shap"].is_monotonic_decreasing


def test_explain_customer_structure(explainer):
    df = engineered_dataset()
    cid = df["customerID"].iloc[0]
    x_row, actual = get_customer_features(cid, df=df)
    exp = explainer.explain_customer(x_row, cid, top_k=5)

    assert exp["customer_id"] == cid
    assert 0.0 <= exp["churn_probability"] <= 1.0
    assert actual in (0, 1)
    assert len(exp["top_positive_factors"]) <= 5
    assert all(f["shap"] > 0 for f in exp["top_positive_factors"])
    assert all(f["shap"] < 0 for f in exp["top_negative_factors"])
    assert len(exp["feature_contributions"]) == len(explainer.feature_names)


def test_known_customer_present():
    df = engineered_dataset()
    x_row, actual = get_customer_features("7590-VHVEG", df=df)
    assert x_row.shape[0] == 1
    assert actual in (0, 1)


def test_missing_customer_raises():
    with pytest.raises(KeyError):
        get_customer_features("does-not-exist")
