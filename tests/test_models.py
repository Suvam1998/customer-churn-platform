"""Phase 6 tests: metric suite + leakage-safe baseline pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.metrics import classification_metrics
from src.models.train import build_pipeline, load_splits, make_logreg, train_estimator


def test_metrics_perfect_prediction():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.98, 0.99])
    m = classification_metrics(y, p)
    assert m["roc_auc"] == 1.0
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["brier"] < 0.01


def test_metrics_confusion_counts_consistent():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.2, 0.9, 0.6, 0.3, 0.8])
    m = classification_metrics(y, p, threshold=0.5)
    cm = m["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y)
    assert cm["tp"] + cm["fn"] == int(y.sum())


@pytest.fixture(scope="module")
def trained():
    return train_estimator(make_logreg(), include_engineered=True)


def test_pipeline_is_single_fittable_estimator():
    pipe = build_pipeline(make_logreg(), include_engineered=True)
    assert pipe.steps[0][0] == "pre"
    assert pipe.steps[-1][0] == "clf"


def test_baseline_trains_and_predicts_probabilities(trained):
    pipe, splits = trained
    proba = pipe.predict_proba(splits.X_val)[:, 1]
    assert proba.shape[0] == len(splits.y_val)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_baseline_beats_random_on_validation(trained):
    pipe, splits = trained
    proba = pipe.predict_proba(splits.X_val)[:, 1]
    m = classification_metrics(splits.y_val, proba)
    # A sane churn baseline should be well above chance.
    assert m["roc_auc"] > 0.75


def test_no_customer_leakage_between_train_and_val():
    splits = load_splits(include_engineered=True)
    assert set(splits.id_train).isdisjoint(set(splits.id_val))
    assert set(splits.id_train).isdisjoint(set(splits.id_test))
    assert set(splits.id_val).isdisjoint(set(splits.id_test))
