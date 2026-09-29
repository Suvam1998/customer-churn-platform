"""Classification metric suite for churn models.

Accuracy is intentionally NOT the headline metric — with ~26.5% positives it is
misleading. We report ROC-AUC, PR-AUC (average precision), recall, precision,
F1, log loss, and Brier score (calibration quality), matching the master plan.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute the full metric suite from true labels and predicted probs.

    ``y_prob`` is the probability of the positive class (churn=1).
    Threshold-independent metrics (ROC-AUC, PR-AUC, log loss, Brier) use the
    probabilities; the rest use the given decision ``threshold``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob, labels=[0, 1])), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }


def metrics_row(name: str, m: dict) -> dict:
    """Flatten selected metrics for a comparison table row."""
    return {
        "model": name,
        "roc_auc": m["roc_auc"],
        "pr_auc": m["pr_auc"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "log_loss": m["log_loss"],
        "brier": m["brier"],
    }
