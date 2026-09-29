"""Model evaluation: metrics + diagnostic figures (confusion, ROC, PR,
calibration). All figures have titles, axis labels, and legends.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    PrecisionRecallDisplay,
    RocCurveDisplay,
    precision_recall_curve,
    roc_curve,
)

from src.models.metrics import classification_metrics  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def plot_confusion_matrix(cm: dict, out: Path, title: str) -> Path:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred Retained", "Pred Churn"])
    ax.set_yticks([0, 1], labels=["True Retained", "True Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center",
                    color="black", fontsize=12)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_roc(y_true, y_prob, out: Path, title: str) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_pr(y_true, y_prob, out: Path, title: str) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=ax)
    base = np.asarray(y_true).mean()
    ax.axhline(base, linestyle="--", color="grey", label=f"Baseline = {base:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_calibration(y_true, y_prob, out: Path, title: str, n_bins: int = 10) -> Path:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.set_title(title)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn frequency")
    ax.legend(loc="upper left")
    fig.savefig(out)
    plt.close(fig)
    return out


def evaluate_and_plot(
    y_true,
    y_prob,
    out_dir: Path,
    prefix: str,
    title_prefix: str,
    threshold: float = 0.5,
) -> dict:
    """Compute metrics and write the four diagnostic figures. Returns metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    m = classification_metrics(y_true, y_prob, threshold=threshold)
    plot_confusion_matrix(m["confusion_matrix"], out_dir / f"{prefix}_confusion.png",
                          f"{title_prefix} — Confusion (thr={threshold})")
    plot_roc(y_true, y_prob, out_dir / f"{prefix}_roc.png",
             f"{title_prefix} — ROC (AUC={m['roc_auc']})")
    plot_pr(y_true, y_prob, out_dir / f"{prefix}_pr.png",
            f"{title_prefix} — Precision-Recall (AP={m['pr_auc']})")
    plot_calibration(y_true, y_prob, out_dir / f"{prefix}_calibration.png",
                     f"{title_prefix} — Calibration (Brier={m['brier']})")
    return m
