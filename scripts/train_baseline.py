"""Phase 6 entrypoint: train the Logistic Regression baseline and evaluate on
the VALIDATION set (test remains untouched until final evaluation).

Trains LR twice — base features vs. engineered features — to directly answer
RQ1 (do engineered features improve churn prediction?). Diagnostic figures are
saved for the engineered baseline.

Usage (repo root, venv active):
    python scripts/train_baseline.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402
from src.models.evaluate import evaluate_and_plot  # noqa: E402
from src.models.metrics import classification_metrics, metrics_row  # noqa: E402
from src.models.train import make_logreg, train_estimator  # noqa: E402


def _val_probs(pipe, splits):
    return pipe.predict_proba(splits.X_val)[:, 1]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()
    results = cfg.resolve_path("paths.results")
    results.mkdir(parents=True, exist_ok=True)
    fig_dir = cfg.resolve_path("paths.figures") / "baseline"

    rows = []
    baseline_metrics = {}

    # Experiment A: base features + LR.
    pipe_a, splits_a = train_estimator(make_logreg(cfg), include_engineered=False, cfg=cfg)
    m_a = classification_metrics(splits_a.y_val, _val_probs(pipe_a, splits_a))
    rows.append(metrics_row("LogReg (base features)", m_a))

    # Experiment B: engineered features + LR (the platform baseline).
    pipe_b, splits_b = train_estimator(make_logreg(cfg), include_engineered=True, cfg=cfg)
    y_prob_b = _val_probs(pipe_b, splits_b)
    m_b = evaluate_and_plot(
        splits_b.y_val, y_prob_b, fig_dir, prefix="baseline_logreg",
        title_prefix="Baseline LogReg (engineered)",
    )
    rows.append(metrics_row("LogReg (engineered features)", m_b))
    baseline_metrics = m_b

    # Persist.
    (results / "baseline_metrics.json").write_text(
        json.dumps({"validation": baseline_metrics,
                    "rq1_comparison": rows}, indent=2), encoding="utf-8")

    # Report.
    print("=" * 78)
    print("BASELINE — Logistic Regression (VALIDATION set; test untouched)")
    print("=" * 78)
    header = f"{'model':<30}{'ROC-AUC':>9}{'PR-AUC':>9}{'Prec':>8}{'Recall':>8}{'F1':>8}{'LogLoss':>9}{'Brier':>8}"
    print(header)
    print("-" * 78)
    for r in rows:
        print(f"{r['model']:<30}{r['roc_auc']:>9}{r['pr_auc']:>9}{r['precision']:>8}"
              f"{r['recall']:>8}{r['f1']:>8}{r['log_loss']:>9}{r['brier']:>8}")
    print("-" * 78)
    d_roc = round(rows[1]["roc_auc"] - rows[0]["roc_auc"], 4)
    d_pr = round(rows[1]["pr_auc"] - rows[0]["pr_auc"], 4)
    print(f"RQ1 (engineered - base): ROC-AUC {d_roc:+}, PR-AUC {d_pr:+}")
    cm = baseline_metrics["confusion_matrix"]
    print(f"Confusion @0.5 (engineered): TN={cm['tn']} FP={cm['fp']} "
          f"FN={cm['fn']} TP={cm['tp']}")
    print(f"Figures: {fig_dir.relative_to(cfg.root)}")
    print(f"Metrics: {(results / 'baseline_metrics.json').relative_to(cfg.root)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
