"""Phase 7 entrypoint: train all candidate models on the SAME engineered split,
evaluate on VALIDATION (test untouched), and write the model comparison.

Usage (repo root, venv active):
    python scripts/train_models.py
Artifacts:
    results/model_comparison.csv, results/model_comparison.json
    docs/figures/models/model_comparison.png
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import get_config  # noqa: E402
from src.models.metrics import metrics_row  # noqa: E402
from src.models.train import train_and_evaluate_all  # noqa: E402


def _comparison_figure(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    d = df.sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(8, 5))
    y = range(len(d))
    ax.barh([i + 0.2 for i in y], d["roc_auc"], height=0.4, label="ROC-AUC", color="#4C72B0")
    ax.barh([i - 0.2 for i in y], d["pr_auc"], height=0.4, label="PR-AUC", color="#DD8452")
    ax.set_yticks(list(y), labels=d["model"])
    for i, (r, p) in enumerate(zip(d["roc_auc"], d["pr_auc"])):
        ax.text(r, i + 0.2, f" {r:.3f}", va="center", fontsize=8)
        ax.text(p, i - 0.2, f" {p:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Score (validation)")
    ax.set_title("Model comparison — ROC-AUC & PR-AUC (validation)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.0)
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()
    results_dir = cfg.resolve_path("paths.results")
    results_dir.mkdir(parents=True, exist_ok=True)

    splits, results, _pipes, skipped = train_and_evaluate_all(
        include_engineered=True, class_weight=None, cfg=cfg
    )

    rows = []
    for name, m in results.items():
        row = metrics_row(name, m)
        row["accuracy"] = m["accuracy"]
        row["train_time_s"] = m["train_time_s"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    df.to_csv(results_dir / "model_comparison.csv", index=False)
    (results_dir / "model_comparison.json").write_text(
        json.dumps({"validation": results, "skipped": skipped,
                    "feature_set": "engineered"}, indent=2), encoding="utf-8")
    _comparison_figure(df, cfg.resolve_path("paths.figures") / "models" / "model_comparison.png")

    print("=" * 92)
    print("MODEL COMPARISON — all candidates, engineered features (VALIDATION; test untouched)")
    print("=" * 92)
    cols = ["model", "roc_auc", "pr_auc", "precision", "recall", "f1",
            "log_loss", "brier", "train_time_s"]
    print(df[cols].to_string(index=False))
    print("-" * 92)
    if skipped:
        print(f"SKIPPED (library unavailable on this interpreter): {skipped}")
    best = df.iloc[0]
    print(f"Best by ROC-AUC: {best['model']} (ROC-AUC={best['roc_auc']}, "
          f"PR-AUC={best['pr_auc']}, F1={best['f1']})")
    print("NOTE: selection is provisional (untuned). Final selection after Phase 8/9 on validation.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
