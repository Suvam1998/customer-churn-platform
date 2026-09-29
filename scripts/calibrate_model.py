"""Phase 9 entrypoint: calibrate the best tuned model and select a method.

Compares uncalibrated / Platt (sigmoid) / isotonic on VALIDATION (test
untouched), draws a reliability overlay, selects by Brier, and persists the
production model artifact.

Usage (repo root, venv active):
    python scripts/calibrate_model.py
Artifacts:
    results/calibration_results.json
    docs/figures/calibration/reliability_comparison.png
    models/production_model.joblib (+ meta)
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
from sklearn.calibration import calibration_curve  # noqa: E402

from src.config import get_config  # noqa: E402
from src.models.calibration import (  # noqa: E402
    evaluate_calibration_variants,
    save_production_model,
    select_best_variant,
)


def _reliability_overlay(splits, variants, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    for label, v in variants.items():
        frac, mean = calibration_curve(splits.y_val, v["probs"],
                                       n_bins=10, strategy="quantile")
        brier = v["metrics"]["brier"]
        ax.plot(mean, frac, marker="o", label=f"{label} (Brier={brier})")
    ax.set_title("Reliability curves — calibration comparison (validation)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn frequency")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()
    results_dir = cfg.resolve_path("paths.results")
    results_dir.mkdir(parents=True, exist_ok=True)

    name, splits, variants = evaluate_calibration_variants(cfg=cfg, cv=5)
    best = select_best_variant(variants)

    _reliability_overlay(
        splits, variants,
        cfg.resolve_path("paths.figures") / "calibration" / "reliability_comparison.png")

    # Persist results (without model objects).
    serializable = {
        label: {"metrics": {k: val for k, val in v["metrics"].items()
                            if k != "confusion_matrix"}}
        for label, v in variants.items()
    }
    (results_dir / "calibration_results.json").write_text(
        json.dumps({"base_model": name, "selected": best,
                    "variants": serializable}, indent=2), encoding="utf-8")

    # Save the selected calibrated model as production artifact.
    artifact, version = save_production_model(
        variants[best]["model"], name, best, variants[best]["metrics"], cfg=cfg)

    # Report.
    print("=" * 84)
    print(f"PROBABILITY CALIBRATION — base model: {name} (VALIDATION; test untouched)")
    print("=" * 84)
    print(f"{'variant':<16}{'ROC-AUC':>9}{'PR-AUC':>9}{'F1':>8}{'LogLoss':>10}{'Brier':>9}")
    print("-" * 84)
    for label, v in variants.items():
        m = v["metrics"]
        mark = "  <-- selected" if label == best else ""
        print(f"{label:<16}{m['roc_auc']:>9}{m['pr_auc']:>9}{m['f1']:>8}"
              f"{m['log_loss']:>10}{m['brier']:>9}{mark}")
    print("-" * 84)
    print(f"Selected calibration : {best} (lowest validation Brier)")
    print(f"Model version        : {version}")
    print(f"Production artifact   : {artifact.relative_to(cfg.root)}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
