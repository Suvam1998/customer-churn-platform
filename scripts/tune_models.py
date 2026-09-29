"""Phase 8 entrypoint: hyperparameter tuning with RandomizedSearchCV.

CV on the TRAINING split; comparison on VALIDATION; TEST untouched. Persists
the tuning results and the single best tuned pipeline as a model artifact.

Usage (repo root, venv active):
    python scripts/tune_models.py
Artifacts:
    results/tuning_results.json
    models/tuned_best_model.joblib  (+ models/tuned_best_meta.json)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402

from src.config import get_config  # noqa: E402
from src.models.tuning import TUNABLE, tune_models  # noqa: E402


def _strip_estimator(res: dict) -> dict:
    """Return a JSON-serialisable copy (drops fitted estimator objects)."""
    return {k: v for k, v in res.items() if k != "best_estimator"}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()
    results_dir = cfg.resolve_path("paths.results")
    models_dir = cfg.resolve_path("paths.models")
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # CatBoost gets fewer iterations to bound wall-clock time.
    results, _splits = tune_models(
        models=TUNABLE,
        include_engineered=True,
        n_iter=15,
        cv=3,
        n_iter_overrides={"catboost": 8},
        cfg=cfg,
    )

    # Rank successful models by validation ROC-AUC.
    ranked = sorted(
        [r for r in results.values() if "val_metrics" in r],
        key=lambda r: r["val_metrics"]["roc_auc"],
        reverse=True,
    )

    serializable = {name: _strip_estimator(r) for name, r in results.items()}
    (results_dir / "tuning_results.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8")

    # Persist the best tuned pipeline as the model artifact.
    best = ranked[0]
    joblib.dump(best["best_estimator"], models_dir / "tuned_best_model.joblib")
    (models_dir / "tuned_best_meta.json").write_text(json.dumps({
        "model": best["model"],
        "feature_set": "engineered",
        "cv_best_score": best["cv_best_score"],
        "best_params": best["best_params"],
        "val_metrics": {k: v for k, v in best["val_metrics"].items()
                        if k != "confusion_matrix"},
    }, indent=2), encoding="utf-8")

    # Report.
    print("=" * 96)
    print("HYPERPARAMETER TUNING — RandomizedSearchCV (CV on train, compared on VALIDATION)")
    print("=" * 96)
    header = (f"{'model':<22}{'cv_roc_auc':>11}{'val_roc_auc':>12}{'val_pr_auc':>11}"
              f"{'val_f1':>8}{'val_brier':>10}{'time_s':>8}")
    print(header)
    print("-" * 96)
    for r in ranked:
        vm = r["val_metrics"]
        print(f"{r['model']:<22}{r['cv_best_score']:>11}{vm['roc_auc']:>12}"
              f"{vm['pr_auc']:>11}{vm['f1']:>8}{vm['brier']:>10}{r['tune_time_s']:>8}")
    errored = [n for n, r in results.items() if "error" in r]
    if errored:
        print(f"\nERRORED: {errored}")
    print("-" * 96)
    print(f"BEST (validation ROC-AUC): {best['model']} "
          f"(cv={best['cv_best_score']}, val_roc={best['val_metrics']['roc_auc']}, "
          f"val_pr={best['val_metrics']['pr_auc']})")
    print(f"Best params: {best['best_params']}")
    print(f"Artifact saved: models/tuned_best_model.joblib")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
