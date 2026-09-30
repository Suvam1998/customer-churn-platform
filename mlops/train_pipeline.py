"""Log the platform's experiments to MLflow.

Logs three families of runs into one experiment:
  * Phase 7 model comparison (untuned candidates) — from live training.
  * Phase 8 tuning results — from results/tuning_results.json.
  * Phase 9 calibration variants — from results/calibration_results.json.

Every metric logged is a real, executed result. Nothing is fabricated.
"""
from __future__ import annotations

import json
import logging

from src.config import Config, get_config
from mlops.mlflow_utils import dataset_version, log_metrics_run, setup_mlflow

logger = logging.getLogger("churn.mlflow")


def log_comparison_runs(cfg: Config, dsv: str) -> int:
    """Train candidates live and log each as an MLflow run."""
    from src.models.train import train_and_evaluate_all

    _splits, results, pipelines, _skipped = train_and_evaluate_all(
        include_engineered=True, cfg=cfg)
    n = 0
    for name, m in results.items():
        params = {k: v for k, v in pipelines[name].named_steps["clf"].get_params().items()}
        log_metrics_run(
            run_name=f"compare-{name}",
            params=params,
            metrics={k: m[k] for k in ["roc_auc", "pr_auc", "precision", "recall",
                                       "f1", "log_loss", "brier", "accuracy"]},
            tags={"phase": "07-comparison", "model": name, "feature_set": "engineered",
                  "dataset_version": dsv, "preprocessing_version": "engineered-v1"},
            duration_s=m.get("train_time_s"),
        )
        n += 1
    return n


def log_tuning_runs(cfg: Config, dsv: str) -> int:
    path = cfg.resolve_path("paths.results") / "tuning_results.json"
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for name, r in data.items():
        if "val_metrics" not in r:
            continue
        vm = r["val_metrics"]
        log_metrics_run(
            run_name=f"tuned-{name}",
            params={**r.get("best_params", {}), "cv_folds": r.get("cv_folds"),
                    "n_iter": r.get("n_iter")},
            metrics={"cv_roc_auc": r.get("cv_best_score"),
                     "roc_auc": vm.get("roc_auc"), "pr_auc": vm.get("pr_auc"),
                     "f1": vm.get("f1"), "brier": vm.get("brier"),
                     "log_loss": vm.get("log_loss")},
            tags={"phase": "08-tuning", "model": name, "feature_set": "engineered",
                  "dataset_version": dsv},
            duration_s=r.get("tune_time_s"),
        )
        n += 1
    return n


def log_calibration_runs(cfg: Config, dsv: str) -> int:
    path = cfg.resolve_path("paths.results") / "calibration_results.json"
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    base = data.get("base_model", "unknown")
    n = 0
    for variant, payload in data.get("variants", {}).items():
        m = payload.get("metrics", {})
        log_metrics_run(
            run_name=f"calib-{variant}",
            params={"base_model": base, "calibration": variant},
            metrics={"roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc"),
                     "f1": m.get("f1"), "brier": m.get("brier"),
                     "log_loss": m.get("log_loss")},
            tags={"phase": "09-calibration", "model": base, "calibration": variant,
                  "selected": str(variant == data.get("selected")),
                  "dataset_version": dsv},
        )
        n += 1
    return n


def run(cfg: Config | None = None) -> dict:
    cfg = cfg or get_config()
    setup_mlflow(cfg)
    dsv = dataset_version(cfg)
    counts = {
        "comparison_runs": log_comparison_runs(cfg, dsv),
        "tuning_runs": log_tuning_runs(cfg, dsv),
        "calibration_runs": log_calibration_runs(cfg, dsv),
    }
    logger.info("logged runs: %s", counts)
    return counts
