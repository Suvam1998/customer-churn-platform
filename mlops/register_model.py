"""Register the production model in the MLflow Model Registry with a
validation-gated stage (alias).

MLflow 3.x removed model-version *stages*, so we express Development / Staging /
Production as registry **aliases** plus a ``stage`` tag. Promotion to Production
is GATED by validation metrics — never automatic/blind.
"""
from __future__ import annotations

import json
import logging

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.config import Config, get_config
from mlops.mlflow_utils import dataset_version, setup_mlflow

logger = logging.getLogger("churn.mlflow")

REGISTERED_MODEL_NAME = "churn-retention-model"

# Promotion gate thresholds (validation metrics).
PROMOTION_GATE = {"min_roc_auc": 0.80, "max_brier": 0.20}


def _load_production_meta(cfg: Config) -> dict:
    meta_path = cfg.resolve_path("paths.models") / "production_model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found — run Phase 9 (calibrate_model) first.")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def passes_gate(val_metrics: dict) -> tuple[bool, str]:
    roc = val_metrics.get("roc_auc", 0.0)
    brier = val_metrics.get("brier", 1.0)
    ok = roc >= PROMOTION_GATE["min_roc_auc"] and brier <= PROMOTION_GATE["max_brier"]
    reason = (f"roc_auc={roc} (>= {PROMOTION_GATE['min_roc_auc']}), "
              f"brier={brier} (<= {PROMOTION_GATE['max_brier']})")
    return ok, reason


def register_production(cfg: Config | None = None) -> dict:
    """Log + register the production model; set stage alias via a validation gate."""
    cfg = cfg or get_config()
    setup_mlflow(cfg)
    meta = _load_production_meta(cfg)
    model = joblib.load(cfg.resolve_path("paths.models") / "production_model.joblib")
    val_metrics = meta.get("val_metrics", {})
    dsv = dataset_version(cfg)

    with mlflow.start_run(run_name=f"register-{meta.get('model_version')}") as run:
        mlflow.set_tags({
            "phase": "19-registry", "base_model": meta.get("base_model"),
            "calibration": meta.get("calibration"), "dataset_version": dsv,
            "model_version_label": meta.get("model_version"),
        })
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))
        # cloudpickle avoids MLflow 3.x's skops "untrusted types" check for
        # the CatBoost estimator inside the pipeline.
        info = mlflow.sklearn.log_model(
            model, name="model", registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="cloudpickle")

    client = MlflowClient()
    # Find the version just created.
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    version = max(int(v.version) for v in versions)

    ok, reason = passes_gate(val_metrics)
    # Default new registrations to development; validation gate promotes further.
    stage = "production" if ok else "staging"
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "development", version)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, stage, version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, str(version), "stage", stage)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, str(version),
                                 "promotion_gate", "pass" if ok else "fail")

    logger.info("registered %s v%s -> %s (%s)", REGISTERED_MODEL_NAME, version, stage, reason)
    return {
        "registered_model": REGISTERED_MODEL_NAME,
        "version": version,
        "stage": stage,
        "gate_passed": ok,
        "gate_reason": reason,
        "run_id": run.info.run_id,
    }
