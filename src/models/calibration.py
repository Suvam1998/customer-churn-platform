"""Probability calibration (Experiment H).

Compares the best tuned model uncalibrated vs. Platt scaling (sigmoid) vs.
isotonic regression. Calibration is learned via cross-validation on the
TRAINING split (``CalibratedClassifierCV`` refits the whole pipeline per fold,
so no leakage), and the method is selected on the VALIDATION set by Brier score
(with log loss as tie-break). The TEST set is never used here.

The selected calibrated pipeline is persisted as the production model artifact
with a ``model_version``.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from src.config import Config, get_config
from src.models.metrics import classification_metrics
from src.models.train import build_pipeline, load_splits
from src.models.tuning import _base_estimator

logger = logging.getLogger("churn.calibration")


def load_best_meta(cfg: Config | None = None) -> dict:
    """Read the tuned-best metadata written by Phase 8."""
    cfg = cfg or get_config()
    meta_path = cfg.resolve_path("paths.models") / "tuned_best_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found — run scripts/tune_models.py (Phase 8) first."
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def build_tuned_estimator(cfg: Config | None = None) -> tuple[str, object]:
    """Reconstruct the tuned best model as an UNFITTED pipeline."""
    cfg = cfg or get_config()
    seed = cfg.get("project.random_seed", 42)
    meta = load_best_meta(cfg)
    name = meta["model"]
    params = meta.get("best_params", {})
    est = _base_estimator(name, seed).set_params(**params)
    return name, build_pipeline(est, include_engineered=True)


def evaluate_calibration_variants(cfg: Config | None = None, cv: int = 5):
    """Fit uncalibrated / sigmoid / isotonic and score each on validation.

    Returns ``(name, splits, variants)`` where ``variants`` maps
    label -> {"metrics": ..., "model": fitted_estimator, "probs": np.ndarray}.
    """
    cfg = cfg or get_config()
    name, base = build_tuned_estimator(cfg)
    splits = load_splits(include_engineered=True, cfg=cfg)

    variants: dict = {}

    # Uncalibrated (base tuned model).
    unc = clone(base).fit(splits.X_train, splits.y_train)
    p_unc = unc.predict_proba(splits.X_val)[:, 1]
    variants["uncalibrated"] = {
        "metrics": classification_metrics(splits.y_val, p_unc),
        "model": unc, "probs": p_unc,
    }

    # Platt scaling (sigmoid) and isotonic — CV-calibrated on training data.
    for label, method in [("platt_sigmoid", "sigmoid"), ("isotonic", "isotonic")]:
        cal = CalibratedClassifierCV(clone(base), method=method, cv=cv)
        cal.fit(splits.X_train, splits.y_train)
        p = cal.predict_proba(splits.X_val)[:, 1]
        variants[label] = {
            "metrics": classification_metrics(splits.y_val, p),
            "model": cal, "probs": p,
        }

    return name, splits, variants


def select_best_variant(variants: dict) -> str:
    """Pick the calibration variant with the lowest validation Brier score
    (log loss breaks ties)."""
    return min(
        variants,
        key=lambda k: (variants[k]["metrics"]["brier"],
                       variants[k]["metrics"]["log_loss"]),
    )


def save_production_model(
    model, base_name: str, method_label: str, val_metrics: dict,
    cfg: Config | None = None,
) -> tuple[Path, str]:
    """Persist the selected calibrated model as the production artifact."""
    cfg = cfg or get_config()
    models_dir = cfg.resolve_path("paths.models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_version = f"{base_name}-{method_label}-{date.today().isoformat()}"
    joblib.dump(model, models_dir / "production_model.joblib")
    meta = {
        "model_version": model_version,
        "base_model": base_name,
        "calibration": method_label,
        "feature_set": "engineered",
        "selected_on": "validation Brier score",
        "val_metrics": {k: v for k, v in val_metrics.items() if k != "confusion_matrix"},
    }
    (models_dir / "production_model_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return models_dir / "production_model.joblib", model_version
