"""Hyperparameter tuning via RandomizedSearchCV.

Tuning uses k-fold CV on the TRAINING split only. The validation set is used
to compare tuned models; the TEST set is never touched here (that is reserved
for the single final evaluation). Search spaces come from
``configs/model_config.yaml``.
"""
from __future__ import annotations

import logging
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config import Config, get_config
from src.models.metrics import classification_metrics
from src.models.train import build_pipeline, load_splits

logger = logging.getLogger("churn.tuning")

# Models eligible for tuning (fast, strong candidates).
TUNABLE = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]


def _base_estimator(name: str, seed: int):
    """Single-threaded base estimator (search parallelises across folds)."""
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(eval_metric="logloss", random_state=seed,
                             n_jobs=1, tree_method="hist")
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(random_seed=seed, verbose=0,
                                  allow_writing_files=False, thread_count=1)
    raise ValueError(f"unknown model: {name}")


def _param_distributions(name: str, cfg: Config) -> dict:
    """Read the YAML search space and prefix keys for the pipeline's ``clf``."""
    raw = cfg.get(f"model.models.{name}.params", {}) or {}
    return {f"clf__{k}": v for k, v in raw.items()}


def tune_one(
    name: str,
    include_engineered: bool = True,
    n_iter: int = 15,
    cv: int = 3,
    cfg: Config | None = None,
    splits=None,
) -> dict:
    """Run RandomizedSearchCV for one model. Returns a results dict."""
    cfg = cfg or get_config()
    seed = cfg.get("project.random_seed", 42)
    scoring = cfg.get("model.common.scoring", "roc_auc")

    splits = splits or load_splits(include_engineered=include_engineered, cfg=cfg)
    pipe = build_pipeline(_base_estimator(name, seed), include_engineered=include_engineered)
    param_dist = _param_distributions(name, cfg)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    t0 = time.perf_counter()
    search.fit(splits.X_train, splits.y_train)
    tune_time = time.perf_counter() - t0

    y_prob_val = search.best_estimator_.predict_proba(splits.X_val)[:, 1]
    val_metrics = classification_metrics(splits.y_val, y_prob_val)

    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    return {
        "model": name,
        "scoring": scoring,
        "cv_folds": cv,
        "n_iter": n_iter,
        "cv_best_score": round(float(search.best_score_), 4),
        "best_params": best_params,
        "val_metrics": val_metrics,
        "tune_time_s": round(tune_time, 2),
        "best_estimator": search.best_estimator_,  # fitted pipeline
    }


def tune_models(
    models: list[str] | None = None,
    include_engineered: bool = True,
    n_iter: int = 15,
    cv: int = 3,
    n_iter_overrides: dict | None = None,
    cfg: Config | None = None,
) -> dict:
    """Tune several models on the SAME split. Returns ``{name: results}``."""
    cfg = cfg or get_config()
    models = models or TUNABLE
    n_iter_overrides = n_iter_overrides or {}
    splits = load_splits(include_engineered=include_engineered, cfg=cfg)

    out: dict = {}
    for name in models:
        try:
            out[name] = tune_one(
                name, include_engineered=include_engineered,
                n_iter=n_iter_overrides.get(name, n_iter), cv=cv,
                cfg=cfg, splits=splits,
            )
            logger.info("tuned %s: cv=%.4f val_roc=%.4f",
                        name, out[name]["cv_best_score"],
                        out[name]["val_metrics"]["roc_auc"])
        except Exception as exc:  # pragma: no cover
            logger.warning("tuning failed for %s: %s", name, exc)
            out[name] = {"model": name, "error": str(exc)}
    return out, splits
