"""Registry of candidate estimators for model comparison (Phase 7).

Defaults are reasonable but light (not yet tuned — tuning is Phase 8). Optional
gradient-boosting libraries are imported defensively: if a wheel is unavailable
for the running interpreter, that model is SKIPPED with a note rather than
fabricated.

Neural network: scikit-learn's ``MLPClassifier`` is used so the platform stays
reproducible on Python 3.14 (TensorFlow/PyTorch wheels are not reliably
available there). This is an explicit, documented substitution.
"""
from __future__ import annotations

import logging

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import Config, get_config

logger = logging.getLogger("churn.models")


def get_estimators(
    cfg: Config | None = None,
    class_weight=None,
) -> tuple[dict, list[str]]:
    """Return ``(estimators, skipped)``.

    ``estimators`` maps model name -> unfitted estimator. ``skipped`` lists any
    optional models whose library could not be imported.
    """
    cfg = cfg or get_config()
    seed = cfg.get("project.random_seed", 42)
    estimators: dict = {}
    skipped: list[str] = []

    estimators["logistic_regression"] = LogisticRegression(
        max_iter=1000, class_weight=class_weight, random_state=seed
    )
    estimators["decision_tree"] = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=20, class_weight=class_weight, random_state=seed
    )
    estimators["random_forest"] = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5, n_jobs=-1,
        class_weight=class_weight, random_state=seed
    )

    # scale_pos_weight for xgboost/lightgbm when balancing is requested.
    spw = None
    if class_weight == "balanced":
        spw = 73.46 / 26.54  # ~ neg/pos ratio from the real target distribution

    try:
        from xgboost import XGBClassifier

        estimators["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            scale_pos_weight=spw, random_state=seed, n_jobs=-1,
            tree_method="hist",
        )
    except Exception as exc:  # pragma: no cover
        skipped.append("xgboost")
        logger.warning("xgboost unavailable: %s", exc)

    try:
        from lightgbm import LGBMClassifier

        estimators["lightgbm"] = LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            subsample=0.9, class_weight=class_weight, random_state=seed,
            n_jobs=-1, verbose=-1,
        )
    except Exception as exc:  # pragma: no cover
        skipped.append("lightgbm")
        logger.warning("lightgbm unavailable: %s", exc)

    try:
        from catboost import CatBoostClassifier

        estimators["catboost"] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced" if class_weight == "balanced" else None,
            random_seed=seed, verbose=0, allow_writing_files=False,
        )
    except Exception as exc:  # pragma: no cover
        skipped.append("catboost")
        logger.warning("catboost unavailable: %s", exc)

    estimators["neural_network"] = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
        max_iter=300, early_stopping=True, random_state=seed,
    )

    return estimators, skipped
