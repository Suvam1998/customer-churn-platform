"""Model training utilities.

Builds leakage-safe sklearn pipelines (preprocessor fit inside the pipeline on
the training split only) and provides split loading with an optional engineered
feature set. Used by the baseline (Phase 6) and later model phases.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import Config, get_config
from src.features.feature_engineering import (
    FEATURED_FEATURE_COLUMNS,
    build_featured_preprocessor,
    engineer_features,
)
from src.ingestion.load_data import load_raw
from src.preprocessing.preprocess import (
    FEATURE_COLUMNS,
    DataSplits,
    build_preprocessor,
    clean_raw,
    split_data,
)


def load_splits(include_engineered: bool = True, cfg: Config | None = None) -> DataSplits:
    """Load -> clean -> [engineer] -> stratified split. Leakage-safe (engineering
    is row-wise; the split precedes any model/preprocessor fitting)."""
    cfg = cfg or get_config()
    df = clean_raw(load_raw(cfg=cfg))
    if include_engineered:
        df = engineer_features(df)
        cols = FEATURED_FEATURE_COLUMNS
    else:
        cols = FEATURE_COLUMNS
    return split_data(df, cfg=cfg, feature_columns=cols)


def build_pipeline(estimator, include_engineered: bool = True) -> Pipeline:
    """Wrap an (unfitted) preprocessor + estimator into one Pipeline.

    Fitting the returned pipeline on X_train fits the preprocessor on the
    training split only — the core leakage control.
    """
    pre = build_featured_preprocessor() if include_engineered else build_preprocessor()
    return Pipeline([("pre", pre), ("clf", estimator)])


def make_logreg(cfg: Config | None = None, class_weight=None) -> LogisticRegression:
    cfg = cfg or get_config()
    seed = cfg.get("project.random_seed", 42)
    return LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=seed)


def train_estimator(estimator, include_engineered: bool = True,
                    cfg: Config | None = None) -> tuple[Pipeline, DataSplits]:
    """Build the pipeline, fit on train, and return (fitted_pipeline, splits)."""
    splits = load_splits(include_engineered=include_engineered, cfg=cfg)
    pipe = build_pipeline(estimator, include_engineered=include_engineered)
    pipe.fit(splits.X_train, splits.y_train)
    return pipe, splits
