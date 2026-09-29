"""Leakage-safe preprocessing: cleaning, splitting, and a sklearn pipeline.

Cleaning decisions (documented):
  * ``TotalCharges`` is text in the raw file with 11 blank values, all for
    ``tenure == 0`` customers. A brand-new customer has accrued ~0 total
    charges, so we deterministically set those blanks to 0.0. This is a
    domain rule (not learned from data) => no leakage. A median imputer
    remains in the pipeline as a safety net for any other NaN.
  * The target ``Churn`` is encoded Yes=1, No=0.
  * No rows are dropped.

Leakage controls:
  * The split happens BEFORE any fitting.
  * The ``ColumnTransformer`` (imputers, scaler, one-hot encoder) is fit ONLY
    on the training split; validation/test are transformed with those fitted
    statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config, get_config
from src.validation.schema import (
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)

# Feature groups for the ColumnTransformer.
NUMERIC_FEATURES = NUMERIC_COLUMNS + BINARY_NUMERIC_COLUMNS
CATEGORICAL_FEATURES = list(CATEGORICAL_COLUMNS)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, non-leaky cleaning. Returns a new DataFrame."""
    out = df.copy()

    # TotalCharges: text -> numeric; blanks (tenure==0) -> 0.0.
    tc = pd.to_numeric(
        out["TotalCharges"].astype(str).str.strip().replace("", np.nan),
        errors="coerce",
    )
    zero_tenure = out["tenure"] == 0
    tc = tc.where(~(tc.isna() & zero_tenure), 0.0)
    out["TotalCharges"] = tc

    # SeniorCitizen -> int.
    out["SeniorCitizen"] = pd.to_numeric(out["SeniorCitizen"], errors="coerce").astype("Int64")

    # Target encoding Yes=1, No=0.
    if TARGET_COLUMN in out.columns:
        out[TARGET_COLUMN] = (out[TARGET_COLUMN].astype(str) == "Yes").astype(int)

    return out


@dataclass
class DataSplits:
    """Container for the train/validation/test splits (features + target + ids)."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    id_train: pd.Series
    id_val: pd.Series
    id_test: pd.Series

    def summary(self) -> dict:
        def churn_rate(y: pd.Series) -> float:
            return round(float(y.mean()), 4)

        return {
            "train": {"n": len(self.y_train), "churn_rate": churn_rate(self.y_train)},
            "val": {"n": len(self.y_val), "churn_rate": churn_rate(self.y_val)},
            "test": {"n": len(self.y_test), "churn_rate": churn_rate(self.y_test)},
        }


def split_data(
    df_clean: pd.DataFrame,
    cfg: Config | None = None,
    feature_columns: list[str] | None = None,
) -> DataSplits:
    """Stratified 70/15/15 split. Requires an already-cleaned DataFrame.

    ``feature_columns`` selects which columns land in X (defaults to the base
    feature set; pass the engineered set to include engineered features).
    """
    cfg = cfg or get_config()
    seed = cfg.get("project.random_seed", 42)
    train_size = cfg.get("split.train_size", 0.70)
    val_size = cfg.get("split.validation_size", 0.15)
    test_size = cfg.get("split.test_size", 0.15)
    stratify_flag = cfg.get("split.stratify", True)

    cols = feature_columns or FEATURE_COLUMNS
    X = df_clean[cols]
    y = df_clean[TARGET_COLUMN]
    ids = df_clean[ID_COLUMN]

    strat = y if stratify_flag else None

    # First carve out the test set.
    X_tmp, X_test, y_tmp, y_test, id_tmp, id_test = train_test_split(
        X, y, ids, test_size=test_size, random_state=seed, stratify=strat
    )
    # Then split the remainder into train/val (val relative to remaining).
    val_relative = val_size / (train_size + val_size)
    strat_tmp = y_tmp if stratify_flag else None
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_tmp, y_tmp, id_tmp,
        test_size=val_relative, random_state=seed, stratify=strat_tmp,
    )

    return DataSplits(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        id_train=id_train, id_val=id_val, id_test=id_test,
    )


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Return an UNFITTED ColumnTransformer (fit only on training data).

    Feature lists default to the base set; pass extended lists to include
    engineered features (see ``features.feature_engineering``).
    """
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return output feature names from a FITTED preprocessor."""
    return list(preprocessor.get_feature_names_out())


def prepare_data(
    cfg: Config | None = None,
    include_engineered: bool = False,
) -> tuple[DataSplits, ColumnTransformer]:
    """Convenience: load raw -> clean -> [engineer] -> split -> fit on TRAIN.

    When ``include_engineered`` is True the engineered features are added
    (row-wise, before split => no leakage) and the preprocessor covers the
    extended column set. Returns the splits and the fitted preprocessor — the
    canonical leakage-safe entrypoint for model-training phases.
    """
    from src.ingestion.load_data import load_raw

    cfg = cfg or get_config()
    df = clean_raw(load_raw(cfg=cfg))

    if include_engineered:
        from src.features.feature_engineering import (
            FEATURED_FEATURE_COLUMNS,
            build_featured_preprocessor,
            engineer_features,
        )

        df = engineer_features(df)
        splits = split_data(df, cfg=cfg, feature_columns=FEATURED_FEATURE_COLUMNS)
        preprocessor = build_featured_preprocessor()
    else:
        splits = split_data(df, cfg=cfg)
        preprocessor = build_preprocessor()

    preprocessor.fit(splits.X_train)  # <-- fit on training data ONLY
    return splits, preprocessor
