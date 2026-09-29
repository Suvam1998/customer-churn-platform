"""Phase 3 tests: schema/quality validation and leakage-safe preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingestion.load_data import load_raw
from src.preprocessing.preprocess import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_preprocessor,
    clean_raw,
    split_data,
)
from src.validation.data_quality import validate
from src.validation.schema import TARGET_COLUMN


@pytest.fixture(scope="module")
def raw_df():
    return load_raw()


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_raw(raw_df)


# ---- validation ----------------------------------------------------------

def test_validation_passes_on_real_data(raw_df):
    report = validate(raw_df)
    assert report.passed, f"validation failures: {[c.name for c in report.failures]}"


def test_validation_flags_unexpected_category(raw_df):
    bad = raw_df.copy()
    bad.loc[bad.index[0], "Contract"] = "Lifetime"  # not an allowed value
    report = validate(bad)
    assert not report.passed
    assert any("Contract" in c.name for c in report.failures)


# ---- cleaning ------------------------------------------------------------

def test_totalcharges_converted_and_zero_tenure_filled(clean_df):
    assert pd.api.types.is_numeric_dtype(clean_df["TotalCharges"])
    # All 11 blanks were tenure==0 and set to 0.0 -> no NaN remain.
    assert int(clean_df["TotalCharges"].isna().sum()) == 0
    assert (clean_df.loc[clean_df["tenure"] == 0, "TotalCharges"] == 0.0).all()


def test_target_encoded_binary(clean_df):
    assert set(clean_df[TARGET_COLUMN].unique()) == {0, 1}


def test_no_rows_dropped(raw_df, clean_df):
    assert len(clean_df) == len(raw_df)


# ---- splitting -----------------------------------------------------------

def test_split_sizes_and_no_overlap(clean_df):
    splits = split_data(clean_df)
    total = len(splits.y_train) + len(splits.y_val) + len(splits.y_test)
    assert total == len(clean_df)
    # No customer appears in more than one split.
    ids = set(splits.id_train) | set(splits.id_val) | set(splits.id_test)
    assert len(ids) == total


def test_split_is_stratified(clean_df):
    splits = split_data(clean_df)
    rates = [splits.y_train.mean(), splits.y_val.mean(), splits.y_test.mean()]
    # Stratification keeps churn rate close across splits.
    assert max(rates) - min(rates) < 0.02


def test_split_is_reproducible(clean_df):
    a = split_data(clean_df)
    b = split_data(clean_df)
    assert list(a.id_test) == list(b.id_test)


# ---- leakage-safe preprocessing -----------------------------------------

def test_preprocessor_fit_on_train_only_no_nan(clean_df):
    splits = split_data(clean_df)
    pre = build_preprocessor()
    pre.fit(splits.X_train)  # fit on TRAIN only
    for X in (splits.X_train, splits.X_val, splits.X_test):
        arr = pre.transform(X)
        assert not np.isnan(arr).any()
        assert arr.shape[0] == len(X)


def test_feature_groups_cover_all_features():
    assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) == set(FEATURE_COLUMNS)


def test_unknown_category_handled_at_transform(clean_df):
    """A category unseen during fit must not crash transform (handle_unknown)."""
    splits = split_data(clean_df)
    pre = build_preprocessor()
    pre.fit(splits.X_train)
    x = splits.X_test.iloc[[0]].copy()
    x.loc[:, "PaymentMethod"] = "Crypto wallet"  # unseen
    arr = pre.transform(x)  # should not raise
    assert arr.shape[0] == 1
