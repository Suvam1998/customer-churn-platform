"""Phase 5 tests: engineered features — correctness and leakage safety."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    ENGINEERED_NUMERIC,
    FEATURED_FEATURE_COLUMNS,
    FeatureEngineer,
    build_featured_preprocessor,
    engineer_features,
)
from src.ingestion.load_data import load_raw
from src.preprocessing.preprocess import clean_raw, split_data
from src.validation.schema import TARGET_COLUMN


@pytest.fixture(scope="module")
def clean_df():
    return clean_raw(load_raw())


@pytest.fixture(scope="module")
def feat_df(clean_df):
    return engineer_features(clean_df)


def test_all_engineered_columns_present(feat_df):
    for col in ENGINEERED_NUMERIC:
        assert col in feat_df.columns


def test_no_rows_added_or_dropped(clean_df, feat_df):
    assert len(feat_df) == len(clean_df)


def test_binary_flags_are_zero_one(feat_df):
    flags = [
        "is_month_to_month", "is_long_term_contract", "has_tech_support",
        "has_online_security", "has_streaming", "payment_risk_indicator",
        "contract_risk_indicator", "tenure_risk_indicator",
    ]
    for f in flags:
        assert set(feat_df[f].unique()).issubset({0, 1})


def test_service_adoption_bounds(feat_df):
    assert feat_df["total_services"].between(0, 9).all()
    assert feat_df["service_adoption_score"].between(0.0, 1.0).all()


def test_average_charge_matches_definition(feat_df):
    pos = feat_df["tenure"] > 0
    expected = feat_df.loc[pos, "TotalCharges"] / feat_df.loc[pos, "tenure"]
    np.testing.assert_allclose(
        feat_df.loc[pos, "average_charge_per_month"].to_numpy(),
        expected.round(2).to_numpy(), atol=0.01,
    )
    # tenure==0 falls back to MonthlyCharges.
    zero = feat_df["tenure"] == 0
    if zero.any():
        assert np.allclose(
            feat_df.loc[zero, "average_charge_per_month"],
            feat_df.loc[zero, "MonthlyCharges"], atol=0.01,
        )


def test_risk_indicators_match_domain_rules(feat_df):
    assert (feat_df["contract_risk_indicator"] ==
            (feat_df["Contract"] == "Month-to-month").astype(int)).all()
    assert (feat_df["payment_risk_indicator"] ==
            (feat_df["PaymentMethod"] == "Electronic check").astype(int)).all()
    assert (feat_df["tenure_risk_indicator"] ==
            (feat_df["tenure"] < 12).astype(int)).all()


def test_engineering_does_not_use_target(feat_df):
    """Dropping the target before engineering must yield identical features."""
    base = feat_df.drop(columns=ENGINEERED_NUMERIC + [
        "has_phone", "has_multiple_lines", "has_internet", "has_online_backup",
        "has_device_protection", "has_streaming_tv", "has_streaming_movies",
    ])
    without_target = base.drop(columns=[TARGET_COLUMN])
    re_eng = engineer_features(without_target)
    for col in ENGINEERED_NUMERIC:
        assert re_eng[col].equals(feat_df[col])


def test_transformer_matches_function(clean_df):
    fe = FeatureEngineer()
    out = fe.fit_transform(clean_df)
    ref = engineer_features(clean_df)
    for col in ENGINEERED_NUMERIC:
        assert out[col].equals(ref[col])


def test_featured_preprocessor_leakage_safe(feat_df):
    splits = split_data(feat_df, feature_columns=FEATURED_FEATURE_COLUMNS)
    pre = build_featured_preprocessor()
    pre.fit(splits.X_train)  # train only
    for X in (splits.X_train, splits.X_val, splits.X_test):
        arr = pre.transform(X)
        assert not np.isnan(arr).any()
    # Engineered set must produce more features than the base 45.
    assert len(pre.get_feature_names_out()) > 45
