"""Phase 11 tests: K-Means segmentation (data-driven k + profiling)."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import get_config
from src.segmentation.clustering import (
    SEGMENT_FEATURES,
    build_segmentation_frame,
    choose_k,
    fit_segmentation,
    profile_clusters,
    suggest_k,
)


def _has_production_model() -> bool:
    return (get_config().resolve_path("paths.models") / "production_model.joblib").exists()


pytestmark = pytest.mark.skipif(
    not _has_production_model(),
    reason="run Phase 9 (calibrate_model) first to create the production model",
)


@pytest.fixture(scope="module")
def seg_df():
    return build_segmentation_frame()


def test_segmentation_frame_has_features(seg_df):
    for col in SEGMENT_FEATURES:
        assert col in seg_df.columns
    assert seg_df["churn_probability"].between(0, 1).all()


def test_choose_k_and_suggest(seg_df):
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(seg_df[SEGMENT_FEATURES].to_numpy(float))
    table = choose_k(X, k_range=range(2, 6))
    # Inertia is non-increasing as k grows.
    assert table["inertia"].is_monotonic_decreasing
    assert table["silhouette"].between(-1, 1).all()
    k = suggest_k(table)
    assert k == int(table.loc[table["silhouette"].idxmax(), "k"])


def test_fit_and_profile(seg_df):
    df_c, scaler, km, X_scaled = fit_segmentation(seg_df, k=3)
    assert df_c["cluster"].nunique() == 3
    profile = profile_clusters(df_c)
    assert profile["size"].sum() == len(seg_df)
    # Labels follow the value/risk scheme.
    for lab in profile["label"]:
        val, risk = lab.split("/")
        assert val in {"High-value", "Low-value"}
        assert risk in {"high-risk", "low-risk"}


def test_reproducible_clusters(seg_df):
    a, *_ = fit_segmentation(seg_df, k=3, seed=42)
    b, *_ = fit_segmentation(seg_df, k=3, seed=42)
    # Same seed -> identical cluster sizes.
    assert sorted(a["cluster"].value_counts().tolist()) == \
        sorted(b["cluster"].value_counts().tolist())
