"""Phase 4 tests: EDA analysis tables and figure generation."""
from __future__ import annotations

import pandas as pd
import pytest

from src.eda import analysis, plots
from src.ingestion.load_data import load_raw
from src.preprocessing.preprocess import clean_raw
from src.validation.schema import TARGET_COLUMN


@pytest.fixture(scope="module")
def clean_df():
    return clean_raw(load_raw())


def test_overall_churn_counts_consistent(clean_df):
    ov = analysis.overall_churn(clean_df)
    assert ov["n_churned"] + ov["n_retained"] == ov["n_customers"]
    assert 0 < ov["churn_rate"] < 1


def test_churn_rate_by_contract_expected_order(clean_df):
    tbl = analysis.churn_rate_by_category(clean_df, "Contract")
    # Month-to-month is well documented as the highest-churn contract.
    assert tbl.iloc[0]["Contract"] == "Month-to-month"
    assert tbl["n_customers"].sum() == len(clean_df)


def test_tenure_band_rates_sum_to_population(clean_df):
    tbl = analysis.churn_rate_by_tenure_band(clean_df)
    assert tbl["n_customers"].sum() == len(clean_df)


def test_correlation_matrix_includes_target(clean_df):
    corr = analysis.correlation_matrix(clean_df)
    assert TARGET_COLUMN in corr.columns
    assert corr.loc[TARGET_COLUMN, TARGET_COLUMN] == pytest.approx(1.0)


def test_generate_all_creates_figures(clean_df, tmp_path, monkeypatch):
    # Redirect figures to a temp dir via a lightweight config shim.
    from src.config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg, "resolve_path", lambda key: tmp_path if key == "paths.figures" else cfg.root)
    saved = plots.generate_all(clean_df, cfg=cfg)
    assert len(saved) >= 10
    for p in saved:
        assert p.exists() and p.stat().st_size > 0
