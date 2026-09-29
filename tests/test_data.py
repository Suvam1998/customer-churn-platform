"""Phase 2 tests: dataset ingestion, validation, and profiling.

These tests require the real dataset to be present locally. They will trigger
a one-time download if it is missing (network required on first run only).
"""
from __future__ import annotations

import pytest

from src.config import get_config
from src.ingestion.download_dataset import REQUIRED_COLUMNS, dataset_path, validate_file
from src.ingestion.load_data import load_raw, profile_dataframe


@pytest.fixture(scope="module")
def raw_df():
    return load_raw()


def test_dataset_file_exists_after_load(raw_df):
    assert dataset_path().exists()


def test_required_columns_present(raw_df):
    assert REQUIRED_COLUMNS.issubset(set(raw_df.columns))


def test_expected_shape(raw_df):
    # The canonical IBM Telco dataset has 7,043 rows and 21 columns.
    assert raw_df.shape == (7043, 21)


def test_target_is_binary_yes_no(raw_df):
    assert set(raw_df["Churn"].unique()) == {"Yes", "No"}


def test_no_duplicate_customer_ids(raw_df):
    assert raw_df["customerID"].duplicated().sum() == 0


def test_validate_file_returns_dataframe():
    df = validate_file(dataset_path())
    assert len(df) > 1000


def test_profile_structure_and_facts(raw_df):
    cfg = get_config()
    profile = profile_dataframe(
        raw_df,
        target=cfg.get("dataset.target_column"),
        id_column=cfg.get("dataset.id_column"),
    )
    assert profile["n_rows"] == 7043
    assert profile["n_columns"] == 21
    assert profile["duplicates"]["full_row"] == 0

    dist = profile["target"]["distribution"]["counts"]
    assert dist["No"] + dist["Yes"] == profile["n_rows"]

    # TotalCharges blank-string quirk must be detected and documented.
    assert any("TotalCharges" in note for note in profile["notes"])
