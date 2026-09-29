"""Load and profile the raw IBM Telco Customer Churn dataset.

Profiling is descriptive only — it never mutates the data. Cleaning decisions
(e.g. the ``TotalCharges`` blank-string issue) are handled in the validation /
preprocessing phases, not here.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import Config, get_config
from src.ingestion.download_dataset import dataset_path, validate_file

logger = logging.getLogger("churn.ingestion")


def load_raw(cfg: Config | None = None, ensure: bool = True) -> pd.DataFrame:
    """Load the raw dataset as a DataFrame.

    If ``ensure`` and the file is absent, it is downloaded first.
    """
    cfg = cfg or get_config()
    path = dataset_path(cfg)
    if not path.exists():
        if ensure:
            from src.ingestion.download_dataset import download_dataset

            download_dataset(cfg=cfg)
        else:
            raise FileNotFoundError(f"raw dataset not found: {path}")
    return validate_file(path)


def profile_dataframe(
    df: pd.DataFrame,
    target: str = "Churn",
    id_column: str = "customerID",
) -> dict[str, Any]:
    """Compute a descriptive profile of the dataset."""
    n_rows, n_cols = df.shape

    missing_counts = df.isna().sum()
    missing = {
        col: {
            "n_missing": int(missing_counts[col]),
            "pct_missing": round(float(missing_counts[col]) / n_rows * 100, 3),
        }
        for col in df.columns
        if missing_counts[col] > 0
    }

    # Duplicates: full-row and by customer id (if present).
    dup_full = int(df.duplicated().sum())
    dup_ids = int(df.duplicated(subset=[id_column]).sum()) if id_column in df.columns else None

    # Target distribution.
    target_dist: dict[str, Any] = {}
    if target in df.columns:
        counts = df[target].value_counts(dropna=False)
        target_dist = {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "proportions": {
                str(k): round(float(v) / n_rows, 4)
                for k, v in counts.items()
            },
        }

    # Note the known TotalCharges quirk (numeric column stored as object with
    # blank strings for some tenure==0 rows).
    notes: list[str] = []
    if "TotalCharges" in df.columns and not is_numeric_dtype(df["TotalCharges"]):
        blanks = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
        notes.append(
            f"TotalCharges is stored as text; {blanks} rows contain blank "
            "strings (all with tenure==0). Handled in preprocessing, not dropped."
        )

    return {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": missing,
        "duplicates": {"full_row": dup_full, "by_id": dup_ids},
        "target": {"column": target, "distribution": target_dist},
        "notes": notes,
    }


def print_profile(profile: dict[str, Any]) -> None:
    """Pretty-print the key profile facts to stdout."""
    print("=" * 70)
    print("IBM TELCO CUSTOMER CHURN — DATA PROFILE (real dataset)")
    print("=" * 70)
    print(f"Rows            : {profile['n_rows']:,}")
    print(f"Columns         : {profile['n_columns']}")
    print(f"Duplicate rows  : {profile['duplicates']['full_row']} "
          f"(by id: {profile['duplicates']['by_id']})")
    print("-" * 70)
    print("Column dtypes:")
    for col, dt in profile["dtypes"].items():
        print(f"  {col:<20} {dt}")
    print("-" * 70)
    if profile["missing_values"]:
        print("Missing values:")
        for col, info in profile["missing_values"].items():
            print(f"  {col:<20} {info['n_missing']} ({info['pct_missing']}%)")
    else:
        print("Missing values: none (NaN); see notes for encoded blanks.")
    print("-" * 70)
    dist = profile["target"]["distribution"]
    if dist:
        print(f"Target '{profile['target']['column']}' distribution:")
        for k, v in dist["counts"].items():
            pct = dist["proportions"][k]
            print(f"  {k:<6} {v:>6,}  ({pct:.2%})")
    if profile["notes"]:
        print("-" * 70)
        print("Notes:")
        for n in profile["notes"]:
            print(f"  - {n}")
    print("=" * 70)


def save_profile(profile: dict[str, Any], cfg: Config | None = None) -> Path:
    """Persist the profile as JSON under the configured results directory."""
    cfg = cfg or get_config()
    results = cfg.resolve_path("paths.results")
    results.mkdir(parents=True, exist_ok=True)
    out = results / "data_profile.json"
    out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    logger.info("profile written to %s", out)
    return out
