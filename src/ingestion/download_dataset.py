"""Acquire the real IBM Telco Customer Churn dataset.

Responsibilities:
  1. Resolve the target path under ``data/raw/`` (never a hardcoded path).
  2. Skip download if the file already exists (unless ``force``).
  3. Download the real dataset from the configured public mirror.
  4. Validate the downloaded file (non-empty, parseable CSV, expected schema).

The dataset is the historical/static IBM Telco Customer Churn dataset. It is
used unmodified as the source of truth for all downstream phases.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import Config, get_config

logger = logging.getLogger("churn.ingestion")

# Minimum columns we require to consider the file a valid Telco churn dataset.
REQUIRED_COLUMNS = {
    "customerID",
    "gender",
    "tenure",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}

# Fallback public mirrors (tried in order) in case the primary URL is down.
FALLBACK_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/nssharmaofficial/telco-customer-churn/main/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]


def dataset_path(cfg: Config | None = None) -> Path:
    """Absolute path to the raw dataset CSV under ``data/raw/``."""
    cfg = cfg or get_config()
    raw_dir = cfg.resolve_path("paths.data_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = cfg.get("dataset.raw_filename", "telco_customer_churn.csv")
    return raw_dir / filename


def _candidate_urls(cfg: Config) -> list[str]:
    urls: list[str] = []
    primary = cfg.get("dataset.source_url")
    if primary:
        urls.append(primary)
    for u in FALLBACK_URLS:
        if u not in urls:
            urls.append(u)
    return urls


def _download_to(url: str, dest: Path, timeout: int = 60) -> None:
    logger.info("downloading dataset from %s", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    content = resp.content
    if not content or len(content) < 1000:
        raise ValueError(f"downloaded file suspiciously small ({len(content)} bytes)")
    dest.write_bytes(content)


def validate_file(path: Path) -> pd.DataFrame:
    """Validate that ``path`` is a parseable CSV with the required columns.

    Returns the parsed DataFrame so callers can avoid re-reading.
    """
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"dataset file missing or empty: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"dataset missing required columns: {sorted(missing)}")
    if len(df) < 1000:
        raise ValueError(f"dataset has too few rows: {len(df)}")
    return df


def download_dataset(force: bool = False, cfg: Config | None = None) -> Path:
    """Ensure the real dataset exists locally; download if needed.

    Returns the path to the validated dataset.
    """
    cfg = cfg or get_config()
    dest = dataset_path(cfg)

    if dest.exists() and not force:
        logger.info("dataset already present at %s (use force=True to refresh)", dest)
        validate_file(dest)
        return dest

    last_err: Exception | None = None
    for url in _candidate_urls(cfg):
        try:
            _download_to(url, dest)
            validate_file(dest)
            logger.info("dataset saved and validated at %s", dest)
            return dest
        except Exception as exc:  # try next mirror
            last_err = exc
            logger.warning("source failed (%s): %s", url, exc)
            if dest.exists():
                dest.unlink(missing_ok=True)

    raise RuntimeError(
        f"failed to download dataset from all sources; last error: {last_err}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    path = download_dataset()
    print(f"Dataset ready: {path}")
