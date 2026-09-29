"""Phase 2 entrypoint: download, load, profile, and document the dataset.

Usage (from repo root, venv active):
    python scripts/download_data.py [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402
from src.ingestion.data_dictionary import save_data_dictionary  # noqa: E402
from src.ingestion.download_dataset import download_dataset  # noqa: E402
from src.ingestion.load_data import (  # noqa: E402
    load_raw,
    print_profile,
    profile_dataframe,
    save_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire & profile the dataset")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()

    path = download_dataset(force=args.force, cfg=cfg)
    print(f"\nDataset location: {path}\n")

    df = load_raw(cfg=cfg)
    profile = profile_dataframe(
        df,
        target=cfg.get("dataset.target_column", "Churn"),
        id_column=cfg.get("dataset.id_column", "customerID"),
    )
    print_profile(profile)

    profile_path = save_profile(profile, cfg=cfg)
    dict_path = save_data_dictionary(df, cfg=cfg)
    print(f"\nProfile JSON     : {profile_path}")
    print(f"Data dictionary  : {dict_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
