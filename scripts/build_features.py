"""Phase 5 entrypoint: engineer features, persist the featured dataset, and
write the feature dictionary.

Row-wise engineering has no leakage, so materialising features on the full
cleaned dataset (for inspection / Databricks Gold parity) is safe. The
train-only fitting still happens later in the model pipeline.

Usage (repo root, venv active):
    python scripts/build_features.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402
from src.features.feature_engineering import (  # noqa: E402
    ENGINEERED_NUMERIC,
    FEATURED_FEATURE_COLUMNS,
    build_feature_dictionary,
    build_featured_preprocessor,
    engineer_features,
)
from src.ingestion.load_data import load_raw  # noqa: E402
from src.preprocessing.preprocess import (  # noqa: E402
    clean_raw,
    split_data,
)
from src.validation.schema import ID_COLUMN, TARGET_COLUMN  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()

    df = engineer_features(clean_raw(load_raw(cfg=cfg)))

    # Persist featured dataset (id + engineered/base features + target).
    features_dir = cfg.resolve_path("paths.data_features")
    features_dir.mkdir(parents=True, exist_ok=True)
    keep = [ID_COLUMN, *FEATURED_FEATURE_COLUMNS, TARGET_COLUMN]
    out_parquet = features_dir / "features.parquet"
    df[keep].to_parquet(out_parquet, index=False)

    # Feature dictionary doc.
    dict_path = cfg.root / "docs" / "feature_dictionary.md"
    dict_path.write_text(build_feature_dictionary(), encoding="utf-8")

    # Demonstrate leakage-safe fit on engineered set.
    splits = split_data(df, cfg=cfg, feature_columns=FEATURED_FEATURE_COLUMNS)
    pre = build_featured_preprocessor()
    pre.fit(splits.X_train)
    n_out = len(pre.get_feature_names_out())

    print("=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    print(f"Engineered numeric features : {len(ENGINEERED_NUMERIC)}")
    print(f"Featured input columns       : {len(FEATURED_FEATURE_COLUMNS)}")
    print(f"Preprocessed output features : {n_out} (fit on TRAIN only)")
    print(f"Featured dataset written     : {out_parquet.relative_to(cfg.root)}  "
          f"shape={df[keep].shape}")
    print(f"Feature dictionary written   : {dict_path.relative_to(cfg.root)}")
    print("-" * 70)
    print("Sample engineered values (first 3 customers):")
    sample_cols = [ID_COLUMN, "tenure", "average_charge_per_month",
                   "total_services", "service_adoption_score",
                   "contract_risk_indicator", "payment_risk_indicator",
                   "tenure_risk_indicator"]
    print(df[sample_cols].head(3).to_string(index=False))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
