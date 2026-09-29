"""Phase 3 entrypoint: validate raw data, then demonstrate leakage-safe
cleaning + split + preprocessing fit.

Usage (from repo root, venv active):
    python scripts/validate_data.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402
from src.ingestion.load_data import load_raw  # noqa: E402
from src.preprocessing.preprocess import (  # noqa: E402
    clean_raw,
    build_preprocessor,
    get_feature_names,
    split_data,
)
from src.validation.data_quality import print_report, validate  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()

    df_raw = load_raw(cfg=cfg)
    report = validate(df_raw)
    print_report(report)

    # Persist the validation report.
    results = cfg.resolve_path("paths.results")
    results.mkdir(parents=True, exist_ok=True)
    (results / "validation_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    # Demonstrate the leakage-safe pipeline.
    df_clean = clean_raw(df_raw)
    splits = split_data(df_clean, cfg=cfg)
    pre = build_preprocessor()
    pre.fit(splits.X_train)  # fit on TRAIN only

    Xtr = pre.transform(splits.X_train)
    Xval = pre.transform(splits.X_val)
    Xte = pre.transform(splits.X_test)
    names = get_feature_names(pre)

    print("\n" + "=" * 70)
    print("PREPROCESSING (leakage-safe: fit on TRAIN only)")
    print("=" * 70)
    print("Split sizes / churn rates:")
    for name, info in splits.summary().items():
        print(f"  {name:<6} n={info['n']:<5} churn_rate={info['churn_rate']:.4f}")
    print(f"TotalCharges NaN after cleaning : {int(df_clean['TotalCharges'].isna().sum())}")
    print(f"Transformed feature matrix shapes: train={Xtr.shape}, "
          f"val={Xval.shape}, test={Xte.shape}")
    print(f"Output feature count             : {len(names)}")
    print(f"First 8 output features          : {names[:8]}")
    print("=" * 70)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
