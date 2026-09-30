"""Phase 19 entrypoint: log experiments to MLflow and register the model.

Usage (repo root, venv active):
    python scripts/run_mlflow.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db      # to view
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402
from mlops import train_pipeline  # noqa: E402
from mlops.mlflow_utils import get_tracking_uri  # noqa: E402
from mlops.register_model import register_production  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()

    print("=" * 80)
    print("MLflow — experiment tracking + model registry")
    print("=" * 80)
    print(f"Tracking URI: {get_tracking_uri(cfg)}")

    counts = train_pipeline.run(cfg)
    print("-" * 80)
    print("Logged runs:")
    for k, v in counts.items():
        print(f"  {k:<18} {v}")

    reg = register_production(cfg)
    print("-" * 80)
    print("Model registry:")
    print(f"  registered model : {reg['registered_model']}")
    print(f"  version          : {reg['version']}")
    print(f"  stage (alias)    : {reg['stage']}")
    print(f"  promotion gate   : {'PASS' if reg['gate_passed'] else 'FAIL'} "
          f"({reg['gate_reason']})")
    print("-" * 80)
    print("View: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
