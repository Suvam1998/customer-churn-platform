"""Phase 10 entrypoint: SHAP global + local explanations.

Builds the SHAP explainer on the tuned CatBoost model, saves a global
importance table + beeswarm summary figure, and generates local explanations
for a few real customers (including 7590-VHVEG). All values are SHAP-generated.

Usage (repo root, venv active):
    python scripts/run_explainability.py
Artifacts:
    results/shap_global_importance.csv
    results/explanations_sample.json
    docs/figures/shap/shap_summary.png
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

from src.config import get_config  # noqa: E402
from src.explainability.shap_explainer import (  # noqa: E402
    ChurnExplainer,
    engineered_dataset,
    get_customer_features,
)
from src.models.train import load_splits  # noqa: E402
from src.validation.schema import ID_COLUMN  # noqa: E402


def _summary_figure(explainer: ChurnExplainer, X_sample, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    Xt = explainer.pipeline.named_steps["pre"].transform(X_sample)
    sv = explainer.shap_values(X_sample)
    plt.figure()
    shap.summary_plot(sv, features=Xt, feature_names=explainer.feature_names,
                      show=False, max_display=15)
    plt.title("SHAP summary — feature impact on churn (validation sample)")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=120)
    plt.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()
    results_dir = cfg.resolve_path("paths.results")
    results_dir.mkdir(parents=True, exist_ok=True)

    explainer = ChurnExplainer.from_tuned(cfg)
    splits = load_splits(include_engineered=True, cfg=cfg)

    # Global importance (sample the validation set for speed).
    X_sample = splits.X_val.iloc[: min(500, len(splits.X_val))]
    gi = explainer.global_importance(X_sample, top_n=20)
    gi.to_csv(results_dir / "shap_global_importance.csv", index=False)
    _summary_figure(explainer, X_sample, cfg.resolve_path("paths.figures") / "shap" / "shap_summary.png")

    # Local explanations for example customers.
    df = engineered_dataset(cfg)
    example_ids = [df[ID_COLUMN].iloc[0], df[ID_COLUMN].iloc[1], df[ID_COLUMN].iloc[3]]
    if "7590-VHVEG" in set(df[ID_COLUMN]) and "7590-VHVEG" not in example_ids:
        example_ids[0] = "7590-VHVEG"

    explanations = []
    for cid in example_ids:
        x_row, actual = get_customer_features(cid, df=df, cfg=cfg)
        exp = explainer.explain_customer(x_row, cid, top_k=5)
        exp["actual_churn"] = actual
        explanations.append(exp)

    (results_dir / "explanations_sample.json").write_text(
        json.dumps(explanations, indent=2), encoding="utf-8")

    # Report.
    print("=" * 84)
    print("SHAP EXPLAINABILITY — global importance (top 10)")
    print("=" * 84)
    print(gi.head(10).to_string(index=False))
    for exp in explanations:
        print("-" * 84)
        print(f"Customer {exp['customer_id']}  |  churn_probability={exp['churn_probability']}  "
              f"|  actual_churn={exp['actual_churn']}")
        print("  Top factors INCREASING churn risk:")
        for f in exp["top_positive_factors"]:
            print(f"    + {f['label']:<32} SHAP={f['shap']:+}")
        print("  Top factors DECREASING churn risk:")
        for f in exp["top_negative_factors"]:
            print(f"    - {f['label']:<32} SHAP={f['shap']:+}")
    print("=" * 84)
    print(f"Artifacts: results/shap_global_importance.csv, "
          f"results/explanations_sample.json, docs/figures/shap/shap_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
