"""Phase 13 entrypoint: generate retention decisions for all customers.

Combines Phase 12 value scoring with the decision engine. Business-simulation
values are ESTIMATES from configurable assumptions, not observed outcomes.

Usage (repo root, venv active):
    python scripts/run_retention.py
Artifacts:
    results/retention_summary.json
    data/features/retention_recommendations.parquet
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
from src.retention.recommendations import decide_for_dataframe  # noqa: E402
from src.retention.scoring import compute_value_table, score_customers  # noqa: E402
from src.validation.schema import ID_COLUMN  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()
    results_dir = cfg.resolve_path("paths.results")
    features_dir = cfg.resolve_path("paths.data_features")
    results_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    scored = score_customers(cfg=cfg)
    vt = compute_value_table(scored, cfg=cfg)

    # Attach segment labels when available (affects high-value escalation).
    seg_path = features_dir / "segments.parquet"
    if seg_path.exists():
        import pandas as pd

        seg = pd.read_parquet(seg_path)[[ID_COLUMN, "segment_label"]]
        vt = vt.merge(seg, on=ID_COLUMN, how="left")

    decisions = decide_for_dataframe(vt, cfg=cfg)
    decisions.to_parquet(features_dir / "retention_recommendations.parquet", index=False)

    # Aggregates.
    action_counts = decisions["recommended_action"].value_counts().to_dict()
    urgency_counts = decisions["urgency"].value_counts().to_dict()
    actionable = decisions[decisions["recommended_action"] != "no_intervention"]
    total_cost = float(actionable["estimated_intervention_cost"].sum())
    total_retained = float(actionable["estimated_expected_retained_value"].sum())

    summary = {
        "DISCLAIMER": "Business-simulation values are ESTIMATES from assumptions, "
                      "not observed retention outcomes.",
        "n_customers": int(len(decisions)),
        "n_actionable": int(len(actionable)),
        "action_distribution": {k: int(v) for k, v in action_counts.items()},
        "urgency_distribution": {k: int(v) for k, v in urgency_counts.items()},
        "estimated_total_intervention_cost": round(total_cost, 2),
        "estimated_total_expected_retained_value": round(total_retained, 2),
        "estimated_net_value": round(total_retained - total_cost, 2),
    }
    (results_dir / "retention_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Report.
    print("=" * 90)
    print("RETENTION DECISION ENGINE  (business-simulation values are ESTIMATES)")
    print("=" * 90)
    print(f"Customers            : {len(decisions):,}")
    print(f"Actionable (non-'no_intervention'): {len(actionable):,}")
    print("-" * 90)
    print("Recommended action distribution:")
    for a, n in sorted(action_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {a:<32} {n:>6,}")
    print("Urgency distribution:")
    for u, n in urgency_counts.items():
        print(f"  {u:<12} {n:>6,}")
    print("-" * 90)
    print(f"Estimated total intervention cost      : {total_cost:,.0f} (EST)")
    print(f"Estimated total expected retained value: {total_retained:,.0f} (EST)")
    print(f"Estimated net value                    : {total_retained - total_cost:,.0f} (EST)")
    print("-" * 90)
    print("Example decisions (top 5 by revenue at risk):")
    cols = ["customer_id", "risk_level", "urgency", "recommended_action",
            "revenue_at_risk", "estimated_net_value"]
    top = decisions.sort_values("revenue_at_risk", ascending=False).head(5)
    print(top[cols].to_string(index=False))
    print("\nReason for #1:")
    print(f"  {top.iloc[0]['reason']}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
