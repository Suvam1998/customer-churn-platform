"""Phase 12 entrypoint: CLV, revenue-at-risk, and priority scoring.

All monetary values are ESTIMATES from configurable assumptions — not actual
financial outcomes.

Usage (repo root, venv active):
    python scripts/run_value.py
Artifacts:
    results/value_summary.json
    data/features/customer_value.parquet
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import get_config  # noqa: E402
from src.retention.scoring import (  # noqa: E402
    compute_value_table,
    prioritization_shift,
    score_customers,
)
from src.validation.schema import ID_COLUMN  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()
    results_dir = cfg.resolve_path("paths.results")
    features_dir = cfg.resolve_path("paths.data_features")
    results_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    lifetime = cfg.get("clv.expected_lifetime_months", 24)
    margin = cfg.get("clv.gross_margin", 0.30)

    scored = score_customers(cfg=cfg)
    vt = compute_value_table(scored, cfg=cfg)

    # Optionally merge segment labels if available.
    seg_path = features_dir / "segments.parquet"
    if seg_path.exists():
        seg = pd.read_parquet(seg_path)[[ID_COLUMN, "segment_label"]]
        vt = vt.merge(seg, on=ID_COLUMN, how="left")

    # Persist per-customer value table (subset of columns).
    keep = [ID_COLUMN, "MonthlyCharges", "tenure", "churn_probability",
            "estimated_clv", "revenue_at_risk", "risk_level", "priority_score",
            "priority_rank", "priority_flag"]
    if "segment_label" in vt.columns:
        keep.append("segment_label")
    vt[keep].to_parquet(features_dir / "customer_value.parquet", index=False)

    # Aggregates (ESTIMATES).
    total_rar = float(vt["revenue_at_risk"].sum())
    total_clv = float(vt["estimated_clv"].sum())
    risk_counts = vt["risk_level"].value_counts().to_dict()
    shift = prioritization_shift(vt, top_percent=cfg.get("retention.priority_top_percent", 0.10))

    by_segment = None
    if "segment_label" in vt.columns:
        by_segment = (vt.groupby("segment_label")
                      .agg(customers=("revenue_at_risk", "size"),
                           total_revenue_at_risk=("revenue_at_risk", "sum"),
                           avg_churn_prob=("churn_probability", "mean"))
                      .round(2).reset_index())

    summary = {
        "DISCLAIMER": "All monetary values are ESTIMATES from configurable "
                      "assumptions, not observed financial outcomes.",
        "assumptions": {"expected_lifetime_months": lifetime, "gross_margin": margin},
        "n_customers": int(len(vt)),
        "total_estimated_clv": round(total_clv, 2),
        "total_estimated_revenue_at_risk": round(total_rar, 2),
        "risk_level_counts": {k: int(v) for k, v in risk_counts.items()},
        "rq3_prioritization_shift": shift,
    }
    (results_dir / "value_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Report.
    print("=" * 88)
    print("CUSTOMER VALUE & REVENUE AT RISK  (ALL VALUES ARE ESTIMATES)")
    print("=" * 88)
    print(f"Assumptions: expected_lifetime={lifetime} months, gross_margin={margin}")
    print(f"Customers                     : {len(vt):,}")
    print(f"Total estimated CLV           : {total_clv:,.0f} (currency units, EST)")
    print(f"Total estimated revenue@risk  : {total_rar:,.0f} (currency units, EST)")
    print(f"Risk level counts             : {risk_counts}")
    print("-" * 88)
    print("Top 5 customers by estimated revenue at risk:")
    cols = [ID_COLUMN, "churn_probability", "estimated_clv", "revenue_at_risk", "risk_level"]
    print(vt.sort_values("revenue_at_risk", ascending=False)[cols].head(5).to_string(index=False))
    if by_segment is not None:
        print("-" * 88)
        print("Estimated revenue at risk by segment:")
        print(by_segment.to_string(index=False))
    print("-" * 88)
    print(f"RQ3 — prioritisation shift (top {int(shift['top_percent']*100)}%): "
          f"{shift['overlap_pct']}% overlap with churn-prob ranking; "
          f"{shift['swapped_pct']}% ({shift['swapped_in_by_value']}) customers "
          f"newly prioritised by VALUE.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
