"""Phase 4 entrypoint: generate EDA figures + a Markdown summary with real
numbers from the cleaned dataset.

Usage (repo root, venv active):
    python scripts/run_eda.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import get_config  # noqa: E402
from src.eda import analysis, plots  # noqa: E402
from src.ingestion.load_data import load_raw  # noqa: E402
from src.preprocessing.preprocess import clean_raw  # noqa: E402


def _md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:  # tabulate missing -> simple fallback
        return "```\n" + df.to_string(index=False) + "\n```"


def build_summary(df: pd.DataFrame) -> str:
    ov = analysis.overall_churn(df)
    lines = [
        "# EDA Summary — IBM Telco Customer Churn",
        "",
        "> Generated from the real cleaned dataset "
        "(`python scripts/run_eda.py`). Figures live in `docs/figures/`.",
        "",
        "## Overall churn",
        "",
        f"- Customers: **{ov['n_customers']:,}**",
        f"- Churned: **{ov['n_churned']:,}** ({ov['churn_rate']:.2%})",
        f"- Retained: **{ov['n_retained']:,}**",
        "",
        "## Churn rate by contract",
        "",
        _md_table(analysis.churn_rate_by_category(df, "Contract")),
        "",
        "## Churn rate by payment method",
        "",
        _md_table(analysis.churn_rate_by_category(df, "PaymentMethod")),
        "",
        "## Churn rate by internet service",
        "",
        _md_table(analysis.churn_rate_by_category(df, "InternetService")),
        "",
        "## Churn rate by tenure band",
        "",
        _md_table(analysis.churn_rate_by_tenure_band(df)),
        "",
        "## Numeric summary by churn outcome",
        "",
        "```",
        analysis.numeric_summary_by_churn(df).to_string(),
        "```",
        "",
        "## Correlation with churn (numeric)",
        "",
        "```",
        analysis.correlation_matrix(df)["Churn"].sort_values(ascending=False).to_string(),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()
    df = clean_raw(load_raw(cfg=cfg))

    saved = plots.generate_all(df, cfg=cfg)
    summary = build_summary(df)
    out = cfg.root / "docs" / "eda_summary.md"
    out.write_text(summary, encoding="utf-8")

    print("Figures generated:")
    for p in saved:
        print(f"  {p.relative_to(cfg.root)}")
    print(f"\nSummary written: {out.relative_to(cfg.root)}")
    print("\n--- Key facts ---")
    ov = analysis.overall_churn(df)
    print(f"Overall churn rate: {ov['churn_rate']:.2%}")
    print("Churn rate by contract:")
    print(analysis.churn_rate_by_category(df, "Contract").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
