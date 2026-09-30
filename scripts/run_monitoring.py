"""Phase 20 entrypoint: compute drift reports and persist monitoring metrics.

Runs two scenarios:
  1. REAL: reference (train) vs current (test) — expected to be STABLE for a
     static dataset (honest baseline).
  2. SIMULATED drift: an injected covariate shift to demonstrate the detector
     firing (RETRAINING_REQUIRED). Clearly labelled as synthetic.

Usage:
    python scripts/run_monitoring.py
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
import joblib  # noqa: E402

from src.config import get_config  # noqa: E402
from src.features.feature_engineering import FEATURED_FEATURE_COLUMNS  # noqa: E402
from src.models.train import load_splits  # noqa: E402
from src.monitoring.monitor import MONITOR_CATEGORICAL, MONITOR_NUMERIC, build_report, inject_drift  # noqa: E402


def _psi_figure(report: dict, out: Path, title: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pf = report["feature_drift"]["per_feature"]
    feats = sorted(pf, key=lambda c: pf[c]["psi"])
    psis = [pf[c]["psi"] for c in feats]
    colors = ["#C44E52" if pf[c]["drifted"] else "#4C72B0" for c in feats]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(feats, psis, color=colors)
    ax.axvline(report["feature_drift"]["threshold"], color="grey", linestyle="--",
               label=f"threshold={report['feature_drift']['threshold']}")
    ax.set_title(title)
    ax.set_xlabel("PSI")
    ax.legend()
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()
    results_dir = cfg.resolve_path("paths.results")
    fig_dir = cfg.resolve_path("paths.figures") / "monitoring"
    results_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(include_engineered=True, cfg=cfg)
    model = joblib.load(cfg.resolve_path("paths.models") / "production_model.joblib")
    ref_probs = model.predict_proba(splits.X_train[FEATURED_FEATURE_COLUMNS])[:, 1]
    cur_probs = model.predict_proba(splits.X_test[FEATURED_FEATURE_COLUMNS])[:, 1]

    # Scenario 1: real train-vs-test.
    real = build_report(splits.X_train, splits.X_test, ref_probs, cur_probs,
                        scenario="real:train-vs-test", cfg=cfg)
    _psi_figure(real, fig_dir / "drift_real.png", "Feature drift — REAL (train vs test)")

    # Scenario 2: simulated covariate shift.
    shifted = inject_drift(splits.X_test)
    shifted_probs = model.predict_proba(shifted[FEATURED_FEATURE_COLUMNS])[:, 1]
    sim = build_report(splits.X_train, shifted, ref_probs, shifted_probs,
                       scenario="SIMULATED:injected-drift", cfg=cfg)
    _psi_figure(sim, fig_dir / "drift_simulated.png",
                "Feature drift — SIMULATED injected shift")

    report = {"real": real, "simulated": sim}
    (results_dir / "monitoring_report.json").write_text(json.dumps(report, indent=2),
                                                        encoding="utf-8")

    def summarize(tag, r):
        fd = r["feature_drift"]
        pdd = r["prediction_drift"]
        print(f"\n[{tag}] status={r['status']}")
        print(f"  features drifted: {fd['n_drifted']}/{fd['n_features']} "
              f"(share={fd['share_drifted']}, threshold={fd['threshold']})")
        if fd["drifted_features"]:
            print(f"  drifted: {fd['drifted_features']}")
        print(f"  prediction drift PSI: {pdd['psi']} ({pdd['band']}, drifted={pdd['drifted']})")

    print("=" * 84)
    print("MODEL / DATA DRIFT MONITORING")
    print("=" * 84)
    summarize("REAL train-vs-test", real)
    summarize("SIMULATED injected drift", sim)
    print("-" * 84)
    print("NOTE: RETRAINING_REQUIRED is a SIGNAL only — no model is auto-deployed.")
    print(f"Artifacts: results/monitoring_report.json, "
          f"docs/figures/monitoring/drift_{{real,simulated}}.png")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
