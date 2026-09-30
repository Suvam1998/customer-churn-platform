"""Monitoring — data quality, prediction distribution, drift status."""
from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in (str(_here.parents[2]), str(_here.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from lib.common import disclaimer, features_dir, get_client, load_json, results_dir  # noqa: E402

st.set_page_config(page_title="Monitoring", page_icon="🩺", layout="wide")
st.title("🩺 Monitoring")
disclaimer(st)
client = get_client()

# --- drift status (real: train vs test) ---
st.subheader("Drift status (reference=train vs current=test)")
drift = client.drift()
if isinstance(drift, dict) and "status" in drift and "__error__" not in drift:
    if drift["status"] == "RETRAINING_REQUIRED":
        st.error(f"🔴 {drift['status']} — {drift['note']}")
    else:
        st.success(f"🟢 {drift['status']} — {drift['note']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Features drifted", f"{drift['n_drifted']}/{drift['n_features']}")
    c2.metric("Share drifted", f"{drift['share_drifted']:.0%}")
    c3.metric("Prediction drift PSI", f"{drift.get('prediction_drift_psi', 0):.3f}")
    if drift["drifted_features"]:
        st.write("Drifted features:", drift["drifted_features"])
    st.caption(f"PSI threshold = {drift['threshold']}. RETRAINING_REQUIRED is a "
               "signal only — no model is auto-deployed.")
else:
    st.warning("Drift endpoint unavailable (start the API).")

# --- injected-drift demo figures (from scripts/run_monitoring.py) ---
from lib.common import figures_dir, show_image_if_exists  # noqa: E402
with st.expander("Drift PSI figures (run scripts/run_monitoring.py)"):
    c1, c2 = st.columns(2)
    with c1:
        show_image_if_exists(st, figures_dir() / "monitoring" / "drift_real.png")
    with c2:
        show_image_if_exists(st, figures_dir() / "monitoring" / "drift_simulated.png")

# --- data quality summary ---
st.subheader("Data quality (validation report)")
report = load_json(results_dir() / "validation_report.json")
if report:
    st.metric("Checks passed", f"{report['n_checks'] - report['n_failures']}/{report['n_checks']}")
    if report["n_failures"]:
        st.error(f"{report['n_failures']} failing checks")
    else:
        st.success("All data-quality checks pass")
else:
    st.info("Run `python scripts/validate_data.py` for the data-quality report.")

# --- prediction distribution ---
st.subheader("Prediction distribution (current scoring)")
val = features_dir() / "customer_value.parquet"
if val.exists():
    df = pd.read_parquet(val)
    fig = px.histogram(df, x="churn_probability", nbins=40)
    fig.update_layout(height=340, xaxis_title="Predicted churn probability",
                      yaxis_title="Customers")
    st.plotly_chart(fig, use_container_width=True)
    rc = df["risk_level"].value_counts().to_dict()
    st.write("Risk level counts:", rc)
else:
    st.info("Run `python scripts/run_value.py` to compute predictions.")

st.caption("Full feature/prediction drift (Evidently + custom metrics) arrives in "
           "Phase 20.")
