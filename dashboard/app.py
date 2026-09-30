"""Streamlit dashboard — Overview (main entry).

Run: streamlit run dashboard/app.py
Other pages live in dashboard/pages/ (native multipage).
"""
from __future__ import annotations

import sys
from pathlib import Path

# bootstrap imports
_here = Path(__file__).resolve()
_root = _here.parent.parent
for _p in (str(_root), str(_here.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from lib.common import api_down_banner, disclaimer, get_client, load_json, results_dir  # noqa: E402

st.set_page_config(page_title="Churn Platform — Overview", page_icon="📉", layout="wide")
client = get_client()

st.title("📉 Customer Churn & Retention — Overview")
disclaimer(st)

metrics = client.dashboard_metrics()
model = client.model_metrics()

if not isinstance(metrics, dict) or "__error__" in metrics:
    api_down_banner(st, metrics)
    st.stop()

# --- KPI tiles ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total customers", f"{metrics['total_customers']:,}")
c2.metric("Churned (historical)", f"{metrics['churned_customers']:,}",
          f"{metrics['churn_rate']:.1%} rate")
c3.metric("High-risk customers", f"{metrics['high_risk_customers']:,}")
c4.metric("Revenue at risk (EST)", f"{metrics['total_revenue_at_risk']:,.0f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Avg churn probability", f"{metrics['average_churn_probability']:.3f}")
c6.metric("Model version", metrics.get("model_version", "—"))
if isinstance(model, dict) and "metrics" in model:
    m = model["metrics"]
    c7.metric("Model ROC-AUC", f"{m.get('roc_auc', float('nan')):.4f}")
    c8.metric("Model Brier", f"{m.get('brier', float('nan')):.4f}")

st.divider()

# --- charts ---
left, right = st.columns(2)

with left:
    st.subheader("Risk distribution")
    rc = metrics.get("risk_level_counts", {})
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    data = {"risk_level": [k for k in order if k in rc],
            "customers": [rc[k] for k in order if k in rc]}
    fig = px.bar(data, x="risk_level", y="customers", color="risk_level",
                 color_discrete_map={"LOW": "#55A868", "MEDIUM": "#4C72B0",
                                     "HIGH": "#DD8452", "CRITICAL": "#C44E52"})
    fig.update_layout(showlegend=False, height=340)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Churn distribution (historical)")
    churned = metrics["churned_customers"]
    retained = metrics["total_customers"] - churned
    fig2 = px.pie(names=["Retained", "Churned"], values=[retained, churned],
                  color=["Retained", "Churned"],
                  color_discrete_map={"Retained": "#4C72B0", "Churned": "#DD8452"})
    fig2.update_layout(height=340)
    st.plotly_chart(fig2, use_container_width=True)

# --- revenue at risk by segment (from artifact) ---
seg = load_json(results_dir() / "segmentation_profile.json")
if seg and seg.get("clusters"):
    st.subheader("Segments — churn risk & value")
    import pandas as pd
    dfc = pd.DataFrame(seg["clusters"])
    st.dataframe(dfc[["label", "size", "avg_churn_probability",
                      "avg_monthly_charges", "avg_tenure"]],
                 use_container_width=True, hide_index=True)

st.caption("Use the sidebar to navigate: Customer Risk · Customer 360 · "
           "Real-Time Monitor · Segmentation · Model Performance · Monitoring.")
