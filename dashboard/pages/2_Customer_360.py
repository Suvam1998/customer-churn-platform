"""Customer 360 — full profile, SHAP explanation, recommendation, live event."""
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

from lib.common import api_down_banner, disclaimer, features_dir, get_client  # noqa: E402

st.set_page_config(page_title="Customer 360", page_icon="🧭", layout="wide")
st.title("🧭 Customer 360")
disclaimer(st)
client = get_client()


@st.cache_data(ttl=300)
def customer_ids():
    val = features_dir() / "customer_value.parquet"
    if val.exists():
        return pd.read_parquet(val)["customerID"].tolist()
    return ["7590-VHVEG"]


ids = customer_ids()
default_idx = ids.index("7590-VHVEG") if "7590-VHVEG" in ids else 0
cid = st.selectbox("Select customer", ids, index=default_idx)

cust = client.customer(cid)
if not isinstance(cust, dict) or "__error__" in cust:
    api_down_banner(st, cust)
    st.stop()

# --- profile + risk ---
left, right = st.columns([2, 3])
with left:
    st.subheader("Profile")
    st.write({
        "Customer": cust["customer_id"],
        "Tenure (months)": cust["tenure"],
        "Contract": cust["contract"],
        "Payment method": cust["payment_method"],
        "Internet": cust["internet_service"],
        "Monthly charges": cust["monthly_charges"],
        "Total charges": cust["total_charges"],
        "Segment": cust.get("segment"),
        "Actual churn (historical)": cust["actual_churn"],
    })
with right:
    st.subheader("Risk & value")
    a, b, c = st.columns(3)
    a.metric("Churn probability", f"{cust['churn_probability']:.3f}")
    b.metric("Risk level", cust["risk_level"])
    c.metric("Revenue at risk (EST)", f"{cust['revenue_at_risk']:,.0f}")
    st.metric("Estimated CLV", f"{cust['estimated_clv']:,.0f}")

st.divider()

# --- SHAP explanation ---
st.subheader("Why? — SHAP explanation")
exp = client.explanation(cid)
if isinstance(exp, dict) and "top_positive_factors" in exp:
    pos = pd.DataFrame(exp["top_positive_factors"])
    neg = pd.DataFrame(exp["top_negative_factors"])
    both = pd.concat([pos, neg], ignore_index=True)
    if not both.empty:
        both["direction"] = ["increases risk"] * len(pos) + ["reduces risk"] * len(neg)
        fig = px.bar(both.sort_values("contribution"), x="contribution", y="factor",
                     color="direction", orientation="h",
                     color_discrete_map={"increases risk": "#C44E52",
                                         "reduces risk": "#55A868"})
        fig.update_layout(height=360, yaxis_title="", xaxis_title="SHAP contribution")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Explanation unavailable (SHAP endpoint). Ensure the API and model are ready.")

# --- recommendation ---
st.subheader("Recommended retention action")
rec = client.recommendation(cid)
if isinstance(rec, dict) and "recommended_action" in rec:
    st.success(f"**{rec['recommended_action']}** · urgency **{rec['urgency']}**")
    st.write(rec["reason"])
    cols = st.columns(3)
    cols[0].metric("Intervention cost (EST)", f"{rec['estimated_intervention_cost']:,.0f}")
    cols[1].metric("Expected retained (EST)", f"{rec['estimated_expected_retained_value']:,.0f}")
    cols[2].metric("Net value (EST)", f"{rec['estimated_net_value']:,.0f}")

st.divider()

# --- simulate a live event ---
st.subheader("Simulate a real-time event (SIMULATED)")
etype = st.selectbox("Event type", [
    "payment_failed", "complaint", "cancellation_attempt", "support_ticket",
    "inactivity", "plan_downgrade", "login", "purchase", "plan_upgrade"])
if st.button("Send event"):
    res = client.send_event(cid, etype)
    if isinstance(res, dict) and "new_probability" in res:
        d = res["new_probability"] - (res["previous_probability"] or 0)
        st.warning(f"SIMULATED {etype}: {res['previous_probability']:.3f} → "
                   f"{res['new_probability']:.3f} ({d:+.3f}) · now {res['risk_level']} · "
                   f"action → {res['recommended_action']}")
    else:
        st.error(f"Event failed: {res}")
