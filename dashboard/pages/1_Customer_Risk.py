"""Customer Risk — interactive, filterable risk table."""
from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve()
_dash = _here.parents[1]
_root = _here.parents[2]
for _p in (str(_root), str(_dash)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib.common import disclaimer, features_dir  # noqa: E402

st.set_page_config(page_title="Customer Risk", page_icon="⚠️", layout="wide")
st.title("⚠️ Customer Risk")
disclaimer(st)


@st.cache_data(ttl=120)
def load_table():
    val = features_dir() / "customer_value.parquet"
    rec = features_dir() / "retention_recommendations.parquet"
    if not val.exists():
        return None
    df = pd.read_parquet(val)
    if rec.exists():
        r = pd.read_parquet(rec)[["customer_id", "recommended_action", "urgency"]]
        df = df.merge(r, left_on="customerID", right_on="customer_id", how="left")
    return df


df = load_table()
if df is None:
    st.warning("Run `python scripts/run_value.py` and `scripts/run_retention.py` "
               "to generate the risk table artifacts.")
    st.stop()

# --- filters ---
with st.sidebar:
    st.header("Filters")
    levels = st.multiselect("Risk level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                            default=["HIGH", "CRITICAL"])
    segs = sorted([s for s in df.get("segment_label", pd.Series()).dropna().unique()])
    seg_sel = st.multiselect("Segment", segs, default=segs)
    tmin, tmax = int(df["tenure"].min()), int(df["tenure"].max())
    tenure_rng = st.slider("Tenure (months)", tmin, tmax, (tmin, tmax))
    min_prob = st.slider("Min churn probability", 0.0, 1.0, 0.0, 0.05)

f = df.copy()
if levels:
    f = f[f["risk_level"].isin(levels)]
if segs:
    f = f[f["segment_label"].isin(seg_sel) | f["segment_label"].isna()]
f = f[(f["tenure"].between(*tenure_rng)) & (f["churn_probability"] >= min_prob)]
f = f.sort_values("revenue_at_risk", ascending=False)

st.metric("Matching customers", f"{len(f):,}")
cols = ["customerID", "churn_probability", "risk_level", "estimated_clv",
        "revenue_at_risk", "tenure", "segment_label"]
if "recommended_action" in f.columns:
    cols += ["recommended_action", "urgency"]
st.dataframe(f[cols].head(500), use_container_width=True, hide_index=True,
             column_config={
                 "churn_probability": st.column_config.ProgressColumn(
                     "Churn prob", min_value=0.0, max_value=1.0, format="%.3f"),
                 "revenue_at_risk": st.column_config.NumberColumn(
                     "Rev@Risk (EST)", format="%.0f"),
                 "estimated_clv": st.column_config.NumberColumn("CLV (EST)", format="%.0f"),
             })
st.caption(f"Showing up to 500 of {len(f):,} matches. Monetary values are ESTIMATES.")
