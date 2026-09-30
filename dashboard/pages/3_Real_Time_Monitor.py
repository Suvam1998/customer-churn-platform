"""Real-Time Monitor — SIMULATED event stream and risk updates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in (str(_here.parents[2]), str(_here.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib.common import disclaimer, features_dir, get_client, stream_dir  # noqa: E402

st.set_page_config(page_title="Real-Time Monitor", page_icon="📡", layout="wide")
st.title("📡 Real-Time Monitor")
st.warning("⚡ The events on this page are **SIMULATED**. The source dataset is "
           "historical, not a live stream.")
disclaimer(st)
client = get_client()


@st.cache_data(ttl=300)
def sample_ids():
    val = features_dir() / "customer_value.parquet"
    if val.exists():
        return pd.read_parquet(val)["customerID"].tolist()
    return ["7590-VHVEG"]


# --- send an event ---
st.subheader("Inject a simulated event")
c1, c2, c3 = st.columns([2, 2, 1])
ids = sample_ids()
cid = c1.selectbox("Customer", ids, index=ids.index("7590-VHVEG") if "7590-VHVEG" in ids else 0)
etype = c2.selectbox("Event", [
    "payment_failed", "complaint", "cancellation_attempt", "support_ticket",
    "inactivity", "plan_downgrade", "login", "purchase", "plan_upgrade"])
if c3.button("Send", use_container_width=True):
    res = client.send_event(cid, etype)
    if isinstance(res, dict) and "new_probability" in res:
        st.session_state.setdefault("events", [])
        st.session_state["events"].insert(0, {
            "customer": cid, "event": etype,
            "prev": res["previous_probability"], "new": res["new_probability"],
            "change": round(res["new_probability"] - (res["previous_probability"] or 0), 4),
            "risk": res["risk_level"], "action": res["recommended_action"],
        })
    else:
        st.error(f"Failed: {res}")

# --- session event log ---
st.subheader("Session event log")
if st.session_state.get("events"):
    st.dataframe(pd.DataFrame(st.session_state["events"]),
                 use_container_width=True, hide_index=True)
else:
    st.info("No events sent this session yet. Use the control above.")

# --- risk updates from the streaming topic (local broker file) ---
st.subheader("Recent risk updates from the streaming pipeline")
risk_file = stream_dir() / "customer-risk-updates.jsonl"
if risk_file.exists():
    raw = [json.loads(l) for l in risk_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    # LocalBroker stores {"key":..., "value": {...}}; unwrap to the payload.
    rows = [r.get("value", r) for r in raw]
    cols = ["customer_id", "event_type", "previous_probability", "new_probability",
            "risk_change", "new_risk_level", "recommended_action"]
    if rows:
        df = pd.DataFrame(rows)
        df = df[[c for c in cols if c in df.columns]]
        st.dataframe(df.tail(25).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("Risk-update topic is empty.")
else:
    st.info("Run `python scripts/run_streaming.py` to populate the "
            "`customer-risk-updates` topic.")
