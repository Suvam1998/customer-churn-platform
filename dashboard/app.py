"""Streamlit dashboard shell (Phase 1).

Provides navigation, the mandatory SIMULATED-events disclaimer, and an API
connectivity check. Page bodies are implemented in later phases under
``dashboard/pages/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

# Make ``src`` importable when run via ``streamlit run dashboard/app.py``.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402

cfg = get_config()
API_BASE = cfg.get("dashboard.api_base_url", "http://localhost:8000")

st.set_page_config(
    page_title="Churn & Retention Platform",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Real-Time Customer Churn Prediction & Intelligent Retention Platform")

st.info(
    "**Data provenance:** Customer records and historical churn labels originate "
    "from the real *IBM Telco Customer Churn* dataset. Any real-time events shown "
    "in this platform are **SIMULATED**, because the source dataset is historical "
    "rather than a live event stream."
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        [
            "Overview",
            "Customer Risk",
            "Customer 360",
            "Real-Time Monitor",
            "Segmentation",
            "Model Performance",
            "Monitoring",
        ],
    )
    st.divider()
    st.caption(f"Version {cfg.get('project.version', '0.1.0')}")

    # API connectivity check.
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=2)
        if resp.ok:
            st.success(f"API: {resp.json().get('status', 'ok')}")
        else:
            st.warning(f"API returned {resp.status_code}")
    except Exception:
        st.error("API unreachable — start it with `uvicorn api.main:app`")

st.subheader(page)
st.write(
    "🚧 This page will be implemented in a later phase. Phase 1 delivers the "
    "project foundation, configuration, health API, and dashboard shell."
)
