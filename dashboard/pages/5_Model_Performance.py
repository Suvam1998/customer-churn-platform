"""Model Performance — comparison table, production metrics, diagnostic figures."""
from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in (str(_here.parents[2]), str(_here.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib.common import (  # noqa: E402
    disclaimer,
    figures_dir,
    get_client,
    results_dir,
    show_image_if_exists,
)

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Model Performance")
disclaimer(st)
client = get_client()

# --- production model metrics ---
model = client.model_metrics()
if isinstance(model, dict) and "metrics" in model:
    st.subheader(f"Production model: `{model['model_version']}`")
    m = model["metrics"]
    cols = st.columns(5)
    for col, key in zip(cols, ["roc_auc", "pr_auc", "f1", "log_loss", "brier"]):
        if key in m:
            col.metric(key.upper(), f"{m[key]:.4f}")
    st.caption(f"base model: {model.get('base_model')} · calibration: "
               f"{model.get('calibration')} (metrics on validation set)")

# --- comparison table ---
st.subheader("Model comparison (validation)")
cmp_path = results_dir() / "model_comparison.csv"
if cmp_path.exists():
    st.dataframe(pd.read_csv(cmp_path), use_container_width=True, hide_index=True)
else:
    st.info("Run `python scripts/train_models.py` for the comparison table.")

# --- figures ---
st.subheader("Diagnostics")
t1, t2, t3 = st.tabs(["Comparison & SHAP", "Baseline curves", "Calibration"])
with t1:
    c1, c2 = st.columns(2)
    with c1:
        show_image_if_exists(st, figures_dir() / "models" / "model_comparison.png")
    with c2:
        show_image_if_exists(st, figures_dir() / "shap" / "shap_summary.png")
with t2:
    c1, c2 = st.columns(2)
    with c1:
        show_image_if_exists(st, figures_dir() / "baseline" / "baseline_logreg_roc.png")
    with c2:
        show_image_if_exists(st, figures_dir() / "baseline" / "baseline_logreg_pr.png")
    show_image_if_exists(st, figures_dir() / "baseline" / "baseline_logreg_confusion.png")
with t3:
    c1, c2 = st.columns(2)
    with c1:
        show_image_if_exists(st, figures_dir() / "baseline" / "baseline_logreg_calibration.png")
    with c2:
        show_image_if_exists(st, figures_dir() / "calibration" / "reliability_comparison.png")
