"""Segmentation — cluster profiles + PCA / k-selection figures."""
from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in (str(_here.parents[2]), str(_here.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib.common import disclaimer, figures_dir, load_json, results_dir, show_image_if_exists  # noqa: E402

st.set_page_config(page_title="Segmentation", page_icon="🧩", layout="wide")
st.title("🧩 Customer Segmentation")
disclaimer(st)

profile = load_json(results_dir() / "segmentation_profile.json")
if not profile:
    st.warning("Run `python scripts/run_segmentation.py` to generate segments.")
    st.stop()

st.metric("Chosen k (silhouette)", profile["chosen_k"])

st.subheader("Cluster profiles")
dfc = pd.DataFrame(profile["clusters"])
st.dataframe(dfc, use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Cluster visualization (PCA)")
    show_image_if_exists(st, figures_dir() / "segmentation" / "pca_clusters.png")
with col2:
    st.subheader("k selection (Elbow + Silhouette)")
    show_image_if_exists(st, figures_dir() / "segmentation" / "k_selection.png")

st.caption("Labels (value/risk) are assigned after profiling, relative to overall "
           "medians. Churn probability shown is model-generated.")
