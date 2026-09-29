"""Customer segmentation with K-Means.

Segmentation features (behaviour + value + risk):
    tenure, MonthlyCharges, TotalCharges, service_adoption_score,
    churn_probability (from the production model).

The number of clusters is chosen from the data using the Elbow (inertia) and
Silhouette scores — not hardcoded. Cluster labels (High/Low value × high/low
risk) are assigned AFTER profiling, relative to overall medians.

This is unsupervised description of the existing customer base, so scaling is
fit on all customers (no target is involved).
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import Config, get_config
from src.explainability.shap_explainer import engineered_dataset
from src.features.feature_engineering import FEATURED_FEATURE_COLUMNS
from src.validation.schema import ID_COLUMN

logger = logging.getLogger("churn.segmentation")

SEGMENT_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "service_adoption_score",
    "churn_probability",
]


def build_segmentation_frame(cfg: Config | None = None) -> pd.DataFrame:
    """Return per-customer segmentation features incl. model churn probability."""
    cfg = cfg or get_config()
    df = engineered_dataset(cfg)

    model_path = cfg.resolve_path("paths.models") / "production_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run scripts/calibrate_model.py (Phase 9) first."
        )
    model = joblib.load(model_path)
    df = df.copy()
    df["churn_probability"] = model.predict_proba(df[FEATURED_FEATURE_COLUMNS])[:, 1]
    return df


def choose_k(X_scaled: np.ndarray, k_range=range(2, 9), seed: int = 42) -> pd.DataFrame:
    """Compute inertia (elbow) and silhouette for each k. Returns a table."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_scaled)
        rows.append({
            "k": k,
            "inertia": round(float(km.inertia_), 2),
            "silhouette": round(float(silhouette_score(X_scaled, labels)), 4),
        })
    return pd.DataFrame(rows)


def suggest_k(k_table: pd.DataFrame) -> int:
    """Pick k with the highest silhouette score."""
    return int(k_table.loc[k_table["silhouette"].idxmax(), "k"])


def fit_segmentation(df: pd.DataFrame, k: int, seed: int = 42):
    """Fit scaler + KMeans on the segmentation features. Returns
    (df_with_cluster, scaler, kmeans, X_scaled)."""
    X = df[SEGMENT_FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_scaled)
    out = df.copy()
    out["cluster"] = km.labels_
    return out, scaler, km, X_scaled


def profile_clusters(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster profile + data-driven value/risk labels."""
    value_median = df_clustered["MonthlyCharges"].median()
    risk_median = df_clustered["churn_probability"].median()

    rows = []
    for c, g in df_clustered.groupby("cluster"):
        mean_monthly = g["MonthlyCharges"].mean()
        mean_risk = g["churn_probability"].mean()
        value = "High-value" if mean_monthly >= value_median else "Low-value"
        risk = "high-risk" if mean_risk >= risk_median else "low-risk"
        rows.append({
            "cluster": int(c),
            "size": int(len(g)),
            "pct": round(len(g) / len(df_clustered) * 100, 1),
            "avg_tenure": round(float(g["tenure"].mean()), 1),
            "avg_monthly_charges": round(float(mean_monthly), 2),
            "avg_total_charges": round(float(g["TotalCharges"].mean()), 2),
            "avg_service_adoption": round(float(g["service_adoption_score"].mean()), 3),
            "avg_churn_probability": round(float(mean_risk), 4),
            "actual_churn_rate": round(float(g["Churn"].mean()), 4),
            "label": f"{value}/{risk}",
        })
    return pd.DataFrame(rows).sort_values("avg_churn_probability", ascending=False).reset_index(drop=True)


def pca_2d(X_scaled: np.ndarray, seed: int = 42) -> np.ndarray:
    """Project scaled features to 2D for visualization."""
    return PCA(n_components=2, random_state=seed).fit_transform(X_scaled)
