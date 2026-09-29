"""Phase 11 entrypoint: K-Means customer segmentation.

Chooses k from Elbow + Silhouette, profiles clusters with data-driven
value/risk labels, and saves a PCA visualization + the per-customer segment
assignment.

Usage (repo root, venv active):
    python scripts/run_segmentation.py
Artifacts:
    results/segmentation_profile.json
    data/features/segments.parquet
    docs/figures/segmentation/{k_selection.png, pca_clusters.png}
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

from src.config import get_config  # noqa: E402
from src.segmentation.clustering import (  # noqa: E402
    SEGMENT_FEATURES,
    build_segmentation_frame,
    choose_k,
    fit_segmentation,
    pca_2d,
    profile_clusters,
    suggest_k,
)
from src.validation.schema import ID_COLUMN  # noqa: E402


def _k_selection_figure(k_table, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(k_table["k"], k_table["inertia"], "o-", color="#4C72B0", label="Inertia (elbow)")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Inertia", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(k_table["k"], k_table["silhouette"], "s--", color="#DD8452", label="Silhouette")
    ax2.set_ylabel("Silhouette score", color="#DD8452")
    ax1.set_title("K selection — Elbow (inertia) & Silhouette")
    ax1.grid(True, alpha=0.3)
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _pca_figure(coords, labels, profile, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    label_map = {r["cluster"]: r["label"] for _, r in profile.iterrows()}
    for c in sorted(set(labels)):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5,
                   label=f"Cluster {c}: {label_map.get(c, '')}")
    ax.set_title("Customer segments (PCA projection of segmentation features)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = get_config()
    seed = cfg.get("project.random_seed", 42)
    results_dir = cfg.resolve_path("paths.results")
    features_dir = cfg.resolve_path("paths.data_features")
    fig_dir = cfg.resolve_path("paths.figures") / "segmentation"
    results_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    df = build_segmentation_frame(cfg)

    # k selection.
    from sklearn.preprocessing import StandardScaler

    X = df[SEGMENT_FEATURES].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    k_table = choose_k(X_scaled, seed=seed)
    k = suggest_k(k_table)
    _k_selection_figure(k_table, fig_dir / "k_selection.png")

    # Fit + profile.
    df_c, _scaler, km, X_scaled = fit_segmentation(df, k=k, seed=seed)
    profile = profile_clusters(df_c)
    coords = pca_2d(X_scaled, seed=seed)
    _pca_figure(coords, km.labels_, profile, fig_dir / "pca_clusters.png")

    # Persist per-customer segment assignment.
    label_map = {r["cluster"]: r["label"] for _, r in profile.iterrows()}
    seg = df_c[[ID_COLUMN, "cluster"]].copy()
    seg["segment_label"] = seg["cluster"].map(label_map)
    seg["churn_probability"] = df_c["churn_probability"].round(4)
    seg.to_parquet(features_dir / "segments.parquet", index=False)

    (results_dir / "segmentation_profile.json").write_text(json.dumps({
        "chosen_k": k,
        "k_selection": k_table.to_dict(orient="records"),
        "clusters": profile.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")

    # Report.
    print("=" * 92)
    print("CUSTOMER SEGMENTATION — K-Means")
    print("=" * 92)
    print("k selection (Elbow inertia + Silhouette):")
    print(k_table.to_string(index=False))
    print(f"\nChosen k = {k} (highest silhouette)")
    print("-" * 92)
    print("Cluster profiles (sorted by churn risk):")
    print(profile.to_string(index=False))
    print("-" * 92)
    print(f"Artifacts: results/segmentation_profile.json, data/features/segments.parquet,")
    print(f"           docs/figures/segmentation/{{k_selection,pca_clusters}}.png")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
