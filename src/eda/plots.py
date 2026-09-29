"""EDA figures. Every figure has a title, axis labels, and (where relevant)
a legend. Saved as PNG under the configured ``docs/figures`` directory.

Uses the non-interactive Agg backend so figures render headlessly (CI/Docker).
A small, consistent, colour-blind-friendly palette is used across all charts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Config, get_config  # noqa: E402
from src.eda import analysis  # noqa: E402
from src.validation.schema import TARGET_COLUMN  # noqa: E402

# Consistent palette: retained (blue-grey) vs churned (orange), plus a sequential.
COLOR_RETAINED = "#4C72B0"
COLOR_CHURNED = "#DD8452"
COLOR_ACCENT = "#55A868"
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def _figures_dir(cfg: Config) -> Path:
    d = cfg.resolve_path("paths.figures")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    return path


def _churn_labels(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COLUMN].map({0: "Retained", 1: "Churned"})


# --- individual figure builders ------------------------------------------

def plot_churn_distribution(df: pd.DataFrame, out: Path) -> Path:
    ov = analysis.overall_churn(df)
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Retained", "Churned"]
    counts = [ov["n_retained"], ov["n_churned"]]
    bars = ax.bar(labels, counts, color=[COLOR_RETAINED, COLOR_CHURNED])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({c/ov['n_customers']:.1%})",
                ha="center", va="bottom")
    ax.set_title(f"Churn Distribution (overall rate = {ov['churn_rate']:.1%})")
    ax.set_xlabel("Customer outcome")
    ax.set_ylabel("Number of customers")
    ax.set_ylim(0, max(counts) * 1.15)
    return _save(fig, out)


def _numeric_by_churn(df: pd.DataFrame, col: str, out: Path, kind: str = "hist") -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    retained = df.loc[df[TARGET_COLUMN] == 0, col].dropna()
    churned = df.loc[df[TARGET_COLUMN] == 1, col].dropna()
    if kind == "hist":
        bins = 30
        ax.hist(retained, bins=bins, alpha=0.6, label="Retained", color=COLOR_RETAINED, density=True)
        ax.hist(churned, bins=bins, alpha=0.6, label="Churned", color=COLOR_CHURNED, density=True)
        ax.set_ylabel("Density")
        ax.legend(title="Outcome")
    else:  # box
        ax.boxplot([retained, churned], tick_labels=["Retained", "Churned"],
                   patch_artist=True,
                   boxprops=dict(facecolor=COLOR_RETAINED, alpha=0.6))
        ax.set_ylabel(col)
    ax.set_title(f"{col} by churn outcome")
    ax.set_xlabel(col if kind == "hist" else "Customer outcome")
    return _save(fig, out)


def plot_churn_rate_by(df: pd.DataFrame, column: str, out: Path,
                       table: pd.DataFrame | None = None) -> Path:
    tbl = table if table is not None else analysis.churn_rate_by_category(df, column)
    overall = df[TARGET_COLUMN].mean()
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(tbl)), 4))
    cats = tbl[column].astype(str).tolist()
    rates = tbl["churn_rate"].tolist()
    bars = ax.bar(cats, rates, color=COLOR_ACCENT)
    for b, r, n in zip(bars, rates, tbl["n_customers"]):
        ax.text(b.get_x() + b.get_width() / 2, r, f"{r:.1%}\n(n={int(n)})",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(overall, color=COLOR_CHURNED, linestyle="--",
               label=f"Overall = {overall:.1%}")
    ax.set_title(f"Churn rate by {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, min(1.0, max(rates) * 1.25))
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, out)


def plot_correlation_matrix(df: pd.DataFrame, out: Path) -> Path:
    corr = analysis.correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=8)
    ax.set_title("Correlation matrix (numeric features + churn)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    return _save(fig, out)


# --- orchestrator ---------------------------------------------------------

def generate_all(df: pd.DataFrame, cfg: Config | None = None) -> list[Path]:
    """Generate the full EDA figure set. Returns the list of saved paths."""
    cfg = cfg or get_config()
    d = _figures_dir(cfg)
    saved: list[Path] = []

    saved.append(plot_churn_distribution(df, d / "churn_distribution.png"))
    saved.append(_numeric_by_churn(df, "tenure", d / "tenure_distribution.png", "hist"))
    saved.append(_numeric_by_churn(df, "MonthlyCharges", d / "monthly_charges_distribution.png", "hist"))
    saved.append(_numeric_by_churn(df, "TotalCharges", d / "total_charges_box.png", "box"))

    for col, fname in [
        ("Contract", "churn_by_contract.png"),
        ("PaymentMethod", "churn_by_payment_method.png"),
        ("InternetService", "churn_by_internet_service.png"),
        ("TechSupport", "churn_by_tech_support.png"),
        ("OnlineSecurity", "churn_by_online_security.png"),
    ]:
        saved.append(plot_churn_rate_by(df, col, d / fname))

    saved.append(plot_churn_rate_by(
        df, "tenure_band", d / "churn_by_tenure_band.png",
        table=analysis.churn_rate_by_tenure_band(df).rename(columns={"tenure_band": "tenure_band"}),
    ))
    saved.append(plot_churn_rate_by(
        df, "charge_band", d / "churn_by_monthly_charge_band.png",
        table=analysis.churn_rate_by_monthly_charge_band(df),
    ))
    saved.append(plot_correlation_matrix(df, d / "correlation_matrix.png"))
    return saved
