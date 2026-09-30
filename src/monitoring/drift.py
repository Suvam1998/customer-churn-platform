"""Data & prediction drift monitoring (custom PSI metrics).

Uses the Population Stability Index (PSI) — a standard, dependency-light drift
metric — for numeric and categorical features and for the prediction
distribution. Evidently can be layered on top optionally (``evidently_report``),
but PSI is the reliable core so monitoring never depends on a heavy optional
library.

Thresholds are configurable (``configs/config.yaml``). If drift exceeds the
threshold the report status becomes ``RETRAINING_REQUIRED`` — a signal only; no
model is auto-deployed (that requires validation, Phase 42/43).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import Config, get_config

_EPS = 1e-6


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI for a numeric feature using reference quantile bin edges."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return 0.0
    # Quantile edges from the reference; dedupe for constant/low-cardinality.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([ref.min() - 1, ref.mean(), ref.max() + 1])
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_pct = np.clip(ref_pct, _EPS, None)
    cur_pct = np.clip(cur_pct, _EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI for a categorical feature using category frequencies."""
    ref = reference.astype(str)
    cur = current.astype(str)
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    ref_pct = ref.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    cur_pct = cur.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    ref_pct = np.clip(ref_pct, _EPS, None)
    cur_pct = np.clip(cur_pct, _EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify_psi(psi: float) -> str:
    """Standard PSI bands: <0.1 stable, 0.1-0.25 moderate, >0.25 significant."""
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    threshold: float | None = None,
    cfg: Config | None = None,
) -> dict:
    """Per-feature PSI + drifted flags and an aggregate share."""
    cfg = cfg or get_config()
    threshold = threshold if threshold is not None else cfg.get("monitoring.drift_threshold", 0.15)

    per_feature = {}
    for col in numeric_features:
        if col in reference.columns and col in current.columns:
            p = psi_numeric(reference[col], current[col])
            per_feature[col] = {"psi": round(p, 4), "type": "numeric",
                                "band": classify_psi(p), "drifted": p > threshold}
    for col in categorical_features:
        if col in reference.columns and col in current.columns:
            p = psi_categorical(reference[col], current[col])
            per_feature[col] = {"psi": round(p, 4), "type": "categorical",
                                "band": classify_psi(p), "drifted": p > threshold}

    n = len(per_feature)
    drifted = [c for c, v in per_feature.items() if v["drifted"]]
    return {
        "threshold": threshold,
        "n_features": n,
        "n_drifted": len(drifted),
        "share_drifted": round(len(drifted) / n, 3) if n else 0.0,
        "drifted_features": sorted(drifted, key=lambda c: -per_feature[c]["psi"]),
        "per_feature": per_feature,
    }


def prediction_drift(reference_probs, current_probs, threshold: float | None = None,
                     cfg: Config | None = None) -> dict:
    cfg = cfg or get_config()
    threshold = threshold if threshold is not None else cfg.get("monitoring.drift_threshold", 0.15)
    psi = psi_numeric(pd.Series(reference_probs), pd.Series(current_probs))
    return {"psi": round(psi, 4), "band": classify_psi(psi),
            "drifted": psi > threshold, "threshold": threshold}


def missing_summary(df: pd.DataFrame) -> dict:
    na = df.isna().sum()
    return {c: int(na[c]) for c in df.columns if na[c] > 0}


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    reference_probs=None,
    current_probs=None,
    cfg: Config | None = None,
    scenario: str = "reference-vs-current",
) -> dict:
    """Assemble a full drift report with an overall status."""
    cfg = cfg or get_config()
    fd = feature_drift(reference, current, numeric_features, categorical_features, cfg=cfg)
    pd_ = (prediction_drift(reference_probs, current_probs, cfg=cfg)
           if reference_probs is not None and current_probs is not None else None)

    max_share = cfg.get("monitoring.max_drifted_share", 0.30)
    retrain = fd["share_drifted"] > max_share or (pd_ is not None and pd_["drifted"])
    status = "RETRAINING_REQUIRED" if retrain else "STABLE"

    return {
        "scenario": scenario,
        "status": status,
        "note": ("Drift exceeds threshold — retraining is RECOMMENDED. No model "
                 "is auto-deployed; promotion requires validation (Phase 42/43)."
                 if retrain else "Drift within configured thresholds."),
        "feature_drift": fd,
        "prediction_drift": pd_,
        "missing_values": missing_summary(current),
    }


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame):  # pragma: no cover
    """Optional Evidently report (returns None if Evidently is unavailable)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        rep = Report(metrics=[DataDriftPreset()])
        return rep.run(reference_data=reference, current_data=current)
    except Exception:
        return None
