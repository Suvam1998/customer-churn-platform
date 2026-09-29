"""SHAP-based explainability for the churn model.

Explanations are computed with SHAP TreeExplainer on the tuned CatBoost model
(the base model behind the production probabilities). Nothing here is
hardcoded: every contribution and factor is produced by SHAP from the fitted
model.

The pipeline is ``Pipeline([("pre", ColumnTransformer), ("clf", CatBoost)])``.
SHAP explains the transformed feature space; feature names come from
``preprocessor.get_feature_names_out()``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from src.config import Config, get_config
from src.features.feature_engineering import (
    FEATURED_FEATURE_COLUMNS,
    engineer_features,
)
from src.ingestion.load_data import load_raw
from src.models.calibration import build_tuned_estimator
from src.models.train import load_splits
from src.preprocessing.preprocess import clean_raw
from src.validation.schema import ID_COLUMN, TARGET_COLUMN

logger = logging.getLogger("churn.explain")


def _prettify(name: str) -> str:
    """Human-friendly label for a transformed feature name."""
    return name.replace("_", " ").replace("  ", " ").strip()


@dataclass
class ChurnExplainer:
    pipeline: object          # fitted Pipeline (pre + clf)
    feature_names: list[str]  # transformed feature names
    explainer: object         # shap.TreeExplainer

    @classmethod
    def from_tuned(cls, cfg: Config | None = None) -> "ChurnExplainer":
        cfg = cfg or get_config()
        _name, pipe = build_tuned_estimator(cfg)
        splits = load_splits(include_engineered=True, cfg=cfg)
        pipe.fit(splits.X_train, splits.y_train)

        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
        feature_names = list(pre.get_feature_names_out())
        explainer = shap.TreeExplainer(clf)
        return cls(pipeline=pipe, feature_names=feature_names, explainer=explainer)

    # --- core SHAP ---
    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.named_steps["pre"].transform(X)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return SHAP values for the positive (churn) class, shape (n, n_feat)."""
        Xt = self._transform(X)
        sv = self.explainer.shap_values(Xt)
        sv = np.asarray(sv)
        if sv.ndim == 3:            # (n, feat, classes)
            sv = sv[:, :, 1]
        if isinstance(sv, list):    # [class0, class1]
            sv = np.asarray(sv[1])
        return sv

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    # --- global ---
    def global_importance(self, X_sample: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        sv = self.shap_values(X_sample)
        mean_abs = np.abs(sv).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": np.round(mean_abs, 5),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return df.head(top_n)

    # --- local ---
    def explain_customer(
        self, x_row: pd.DataFrame, customer_id: str, top_k: int = 5,
    ) -> dict:
        """Explain one customer. All values are SHAP-generated."""
        sv = self.shap_values(x_row)[0]           # (n_feat,)
        prob = float(self.predict_proba(x_row)[0])
        base = self.explainer.expected_value
        base = float(np.ravel(base)[-1]) if np.ndim(base) else float(base)

        contribs = sorted(
            [
                {"feature": self.feature_names[i],
                 "label": _prettify(self.feature_names[i]),
                 "shap": round(float(sv[i]), 5)}
                for i in range(len(sv))
            ],
            key=lambda d: d["shap"],
        )
        positive = [c for c in reversed(contribs) if c["shap"] > 0][:top_k]
        negative = [c for c in contribs if c["shap"] < 0][:top_k]

        return {
            "customer_id": customer_id,
            "churn_probability": round(prob, 4),
            "base_value_logodds": round(base, 4),
            "top_positive_factors": positive,   # push toward churn
            "top_negative_factors": negative,   # protective
            "feature_contributions": {c["feature"]: c["shap"] for c in contribs},
        }


# --- customer row helpers ---

def engineered_dataset(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or get_config()
    return engineer_features(clean_raw(load_raw(cfg=cfg)))


def get_customer_features(
    customer_id: str, df: pd.DataFrame | None = None, cfg: Config | None = None,
) -> tuple[pd.DataFrame, int]:
    """Return (single-row feature frame, actual churn label) for a customer."""
    df = df if df is not None else engineered_dataset(cfg)
    match = df[df[ID_COLUMN] == customer_id]
    if match.empty:
        raise KeyError(f"customer_id {customer_id!r} not in dataset")
    row = match.iloc[[0]]
    return row[FEATURED_FEATURE_COLUMNS], int(row[TARGET_COLUMN].iloc[0])
