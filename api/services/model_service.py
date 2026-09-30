"""Model service layer for the API.

Loads the production model and precomputes a per-customer scored table
(churn probability, value, risk, recommendation) at startup so read endpoints
are fast and available even without a database. The SHAP explainer is built
lazily (only when an explanation/prediction factor set is first requested).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from functools import lru_cache

import joblib
import pandas as pd

from src.config import Config, get_config
from src.explainability.shap_explainer import engineered_dataset
from src.features.feature_engineering import FEATURED_FEATURE_COLUMNS, engineer_features
from src.preprocessing.preprocess import clean_raw
from src.retention.recommendations import make_decision
from src.retention.scoring import compute_value_table, estimate_clv, revenue_at_risk
from src.risk import classify_risk
from src.validation.schema import ID_COLUMN, TARGET_COLUMN

logger = logging.getLogger("churn.service")

_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


class ModelNotReadyError(RuntimeError):
    pass


class ChurnService:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or get_config()
        model_path = self.cfg.resolve_path("paths.models") / "production_model.joblib"
        if not model_path.exists():
            raise ModelNotReadyError(
                "production_model.joblib missing — run Phase 9 (calibrate_model)."
            )
        self.model = joblib.load(model_path)

        meta_path = self.cfg.resolve_path("paths.models") / "production_model_meta.json"
        self.meta = {}
        if meta_path.exists():
            import json
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.model_version = self.meta.get("model_version", "unknown")

        # Precompute scored table + decisions.
        df = engineered_dataset(self.cfg)
        df["churn_probability"] = self.model.predict_proba(df[FEATURED_FEATURE_COLUMNS])[:, 1]
        vt = compute_value_table(df, cfg=self.cfg)
        self._attach_segments(vt)
        self.table = vt.set_index(ID_COLUMN, drop=False)
        self._explainer = None
        self._drift = None

    def _attach_segments(self, vt: pd.DataFrame) -> None:
        seg_path = self.cfg.resolve_path("paths.data_features") / "segments.parquet"
        if seg_path.exists():
            seg = pd.read_parquet(seg_path)[[ID_COLUMN, "segment_label"]]
            merged = vt.merge(seg, on=ID_COLUMN, how="left")
            vt["segment_label"] = merged["segment_label"].values
        else:
            vt["segment_label"] = None

    # ---- lazy SHAP ----
    @property
    def explainer(self):
        if self._explainer is None:
            from src.explainability.shap_explainer import ChurnExplainer
            self._explainer = ChurnExplainer.from_tuned(self.cfg)
        return self._explainer

    # ---- helpers ----
    def _row(self, customer_id: str) -> pd.Series:
        if customer_id not in self.table.index:
            raise KeyError(customer_id)
        return self.table.loc[customer_id]

    def _decision_for(self, row: pd.Series):
        return make_decision(row.to_dict(), cfg=self.cfg)

    def _top_factors(self, feature_row: pd.DataFrame, customer_id: str, k: int = 3) -> list[str]:
        exp = self.explainer.explain_customer(feature_row, customer_id, top_k=k)
        return [f["label"] for f in exp["top_positive_factors"]]

    # ---- public API ----
    def predict_existing(self, customer_id: str) -> dict:
        start = time.perf_counter()
        row = self._row(customer_id)
        feats = self.table.loc[[customer_id], FEATURED_FEATURE_COLUMNS]
        decision = self._decision_for(row)
        top = self._top_factors(feats, customer_id)
        latency = round((time.perf_counter() - start) * 1000, 2)
        return {
            "customer_id": customer_id,
            "churn_probability": round(float(row["churn_probability"]), 4),
            "risk_level": row["risk_level"],
            "estimated_clv": round(float(row["estimated_clv"]), 2),
            "revenue_at_risk": round(float(row["revenue_at_risk"]), 2),
            "top_risk_factors": top,
            "recommended_action": decision.recommended_action,
            "model_version": self.model_version,
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction_latency_ms": latency,
        }

    def predict_raw(self, raw: dict) -> dict:
        start = time.perf_counter()
        cid = raw.get("customerID", "adhoc")
        df = pd.DataFrame([{**raw, "customerID": cid, "Churn": "No"}])
        df = engineer_features(clean_raw(df))
        prob = float(self.model.predict_proba(df[FEATURED_FEATURE_COLUMNS])[:, 1][0])
        clv = estimate_clv(df["MonthlyCharges"].iloc[0],
                           self.cfg.get("clv.expected_lifetime_months", 24),
                           self.cfg.get("clv.gross_margin", 0.30))
        rar = revenue_at_risk(prob, clv)
        level = classify_risk(prob)
        feat = df.iloc[0].to_dict()
        feat.update({"churn_probability": prob, "estimated_clv": clv,
                     "revenue_at_risk": rar, "risk_level": level, "priority_score": rar})
        decision = make_decision(feat, cfg=self.cfg)
        top = self._top_factors(df[FEATURED_FEATURE_COLUMNS], cid)
        latency = round((time.perf_counter() - start) * 1000, 2)
        return {
            "customer_id": cid,
            "churn_probability": round(prob, 4),
            "risk_level": level,
            "estimated_clv": round(clv, 2),
            "revenue_at_risk": round(rar, 2),
            "top_risk_factors": top,
            "recommended_action": decision.recommended_action,
            "model_version": self.model_version,
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction_latency_ms": latency,
        }

    def get_customer(self, customer_id: str) -> dict:
        row = self._row(customer_id)
        return {
            "customer_id": customer_id,
            "tenure": int(row["tenure"]),
            "contract": row["Contract"],
            "payment_method": row["PaymentMethod"],
            "internet_service": row["InternetService"],
            "monthly_charges": round(float(row["MonthlyCharges"]), 2),
            "total_charges": round(float(row["TotalCharges"]), 2),
            "churn_probability": round(float(row["churn_probability"]), 4),
            "risk_level": row["risk_level"],
            "estimated_clv": round(float(row["estimated_clv"]), 2),
            "revenue_at_risk": round(float(row["revenue_at_risk"]), 2),
            "segment": row.get("segment_label"),
            "actual_churn": int(row[TARGET_COLUMN]),
        }

    def explain(self, customer_id: str, k: int = 5) -> dict:
        if customer_id not in self.table.index:
            raise KeyError(customer_id)
        feats = self.table.loc[[customer_id], FEATURED_FEATURE_COLUMNS]
        exp = self.explainer.explain_customer(feats, customer_id, top_k=k)
        return {
            "customer_id": customer_id,
            "churn_probability": exp["churn_probability"],
            "top_positive_factors": [
                {"factor": f["label"], "contribution": f["shap"]}
                for f in exp["top_positive_factors"]],
            "top_negative_factors": [
                {"factor": f["label"], "contribution": f["shap"]}
                for f in exp["top_negative_factors"]],
        }

    def recommend(self, customer_id: str) -> dict:
        row = self._row(customer_id)
        d = self._decision_for(row)
        return {
            "customer_id": customer_id,
            "churn_probability": d.churn_probability,
            "risk_level": d.risk_level,
            "urgency": d.urgency,
            "recommended_action": d.recommended_action,
            "reason": d.reason,
            "revenue_at_risk": d.revenue_at_risk,
            "estimated_intervention_cost": d.estimated_intervention_cost,
            "estimated_expected_retained_value": d.estimated_expected_retained_value,
            "estimated_net_value": d.estimated_net_value,
        }

    def high_risk(self, min_level: str = "HIGH", limit: int = 50) -> list[dict]:
        floor = _RISK_ORDER.get(min_level.upper(), 2)
        mask = self.table["risk_level"].map(_RISK_ORDER).fillna(0) >= floor
        sub = self.table[mask].sort_values("revenue_at_risk", ascending=False).head(limit)
        out = []
        for _, row in sub.iterrows():
            out.append({
                "customer_id": row[ID_COLUMN],
                "churn_probability": round(float(row["churn_probability"]), 4),
                "risk_level": row["risk_level"],
                "revenue_at_risk": round(float(row["revenue_at_risk"]), 2),
                "segment": row.get("segment_label"),
                "recommended_action": self._decision_for(row).recommended_action,
            })
        return out

    def dashboard_metrics(self) -> dict:
        t = self.table
        n = len(t)
        risk_counts = t["risk_level"].value_counts().to_dict()
        high = int((t["risk_level"].map(_RISK_ORDER).fillna(0) >= 2).sum())
        return {
            "total_customers": n,
            "churned_customers": int(t[TARGET_COLUMN].sum()),
            "churn_rate": round(float(t[TARGET_COLUMN].mean()), 4),
            "high_risk_customers": high,
            "total_revenue_at_risk": round(float(t["revenue_at_risk"].sum()), 2),
            "average_churn_probability": round(float(t["churn_probability"].mean()), 4),
            "model_version": self.model_version,
            "risk_level_counts": {k: int(v) for k, v in risk_counts.items()},
        }

    def model_metrics(self) -> dict:
        return {
            "model_version": self.model_version,
            "base_model": self.meta.get("base_model"),
            "calibration": self.meta.get("calibration"),
            "metrics": self.meta.get("val_metrics", {}),
        }

    def drift(self) -> dict:
        """Real feature + prediction drift (reference=train vs current=test)."""
        if self._drift is None:
            from src.models.train import load_splits
            from src.monitoring.monitor import build_report

            splits = load_splits(include_engineered=True, cfg=self.cfg)
            ref = self.model.predict_proba(splits.X_train[FEATURED_FEATURE_COLUMNS])[:, 1]
            cur = self.model.predict_proba(splits.X_test[FEATURED_FEATURE_COLUMNS])[:, 1]
            r = build_report(splits.X_train, splits.X_test, ref, cur,
                             scenario="real:train-vs-test", cfg=self.cfg)
            fd = r["feature_drift"]
            self._drift = {
                "status": r["status"],
                "scenario": r["scenario"],
                "note": r["note"],
                "n_features": fd["n_features"],
                "n_drifted": fd["n_drifted"],
                "share_drifted": fd["share_drifted"],
                "threshold": fd["threshold"],
                "drifted_features": fd["drifted_features"],
                "prediction_drift_psi": r["prediction_drift"]["psi"] if r["prediction_drift"] else None,
            }
        return self._drift


@lru_cache(maxsize=1)
def get_service() -> ChurnService:
    return ChurnService()
