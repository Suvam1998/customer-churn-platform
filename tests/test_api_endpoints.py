"""Phase 15 tests: full API endpoints against the real production model.

These require the production model artifact (Phase 9). Guarded to skip if
absent so the suite still passes on a fresh clone.
"""
from __future__ import annotations

import pytest

from src.config import get_config


def _model_ready() -> bool:
    return (get_config().resolve_path("paths.models") / "production_model.joblib").exists()


pytestmark = pytest.mark.skipif(not _model_ready(), reason="run Phase 9 first")

KNOWN_ID = "7590-VHVEG"


def test_dashboard_metrics(api_client):
    r = api_client.get("/dashboard/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["total_customers"] == 7043
    assert 0 < body["churn_rate"] < 1
    assert "risk_level_counts" in body


def test_model_metrics(api_client):
    r = api_client.get("/model/metrics")
    assert r.status_code == 200
    assert r.json()["model_version"]


def test_predict_existing_customer(api_client):
    r = api_client.post("/predict", json={"customer_id": KNOWN_ID})
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == KNOWN_ID
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert len(body["top_risk_factors"]) >= 1
    assert body["prediction_latency_ms"] >= 0


def test_predict_unknown_customer_404(api_client):
    r = api_client.post("/predict", json={"customer_id": "NOPE-000"})
    assert r.status_code == 404


def test_predict_requires_input(api_client):
    r = api_client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_raw_customer(api_client):
    r = api_client.post("/predict", json={"customer": {
        "tenure": 1, "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 90.0,
        "TotalCharges": 90.0, "InternetService": "Fiber optic",
        "TechSupport": "No", "OnlineSecurity": "No",
    }})
    assert r.status_code == 200
    assert r.json()["risk_level"] in {"MEDIUM", "HIGH", "CRITICAL", "LOW"}


def test_get_customer(api_client):
    r = api_client.get(f"/customer/{KNOWN_ID}")
    assert r.status_code == 200
    assert r.json()["contract"]


def test_customer_explanation(api_client):
    r = api_client.get(f"/customer/{KNOWN_ID}/explanation")
    assert r.status_code == 200
    body = r.json()
    assert len(body["top_positive_factors"]) >= 1
    assert all("contribution" in f for f in body["top_positive_factors"])


def test_customer_recommendation(api_client):
    r = api_client.get(f"/customer/{KNOWN_ID}/recommendation")
    assert r.status_code == 200
    assert r.json()["recommended_action"]


def test_high_risk_list(api_client):
    r = api_client.get("/customers/high-risk?min_level=HIGH&limit=10")
    assert r.status_code == 200
    items = r.json()
    assert len(items) <= 10
    assert all(i["risk_level"] in {"HIGH", "CRITICAL"} for i in items)


def test_event_increases_risk(api_client):
    r = api_client.post("/event", json={"customer_id": KNOWN_ID,
                                        "event_type": "payment_failed"})
    assert r.status_code == 200
    body = r.json()
    assert body["is_simulated"] is True
    assert body["new_probability"] >= body["previous_probability"]


def test_event_invalid_type_422(api_client):
    r = api_client.post("/event", json={"customer_id": KNOWN_ID,
                                        "event_type": "not_a_real_event"})
    assert r.status_code == 422


def test_drift_and_retrain_stubs(api_client):
    assert api_client.get("/monitoring/drift").status_code == 200
    assert api_client.post("/retrain").status_code == 202
