"""Thin HTTP client for the platform API (used by the dashboard pages).

Decoupled from ``src`` — reads the base URL from the environment so the
dashboard can point at a local or containerised API. Every call degrades
gracefully: on error it returns ``None`` (pages then show a friendly message).
"""
from __future__ import annotations

import os

import requests

API_BASE = os.getenv("DASHBOARD_API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 8


class ApiClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base = (base_url or API_BASE).rstrip("/")

    def _get(self, path: str, params: dict | None = None):
        try:
            r = requests.get(f"{self.base}{path}", params=params, timeout=_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            return {"__error__": r.status_code, "detail": _safe_detail(r)}
        except requests.RequestException as exc:
            return {"__error__": "unreachable", "detail": str(exc)}

    def _post(self, path: str, body: dict):
        try:
            r = requests.post(f"{self.base}{path}", json=body, timeout=_TIMEOUT)
            if r.status_code in (200, 202):
                return r.json()
            return {"__error__": r.status_code, "detail": _safe_detail(r)}
        except requests.RequestException as exc:
            return {"__error__": "unreachable", "detail": str(exc)}

    # endpoints
    def health(self):
        return self._get("/health")

    def dashboard_metrics(self):
        return self._get("/dashboard/metrics")

    def model_metrics(self):
        return self._get("/model/metrics")

    def high_risk(self, min_level="HIGH", limit=100):
        return self._get("/customers/high-risk",
                         {"min_level": min_level, "limit": limit})

    def customer(self, cid: str):
        return self._get(f"/customer/{cid}")

    def explanation(self, cid: str):
        return self._get(f"/customer/{cid}/explanation")

    def recommendation(self, cid: str):
        return self._get(f"/customer/{cid}/recommendation")

    def predict(self, customer_id: str):
        return self._post("/predict", {"customer_id": customer_id})

    def send_event(self, customer_id: str, event_type: str):
        return self._post("/event", {"customer_id": customer_id, "event_type": event_type})

    def drift(self):
        return self._get("/monitoring/drift")


def _safe_detail(resp) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except Exception:
        return resp.text


def is_error(payload) -> bool:
    return not isinstance(payload, (dict, list)) or (
        isinstance(payload, dict) and "__error__" in payload
    )
