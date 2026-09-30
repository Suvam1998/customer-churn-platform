"""Thin client for the platform API & services (used by the dashboard pages).

Provides a dual-mode client:
1. Direct high-speed in-process service execution (zero-latency, zero-configuration,
   100% standalone & Streamlit Cloud ready).
2. Remote HTTP execution when pointing at a custom external API URL (e.g. Docker / Kubernetes).
Every call degrades gracefully: on error it returns an error dict.
"""
from __future__ import annotations

import logging
import os
import requests

logger = logging.getLogger("churn.dashboard.client")

API_BASE = os.getenv("DASHBOARD_API_BASE_URL", "")
_TIMEOUT = 5


class ApiClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.is_custom_remote = bool(base_url)
        self.base = (base_url or API_BASE).rstrip("/")
        self._service = None
        self._risk_state = None

    def _get_service(self):
        if self._service is None:
            from api.services.model_service import get_service
            self._service = get_service()
        return self._service

    def _get_risk_state(self):
        if self._risk_state is None:
            from src.streaming.risk_update import RiskState
            self._risk_state = RiskState(self._get_service())
        return self._risk_state

    # --------------------------------------------------------------------------
    # Remote HTTP handlers (used when an explicit remote URL is specified)
    # --------------------------------------------------------------------------
    def _http_get(self, path: str, params: dict | None = None):
        try:
            r = requests.get(f"{self.base}{path}", params=params, timeout=_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            return {"__error__": r.status_code, "detail": _safe_detail(r)}
        except requests.RequestException as exc:
            return {"__error__": "unreachable", "detail": str(exc)}

    def _http_post(self, path: str, body: dict):
        try:
            r = requests.post(f"{self.base}{path}", json=body, timeout=_TIMEOUT)
            if r.status_code in (200, 202):
                return r.json()
            return {"__error__": r.status_code, "detail": _safe_detail(r)}
        except requests.RequestException as exc:
            return {"__error__": "unreachable", "detail": str(exc)}

    # --------------------------------------------------------------------------
    # API endpoints (Seamlessly dispatches to direct service or remote HTTP)
    # --------------------------------------------------------------------------
    def health(self):
        if self.is_custom_remote:
            return self._http_get("/health")
        return {"status": "ok", "version": "1.0.0"}

    def dashboard_metrics(self):
        if self.is_custom_remote:
            return self._http_get("/dashboard/metrics")
        try:
            return self._get_service().dashboard_metrics()
        except Exception as exc:
            return {"__error__": "internal_error", "detail": str(exc)}

    def model_metrics(self):
        if self.is_custom_remote:
            return self._http_get("/model/metrics")
        try:
            return self._get_service().model_metrics()
        except Exception as exc:
            return {"__error__": "internal_error", "detail": str(exc)}

    def high_risk(self, min_level: str = "HIGH", limit: int = 100):
        if self.is_custom_remote:
            return self._http_get("/customers/high-risk", {"min_level": min_level, "limit": limit})
        try:
            return self._get_service().high_risk(min_level=min_level, limit=limit)
        except Exception as exc:
            return {"__error__": "internal_error", "detail": str(exc)}

    def customer(self, cid: str):
        if self.is_custom_remote:
            return self._http_get(f"/customer/{cid}")
        try:
            return self._get_service().get_customer(cid)
        except KeyError:
            return {"__error__": 404, "detail": f"customer_id {cid} not found"}
        except Exception as exc:
            return {"__error__": "internal_error", "detail": str(exc)}

    def explanation(self, cid: str):
        if self.is_custom_remote:
            return self._http_get(f"/customer/{cid}/explanation")
        try:
            return self._get_service().explain(cid)
        except KeyError:
            return {"__error__": 404, "detail": f"customer_id {cid} not found"}
        except Exception as exc:
            return {"__error__": "internal_error", "detail": str(exc)}

    def recommendation(self, cid: str):
        if self.is_custom_remote:
            return self._http_get(f"/customer/{cid}/recommendation")
        try:
            return self._get_service().recommend(cid)
        except KeyError:
            return {"__error__": 404, "detail": f"customer_id {cid} not found"}
        except Exception as exc:
            return {"__error__": "internal_error", "detail": str(exc)}

    def predict(self, customer_id: str):
        if self.is_custom_remote:
            return self._http_post("/predict", {"customer_id": customer_id})
        try:
            return self._get_service().predict_existing(customer_id)
        except KeyError:
            return {"__error__": 404, "detail": f"customer_id {customer_id} not found"}
        except Exception as exc:
            return {"__error__": "internal_error", "detail": str(exc)}

    def send_event(self, customer_id: str, event_type: str):
        if self.is_custom_remote:
            return self._http_post("/event", {"customer_id": customer_id, "event_type": event_type})
        try:
            state = self._get_risk_state()
            if customer_id not in state.service.table.index:
                return {"__error__": 404, "detail": f"customer_id {customer_id} not found"}
            upd = state.apply({
                "customer_id": customer_id,
                "event_type": event_type,
                "timestamp": None,
                "value": 1.0,
            })
            return {
                "accepted": True,
                "is_simulated": True,
                "customer_id": upd.customer_id,
                "event_type": upd.event_type,
                "previous_probability": upd.previous_probability,
                "new_probability": upd.new_probability,
                "risk_change": upd.risk_change,
                "risk_level": upd.new_risk_level,
                "recommended_action": upd.recommended_action,
                "note": "SIMULATED event: risk updated in real-time.",
            }
        except Exception as exc:
            return {"__error__": "internal_error", "detail": str(exc)}

    def drift(self):
        if self.is_custom_remote:
            return self._http_get("/monitoring/drift")
        try:
            return self._get_service().drift()
        except Exception as exc:
            return {"__error__": "internal_error", "detail": str(exc)}


def _safe_detail(resp) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except Exception:
        return resp.text


def is_error(payload) -> bool:
    return not isinstance(payload, (dict, list)) or (
        isinstance(payload, dict) and "__error__" in payload
    )
