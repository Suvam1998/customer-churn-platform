"""Tests for the FastAPI application (Phase 1: root + health)."""
from __future__ import annotations

from src import __version__


def test_root_ok(api_client):
    resp = api_client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == __version__
    assert body["health"] == "/health"


def test_health_ok(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert "timestamp" in body


def test_latency_header_present(api_client):
    resp = api_client.get("/health")
    assert "X-Process-Time-ms" in resp.headers
