"""Shared pytest fixtures and path setup."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def api_client():
    """A FastAPI TestClient against the app (no network required)."""
    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app)
