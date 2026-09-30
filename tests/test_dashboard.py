"""Phase 18 tests: dashboard pages render without error + API client behaviour.

Uses Streamlit's AppTest to execute each page headlessly. Pages degrade
gracefully when the API is down (banner + st.stop), so these pass with or
without a running API.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"
for _p in (str(ROOT), str(DASH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.api_client import ApiClient, is_error  # noqa: E402

PAGES = [
    "dashboard/app.py",
    "dashboard/pages/1_Customer_Risk.py",
    "dashboard/pages/2_Customer_360.py",
    "dashboard/pages/3_Real_Time_Monitor.py",
    "dashboard/pages/4_Segmentation.py",
    "dashboard/pages/5_Model_Performance.py",
    "dashboard/pages/6_Monitoring.py",
]


def test_api_client_unreachable_returns_error():
    client = ApiClient("http://127.0.0.1:59999")  # nothing listening
    res = client.dashboard_metrics()
    assert is_error(res)
    assert res["__error__"] == "unreachable"


def test_is_error_helper():
    assert is_error({"__error__": 404})
    assert not is_error({"total_customers": 10})
    assert not is_error([1, 2, 3])


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / page), default_timeout=60)
    at.run()
    # at.exception is an ElementList (empty when no uncaught exception).
    assert len(at.exception) == 0, f"{page} raised: {list(at.exception)}"
