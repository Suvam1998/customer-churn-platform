"""Shared dashboard helpers: project paths, the simulated-events disclaimer,
artifact loaders, and a cached ApiClient.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


def project_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent and not (p / "configs" / "config.yaml").exists():
        p = p.parent
    return p


ROOT = project_root()


def bootstrap_path() -> None:
    """Ensure project root and dashboard dir are importable from any page."""
    dash = ROOT / "dashboard"
    for p in (str(ROOT), str(dash)):
        if p not in sys.path:
            sys.path.insert(0, p)


def figures_dir() -> Path:
    return ROOT / "docs" / "figures"


def results_dir() -> Path:
    return ROOT / "results"


def features_dir() -> Path:
    return ROOT / "data" / "features"


def stream_dir() -> Path:
    return ROOT / "data" / "stream"


@lru_cache(maxsize=1)
def get_client():
    from lib.api_client import ApiClient
    return ApiClient()


def disclaimer(st) -> None:
    st.caption(
        "🔵 Customer records & historical churn labels are from the **real IBM "
        "Telco** dataset. Real-time events shown here are **SIMULATED** (the "
        "source dataset is historical)."
    )


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def show_image_if_exists(st, path: Path, caption: str = "") -> bool:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
        return True
    st.info(f"Figure not found: `{path.name}` — run the relevant phase script.")
    return False


def api_down_banner(st, payload) -> None:
    detail = payload.get("detail", "") if isinstance(payload, dict) else ""
    st.error(
        "API unreachable or returned an error. Start it with "
        "`uvicorn api.main:app --reload`.\n\n"
        f"Detail: {detail}"
    )
