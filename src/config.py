"""Central configuration loader.

Loads ``configs/config.yaml`` and ``configs/model_config.yaml`` once, resolves
the project root robustly (never a hardcoded absolute path), and exposes a
cached :class:`Config` object. Environment variables from a ``.env`` file are
loaded so secrets/overrides never live in code.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:  # optional: .env support without a hard dependency at import time
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False


def get_project_root() -> Path:
    """Return the repository root (the dir containing ``configs/``).

    Walks upward from this file so the code works regardless of the current
    working directory or OS.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "configs" / "config.yaml").exists():
            return parent
    # Fallback: two levels up from src/config.py
    return here.parent.parent


class Config:
    """Immutable-ish view over merged YAML config with dotted-key access."""

    def __init__(self, data: dict[str, Any], root: Path) -> None:
        self._data = data
        self.root = root

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a nested value using ``"a.b.c"`` notation."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def resolve_path(self, dotted_key: str) -> Path:
        """Resolve a config path value (relative to project root) to a Path."""
        rel = self.get(dotted_key)
        if rel is None:
            raise KeyError(f"No path configured at '{dotted_key}'")
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    @property
    def data(self) -> dict[str, Any]:
        return self._data


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and cache the merged platform configuration."""
    root = get_project_root()
    load_dotenv(root / ".env")

    data = _read_yaml(root / "configs" / "config.yaml")
    model_cfg = _read_yaml(root / "configs" / "model_config.yaml")
    data["model"] = model_cfg

    # Allow selected env overrides (never secrets in YAML).
    if os.getenv("MLFLOW_TRACKING_URI"):
        data.setdefault("mlflow", {})["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    if os.getenv("STREAMING_MODE"):
        data.setdefault("streaming", {})["mode"] = os.environ["STREAMING_MODE"]

    return Config(data, root)
