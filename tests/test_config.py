"""Tests for the configuration loader."""
from __future__ import annotations

from pathlib import Path

from src.config import get_config, get_project_root


def test_project_root_contains_configs():
    root = get_project_root()
    assert (root / "configs" / "config.yaml").exists()


def test_config_loads_core_keys():
    cfg = get_config()
    assert cfg.get("project.name")
    assert cfg.get("dataset.target_column") == "Churn"
    assert cfg.get("split.train_size") == 0.70


def test_risk_thresholds_are_ordered():
    t = get_config().get("risk_thresholds")
    assert t["low"] < t["medium"] < t["high"] < t["critical"]


def test_resolve_path_returns_absolute():
    p = get_config().resolve_path("paths.data_raw")
    assert isinstance(p, Path)
    assert p.is_absolute()


def test_model_config_merged():
    cfg = get_config()
    assert cfg.get("model.common.random_seed") == 42
    assert cfg.get("model.models.xgboost.enabled") is True
