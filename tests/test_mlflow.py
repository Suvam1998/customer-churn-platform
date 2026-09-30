"""Phase 19 tests: MLflow logging + promotion gate."""
from __future__ import annotations

from mlops.register_model import passes_gate


def test_promotion_gate_pass():
    ok, _ = passes_gate({"roc_auc": 0.8371, "brier": 0.1372})
    assert ok is True


def test_promotion_gate_fail_low_auc():
    ok, _ = passes_gate({"roc_auc": 0.70, "brier": 0.13})
    assert ok is False


def test_promotion_gate_fail_high_brier():
    ok, _ = passes_gate({"roc_auc": 0.85, "brier": 0.25})
    assert ok is False


def test_log_run_roundtrip(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    from mlflow.tracking import MlflowClient
    from mlops.mlflow_utils import log_metrics_run, setup_mlflow

    setup_mlflow()
    run_id = log_metrics_run(
        run_name="unit-test",
        params={"C": 1.0, "penalty": "l2"},
        metrics={"roc_auc": 0.8, "brier": 0.14},
        tags={"phase": "test"},
        duration_s=1.5,
    )

    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.data.metrics["roc_auc"] == 0.8
    assert run.data.metrics["training_duration_s"] == 1.5
    assert run.data.params["penalty"] == "l2"
    assert run.data.tags["phase"] == "test"
