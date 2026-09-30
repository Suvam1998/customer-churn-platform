"""MLflow setup + logging helpers.

Registry features require a database-backed store, so when the configured URI
is a file store (or unset) we upgrade to a local SQLite DB. Artifacts land under
``mlartifacts/``.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager

import mlflow

from src.config import Config, get_config

logger = logging.getLogger("churn.mlflow")


def get_tracking_uri(cfg: Config | None = None) -> str:
    cfg = cfg or get_config()
    uri = os.getenv("MLFLOW_TRACKING_URI") or cfg.get("mlflow.tracking_uri")
    # Model Registry needs a DB backend; upgrade file/empty URIs to SQLite.
    if not uri or uri.startswith("file:"):
        uri = f"sqlite:///{(cfg.root / 'mlflow.db').as_posix()}"
    return uri


def setup_mlflow(cfg: Config | None = None) -> str:
    """Configure tracking URI + experiment. Returns the experiment id."""
    cfg = cfg or get_config()
    mlflow.set_tracking_uri(get_tracking_uri(cfg))
    name = cfg.get("mlflow.experiment_name", "churn-retention")

    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        artifact_loc = (cfg.root / "mlartifacts").as_uri()
        exp_id = mlflow.create_experiment(name, artifact_location=artifact_loc)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(name)
    logger.info("mlflow tracking=%s experiment=%s", mlflow.get_tracking_uri(), name)
    return exp_id


def dataset_version(cfg: Config | None = None) -> str:
    """A simple, reproducible dataset version tag from the real data row count."""
    from src.ingestion.load_data import load_raw

    cfg = cfg or get_config()
    try:
        n = len(load_raw(cfg=cfg))
    except Exception:
        n = -1
    return f"ibm-telco-{n}"


@contextmanager
def start_run(run_name: str, tags: dict | None = None):
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_metrics_run(
    run_name: str,
    params: dict,
    metrics: dict,
    tags: dict | None = None,
    duration_s: float | None = None,
) -> str:
    """Log a single run (params + metrics + tags). Returns the run id."""
    with start_run(run_name, tags) as run:
        # MLflow params must be scalar-ish; stringify complex values.
        mlflow.log_params({k: _scalar(v) for k, v in params.items()})
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))
        if duration_s is not None:
            mlflow.log_metric("training_duration_s", float(duration_s))
        return run.info.run_id


def _scalar(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
