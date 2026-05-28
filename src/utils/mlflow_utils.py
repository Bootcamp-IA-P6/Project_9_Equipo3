"""
Shared MLflow helpers for local experiment tracking.

This module is intentionally additive and lightweight so existing notebook/script
MLflow code can continue working unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow


def configure_local_tracking(
    project_root: Path | str,
    experiment_name: str,
    tracking_subdir: str = "mlruns",
) -> Path:
    """
    Configure MLflow to use a local file-based tracking directory.
    """
    root = Path(project_root).resolve()
    tracking_dir = root / tracking_subdir
    tracking_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    mlflow.set_experiment(experiment_name)
    return tracking_dir


@contextmanager
def start_run_context(run_name: str) -> Iterator[Any]:
    """
    Thin context wrapper around mlflow.start_run for consistency.
    """
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_core_params(params: dict[str, Any]) -> None:
    """
    Log non-null parameters safely.
    """
    for key, value in params.items():
        if value is None:
            continue
        mlflow.log_param(str(key), value)


def log_core_metrics(metrics: dict[str, Any]) -> None:
    """
    Log numeric metrics safely (casts to float when possible).
    """
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            mlflow.log_metric(str(key), float(value))
        except (TypeError, ValueError):
            continue
