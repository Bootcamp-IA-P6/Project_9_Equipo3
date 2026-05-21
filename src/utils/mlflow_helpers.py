"""MLflow experiment logging (AGENTS.md Phase 5)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = "youtube-toxic-detector"


def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
) -> str:
    """Set tracking URI (local ./mlruns by default) and experiment."""
    uri = tracking_uri or str(ROOT / "mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri


@contextmanager
def start_run(
    run_name: str,
    *,
    tags: dict[str, str] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Context manager for an MLflow run."""
    setup_mlflow()
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        yield run


def log_metric_dict(metrics: dict[str, Any], *, prefix: str = "") -> None:
    """Flatten nested metric dicts for MLflow."""
    for key, value in metrics.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            if "train" in value and "test" in value:
                mlflow.log_metric(f"{name}_train", float(value["train"]))
                mlflow.log_metric(f"{name}_test", float(value["test"]))
                if "gap_pct" in value:
                    mlflow.log_metric(f"{name}_gap_pct", float(value["gap_pct"]))
            else:
                log_metric_dict(value, prefix=f"{name}_")
        elif isinstance(value, (int, float)):
            mlflow.log_metric(name, float(value))


def log_overfitting_metrics(metrics: dict[str, Any]) -> None:
    """Log standard train/test/gap metrics from evaluate bundle."""
    log_metric_dict(metrics.get("accuracy", {}), prefix="accuracy_")
    log_metric_dict(metrics.get("f1_toxic", {}), prefix="f1_toxic_")
    mlflow.log_metric("passes_overfitting", float(metrics.get("passes_overfitting", False)))
