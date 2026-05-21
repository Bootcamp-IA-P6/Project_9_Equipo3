"""Plan A training with MLflow (uv run python -m src.pipeline.plan_a_train)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow

from src.evaluation.phase1_audit import BINARY_TARGET
from src.features.plan_a_dataset import (
    build_feature_matrices,
    load_raw_split,
    maybe_augment_train,
)
from src.models.plan_a_tuning import select_best, tune_plan_a
from src.utils.execution_log import log_event
from src.utils.mlflow_helpers import log_overfitting_metrics, setup_mlflow, start_run

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/youtoxic_english_1000.csv"
MODEL_PATH = ROOT / "models/plan_a_best.joblib"
REPORT_PATH = ROOT / "reports/phase5/plan_a_report.json"
LOG_PATH = ROOT / "logs/execution.log"


def save_plan_a_bundle(
    best: dict,
    matrices: dict,
    path: Path,
    selection_reason: str,
) -> None:
    payload = {
        "bundle_type": "plan_a_hybrid",
        "vectorizer": matrices["vectorizer"],
        "scaler": matrices["scaler"],
        "classifier": best["classifier"],
        "model_name": best["model_name"],
        "metrics": best["metrics"],
        "params": best.get("params", {}),
        "selection_reason": selection_reason,
        "target": BINARY_TARGET,
        "mode": "binary",
        "version": "plan-a",
        "borderline_low": 0.4,
        "borderline_high": 0.6,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def run_plan_a(*, augment: bool = True, n_trials: int = 80) -> dict:
    log_event("Plan A training started", phase="5", log_path=LOG_PATH)
    setup_mlflow()

    train_df, test_df, label_cols = load_raw_split(str(RAW_PATH))
    if augment:
        train_df = maybe_augment_train(train_df, label_cols, multiplier=1, enabled=True)

    matrices = build_feature_matrices(train_df, test_df)
    candidates = tune_plan_a(
        matrices["X_train"],
        matrices["X_test"],
        matrices["y_train"],
        matrices["y_test"],
        n_trials=n_trials,
    )
    best, reason = select_best(candidates)

    with start_run("plan_a_hybrid", tags={"plan": "A", "augment": str(augment)}):
        mlflow.log_param("augmentation", augment)
        mlflow.log_param("n_train", matrices["n_train"])
        mlflow.log_param("n_test", matrices["n_test"])
        mlflow.log_param("model_name", best["model_name"])
        mlflow.log_params({f"clf_{k}": v for k, v in best.get("params", {}).items()})
        log_overfitting_metrics(best["metrics"])
        mlflow.log_metric("test_accuracy", best["test_accuracy"])
        mlflow.log_metric("test_f1_toxic", best["test_f1_toxic"])
        save_plan_a_bundle(best, matrices, MODEL_PATH, reason)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")

    report = {
        "plan": "A",
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "best_model": best["model_name"],
        "selection_reason": reason,
        "passes_overfitting": best["passes_overfitting"],
        "metrics": best["metrics"],
        "augmentation": augment,
        "n_train": matrices["n_train"],
        "n_test": matrices["n_test"],
        "all_models": [
            {
                "name": c["model_name"],
                "test_accuracy": c["test_accuracy"],
                "test_f1_toxic": c["test_f1_toxic"],
                "passes_overfitting": c["passes_overfitting"],
            }
            for c in candidates
        ],
        "mlflow_experiment": "youtube-toxic-detector",
        "mlflow_run": "plan_a_hybrid",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_event(f"Plan A done: {best['model_name']} acc={best['test_accuracy']:.3f}", phase="5", log_path=LOG_PATH)
    return report


if __name__ == "__main__":
    print(json.dumps(run_plan_a(), indent=2))
