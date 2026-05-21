"""Train Phase 3 baseline and write reports (uv run python -m src.pipeline.phase3_train_baseline)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.evaluation.phase1_audit import BINARY_TARGET
from src.models.phase3_baseline import save_model_bundle, train_and_compare
from src.utils.execution_log import log_event

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_CSV = ROOT / "data/processed/youtoxic_phase2.csv"
VECTORIZER_PATH = ROOT / "data/processed/vectorizers/tfidf_vectorizer.joblib"
MODEL_PATH = ROOT / "models/phase3_baseline.joblib"
REPORT_PATH = ROOT / "reports/phase3/phase3_training.json"
LOG_PATH = ROOT / "logs/execution.log"


def train_phase3() -> dict:
    log_event("Phase 3 training started", phase="3", log_path=LOG_PATH)

    df = pd.read_csv(PROCESSED_CSV)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_texts = train_df["ProcessedText"].astype(str).tolist()
    test_texts = test_df["ProcessedText"].astype(str).tolist()
    y_train = train_df[BINARY_TARGET].values
    y_test = test_df[BINARY_TARGET].values

    vectorizer = joblib.load(VECTORIZER_PATH)
    bundle = train_and_compare(
        train_texts, test_texts, y_train, y_test, vectorizer=vectorizer
    )
    save_model_bundle(bundle, MODEL_PATH)

    report = {
        "phase": 3,
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "best_model": bundle["best"]["model_name"],
        "selection_reason": bundle["selection_reason"],
        "passes_overfitting": bundle["best"]["passes_overfitting"],
        "metrics": bundle["best"]["metrics"],
        "all_models": [
            {
                "name": r["model_name"],
                "test_f1_toxic": r["test_f1_toxic"],
                "passes_overfitting": r["passes_overfitting"],
                "metrics": r["metrics"],
            }
            for r in bundle["all_results"]
        ],
        "target": BINARY_TARGET,
        "mode": "binary",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log_event(
        f"Phase 3 model saved: {bundle['best']['model_name']} -> {MODEL_PATH}",
        phase="3",
        log_path=LOG_PATH,
    )
    log_event("Phase 3 training completed", phase="3", log_path=LOG_PATH)
    return report


if __name__ == "__main__":
    print(json.dumps(train_phase3(), indent=2))
