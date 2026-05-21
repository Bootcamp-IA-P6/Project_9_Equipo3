"""Phase 4: Optuna tuning, ensemble, report (uv run python -m src.pipeline.phase4_optimize)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.evaluation.phase1_audit import BINARY_TARGET
from src.models.phase3_baseline import save_model_bundle
from src.models.phase4_ensemble import build_voting_ensemble
from src.models.phase4_tuning import (
    train_mlp_baseline,
    tune_logistic_regression,
    tune_random_forest,
)
from src.utils.execution_log import log_event

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_CSV = ROOT / "data/processed/youtoxic_phase2.csv"
VECTORIZER_PATH = ROOT / "data/processed/vectorizers/tfidf_vectorizer.joblib"
MODEL_PATH = ROOT / "models/phase4_best.joblib"
REPORT_PATH = ROOT / "reports/phase4/phase4_optimization.json"
LOG_PATH = ROOT / "logs/execution.log"
CONSULT_PATH = ROOT / "reports/phase4/OVERFITTING_CONSULT.md"


def _load_xy():
    df = pd.read_csv(PROCESSED_CSV)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    vectorizer = joblib.load(VECTORIZER_PATH)
    train_texts = train_df["ProcessedText"].astype(str).tolist()
    test_texts = test_df["ProcessedText"].astype(str).tolist()
    X_train = vectorizer.transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    y_train = train_df[BINARY_TARGET].values
    y_test = test_df[BINARY_TARGET].values
    return vectorizer, X_train, X_test, y_train, y_test


def run_phase4(*, logreg_trials: int = 60, rf_trials: int = 50) -> dict:
    log_event("Phase 4 optimization started", phase="4", log_path=LOG_PATH)

    vectorizer, X_train, X_test, y_train, y_test = _load_xy()

    candidates = [
        tune_logistic_regression(X_train, X_test, y_train, y_test, n_trials=logreg_trials),
        tune_random_forest(X_train, X_test, y_train, y_test, n_trials=rf_trials),
        train_mlp_baseline(X_train, X_test, y_train, y_test),
    ]

    passing = [c for c in candidates if c["passes_overfitting"]]
    if len(passing) >= 2:
        ensemble = build_voting_ensemble(
            [(c["model_name"], c["classifier"]) for c in passing[:3]],
            X_train,
            X_test,
            y_train,
            y_test,
        )
        candidates.append(ensemble)

    if passing:
        best = max(passing, key=lambda c: c["test_f1_toxic"])
        selection_reason = "best test F1 among models passing 5% overfitting gap"
    else:
        best = min(
            candidates,
            key=lambda c: c["metrics"]["f1_toxic"]["gap_pct"],
        )
        selection_reason = (
            "smallest F1 overfitting gap — **does not meet 5% standard**; user consultation required"
        )
        _write_consult_note(candidates, best)

    bundle = {
        "vectorizer": vectorizer,
        "best": best,
        "all_results": candidates,
        "selection_reason": selection_reason,
        "target": BINARY_TARGET,
    }
    save_model_bundle(
        bundle,
        MODEL_PATH,
    )

    report = {
        "phase": 4,
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "best_model": best["model_name"],
        "selection_reason": selection_reason,
        "passes_overfitting": best["passes_overfitting"],
        "metrics": best["metrics"],
        "overfitting_standard_pct": 5.0,
        "requires_user_consultation": not best["passes_overfitting"],
        "all_models": [
            {
                "name": c["model_name"],
                "test_f1_toxic": c["test_f1_toxic"],
                "passes_overfitting": c["passes_overfitting"],
                "f1_gap_pct": c["metrics"]["f1_toxic"]["gap_pct"],
                "params": c.get("params"),
            }
            for c in candidates
        ],
        "target": BINARY_TARGET,
        "mode": "binary",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log_event(
        f"Phase 4 best={best['model_name']} passes_overfitting={best['passes_overfitting']}",
        phase="4",
        log_path=LOG_PATH,
    )
    return report


def _write_consult_note(candidates: list, best: dict) -> None:
    lines = [
        "# Overfitting standard not met — consultation needed",
        "",
        "AGENTS.md requires `|Train Metric - Test Metric| < 5%` (F1 toxic and accuracy).",
        "No Phase 4 candidate satisfied both gaps on this dataset (~800 train rows).",
        "",
        "## Candidates (F1 toxic gap %)",
        "",
    ]
    for c in sorted(candidates, key=lambda x: x["metrics"]["f1_toxic"]["gap_pct"]):
        m = c["metrics"]["f1_toxic"]
        lines.append(
            f"- **{c['model_name']}**: test F1={m['test']:.4f}, gap={m['gap_pct']:.2f}%"
        )
    lines.extend(
        [
            "",
            f"## Auto-selected fallback (smallest F1 gap): **{best['model_name']}**",
            "",
            "Please choose how to proceed:",
            "1. **Accept** this model for Medio deliverable despite gap > 5%",
            "2. **Relax** monitoring metric (e.g. track gap < 10% until more data)",
            "3. **Invest** in more data / augmentation before deploying",
            "4. **Retry** Phase 4 with different constraints (tell the team)",
            "",
        ]
    )
    CONSULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONSULT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_phase4(), indent=2))
