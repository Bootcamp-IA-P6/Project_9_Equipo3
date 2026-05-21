"""Phase 4 voting ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.overfitting_check import format_gap_report


def build_voting_ensemble(
    estimators: list[tuple[str, Any]],
    X_train,
    X_test,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Soft-voting ensemble over tuned base models."""
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X_train, y_train)
    train_pred = ensemble.predict(X_train)
    test_pred = ensemble.predict(X_test)

    acc = format_gap_report(
        "accuracy",
        accuracy_score(y_train, train_pred),
        accuracy_score(y_test, test_pred),
    )
    f1 = format_gap_report(
        "f1_toxic",
        f1_score(y_train, train_pred, pos_label=1, zero_division=0),
        f1_score(y_test, test_pred, pos_label=1, zero_division=0),
    )
    passes = acc["passes"] and f1["passes"]
    return {
        "model_name": "voting_ensemble",
        "classifier": ensemble,
        "metrics": {"accuracy": acc, "f1_toxic": f1, "passes_overfitting": passes},
        "passes_overfitting": passes,
        "test_f1_toxic": f1["test"],
        "estimators": [name for name, _ in estimators],
    }
