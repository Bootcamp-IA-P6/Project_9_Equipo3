from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def metrics_gap(train_metrics: dict[str, float], test_metrics: dict[str, float]) -> dict[str, float]:
    keys = set(train_metrics) & set(test_metrics)
    return {key: abs(train_metrics[key] - test_metrics[key]) for key in keys}


def confusion_matrix_list(y_true, y_pred) -> list[list[int]]:
    matrix = confusion_matrix(y_true, y_pred)
    return matrix.tolist()


def passes_gap_constraint(
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    max_gap: float,
    primary: str = "f1",
) -> bool:
    gap = metrics_gap(train_metrics, test_metrics)[primary]
    return gap < max_gap
