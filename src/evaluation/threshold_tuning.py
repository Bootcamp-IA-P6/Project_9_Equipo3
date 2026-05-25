"""
Validation-set threshold search to maximize F1 (default: toxic class).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def search_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    metric: str = "f1_toxic",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.01,
) -> tuple[float, float]:
    """
    Grid-search classification threshold on validation probabilities.

    Returns (best_threshold, best_score).
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=float)
    thresholds = np.arange(min_threshold, max_threshold + step / 2, step)

    best_t = 0.5
    best_score = -1.0
    average = "weighted" if metric == "f1_weighted" else "binary"
    pos_label = 1

    for t in thresholds:
        preds = (p >= t).astype(int)
        if metric in ("f1_toxic", "f1_binary"):
            score = float(f1_score(y, preds, pos_label=pos_label, zero_division=0))
        else:
            score = float(
                f1_score(y, preds, average=average, zero_division=0)
            )
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, best_score


def predict_with_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(probs, dtype=float) >= threshold).astype(int)
