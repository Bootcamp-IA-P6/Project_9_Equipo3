"""Tests for expert threshold tuning (no model download)."""

import numpy as np

from src.evaluation.threshold_tuning import predict_with_threshold, search_best_threshold
from src.models.hybrid_ensemble import tune_ensemble_threshold


def test_search_best_threshold_prefers_high_recall_slice():
    y = np.array([0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.55, 0.7, 0.9])
    t, score = search_best_threshold(y, probs, metric="f1_toxic")
    preds = predict_with_threshold(probs, t)
    assert score > 0
    assert 0.05 <= t <= 0.95
    assert preds.shape == y.shape


def test_tune_ensemble_threshold():
    y_val = np.array([0, 1, 1, 0])
    a = np.array([0.2, 0.8, 0.7, 0.3])
    b = np.array([0.3, 0.6, 0.75, 0.25])
    t, _ = tune_ensemble_threshold(a, b, y_val, bert_weight=0.7, lr_weight=0.3)
    assert 0.05 <= t <= 0.95
