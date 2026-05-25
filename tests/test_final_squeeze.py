"""Final Squeeze protocol — freeze mode and threshold grid."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.threshold_tuning import search_best_threshold


def test_threshold_grid_030_to_070():
    y = np.array([0, 0, 1, 1, 1, 0])
    probs = np.array([0.2, 0.4, 0.55, 0.62, 0.8, 0.35])
    t, score = search_best_threshold(
        y,
        probs,
        metric="f1_weighted",
        min_threshold=0.30,
        max_threshold=0.70,
        step=0.01,
    )
    assert 0.30 <= t <= 0.70
    assert score >= 0.0


def test_average_state_dicts():
    import torch

    from src.models.transformer_trainer import _average_state_dicts

    a = {"w": torch.tensor([1.0, 3.0])}
    b = {"w": torch.tensor([3.0, 5.0])}
    avg = _average_state_dicts([a, b])
    assert torch.allclose(avg["w"], torch.tensor([2.0, 4.0]))


def test_apply_model_freeze_full_mode():
    pytest.importorskip("transformers")
    from transformers import AutoModelForSequenceClassification

    from src.models.transformer_trainer import _apply_model_freeze

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    )
    mode, freeze_n = _apply_model_freeze(model, {"freeze_mode": "full"})
    assert mode == "full_unfreeze"
    assert freeze_n == 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable == total
