"""Golden Baseline — R-Drop loss and frozen inference mode."""

from __future__ import annotations

import torch

from src.models.transformer_trainer import _symmetric_kl


def test_symmetric_kl_zero_for_identical_logits():
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.5]])
    kl = _symmetric_kl(logits, logits)
    assert kl.item() < 1e-5
