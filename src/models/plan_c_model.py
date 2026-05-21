"""DistilBERT builder with layer freezing and head regularization (Plan C v2)."""

from __future__ import annotations

import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DISTILBERT_NUM_LAYERS = 6


def build_distilbert_classifier(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    num_labels: int = 2,
    frozen_layers: int = 4,
    head_dropout: float = 0.4,
    freeze_embeddings: bool | None = None,
) -> AutoModelForSequenceClassification:
    """
    Load DistilBERT for sequence classification with regularization.

    Parameters
    ----------
    frozen_layers:
        Number of transformer layers to freeze from the bottom (0–6).
        6 freezes the full backbone (head-only training).
    head_dropout:
        Dropout on the classification path (`seq_classif_dropout` and `dropout`).
    freeze_embeddings:
        If True, freeze embedding layer. Defaults to True when ``frozen_layers >= 6``.
    """
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    config.dropout = head_dropout
    config.seq_classif_dropout = head_dropout

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    apply_layer_freeze(model, frozen_layers=frozen_layers, freeze_embeddings=freeze_embeddings)
    return model


def apply_layer_freeze(
    model: AutoModelForSequenceClassification,
    *,
    frozen_layers: int,
    freeze_embeddings: bool | None = None,
) -> None:
    """Freeze bottom ``frozen_layers`` transformer blocks (and optionally embeddings)."""
    frozen_layers = max(0, min(frozen_layers, DISTILBERT_NUM_LAYERS))
    if freeze_embeddings is None:
        freeze_embeddings = frozen_layers >= DISTILBERT_NUM_LAYERS

    for param in model.parameters():
        param.requires_grad = True

    for i, layer in enumerate(model.distilbert.transformer.layer):
        if i < frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

    if freeze_embeddings:
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False

    if frozen_layers >= DISTILBERT_NUM_LAYERS:
        for param in model.distilbert.parameters():
            param.requires_grad = False
        for param in model.pre_classifier.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
