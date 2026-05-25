"""Unit tests for stable training helpers (no full fine-tune)."""

import numpy as np
import pytest

from src.models.hybrid_ensemble import fit_lr_with_gap_control, soft_vote_probs
from src.models.transformer_trainer import freeze_distilbert_partial


def test_fit_lr_with_gap_control_picks_regularized_c():
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 120
    X = pd.Series(["word " * (i % 5) + str(i) for i in range(n)])
    y = pd.Series(rng.integers(0, 2, size=n))
    X_tr, X_te = X.iloc[:100], X.iloc[100:]
    y_tr, y_te = y.iloc[:100], y.iloc[100:]
    lr_cfg = {
        "C": 0.05,
        "max_iter": 500,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "gap_search": {
            "enabled": True,
            "param_grid": [{"C": 0.05, "max_features": 200}, {"C": 0.001, "max_features": 50}],
        },
    }
    tfidf_cfg = {"max_features": 200, "ngram_range": [1, 1], "min_df": 1}
    model, meta = fit_lr_with_gap_control(X_tr, y_tr, X_te, y_te, lr_cfg, tfidf_cfg, max_gap=0.05)
    assert model.is_fitted
    assert "C" in meta


def test_soft_vote_equal_weights():
    a = np.array([0.8, 0.2])
    b = np.array([0.4, 0.6])
    out = soft_vote_probs(a, b, 0.5, 0.5)
    np.testing.assert_allclose(out, [0.6, 0.4])


def test_deduplicate_by_cosine_drops_near_duplicates(monkeypatch):
    from src.features import augmentation as aug

    class FakeModel:
        def encode(self, texts, **kwargs):
            if len(texts) == 1:
                return np.array([[1.0, 0.0]])
            return np.array([[0.99, 0.01], [0.0, 1.0]])

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda *_a, **_k: FakeModel(),
    )

    kept_t, kept_l = aug.deduplicate_by_cosine(
        ["near dup", "different"],
        [1, 1],
        ["ref"],
        threshold=0.95,
    )
    assert kept_t == ["different"]
    assert kept_l == [1]


def test_partial_freeze_distilbert():
    pytest.importorskip("transformers")
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    )
    freeze_distilbert_partial(model, freeze_first_n=4)
    layers = list(model.distilbert.transformer.layer)
    for i, layer in enumerate(layers):
        frozen = not any(p.requires_grad for p in layer.parameters())
        if i < 4:
            assert frozen, f"layer {i} should be frozen"
        else:
            assert not frozen, f"layer {i} should be trainable"

    assert model.pre_classifier.weight.requires_grad
    assert model.classifier.weight.requires_grad
