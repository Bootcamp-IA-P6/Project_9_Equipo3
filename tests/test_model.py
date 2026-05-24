"""Tests de salida binaria de modelos sklearn."""

import numpy as np
import pytest

from src.models.baseline import build_model

X_TRAIN = [
    "the quick brown fox is nice",
    "the lazy dog sleeps well",
    "the fox and dog are friends",
    "another calm peaceful day today",
    "you are stupid and worthless idiot",
    "kill them all right now attack",
]
Y_TRAIN = [0, 0, 0, 0, 1, 1]
X_PRED = ["the fox is calm", "you idiot fool"]


@pytest.fixture(scope="module")
def trained_lr(models_config: str, features_config: str, best_params_config: str):
    model = build_model(
        "lr",
        config_path=models_config,
        feat_config_path=features_config,
        best_params_path=best_params_config,
    )
    model.fit(X_TRAIN, Y_TRAIN)
    return model


def test_predict_binary_labels(trained_lr):
    preds = trained_lr.predict(X_PRED)

    assert preds.shape == (len(X_PRED),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_predict_proba_valid_binary_distribution(trained_lr):
    proba = trained_lr.predict_proba(X_PRED)

    assert proba.shape == (len(X_PRED), 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
