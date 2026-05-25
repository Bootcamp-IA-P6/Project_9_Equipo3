"""Tests del vectorizador TF-IDF."""

import pytest

from src.features.vectorizer import Vectorizer


# min_df=3 en configs/features.yaml: términos deben aparecer en ≥3 documentos
CORPUS_TRAIN = [
    "the quick brown fox jumps",
    "the lazy dog runs fast",
    "the fox and dog play together",
    "another quick fox story here",
]
CORPUS_TEST = ["the fox is quick today"]


@pytest.fixture(scope="module")
def vectorizer(features_config: str) -> Vectorizer:
    return Vectorizer(config_path=features_config, method="tfidf")


def test_fit_transform_output_shape(vectorizer: Vectorizer):
    matrix = vectorizer.fit_transform(CORPUS_TRAIN)

    assert matrix.shape[0] == len(CORPUS_TRAIN)
    assert matrix.shape[1] > 0
    assert matrix.shape[1] <= 5000


def test_transform_preserves_sample_count(vectorizer: Vectorizer):
    train_matrix = vectorizer.fit_transform(CORPUS_TRAIN)
    test_matrix = vectorizer.transform(CORPUS_TEST)

    assert test_matrix.shape[0] == len(CORPUS_TEST)
    assert test_matrix.shape[1] == train_matrix.shape[1]
