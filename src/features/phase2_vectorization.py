"""Phase 2 TF-IDF and CountVectorizer helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


DEFAULT_MAX_FEATURES = 5000
DEFAULT_NGRAM_RANGE = (1, 2)


def fit_vectorizers(
    train_texts: list[str] | np.ndarray,
    *,
    max_features: int = DEFAULT_MAX_FEATURES,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> dict[str, Any]:
    """Fit TF-IDF and CountVectorizer on training processed text."""
    train = list(train_texts)
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True,
    )
    count = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
    )
    X_tfidf = tfidf.fit_transform(train)
    X_count = count.fit_transform(train)
    return {
        "tfidf_vectorizer": tfidf,
        "count_vectorizer": count,
        "tfidf_shape": X_tfidf.shape,
        "count_shape": X_count.shape,
        "vocabulary_size_tfidf": len(tfidf.vocabulary_),
        "vocabulary_size_count": len(count.vocabulary_),
    }


def transform_texts(
    texts: list[str] | np.ndarray,
    tfidf_vectorizer: TfidfVectorizer,
    count_vectorizer: CountVectorizer,
) -> dict[str, Any]:
    """Transform texts with fitted vectorizers."""
    texts = list(texts)
    return {
        "X_tfidf": tfidf_vectorizer.transform(texts),
        "X_count": count_vectorizer.transform(texts),
    }


def save_vectorizers(
    tfidf_vectorizer: TfidfVectorizer,
    count_vectorizer: CountVectorizer,
    directory: str | Path,
) -> dict[str, Path]:
    """Persist vectorizers for Phase 3 training."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "tfidf": out / "tfidf_vectorizer.joblib",
        "count": out / "count_vectorizer.joblib",
    }
    joblib.dump(tfidf_vectorizer, paths["tfidf"])
    joblib.dump(count_vectorizer, paths["count"])
    return paths


def vectorizer_summary(
    tfidf_vectorizer: TfidfVectorizer,
    count_vectorizer: CountVectorizer,
) -> dict[str, Any]:
    """Metadata for reports."""
    return {
        "tfidf": {
            "max_features": tfidf_vectorizer.max_features,
            "ngram_range": list(tfidf_vectorizer.ngram_range),
            "vocabulary_size": len(tfidf_vectorizer.vocabulary_),
        },
        "count": {
            "max_features": count_vectorizer.max_features,
            "ngram_range": list(count_vectorizer.ngram_range),
            "vocabulary_size": len(count_vectorizer.vocabulary_),
        },
    }
