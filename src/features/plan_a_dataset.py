"""Plan A: TF-IDF + advanced numeric features + optional augmentation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_processed_data
from src.evaluation.phase1_audit import BINARY_TARGET, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from src.features.augmentation import augment_text
from src.features.phase2_text_pipeline import process_text
from src.features.preprocessing import clean_text, extract_advanced_features

ROOT_FEATURES = ["polarity", "subjectivity", "caps_ratio", "excl_count", "quest_count", "text_len", "avg_word_len"]


def load_raw_split(
    raw_path: str,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """Load CSV and stratified train/test split on IsToxic."""
    df, label_cols = load_processed_data(raw_path)
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df[BINARY_TARGET],
    )
    return df.loc[train_idx], df.loc[test_idx], label_cols


def maybe_augment_train(
    train_df: pd.DataFrame,
    label_cols: pd.Index,
    *,
    multiplier: int = 1,
    enabled: bool = True,
) -> pd.DataFrame:
    """Synonym augmentation on train toxic rows only (Phase 1 recommendation)."""
    if not enabled or multiplier < 1:
        return train_df.reset_index(drop=True)
    return augment_text(train_df.copy(), list(label_cols), multiplier=multiplier)


def prepare_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add CleanText and ProcessedText."""
    out = df.copy()
    out["CleanText"] = out["Text"].apply(clean_text)
    out["ProcessedText"] = out["Text"].apply(lambda t: process_text(t)[1])
    return out


def build_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    max_features: int = 8000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> dict[str, Any]:
    """Fit TF-IDF + scaler on train; return sparse hstack matrices."""
    train_df = prepare_text_columns(train_df)
    test_df = prepare_text_columns(test_df)

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(train_df["ProcessedText"].astype(str))
    X_test_tfidf = tfidf.transform(test_df["ProcessedText"].astype(str))

    adv_train = extract_advanced_features(train_df)
    adv_test = extract_advanced_features(test_df)
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(adv_train[ROOT_FEATURES].values)
    X_test_num = scaler.transform(adv_test[ROOT_FEATURES].values)

    X_train = hstack([X_train_tfidf, csr_matrix(X_train_num)])
    X_test = hstack([X_test_tfidf, csr_matrix(X_test_num)])

    y_train = train_df[BINARY_TARGET].values.astype(int)
    y_test = test_df[BINARY_TARGET].values.astype(int)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "vectorizer": tfidf,
        "scaler": scaler,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
