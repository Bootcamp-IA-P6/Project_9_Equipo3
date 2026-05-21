"""Phase 3 baseline classifiers: Safe vs Toxic (IsToxic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.overfitting_check import format_gap_report, passes_overfitting_check
from src.evaluation.phase1_audit import BINARY_TARGET

MODEL_BUILDERS: dict[str, Any] = {
    "naive_bayes": lambda: MultinomialNB(),
    "logistic_regression": lambda: LogisticRegression(
        C=0.02,
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=80,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}


def evaluate_classifier(
    clf,
    X_train,
    X_test,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Train and return metrics with overfitting gaps."""
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    metrics = {}
    for name, fn in [
        ("accuracy", accuracy_score),
        ("f1_toxic", lambda yt, yp: f1_score(yt, yp, pos_label=1, zero_division=0)),
    ]:
        train_v = fn(y_train, train_pred)
        test_v = fn(y_test, test_pred)
        metrics[name] = format_gap_report(name, train_v, test_v)

    metrics["passes_overfitting"] = all(
        metrics[k]["passes"] for k in ("accuracy", "f1_toxic")
    )
    return metrics


def train_and_compare(
    train_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    vectorizer: TfidfVectorizer | None = None,
) -> dict[str, Any]:
    """Fit vectorizer (if needed), compare baselines, return best result bundle."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X_train = vectorizer.fit_transform(train_texts)
    else:
        X_train = vectorizer.transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    results: list[dict[str, Any]] = []
    for name, builder in MODEL_BUILDERS.items():
        clf = builder()
        metrics = evaluate_classifier(clf, X_train, X_test, y_train, y_test)
        clf.fit(X_train, y_train)
        results.append(
            {
                "model_name": name,
                "classifier": clf,
                "metrics": metrics,
                "test_f1_toxic": metrics["f1_toxic"]["test"],
                "passes_overfitting": metrics["passes_overfitting"],
            }
        )

    passing = [r for r in results if r["passes_overfitting"]]
    if passing:
        best = max(passing, key=lambda r: r["test_f1_toxic"])
        selection_reason = "best test F1 among models passing overfitting check"
    else:
        best = max(results, key=lambda r: r["test_f1_toxic"])
        selection_reason = (
            "best test F1 (no model met 5% overfitting gap — increase regularization in Phase 4)"
        )

    return {
        "vectorizer": vectorizer,
        "best": best,
        "all_results": results,
        "selection_reason": selection_reason,
        "target": BINARY_TARGET,
    }


def save_model_bundle(
    bundle: dict[str, Any],
    path: str | Path,
    *,
    borderline_low: float = 0.4,
    borderline_high: float = 0.6,
) -> Path:
    """Persist vectorizer + classifier + metadata for inference."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vectorizer": bundle["vectorizer"],
        "classifier": bundle["best"]["classifier"],
        "model_name": bundle["best"]["model_name"],
        "metrics": bundle["best"]["metrics"],
        "selection_reason": bundle["selection_reason"],
        "target": bundle["target"],
        "mode": "binary",
        "version": "phase3-esencial",
        "borderline_low": borderline_low,
        "borderline_high": borderline_high,
    }
    joblib.dump(payload, out)
    return out


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)
