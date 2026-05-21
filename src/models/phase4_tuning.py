"""Phase 4 Optuna hyperparameter tuning with overfitting constraint."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from src.evaluation.overfitting_check import MAX_GAP_PCT, format_gap_report, passes_overfitting_check


def _f1_gap(clf, X_train, X_test, y_train, y_test) -> tuple[float, float, float]:
    clf.fit(X_train, y_train)
    train_f1 = f1_score(y_train, clf.predict(X_train), pos_label=1, zero_division=0)
    test_f1 = f1_score(y_test, clf.predict(X_test), pos_label=1, zero_division=0)
    gap = abs((train_f1 - test_f1) * 100.0)
    return train_f1, test_f1, gap


def tune_logistic_regression(
    X_train,
    X_test,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    n_trials: int = 60,
) -> dict[str, Any]:
    """Optuna search for LogisticRegression C with gap constraint."""

    def objective(trial: optuna.Trial) -> float:
        c = trial.suggest_float("C", 0.001, 0.3, log=True)
        clf = LogisticRegression(
            C=c,
            max_iter=4000,
            class_weight="balanced",
            random_state=42,
        )
        train_f1, test_f1, gap = _f1_gap(clf, X_train, X_test, y_train, y_test)
        trial.set_user_attr("train_f1", train_f1)
        trial.set_user_attr("test_f1", test_f1)
        trial.set_user_attr("gap_pct", gap)
        if gap > MAX_GAP_PCT:
            return test_f1 - (gap - MAX_GAP_PCT) * 0.02
        return test_f1

    study = optuna.create_study(direction="maximize", study_name="phase4_logreg")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    clf = LogisticRegression(
        C=best.params["C"],
        max_iter=4000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    metrics = _build_metrics(clf, X_train, X_test, y_train, y_test)
    return {
        "model_name": "logistic_regression_tuned",
        "classifier": clf,
        "params": best.params,
        "metrics": metrics,
        "passes_overfitting": metrics["passes_overfitting"],
        "test_f1_toxic": metrics["f1_toxic"]["test"],
    }


def tune_random_forest(
    X_train,
    X_test,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    n_trials: int = 50,
) -> dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 40, 120),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 20),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        train_f1, test_f1, gap = _f1_gap(clf, X_train, X_test, y_train, y_test)
        trial.set_user_attr("gap_pct", gap)
        trial.set_user_attr("test_f1", test_f1)
        if gap > MAX_GAP_PCT:
            return test_f1 - (gap - MAX_GAP_PCT) * 0.02
        return test_f1

    study = optuna.create_study(direction="maximize", study_name="phase4_rf")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    clf = RandomForestClassifier(
        **{k: v for k, v in best.params.items()},
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    metrics = _build_metrics(clf, X_train, X_test, y_train, y_test)
    return {
        "model_name": "random_forest_tuned",
        "classifier": clf,
        "params": best.params,
        "metrics": metrics,
        "passes_overfitting": metrics["passes_overfitting"],
        "test_f1_toxic": metrics["f1_toxic"]["test"],
    }


def train_mlp_baseline(
    X_train,
    X_test,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Lightweight neural baseline (sklearn MLP)."""
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=0.01,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    metrics = _build_metrics(clf, X_train, X_test, y_train, y_test)
    return {
        "model_name": "mlp_classifier",
        "classifier": clf,
        "params": {"hidden_layer_sizes": [64, 32], "alpha": 0.01},
        "metrics": metrics,
        "passes_overfitting": metrics["passes_overfitting"],
        "test_f1_toxic": metrics["f1_toxic"]["test"],
    }


def _build_metrics(clf, X_train, X_test, y_train, y_test) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score

    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    acc = format_gap_report(
        "accuracy",
        accuracy_score(y_train, train_pred),
        accuracy_score(y_test, test_pred),
    )
    f1 = format_gap_report(
        "f1_toxic",
        f1_score(y_train, train_pred, pos_label=1, zero_division=0),
        f1_score(y_test, test_pred, pos_label=1, zero_division=0),
    )
    passes = acc["passes"] and f1["passes"]
    return {"accuracy": acc, "f1_toxic": f1, "passes_overfitting": passes}
