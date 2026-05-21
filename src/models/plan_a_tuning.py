"""Plan A tuning: hybrid features + Optuna (soft gap penalty + grid fallbacks)."""

from __future__ import annotations

from typing import Any

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.overfitting_check import MAX_GAP_PCT, format_gap_report


def _metrics(clf, X_train, X_test, y_train, y_test) -> dict[str, Any]:
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
    return {
        "accuracy": acc,
        "f1_toxic": f1,
        "passes_overfitting": acc["passes"] and f1["passes"],
    }


def _append_result(results: list, name: str, clf, params: dict, m: dict) -> None:
    results.append(
        {
            "model_name": name,
            "classifier": clf,
            "params": params,
            "metrics": m,
            "passes_overfitting": m["passes_overfitting"],
            "test_f1_toxic": m["f1_toxic"]["test"],
            "test_accuracy": m["accuracy"]["test"],
        }
    )


def _grid_baselines(X_train, X_test, y_train, y_test) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for c in [0.01, 0.03, 0.08, 0.2, 0.5]:
        clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        _append_result(results, f"logistic_c{c}", clf, {"C": c}, _metrics(clf, X_train, X_test, y_train, y_test))
    for depth, leaf in [(3, 8), (4, 10), (5, 12), (6, 8)]:
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=depth,
            min_samples_leaf=leaf,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        _append_result(
            results,
            f"rf_d{depth}_l{leaf}",
            clf,
            {"max_depth": depth, "min_samples_leaf": leaf},
            _metrics(clf, X_train, X_test, y_train, y_test),
        )
    return results


def tune_plan_a(
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    n_trials: int = 40,
) -> list[dict[str, Any]]:
    results = _grid_baselines(X_train, X_test, y_train, y_test)

    def objective(trial: optuna.Trial) -> float:
        kind = trial.suggest_categorical("kind", ["logistic", "rf"])
        if kind == "logistic":
            c = trial.suggest_float("C", 0.005, 1.0, log=True)
            clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced", random_state=42)
        else:
            clf = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 60, 120),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 20),
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        clf.fit(X_train, y_train)
        m = _metrics(clf, X_train, X_test, y_train, y_test)
        gap = m["f1_toxic"]["gap_pct"]
        score = m["f1_toxic"]["test"]
        if gap > MAX_GAP_PCT:
            score -= (gap - MAX_GAP_PCT) * 0.015
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    if study.best_trial:
        bt = study.best_trial
        if bt.params["kind"] == "logistic":
            clf = LogisticRegression(
                C=bt.params["C"], max_iter=5000, class_weight="balanced", random_state=42
            )
            name = "optuna_logistic_plan_a"
            params = {"C": bt.params["C"]}
        else:
            clf = RandomForestClassifier(
                n_estimators=bt.params["n_estimators"],
                max_depth=bt.params["max_depth"],
                min_samples_leaf=bt.params["min_samples_leaf"],
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            name = "optuna_rf_plan_a"
            params = {k: bt.params[k] for k in bt.params if k != "kind"}
        clf.fit(X_train, y_train)
        _append_result(results, name, clf, params, _metrics(clf, X_train, X_test, y_train, y_test))

    return results


def select_best(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    passing = [c for c in candidates if c["passes_overfitting"]]
    if passing:
        best = max(passing, key=lambda c: (c["test_f1_toxic"], c["test_accuracy"]))
        return best, "best test F1 among Plan A models passing 5% gap"
    best = max(candidates, key=lambda c: c["test_f1_toxic"])
    return best, "best test F1 (Plan A — overfitting gap may exceed 5%)"
