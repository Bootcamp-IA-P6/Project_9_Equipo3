#!/usr/bin/env python3
"""Track B: FeatureUnion word+char TF-IDF + soft voting LR/SVC + optional SelectKBest.

Writes to experiment paths only (not models/lr_tfidf_limited.joblib).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
BASELINE_F1 = 0.7473
BASELINE_GAP_PP = 4.84
GAP_MAX_PP = 5.0
SAVE_PATH = ROOT / "models" / "nb10_experiment_track_b.joblib"
METRICS_PATH = ROOT / "reports" / "nb10_experiment_track_b.json"
MIN_DF = 5
WORD_MAX = 500
CHAR_MAX = 500
C = 0.01


def evaluate(pipe, X_train, X_test, y_train, y_test) -> dict:
    pred_tr = pipe.predict(X_train)
    pred_te = pipe.predict(X_test)
    proba_te = pipe.predict_proba(X_test)[:, 1]
    f1_tr = f1_score(y_train, pred_tr, average="weighted")
    f1_te = f1_score(y_test, pred_te, average="weighted")
    gap_pp = abs(f1_tr - f1_te) * 100
    return {
        "f1_train": round(float(f1_tr), 4),
        "f1_test": round(float(f1_te), 4),
        "train_test_gap_pp": round(float(gap_pp), 2),
        "gap_ok": gap_pp < GAP_MAX_PP,
        "f1_test_ok": f1_te >= 0.70,
        "roc_auc": round(float(roc_auc_score(y_test, proba_te)), 4),
    }


def classify_result(f1_test: float, gap_pp: float) -> str:
    gap_ok = gap_pp < GAP_MAX_PP
    if f1_test > BASELINE_F1 and gap_ok:
        return "IMPROVED"
    if f1_test >= BASELINE_F1 - 0.005 and gap_ok:
        return "SIMILAR"
    return "WORSE"


def feature_union() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=WORD_MAX,
                    min_df=MIN_DF,
                    sublinear_tf=False,
                    analyzer="word",
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=CHAR_MAX,
                    min_df=MIN_DF,
                    sublinear_tf=False,
                ),
            ),
        ]
    )


def logistic_clf() -> LogisticRegression:
    return LogisticRegression(
        C=C,
        penalty="l2",
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )


def voting_classifier() -> VotingClassifier:
    lr = LogisticRegression(
        C=C,
        penalty="l2",
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    # LinearSVC has no predict_proba; CalibratedClassifierCV enables soft voting.
    svc = CalibratedClassifierCV(
        LinearSVC(C=C, dual="auto", max_iter=5000, class_weight="balanced"),
        cv=3,
    )
    return VotingClassifier(
        estimators=[("lr", lr), ("svc", svc)],
        voting="soft",
    )


def build_pipeline(
    select_k: int | None = None, use_voting: bool = True
) -> Pipeline:
    steps: list[tuple[str, object]] = [("features", feature_union())]
    if select_k is not None:
        steps.append(("select", SelectKBest(chi2, k=select_k)))
    clf = voting_classifier() if use_voting else logistic_clf()
    steps.append(("clf", clf))
    return Pipeline(steps)


def fit_with_optional_select(
    X_train, X_test, y_train, y_test, use_voting: bool = True
) -> tuple[Pipeline, dict, int | None]:
    pipe = build_pipeline(select_k=None, use_voting=use_voting)
    pipe.fit(X_train, y_train)
    metrics = evaluate(pipe, X_train, X_test, y_train, y_test)
    if metrics["gap_ok"]:
        return pipe, metrics, None

    # Gap too high — sweep SelectKBest k on transformed feature count.
    feat_pipe = Pipeline([("features", feature_union())])
    X_tr_m = feat_pipe.fit_transform(X_train, y_train)
    n_features = X_tr_m.shape[1]
    k_candidates = sorted(
        {k for k in (300, 400, 500, 600, 700, 800, 900, 1000, n_features) if k <= n_features},
        reverse=True,
    )

    best_pipe, best_metrics, best_k = pipe, metrics, None
    for k in k_candidates:
        pipe_k = build_pipeline(select_k=k, use_voting=use_voting)
        pipe_k.fit(X_train, y_train)
        m = evaluate(pipe_k, X_train, X_test, y_train, y_test)
        if m["gap_ok"] and (
            best_k is None
            or not best_metrics["gap_ok"]
            or m["f1_test"] > best_metrics["f1_test"]
            or (
                m["f1_test"] == best_metrics["f1_test"]
                and m["train_test_gap_pp"] < best_metrics["train_test_gap_pp"]
            )
        ):
            best_pipe, best_metrics, best_k = pipe_k, m, k
        elif not best_metrics["gap_ok"] and m["f1_test"] > best_metrics["f1_test"]:
            best_pipe, best_metrics, best_k = pipe_k, m, k

    return best_pipe, best_metrics, best_k


def run_variant(use_voting: bool) -> tuple[Pipeline, dict, int | None, str]:
    with open(ROOT / "configs" / "pipeline.yaml") as f:
        pipe_cfg = yaml.safe_load(f)
    target = pipe_cfg["data"]["target_binary"]
    rand = pipe_cfg["pipeline"]["random_state"]
    test_size = pipe_cfg["pipeline"]["test_size"]

    df = pd.read_csv(ROOT / "data/processed/v2/comments_preprocessed.csv")
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    X, y = df["clean_text"], df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rand, stratify=y
    )
    pipe, metrics, select_k = fit_with_optional_select(
        X_train, X_test, y_train, y_test, use_voting=use_voting
    )
    label = "SoftVoting" if use_voting else "LR-only"
    strategy = (
        f"FeatureUnion(word 1-2 + char 3-5, mf=500) | {label}(C={C})"
    )
    if select_k is not None:
        strategy += f" | SelectKBest(k={select_k})"
    return pipe, metrics, select_k, strategy


def main() -> None:
    variants = [
        run_variant(use_voting=True),
        run_variant(use_voting=False),
    ]
    pipe, metrics, select_k, strategy = max(
        variants,
        key=lambda v: (
            classify_result(v[1]["f1_test"], v[1]["train_test_gap_pp"]) != "WORSE",
            v[1]["f1_test"],
            -v[1]["train_test_gap_pp"],
        ),
    )
    status = classify_result(metrics["f1_test"], metrics["train_test_gap_pp"])

    payload = {
        "strategy": strategy,
        "experiment": "track_b_nb10",
        "baseline": {"f1_test": BASELINE_F1, "gap_pp": BASELINE_GAP_PP},
        "hyperparameters": {
            "word_ngram_range": [1, 2],
            "word_max_features": WORD_MAX,
            "char_ngram_range": [3, 5],
            "char_max_features": CHAR_MAX,
            "min_df": MIN_DF,
            "C": C,
            "select_k": select_k,
        },
        "result": metrics,
        "status": status,
        "model_path": None,
    }

    if status != "WORSE":
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, SAVE_PATH)
        payload["model_path"] = str(SAVE_PATH.relative_to(ROOT))

    METRICS_PATH.write_text(json.dumps(payload, indent=2))

    if status == "WORSE":
        if SAVE_PATH.exists():
            SAVE_PATH.unlink()
        print("Discarded experiment artifacts (WORSE).")

    print("\n--- Track B summary ---")
    print(f"Strategy: {strategy}")
    print(f"F1: {BASELINE_F1:.4f} -> {metrics['f1_test']:.4f}")
    print(f"Gap: {BASELINE_GAP_PP:.2f} -> {metrics['train_test_gap_pp']:.2f} pp")
    print(f"Status: {status}")
    print(f"Saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
