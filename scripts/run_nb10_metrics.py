#!/usr/bin/env python3
"""Train NB 10: FeatureUnion word+char TF-IDF + L2 logistic regression."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lr_tfidf_limited.joblib"
METRICS_PATH = ROOT / "reports" / "nb10_metrics.json"
GAP_MAX_PP = 5.0
MIN_DF = 5
WORD_MAX = 500
CHAR_MAX = 500
C = 0.01


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


def build_pipeline(select_k: int | None = None) -> Pipeline:
    steps: list[tuple[str, object]] = [("features", feature_union())]
    if select_k is not None:
        steps.append(("select", SelectKBest(chi2, k=select_k)))
    steps.append(
        (
            "clf",
            LogisticRegression(
                C=C,
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
        )
    )
    return Pipeline(steps)


def evaluate(pipe, X_train, X_test, y_train, y_test) -> dict:
    pred_tr = pipe.predict(X_train)
    pred_te = pipe.predict(X_test)
    proba_te = pipe.predict_proba(X_test)[:, 1]
    f1_tr = f1_score(y_train, pred_tr, average="weighted")
    f1_te = f1_score(y_test, pred_te, average="weighted")
    gap_pp = abs(f1_tr - f1_te) * 100
    return {
        "name": "Hybrid TF-IDF + LR (nb10)",
        "f1_train": round(float(f1_tr), 4),
        "f1_test": round(float(f1_te), 4),
        "train_test_gap_pp": round(float(gap_pp), 2),
        "gap_ok": gap_pp < GAP_MAX_PP,
        "f1_test_ok": f1_te >= 0.70,
        "roc_auc": round(float(roc_auc_score(y_test, proba_te)), 4),
    }


def main() -> None:
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

    pipe = build_pipeline(select_k=None)
    pipe.fit(X_train, y_train)
    metrics = evaluate(pipe, X_train, X_test, y_train, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    payload = {
        "notebook": "10_tfidf_limited_v2",
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "hyperparameters": {
            "word_ngram_range": [1, 2],
            "word_max_features": WORD_MAX,
            "char_ngram_range": [3, 5],
            "char_max_features": CHAR_MAX,
            "min_df": MIN_DF,
            "C": C,
            "classifier": "logistic_regression_l2",
            "select_k": None,
        },
        "metrics": metrics,
        "constraints": {"gap_max_pp": GAP_MAX_PP, "f1_test_min": 0.70},
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()
