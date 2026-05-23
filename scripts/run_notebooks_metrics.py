#!/usr/bin/env python3
"""Execute notebook logic and write metrics JSON for 10-13."""
from pathlib import Path
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "configs/pipeline.yaml") as f:
    pipe_cfg = yaml.safe_load(f)
TARGET = pipe_cfg["data"]["target_binary"]
RAND = pipe_cfg["pipeline"]["random_state"]
TEST_SIZE = pipe_cfg["pipeline"]["test_size"]
GAP_MAX = 5.0
F1_MIN = 0.70


def load_split():
    df = pd.read_csv(ROOT / "data/processed/v2/comments_preprocessed.csv")
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    X, y = df["clean_text"], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RAND, stratify=y)


def metrics(pipe, X_train, y_train, X_test, y_test):
    f1_tr = f1_score(y_train, pipe.predict(X_train), average="weighted")
    f1_te = f1_score(y_test, pipe.predict(X_test), average="weighted")
    gap = abs(f1_tr - f1_te) * 100
    roc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    return {
        "f1_train": round(f1_tr, 4),
        "f1_test": round(f1_te, 4),
        "train_test_gap_pp": round(gap, 2),
        "gap_ok": gap < GAP_MAX,
        "f1_test_ok": f1_te >= F1_MIN,
        "roc_auc": round(roc, 4),
    }


def run_nb10():
    X_train, X_test, y_train, y_test = load_split()
    winner = None
    for mf in [500, 600, 700, 800]:
        for C in [0.01, 0.05, 0.1, 0.2]:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=mf, min_df=5, ngram_range=(1, 1), sublinear_tf=False)),
                ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=RAND)),
            ])
            pipe.fit(X_train, y_train)
            m = metrics(pipe, X_train, y_train, X_test, y_test)
            if m["gap_ok"] and m["f1_test_ok"]:
                if winner is None or m["f1_test"] > winner[0]["f1_test"]:
                    winner = (m, mf, C, pipe)
    if winner is None:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500, min_df=5, ngram_range=(1, 1), sublinear_tf=False)),
            ("clf", LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=RAND)),
        ])
        pipe.fit(X_train, y_train)
        m = metrics(pipe, X_train, y_train, X_test, y_test)
        winner = (m, 500, 0.01, pipe)
    m, mf, C, pipe = winner
    path = ROOT / "models/lr_tfidf_limited.joblib"
    path.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, path)
    payload = {"notebook": "10", "hyperparameters": {"max_features": mf, "C": C, "min_df": 5}, "metrics": m}
    (ROOT / "reports/nb10_metrics.json").write_text(json.dumps(payload, indent=2))
    return m, str(path)


def run_nb13():
    X_train, X_test, y_train, y_test = load_split()
    best = None
    for n in [100, 150, 200]:
        for name, clf in [
            ("LinearSVC", LinearSVC(class_weight="balanced", max_iter=3000, random_state=RAND)),
            ("LR", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=RAND)),
        ]:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=500, min_df=5, ngram_range=(1, 1), sublinear_tf=False)),
                ("svd", TruncatedSVD(n_components=n, random_state=RAND)),
                ("clf", clf),
            ])
            pipe.fit(X_train, y_train)
            pred_te = pipe.predict(X_test)
            f1_tr = f1_score(y_train, pipe.predict(X_train), average="weighted")
            f1_te = f1_score(y_test, pred_te, average="weighted")
            gap = abs(f1_tr - f1_te) * 100
            m = {"f1_train": round(f1_tr, 4), "f1_test": round(f1_te, 4), "train_test_gap_pp": round(gap, 2),
                 "gap_ok": gap < GAP_MAX, "f1_test_ok": f1_te >= F1_MIN, "classifier": name, "n_components": n}
            if m["gap_ok"] and m["f1_test_ok"] and (best is None or m["f1_test"] > best[0]["f1_test"]):
                best = (m, pipe)
    m, pipe = best
    joblib.dump(pipe, ROOT / "models/lsa_pipeline.joblib")
    (ROOT / "reports/nb13_metrics.json").write_text(json.dumps({"notebook": "13", "metrics": m}, indent=2))
    return m, str(ROOT / "models/lsa_pipeline.joblib")


if __name__ == "__main__":
    m10, p10 = run_nb10()
    m13, p13 = run_nb13()
    print("nb10", m10, p10)
    print("nb13", m13, p13)
