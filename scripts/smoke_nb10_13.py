#!/usr/bin/env python3
"""Smoke tests for notebooks 10 and 13 (sklearn pipelines)."""
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "configs/pipeline.yaml") as f:
    pipe_cfg = yaml.safe_load(f)
TARGET = pipe_cfg["data"]["target_binary"]
RAND = pipe_cfg["pipeline"]["random_state"]
TEST_SIZE = pipe_cfg["pipeline"]["test_size"]
GAP_MAX = 5.0
F1_MIN = 0.70

df = pd.read_csv(ROOT / "data/processed/v2/comments_preprocessed.csv")
df["clean_text"] = df["clean_text"].fillna("").astype(str)
X, y = df["clean_text"], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RAND, stratify=y
)

results = []

# --- NB10 ---
for mf in [500, 600, 700, 800]:
    for C in [0.01, 0.05, 0.1]:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=mf, min_df=5, ngram_range=(1, 1),
                sublinear_tf=False, analyzer="word", strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                C=C, max_iter=2000, class_weight="balanced",
                solver="lbfgs", random_state=RAND,
            )),
        ])
        pipe.fit(X_train, y_train)
        f1_tr = f1_score(y_train, pipe.predict(X_train), average="weighted")
        f1_te = f1_score(y_test, pipe.predict(X_test), average="weighted")
        gap = abs(f1_tr - f1_te) * 100
        ok = gap < GAP_MAX and f1_te >= F1_MIN
        results.append(("nb10", mf, C, f1_te, gap, ok))
        if ok:
            print(f"nb10 PASS mf={mf} C={C} f1_test={f1_te:.4f} gap={gap:.2f}pp")

# --- NB13 ---
for n_comp in [100, 150, 200]:
    for clf_name, clf in [
        ("LinearSVC", LinearSVC(class_weight="balanced", max_iter=3000, random_state=RAND)),
        ("LR", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=RAND)),
    ]:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=500, min_df=5, ngram_range=(1, 1),
                sublinear_tf=False, analyzer="word", strip_accents="unicode",
            )),
            ("svd", TruncatedSVD(n_components=n_comp, random_state=RAND)),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        f1_tr = f1_score(y_train, pipe.predict(X_train), average="weighted")
        f1_te = f1_score(y_test, pipe.predict(X_test), average="weighted")
        gap = abs(f1_tr - f1_te) * 100
        ok = gap < GAP_MAX and f1_te >= F1_MIN
        results.append(("nb13", n_comp, clf_name, f1_te, gap, ok))
        if ok:
            print(f"nb13 PASS n={n_comp} {clf_name} f1_test={f1_te:.4f} gap={gap:.2f}pp")

best10 = [r for r in results if r[0] == "nb10" and r[5]]
best13 = [r for r in results if r[0] == "nb13" and r[5]]
print("\nnb10 passing:", len(best10), "nb13 passing:", len(best13))
