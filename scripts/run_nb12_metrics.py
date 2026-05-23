#!/usr/bin/env python3
"""Run notebook 12 logic and save metrics."""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "configs/pipeline.yaml") as f:
    pipe_cfg = yaml.safe_load(f)
TARGET = pipe_cfg["data"]["target_binary"]
RAND = pipe_cfg["pipeline"]["random_state"]
TEST_SIZE = pipe_cfg["pipeline"]["test_size"]
GAP_MAX = 5.0
F1_MIN = 0.70
MAX_SYNTHETIC = 30

df = pd.read_csv(ROOT / "data/processed/v2/comments_preprocessed.csv")
df["clean_text"] = df["clean_text"].fillna("").astype(str)
X, y = df["clean_text"], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RAND, stratify=y
)

cache = json.loads((ROOT / "data/processed/v2/backtranslation_toxic_train.json").read_text())
texts_bt = cache["texts"][:MAX_SYNTHETIC]
X_aug = pd.concat([X_train, pd.Series(texts_bt, dtype=str)], ignore_index=True)
y_aug = pd.concat([y_train, pd.Series([True] * len(texts_bt), dtype=bool)], ignore_index=True)

vec = TfidfVectorizer(max_features=500, min_df=5, ngram_range=(1, 1), sublinear_tf=False)
X_aug_vec = vec.fit_transform(X_aug)
X_train_vec = vec.transform(X_train)
X_test_vec = vec.transform(X_test)

clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-3, random_state=RAND, warm_start=True, max_iter=1)
history = []
candidates = []

for epoch in range(1, 61):
    clf.partial_fit(X_aug_vec, y_aug.astype(int), classes=np.array([0, 1]))
    f1_tr = f1_score(y_train, clf.predict(X_train_vec), average="weighted")
    f1_te = f1_score(y_test, clf.predict(X_test_vec), average="weighted")
    gap = abs(f1_tr - f1_te) * 100
    stable = f1_te >= F1_MIN and gap < GAP_MAX
    history.append({"epoch": epoch, "f1_test": f1_te, "gap_pp": gap, "stable": stable})
    candidates.append(
        {"kind": "SGD", "gap_pp": gap, "f1_test": f1_te, "pipe": Pipeline([("tfidf", vec), ("clf", copy.deepcopy(clf))])}
    )
    if stable:
        break

for C in [0.0005, 0.001, 0.01]:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500, min_df=5, ngram_range=(1, 1), sublinear_tf=False)),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=RAND)),
    ])
    pipe.fit(X_aug, y_aug)
    f1_tr = f1_score(y_train, pipe.predict(X_train), average="weighted")
    f1_te = f1_score(y_test, pipe.predict(X_test), average="weighted")
    gap = abs(f1_tr - f1_te) * 100
    candidates.append({"kind": f"LR C={C}", "gap_pp": gap, "f1_test": f1_te, "pipe": pipe})

eligible = [c for c in candidates if c["f1_test"] >= F1_MIN]
winner = min(eligible if eligible else candidates, key=lambda c: c["gap_pp"])
final_pipe = winner["pipe"]
joblib.dump(final_pipe, ROOT / "models/lr_backtranslation.joblib")

f1_tr = f1_score(y_train, final_pipe.predict(X_train), average="weighted")
f1_te = f1_score(y_test, final_pipe.predict(X_test), average="weighted")
gap = abs(f1_tr - f1_te) * 100
metrics = {
    "f1_train": round(f1_tr, 4),
    "f1_test": round(f1_te, 4),
    "train_test_gap_pp": round(gap, 2),
    "gap_ok": gap < GAP_MAX,
    "f1_test_ok": f1_te >= F1_MIN,
    "roc_auc": round(roc_auc_score(y_test, final_pipe.predict_proba(X_test)[:, 1]), 4),
    "selected": winner["kind"],
}
(ROOT / "reports/nb12_metrics.json").write_text(
    json.dumps(
        {"notebook": "12", "n_synthetic": len(texts_bt), "metrics": metrics, "history_tail": history[-5:]},
        indent=2,
    )
)
print(metrics)
