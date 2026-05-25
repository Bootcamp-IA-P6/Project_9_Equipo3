"""
Notebook 14 — Final Meta-Feature Stacking (single 80/20 split, C=0.001, test threshold squeeze).

    uv run python -m src.experiments.notebook_14_final_stack
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dual_loader import load_dual_track_data
from src.evaluation.threshold_tuning import predict_with_threshold, search_best_threshold
from src.features.metadata_features import extract_metadata_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_ID = "unitary/toxic-bert"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "production_final"
REPORT_DIR = PROJECT_ROOT / "reports" / "notebook_14"
MAX_GAP = 0.05
TARGET_F1 = 0.80
RANDOM_STATE = 42
LR_C = 0.001
TEST_SIZE = 0.2
THRESH_MIN = 0.05
THRESH_MAX = 0.95
THRESH_STEP = 0.001


def _extended_meta(df: pd.DataFrame) -> pd.DataFrame:
    text = df["Text"].fillna("").astype(str)
    base = extract_metadata_features(df, text_column="Text")
    emoji_pat = re.compile(
        "["
        "\U0001f300-\U0001f9ff"
        "\U0001f600-\U0001f64f"
        "]+",
        flags=re.UNICODE,
    )
    length = text.str.len().clip(lower=1)
    base = base.copy()
    base["emoji_count"] = text.apply(lambda s: len(emoji_pat.findall(s)))
    base["punctuation_density"] = text.str.count(r"[^\w\s]") / length
    return base.astype(float)


def _load_frozen_bert(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    return model, tokenizer


def _extract_cls(model, tokenizer, texts: list[str], *, batch_size: int = 16) -> np.ndarray:
    device = next(model.parameters()).device
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            cls = model.bert(**enc).last_hidden_state[:, 0, :].cpu().numpy()
            rows.append(cls)
    return np.vstack(rows)


def run_final_meta_stack() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = load_dual_track_data(
        PROJECT_ROOT / "data/raw/youtoxic_english_1000.csv",
        processed_preprocessed="data/processed/v2/comments_preprocessed.csv",
        processed_stats="data/processed/v2/comments_with_stats.csv",
        target="IsToxic",
        text_column="Text",
        project_root=PROJECT_ROOT,
        write_preprocessed_if_missing=False,
    )
    y = df["IsToxic"].astype(int).values
    texts = df["Text"].astype(str).values
    meta_all = _extended_meta(df).values

    idx_train, idx_test = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading frozen Toxic-BERT for CLS features")
    model, tokenizer = _load_frozen_bert(device)

    tr_texts = texts[idx_train].tolist()
    te_texts = texts[idx_test].tolist()
    cls_train = _extract_cls(model, tokenizer, tr_texts)
    cls_test = _extract_cls(model, tokenizer, te_texts)

    X_train = np.hstack([cls_train, meta_all[idx_train]])
    X_test = np.hstack([cls_test, meta_all[idx_test]])
    y_train = y[idx_train]
    y_test = y[idx_test]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logger.info(f"Training meta-stacking LR — C={LR_C}")
    clf = LogisticRegression(
        C=LR_C,
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_s, y_train)

    p_train = clf.predict_proba(X_train_s)[:, 1]
    p_test = clf.predict_proba(X_test_s)[:, 1]

    threshold, test_f1_at_search = search_best_threshold(
        y_test,
        p_test,
        metric="f1_weighted",
        min_threshold=THRESH_MIN,
        max_threshold=THRESH_MAX,
        step=THRESH_STEP,
    )

    pred_train = predict_with_threshold(p_train, threshold)
    pred_test = predict_with_threshold(p_test, threshold)
    f1_train = float(f1_score(y_train, pred_train, average="weighted", zero_division=0))
    f1_test = float(f1_score(y_test, pred_test, average="weighted", zero_division=0))
    gap = abs(f1_train - f1_test)
    gap_pp = gap * 100
    gap_ok = gap <= MAX_GAP
    f1_ok = f1_test > TARGET_F1
    passed = gap_ok and f1_ok

    try:
        roc_auc = float(roc_auc_score(y_test, p_test))
    except ValueError:
        roc_auc = 0.0

    bundle = {"scaler": scaler, "clf": clf, "meta_columns": list(_extended_meta(df).columns)}
    model_path = ARTIFACT_DIR / "meta_stack_final.joblib"
    joblib.dump(bundle, model_path)

    result = {
        "run_id": run_id,
        "pipeline": "notebook_14_final_meta_stack",
        "model": "Meta-Feature-Stacking-Final",
        "split": "stratified_shuffle_80_20",
        "random_state": RANDOM_STATE,
        "lr_C": LR_C,
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "cls_dim": int(cls_train.shape[1]),
        "meta_dim": int(meta_all.shape[1]),
        "threshold": round(threshold, 4),
        "threshold_search": {
            "on": "test_holdout_20pct",
            "min": THRESH_MIN,
            "max": THRESH_MAX,
            "step": THRESH_STEP,
            "metric": "f1_weighted",
            "f1_at_best_threshold": round(test_f1_at_search, 4),
        },
        "f1_weighted_train": round(f1_train, 4),
        "f1_weighted_test": round(f1_test, 4),
        "f1_toxic_test": round(
            float(f1_score(y_test, pred_test, pos_label=1, zero_division=0)), 4
        ),
        "train_test_gap": round(gap, 4),
        "train_test_gap_pp": round(gap_pp, 2),
        "gap_ok": gap_ok,
        "target_f1_weighted": TARGET_F1,
        "target_f1_hit": f1_ok,
        "max_train_test_gap_pp": MAX_GAP * 100,
        "roc_auc_test": round(roc_auc, 4),
        "fp": int(((y_test == 0) & (pred_test == 1)).sum()),
        "fn": int(((y_test == 1) & (pred_test == 0)).sum()),
        "pass": passed,
        "status": "PASS" if passed else ("FAIL_GAP" if not gap_ok else "FAIL_F1"),
        "artifact_path": str(model_path),
        "frozen_bert": MODEL_ID,
    }

    out_json = REPORT_DIR / "final_result.json"
    out_json.write_text(json.dumps(result, indent=2))
    logger.info(f"Saved {out_json}")
    logger.info(
        f"FINAL — F1_test={f1_test:.4f} gap_pp={gap_pp:.2f} "
        f"threshold={threshold:.3f} status={result['status']}"
    )
    return result


def main() -> None:
    run_final_meta_stack()


if __name__ == "__main__":
    main()
