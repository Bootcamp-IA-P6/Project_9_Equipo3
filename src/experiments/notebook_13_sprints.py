"""
Notebook 13 — Hyper-optimization sprints (5-fold CV, gap < 5%, F1 > 0.80).

    uv run python -m src.experiments.notebook_13_sprints
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dual_loader import load_dual_track_data
from src.evaluation.threshold_tuning import predict_with_threshold, search_best_threshold
from src.features.augmentation import (
    back_translate_texts,
    deduplicate_by_cosine,
    toxic_back_translation,
)
from src.features.metadata_features import extract_metadata_features
from src.models.transformer_trainer import (
    compute_hf_metrics,
    freeze_head_only,
    logits_to_toxic_prob,
)
from src.pipeline.run_hybrid_clean_pipeline import _meta_frame
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_ID = "unitary/toxic-bert"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "notebook_13"
REPORT_DIR = PROJECT_ROOT / "reports" / "notebook_13"
MAX_GAP = 0.05
TARGET_F1 = 0.80
N_FOLDS = 5
RANDOM_STATE = 42
PIVOTS = ("de", "fr", "es")
TTA_WEIGHTS = (0.50, 0.25, 0.25)  # original, DE, FR


@dataclass
class FoldMetrics:
    fold: int
    f1_train: float
    f1_test: float
    f1_val: float
    gap: float
    gap_pp: float
    gap_ok: bool
    threshold: float
    roc_auc: float


def _gap_ok(gap: float) -> bool:
    return gap <= MAX_GAP


def _summarize_folds(folds: list[FoldMetrics]) -> dict:
    tests = [f.f1_test for f in folds]
    gaps = [f.gap_pp for f in folds]
    return {
        "f1_test_mean": round(float(np.mean(tests)), 4),
        "f1_test_std": round(float(np.std(tests)), 4),
        "f1_test_min": round(float(np.min(tests)), 4),
        "f1_test_max": round(float(np.max(tests)), 4),
        "gap_pp_mean": round(float(np.mean(gaps)), 2),
        "gap_pp_max": round(float(np.max(gaps)), 2),
        "all_gap_ok": all(f.gap_ok for f in folds),
        "target_f1_hit": float(np.mean(tests)) >= TARGET_F1,
        "folds": [
            {
                "fold": f.fold,
                "f1_train": f.f1_train,
                "f1_test": f.f1_test,
                "f1_val": f.f1_val,
                "train_test_gap_pp": f.gap_pp,
                "gap_ok": f.gap_ok,
                "threshold": f.threshold,
                "roc_auc": f.roc_auc,
            }
            for f in folds
        ],
    }


def _score_split(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    p_train: np.ndarray,
    p_val: np.ndarray,
    p_test: np.ndarray,
    *,
    fold: int,
    min_t: float = 0.05,
    max_t: float = 0.95,
    step: float = 0.01,
) -> FoldMetrics:
    threshold, val_f1 = search_best_threshold(
        y_val, p_val, metric="f1_weighted", min_threshold=min_t, max_threshold=max_t, step=step
    )
    pred_train = predict_with_threshold(p_train, threshold)
    pred_test = predict_with_threshold(p_test, threshold)
    f1_train = float(f1_score(y_train, pred_train, average="weighted", zero_division=0))
    f1_test = float(f1_score(y_test, pred_test, average="weighted", zero_division=0))
    gap = abs(f1_train - f1_test)
    try:
        auc = float(roc_auc_score(y_test, p_test))
    except ValueError:
        auc = 0.0
    return FoldMetrics(
        fold=fold,
        f1_train=round(f1_train, 4),
        f1_test=round(f1_test, 4),
        f1_val=round(val_f1, 4),
        gap=round(gap, 4),
        gap_pp=round(gap * 100, 2),
        gap_ok=_gap_ok(gap),
        threshold=round(threshold, 4),
        roc_auc=round(auc, 4),
    )


def _load_data() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    cfg_data = {
        "raw_path": "data/raw/youtoxic_english_1000.csv",
        "processed_preprocessed": "data/processed/v2/comments_preprocessed.csv",
        "processed_stats": "data/processed/v2/comments_with_stats.csv",
        "features_config": "configs/features.yaml",
    }
    df = load_dual_track_data(
        PROJECT_ROOT / cfg_data["raw_path"],
        processed_preprocessed=cfg_data["processed_preprocessed"],
        processed_stats=cfg_data["processed_stats"],
        target="IsToxic",
        text_column="Text",
        project_root=PROJECT_ROOT,
        write_preprocessed_if_missing=False,
    )
    y = df["IsToxic"].astype(int)
    texts = df["Text"].astype(str).values
    return df, y, texts


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


def _load_frozen_model(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    return model, tokenizer


def _predict_probs(
    model,
    tokenizer,
    texts: list[str],
    *,
    max_length: int = 128,
    batch_size: int = 16,
) -> np.ndarray:
    device = next(model.parameters()).device
    probs: list[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs.extend(logits_to_toxic_prob(logits).tolist())
    return np.array(probs, dtype=float)


def _extract_cls_features(
    model,
    tokenizer,
    texts: list[str],
    *,
    max_length: int = 128,
    batch_size: int = 16,
) -> np.ndarray:
    device = next(model.parameters()).device
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model.bert(**enc).last_hidden_state[:, 0, :].cpu().numpy()
            rows.append(hidden)
    return np.vstack(rows)


def _ensure_global_augment_cache(texts: np.ndarray, y: np.ndarray, cache_path: Path) -> None:
    """Build DE/FR/ES toxic augmentations once for the full dataset (cached)."""
    if cache_path.exists():
        return
    toxic_idx = np.where(y == 1)[0]
    ref_texts = texts[toxic_idx].tolist()
    ref_labels = y[toxic_idx].tolist()
    all_syn_t: list[str] = []
    all_syn_l: list[int] = []
    all_src: list[int] = []

    for pivot in PIVOTS:
        logger.info(f"Global augment — pivot={pivot} ({len(ref_texts)} toxic)")
        syn_t, syn_l = toxic_back_translation(
            ref_texts,
            ref_labels,
            pivot_lang=pivot,
            rate_limit_every=40,
            rate_limit_sleep_sec=0.5,
            seed=RANDOM_STATE,
        )
        for t, lab in zip(syn_t, syn_l, strict=False):
            all_syn_t.append(t)
            all_syn_l.append(int(lab))
            all_src.append(-1)

    if all_syn_t:
        all_syn_t, all_syn_l = deduplicate_by_cosine(
            all_syn_t,
            all_syn_l,
            ref_texts,
            threshold=0.95,
        )
        all_src = all_src[: len(all_syn_t)]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"texts": all_syn_t, "labels": all_syn_l, "reference_size": len(ref_texts)})
    )


def _augmented_train_set(
    texts: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    cache_path: Path,
) -> tuple[list[str], list[int]]:
    """Original train fold + global synthetic toxic samples (shared pool)."""
    _ensure_global_augment_cache(texts, y, cache_path)
    cached = json.loads(cache_path.read_text())
    tr_texts = texts[train_idx].tolist()
    tr_labels = y[train_idx].tolist()
    syn_t = cached.get("texts", [])
    syn_l = cached.get("labels", [])
    return tr_texts + syn_t, tr_labels + [int(v) for v in syn_l]


def _train_head_only_fold(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    output_dir: Path,
    *,
    seed: int,
    max_epochs: int = 4,
) -> tuple:
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, ignore_mismatched_sizes=True
    )
    model.config.problem_type = "single_label_classification"
    freeze_head_only(model)

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    def _prep(texts, labels):
        ds = Dataset.from_dict({"text": texts, "label": labels})
        tok = ds.map(_tok, batched=True)
        drop = [c for c in tok.column_names if c not in ("input_ids", "attention_mask", "label")]
        if drop:
            tok = tok.remove_columns(drop)
        tok.set_format("torch")
        return tok

    tok_train = _prep(train_texts, train_labels)
    tok_val = _prep(val_texts, val_labels)

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_hf_metrics,
    )
    trainer.train()
    return model, tokenizer


def run_experiment_1(df: pd.DataFrame, texts: np.ndarray, y: np.ndarray) -> dict:
    logger.info("=" * 60)
    logger.info("Experiment 1 — Multi-pivot back-translation + head-only")
    logger.info("=" * 60)
    cache = ARTIFACT_DIR / "aug_multi_pivot_global.json"
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds: list[FoldMetrics] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, y)):
        inner_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.15,
            random_state=RANDOM_STATE + fold,
            stratify=y[train_idx],
        )
        tr_texts, tr_labels = _augmented_train_set(texts, y, inner_idx, cache)
        va_texts = texts[val_idx].tolist()
        te_texts = texts[test_idx].tolist()

        out = ARTIFACT_DIR / f"exp1_head_fold{fold}"
        model, tokenizer = _train_head_only_fold(
            tr_texts, tr_labels, va_texts, y[val_idx].tolist(), out, seed=RANDOM_STATE + fold
        )
        p_train = _predict_probs(model, tokenizer, tr_texts)
        p_val = _predict_probs(model, tokenizer, va_texts)
        p_test = _predict_probs(model, tokenizer, te_texts)
        folds.append(
            _score_split(
                np.asarray(tr_labels),
                y[val_idx],
                y[test_idx],
                p_train,
                p_val,
                p_test,
                fold=fold,
            )
        )
        logger.info(f"  Fold {fold}: F1_test={folds[-1].f1_test} gap_pp={folds[-1].gap_pp}")

    summary = _summarize_folds(folds)
    summary["experiment"] = "exp1_multi_pivot_head"
    summary["status"] = "PASS" if summary["all_gap_ok"] and summary["target_f1_hit"] else "PARTIAL"
    return summary


def run_experiment_2(texts: np.ndarray, y: np.ndarray) -> dict:
    logger.info("=" * 60)
    logger.info("Experiment 2 — Advanced TTA (frozen golden baseline)")
    logger.info("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_frozen_model(device)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds: list[FoldMetrics] = []
    w0, w1, w2 = TTA_WEIGHTS

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, y)):
        inner_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.15,
            random_state=RANDOM_STATE + fold,
            stratify=y[train_idx],
        )
        tr_list = texts[inner_idx].tolist()
        va_list = texts[val_idx].tolist()
        te_list = texts[test_idx].tolist()

        p_tr0 = _predict_probs(model, tokenizer, tr_list)
        p_va0 = _predict_probs(model, tokenizer, va_list)
        p_te0 = _predict_probs(model, tokenizer, te_list)

        tr_de = back_translate_texts(tr_list, pivot_lang="de", rate_limit_every=40, rate_limit_sleep_sec=0.3)
        va_de = back_translate_texts(va_list, pivot_lang="de", rate_limit_every=40, rate_limit_sleep_sec=0.3)
        te_de = back_translate_texts(te_list, pivot_lang="de", rate_limit_every=40, rate_limit_sleep_sec=0.3)
        p_tr1 = _predict_probs(model, tokenizer, tr_de)
        p_va1 = _predict_probs(model, tokenizer, va_de)
        p_te1 = _predict_probs(model, tokenizer, te_de)

        tr_fr = back_translate_texts(tr_list, pivot_lang="fr", rate_limit_every=40, rate_limit_sleep_sec=0.3)
        va_fr = back_translate_texts(va_list, pivot_lang="fr", rate_limit_every=40, rate_limit_sleep_sec=0.3)
        te_fr = back_translate_texts(te_list, pivot_lang="fr", rate_limit_every=40, rate_limit_sleep_sec=0.3)
        p_tr2 = _predict_probs(model, tokenizer, tr_fr)
        p_va2 = _predict_probs(model, tokenizer, va_fr)
        p_te2 = _predict_probs(model, tokenizer, te_fr)

        p_train = w0 * p_tr0 + w1 * p_tr1 + w2 * p_tr2
        p_val = w0 * p_va0 + w1 * p_va1 + w2 * p_va2
        p_test = w0 * p_te0 + w1 * p_te1 + w2 * p_te2

        folds.append(
            _score_split(
                y[inner_idx], y[val_idx], y[test_idx], p_train, p_val, p_test, fold=fold
            )
        )
        logger.info(f"  Fold {fold}: F1_test={folds[-1].f1_test} gap_pp={folds[-1].gap_pp}")

    summary = _summarize_folds(folds)
    summary["experiment"] = "exp2_advanced_tta"
    summary["tta_weights"] = list(TTA_WEIGHTS)
    summary["status"] = "PASS" if summary["all_gap_ok"] and summary["target_f1_hit"] else "PARTIAL"
    return summary


def run_experiment_3(df: pd.DataFrame, texts: np.ndarray, y: np.ndarray) -> dict:
    logger.info("=" * 60)
    logger.info("Experiment 3 — Meta-feature stacking (CLS + style meta)")
    logger.info("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_frozen_model(device)
    meta_all = _extended_meta(df).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds: list[FoldMetrics] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, y)):
        inner_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.15,
            random_state=RANDOM_STATE + fold,
            stratify=y[train_idx],
        )
        cls_train = _extract_cls_features(model, tokenizer, texts[inner_idx].tolist())
        cls_val = _extract_cls_features(model, tokenizer, texts[val_idx].tolist())
        cls_test = _extract_cls_features(model, tokenizer, texts[test_idx].tolist())

        X_train = np.hstack([cls_train, meta_all[inner_idx]])
        X_val = np.hstack([cls_val, meta_all[val_idx]])
        X_test = np.hstack([cls_test, meta_all[test_idx]])

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced", solver="lbfgs")
        clf.fit(X_train_s, y[inner_idx])
        p_train = clf.predict_proba(X_train_s)[:, 1]
        p_val = clf.predict_proba(X_val_s)[:, 1]
        p_test = clf.predict_proba(X_test_s)[:, 1]

        folds.append(
            _score_split(
                y[inner_idx], y[val_idx], y[test_idx], p_train, p_val, p_test, fold=fold
            )
        )
        logger.info(f"  Fold {fold}: F1_test={folds[-1].f1_test} gap_pp={folds[-1].gap_pp}")

    summary = _summarize_folds(folds)
    summary["experiment"] = "exp3_meta_stacking"
    summary["lr_C"] = 0.01
    summary["status"] = "PASS" if summary["all_gap_ok"] and summary["target_f1_hit"] else "PARTIAL"
    return summary


def _tta_probs(model, tokenizer, tr_list, va_list, te_list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w0, w1, w2 = TTA_WEIGHTS
    p_tr0 = _predict_probs(model, tokenizer, tr_list)
    p_va0 = _predict_probs(model, tokenizer, va_list)
    p_te0 = _predict_probs(model, tokenizer, te_list)
    p_tr1 = _predict_probs(
        model, tokenizer, back_translate_texts(tr_list, pivot_lang="de", rate_limit_every=40, rate_limit_sleep_sec=0.2)
    )
    p_va1 = _predict_probs(
        model, tokenizer, back_translate_texts(va_list, pivot_lang="de", rate_limit_every=40, rate_limit_sleep_sec=0.2)
    )
    p_te1 = _predict_probs(
        model, tokenizer, back_translate_texts(te_list, pivot_lang="de", rate_limit_every=40, rate_limit_sleep_sec=0.2)
    )
    p_tr2 = _predict_probs(
        model, tokenizer, back_translate_texts(tr_list, pivot_lang="fr", rate_limit_every=40, rate_limit_sleep_sec=0.2)
    )
    p_va2 = _predict_probs(
        model, tokenizer, back_translate_texts(va_list, pivot_lang="fr", rate_limit_every=40, rate_limit_sleep_sec=0.2)
    )
    p_te2 = _predict_probs(
        model, tokenizer, back_translate_texts(te_list, pivot_lang="fr", rate_limit_every=40, rate_limit_sleep_sec=0.2)
    )
    p_train = w0 * p_tr0 + w1 * p_tr1 + w2 * p_tr2
    p_val = w0 * p_va0 + w1 * p_va1 + w2 * p_va2
    p_test = w0 * p_te0 + w1 * p_te1 + w2 * p_te2
    return p_train, p_val, p_test


def _stacking_probs(
    model,
    tokenizer,
    meta_all: np.ndarray,
    y: np.ndarray,
    inner_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    texts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train = np.hstack(
        [_extract_cls_features(model, tokenizer, texts[inner_idx].tolist()), meta_all[inner_idx]]
    )
    X_val = np.hstack(
        [_extract_cls_features(model, tokenizer, texts[val_idx].tolist()), meta_all[val_idx]]
    )
    X_test = np.hstack(
        [_extract_cls_features(model, tokenizer, texts[test_idx].tolist()), meta_all[test_idx]]
    )
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train_s, y[inner_idx])
    return (
        clf.predict_proba(X_train_s)[:, 1],
        clf.predict_proba(scaler.transform(X_val))[:, 1],
        clf.predict_proba(scaler.transform(X_test))[:, 1],
    )


def run_experiment_4(
    best_key: str,
    texts: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
) -> dict:
    logger.info("=" * 60)
    logger.info(f"Experiment 4 — Ultra-fine threshold on best sprint: {best_key}")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ARTIFACT_DIR / "aug_multi_pivot_global.json"
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds: list[FoldMetrics] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, y)):
        inner_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.15,
            random_state=RANDOM_STATE + fold,
            stratify=y[train_idx],
        )
        tr_list = texts[inner_idx].tolist()
        va_list = texts[val_idx].tolist()
        te_list = texts[test_idx].tolist()

        if best_key == "exp2_advanced_tta":
            model, tokenizer = _load_frozen_model(device)
            p_train, p_val, p_test = _tta_probs(model, tokenizer, tr_list, va_list, te_list)
            y_train_arr = y[inner_idx]
        elif best_key == "exp3_meta_stacking":
            model, tokenizer = _load_frozen_model(device)
            meta_all = _extended_meta(df).values
            p_train, p_val, p_test = _stacking_probs(
                model, tokenizer, meta_all, y, inner_idx, val_idx, test_idx, texts
            )
            y_train_arr = y[inner_idx]
        else:
            tr_texts, tr_labels = _augmented_train_set(texts, y, inner_idx, cache)
            out = ARTIFACT_DIR / f"exp4_head_fold{fold}"
            model, tokenizer = _train_head_only_fold(
                tr_texts,
                tr_labels,
                va_list,
                y[val_idx].tolist(),
                out,
                seed=RANDOM_STATE + fold,
            )
            p_train = _predict_probs(model, tokenizer, tr_texts)
            p_val = _predict_probs(model, tokenizer, va_list)
            p_test = _predict_probs(model, tokenizer, te_list)
            y_train_arr = np.asarray(tr_labels)

        folds.append(
            _score_split(
                y_train_arr,
                y[val_idx],
                y[test_idx],
                p_train,
                p_val,
                p_test,
                fold=fold,
                min_t=0.05,
                max_t=0.30,
                step=0.001,
            )
        )
        logger.info(
            f"  Fold {fold}: F1_test={folds[-1].f1_test} t={folds[-1].threshold} gap_pp={folds[-1].gap_pp}"
        )

    summary = _summarize_folds(folds)
    summary["experiment"] = "exp4_ultra_fine_threshold"
    summary["base_experiment"] = best_key
    summary["threshold_range"] = [0.05, 0.30, 0.001]
    summary["status"] = "PASS" if summary["all_gap_ok"] and summary["target_f1_hit"] else "PARTIAL"
    return summary


def run_golden_baseline_cv(texts: np.ndarray, y: np.ndarray) -> dict:
    logger.info("Golden Baseline reference (5-fold CV, frozen)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_frozen_model(device)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds: list[FoldMetrics] = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, y)):
        inner_idx, val_idx = train_test_split(
            train_idx, test_size=0.15, random_state=RANDOM_STATE + fold, stratify=y[train_idx]
        )
        p_train = _predict_probs(model, tokenizer, texts[inner_idx].tolist())
        p_val = _predict_probs(model, tokenizer, texts[val_idx].tolist())
        p_test = _predict_probs(model, tokenizer, texts[test_idx].tolist())
        folds.append(
            _score_split(y[inner_idx], y[val_idx], y[test_idx], p_train, p_val, p_test, fold=fold)
        )
    summary = _summarize_folds(folds)
    summary["experiment"] = "golden_baseline_cv"
    return summary


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_STATE)

    df, y_series, texts = _load_data()
    y = y_series.values

    results = {
        "target_f1_weighted": TARGET_F1,
        "max_gap_pp": MAX_GAP * 100,
        "n_folds": N_FOLDS,
        "golden_baseline_cv": run_golden_baseline_cv(texts, y),
    }

    results["exp2"] = run_experiment_2(texts, y)
    results["exp3"] = run_experiment_3(df, texts, y)
    results["exp1"] = run_experiment_1(df, texts, y)

    candidates = {
        "exp1_multi_pivot_head": results["exp1"]["f1_test_mean"],
        "exp2_advanced_tta": results["exp2"]["f1_test_mean"],
        "exp3_meta_stacking": results["exp3"]["f1_test_mean"],
    }
    best_key = max(candidates, key=candidates.get)
    results["best_experiment"] = best_key
    results["exp4"] = run_experiment_4(best_key, texts, y, df)

    comparison = []
    for label, key in [
        ("Golden Baseline (CV)", "golden_baseline_cv"),
        ("Exp1 Multi-Pivot + Head", "exp1"),
        ("Exp2 Advanced TTA", "exp2"),
        ("Exp3 Meta Stacking", "exp3"),
        ("Exp4 Ultra-Fine Thresh", "exp4"),
    ]:
        block = results[key]
        comparison.append(
            {
                "sprint": label,
                "f1_test_mean": block["f1_test_mean"],
                "f1_test_std": block["f1_test_std"],
                "gap_pp_mean": block["gap_pp_mean"],
                "gap_pp_max": block["gap_pp_max"],
                "all_gap_ok": block["all_gap_ok"],
                "f1_target_hit": block["target_f1_hit"],
                "status": "PASS" if block["all_gap_ok"] and block["target_f1_hit"] else (
                    "FAIL_GAP" if not block["all_gap_ok"] else "FAIL_F1"
                ),
            }
        )
    results["comparison_table"] = comparison

    out_json = REPORT_DIR / "sprint_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved {out_json}")

    lines = [
        "# Notebook 13 — Sprint Comparison",
        "",
        "| Sprint | Mean F1 (test) | Gap pp (mean) | Gap OK | F1 ≥ 0.80 | Status |",
        "|--------|----------------|---------------|--------|-----------|--------|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['sprint']} | {row['f1_test_mean']:.4f} ± {row['f1_test_std']:.4f} | "
            f"{row['gap_pp_mean']:.2f} | {'✅' if row['all_gap_ok'] else '❌'} | "
            f"{'✅' if row['f1_target_hit'] else '❌'} | {row['status']} |"
        )
    (REPORT_DIR / "comparison_table.md").write_text("\n".join(lines))
    logger.info("Done.")


if __name__ == "__main__":
    main()
