"""
Clean-Signal Dual-Input Hybrid: Toxic-BERT on raw Text + LR on clean_text + metadata.

    uv run python -m src.pipeline.run_hybrid_clean_pipeline
    uv run python -m src.pipeline.run_hybrid_clean_pipeline --skip-augmentation
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dual_loader import load_dual_track_data
from src.evaluation.hybrid_clean_report import write_hybrid_clean_report
from src.evaluation.threshold_tuning import predict_with_threshold, search_best_threshold
from src.features.augmentation import augment_toxic_train
from src.features.metadata_features import DEFAULT_METADATA_COLUMNS
from src.features.text_preprocessor import TextPreprocessor
from src.models.hybrid_ensemble import (
    compute_performance_weights,
    evaluate_ensemble,
    save_ensemble_meta,
    tune_ensemble_threshold,
)
from src.models.metadata_lr import MetadataLRModel, fit_metadata_lr_with_gap_control
from src.models.transformer_trainer import infer_transformer_probs, train_transformer_stable
from src.utils.logger import get_logger
from datasets import Dataset

logger = get_logger(__name__)


def _build_hf_dataset(X: pd.Series, y: pd.Series) -> Dataset:
    return Dataset.from_pandas(
        pd.DataFrame({"text": X.values, "label": y.astype(int).values})
    )


def _meta_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DEFAULT_METADATA_COLUMNS if c in df.columns]
    return df[cols].copy()


def augment_dual_track(
    X_raw_train: pd.Series,
    X_clean_train: pd.Series,
    meta_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict,
    preprocessor: TextPreprocessor,
    *,
    seed: int,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Back-translate toxic raw samples; preprocess new rows for LR track."""
    from src.features.metadata_features import extract_metadata_features

    X_raw_aug, y_aug = augment_toxic_train(X_raw_train, y_train, cfg, seed=seed)
    if len(X_raw_aug) <= len(X_raw_train):
        return X_raw_aug, X_clean_train, meta_train, y_aug

    new_raw = X_raw_aug.iloc[len(X_raw_train) :].reset_index(drop=True)
    new_clean = preprocessor.transform(new_raw)
    new_clean = pd.Series(new_clean.values, name="clean_text")
    new_meta = extract_metadata_features(
        pd.DataFrame({"Text": new_raw.values}),
        text_column="Text",
    )

    X_clean_aug = pd.concat(
        [X_clean_train.reset_index(drop=True), new_clean],
        ignore_index=True,
    )
    meta_aug = pd.concat(
        [meta_train.reset_index(drop=True), new_meta],
        ignore_index=True,
    )
    return X_raw_aug, X_clean_aug, meta_aug, y_aug


def _branch_metrics(
    y_train,
    y_test,
    y_val,
    train_probs,
    val_probs,
    test_probs,
    *,
    model_name: str,
    gap_meta: dict | None = None,
    fixed_threshold: float | None = None,
) -> dict:
    y_train_arr = np.asarray(y_train).astype(int)
    y_val_arr = np.asarray(y_val).astype(int)
    y_test_arr = np.asarray(y_test).astype(int)

    if fixed_threshold is not None:
        threshold = fixed_threshold
    else:
        threshold, _ = search_best_threshold(y_val_arr, val_probs, metric="f1_weighted")

    train_preds = predict_with_threshold(train_probs, threshold)
    test_preds = predict_with_threshold(test_probs, threshold)

    f1_train = float(f1_score(y_train_arr, train_preds, average="weighted", zero_division=0))
    f1_test = float(f1_score(y_test_arr, test_preds, average="weighted", zero_division=0))
    gap = abs(f1_train - f1_test)

    out = {
        "model": model_name,
        "threshold": round(threshold, 4),
        "f1_weighted": round(f1_test, 4),
        "f1_toxic": round(float(f1_score(y_test_arr, test_preds, pos_label=1, zero_division=0)), 4),
        "f1_train": round(f1_train, 4),
        "train_test_gap": round(gap, 4),
        "train_test_gap_pp": round(gap * 100, 2),
        "gap_ok": gap < 0.05,
        "roc_auc": round(float(roc_auc_score(y_test_arr, test_probs)), 4),
    }
    if gap_meta:
        out["lr_C"] = gap_meta.get("C")
        out["max_features"] = gap_meta.get("max_features")
    return out


def run_hybrid_clean_pipeline(
    *,
    config_path: Path | None = None,
    skip_augmentation: bool = False,
) -> dict:
    cfg_path = config_path or (PROJECT_ROOT / "configs" / "hybrid_clean_training.yaml")
    cfg = yaml.safe_load(open(cfg_path))

    rand = int(cfg["pipeline"]["random_state"])
    test_size = float(cfg["pipeline"]["test_size"])
    val_size = float(cfg["pipeline"]["val_size"])
    max_gap = float(cfg["pipeline"].get("max_train_test_gap", 0.05))
    target_f1 = float(cfg["pipeline"].get("target_f1_weighted", 0.80))
    target_col = cfg["data"]["target_binary"]
    text_col = cfg["data"].get("text_column", "Text")

    out_cfg = cfg["output"]
    reports_dir = PROJECT_ROOT / out_cfg["reports_dir"]
    lr_path = PROJECT_ROOT / out_cfg["lr_path"]
    meta_path = PROJECT_ROOT / out_cfg["ensemble_meta_path"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 60)
    logger.info(f"CLEAN-SIGNAL HYBRID — run={run_id}")
    logger.info("=" * 60)

    df = load_dual_track_data(
        PROJECT_ROOT / cfg["data"]["raw_path"],
        processed_preprocessed=cfg["data"]["processed_preprocessed"],
        processed_stats=cfg["data"]["processed_stats"],
        target=target_col,
        text_column=text_col,
        id_column=cfg["data"].get("id_column", "CommentId"),
        features_config=cfg["data"]["features_config"],
        project_root=PROJECT_ROOT,
    )

    y = df[target_col].astype(int)
    idx_trainval, idx_test = train_test_split(
        df.index, test_size=test_size, random_state=rand, stratify=y
    )
    y_trainval = y.loc[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size,
        random_state=rand,
        stratify=y_trainval,
    )

    def _slice(index):
        return {
            "raw": df.loc[index, text_col],
            "clean": df.loc[index, "clean_text"],
            "meta": _meta_frame(df.loc[index]),
            "y": y.loc[index],
        }

    tr, va, te = _slice(idx_train), _slice(idx_val), _slice(idx_test)

    preprocessor = TextPreprocessor(
        config_path=str(PROJECT_ROOT / cfg["data"]["features_config"])
    )

    aug_info = {"enabled": False}
    if not skip_augmentation and cfg.get("augmentation", {}).get("enabled", True):
        logger.info("Dual-track augmentation (raw BT + clean preprocess)")
        tr["raw"], tr["clean"], tr["meta"], tr["y"] = augment_dual_track(
            tr["raw"],
            tr["clean"],
            tr["meta"],
            tr["y"],
            cfg,
            preprocessor,
            seed=rand,
        )
        aug_info = {
            "enabled": True,
            "pivot_lang": cfg["augmentation"].get("pivot_lang", "de"),
            "train_size_after": len(tr["y"]),
        }

    y_test_arr = te["y"].astype(int).values
    y_val_arr = va["y"].astype(int).values

    all_metrics: dict = {
        "run_id": run_id,
        "config": str(cfg_path),
        "target_f1_weighted": target_f1,
        "augmentation": aug_info,
        "data_sources": {
            "raw": cfg["data"]["raw_path"],
            "processed_stats": cfg["data"]["processed_stats"],
            "processed_preprocessed": cfg["data"]["processed_preprocessed"],
        },
    }

    # ── LR (clean + metadata) ─────────────────────────────────────────────
    lr_cfg = cfg["logistic_regression"]
    tfidf_cfg = lr_cfg.get("tfidf", {})
    use_orig_gap = lr_cfg.get("gap_search", {}).get("use_original_train_for_gap", True)

    logger.info("Training Metadata LR on clean_text + stats features")
    lr_model, lr_gap_meta = fit_metadata_lr_with_gap_control(
        tr["clean"],
        tr["meta"],
        tr["y"],
        te["clean"],
        te["meta"],
        te["y"],
        lr_cfg,
        tfidf_cfg,
        max_gap=max_gap,
        X_train_gap_clean=df.loc[idx_train, "clean_text"] if use_orig_gap else tr["clean"],
        meta_train_gap=_meta_frame(df.loc[idx_train]) if use_orig_gap else tr["meta"],
        y_train_gap=y.loc[idx_train] if use_orig_gap else tr["y"],
    )
    lr_model.save(lr_path)
    all_metrics["lr_gap_search"] = lr_gap_meta

    lr_val_probs = lr_model.predict_proba(va["clean"], va["meta"])[:, 1]
    lr_test_probs = lr_model.predict_proba(te["clean"], te["meta"])[:, 1]
    lr_train_probs = lr_model.predict_proba(
        df.loc[idx_train, "clean_text"] if use_orig_gap else tr["clean"],
        _meta_frame(df.loc[idx_train]) if use_orig_gap else tr["meta"],
    )[:, 1]

    lr_metrics = _branch_metrics(
        y.loc[idx_train] if use_orig_gap else tr["y"],
        te["y"],
        va["y"],
        lr_train_probs,
        lr_val_probs,
        lr_test_probs,
        model_name="LR-clean+meta",
        gap_meta=lr_gap_meta,
    )
    all_metrics["logistic_regression"] = lr_metrics

    # ── Toxic-BERT (raw) ────────────────────────────────────────────────────
    bert_cfg = cfg["transformer"]
    ckpt = PROJECT_ROOT / bert_cfg.get("reuse_checkpoint", "models/expert_toxic_bert")
    fixed_t = float(bert_cfg.get("fixed_threshold", 0.33))
    train_if_missing = bool(bert_cfg.get("train_if_missing", False))

    if ckpt.exists() and (ckpt / "config.json").exists():
        logger.info(f"Toxic-BERT inference from checkpoint (threshold={fixed_t})")
        bert_val_probs = infer_transformer_probs(ckpt, va["raw"], max_length=int(bert_cfg.get("max_length", 128)))
        bert_test_probs = infer_transformer_probs(ckpt, te["raw"], max_length=int(bert_cfg.get("max_length", 128)))
        bert_train_probs = infer_transformer_probs(ckpt, tr["raw"], max_length=int(bert_cfg.get("max_length", 128)))
        bert_metrics = _branch_metrics(
            tr["y"],
            te["y"],
            va["y"],
            bert_train_probs,
            bert_val_probs,
            bert_test_probs,
            model_name="Toxic-BERT-raw",
            fixed_threshold=fixed_t,
        )
        bert_metrics["checkpoint"] = str(ckpt)
        bert_metrics["reused_checkpoint"] = True
    elif train_if_missing:
        logger.info("Training Toxic-BERT (no checkpoint found)")
        expert_cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "expert_training.yaml"))
        hf_train = _build_hf_dataset(tr["raw"], tr["y"])
        hf_val = _build_hf_dataset(va["raw"], va["y"])
        hf_test = _build_hf_dataset(te["raw"], te["y"])
        bert_result = train_transformer_stable(
            hf_train,
            hf_val,
            hf_test,
            y_test_arr,
            y_val_arr,
            expert_cfg,
            ckpt,
            seed=rand,
            model_label="Toxic-BERT-raw",
        )
        bert_metrics = bert_result["metrics"]
        fixed_t = float(bert_metrics.get("threshold", fixed_t))
        bert_val_probs = bert_result["val_probs"]
        bert_test_probs = bert_result["test_probs"]
        bert_train_probs = infer_transformer_probs(ckpt, tr["raw"])
    else:
        raise FileNotFoundError(
            f"Toxic-BERT checkpoint not found at {ckpt}. "
            "Run expert pipeline first or set transformer.train_if_missing: true"
        )

    all_metrics["transformer"] = bert_metrics

    # ── Dynamic weights + ensemble ──────────────────────────────────────────
    ens_cfg = cfg["ensemble"]
    bw, lw, weight_info = compute_performance_weights(
        bert_val_probs,
        lr_val_probs,
        y_val_arr,
        bert_threshold=fixed_t,
        lr_threshold=lr_metrics["threshold"],
        metric=ens_cfg.get("weight_metric", "f1_weighted"),
        min_lr_weight=float(ens_cfg.get("min_lr_weight", 0.15)),
        max_lr_weight=float(ens_cfg.get("max_lr_weight", 0.45)),
    )
    all_metrics["ensemble_weights"] = weight_info
    logger.info(f"Dynamic weights — BERT={bw:.3f} LR={lw:.3f}")

    th_cfg = ens_cfg.get("threshold_tuning", {})
    if th_cfg.get("enabled", True):
        ens_threshold, _ = tune_ensemble_threshold(
            bert_val_probs,
            lr_val_probs,
            y_val_arr,
            bert_weight=bw,
            lr_weight=lw,
            metric=th_cfg.get("metric", "f1_weighted"),
        )
    else:
        ens_threshold = 0.5

    ensemble_metrics = evaluate_ensemble(
        bert_test_probs,
        lr_test_probs,
        y_test_arr,
        bert_weight=bw,
        lr_weight=lw,
        model_name="Clean-Signal-Hybrid",
        threshold=ens_threshold,
    )

    from src.models.hybrid_ensemble import soft_vote_probs

    ens_train_probs = soft_vote_probs(
        bert_train_probs,
        lr_model.predict_proba(tr["clean"], tr["meta"])[:, 1],
        bw,
        lw,
    )
    ens_train_preds = predict_with_threshold(ens_train_probs, ens_threshold)
    y_tr_arr = tr["y"].astype(int).values
    f1_train_ens = float(f1_score(y_tr_arr, ens_train_preds, average="weighted", zero_division=0))
    gap_ens = abs(f1_train_ens - ensemble_metrics["f1_weighted"])
    ensemble_metrics["f1_train"] = round(f1_train_ens, 4)
    ensemble_metrics["train_test_gap"] = round(gap_ens, 4)
    ensemble_metrics["train_test_gap_pp"] = round(gap_ens * 100, 2)
    ensemble_metrics["gap_ok"] = gap_ens < 0.05
    ensemble_metrics["bert_weight"] = bw
    ensemble_metrics["lr_weight"] = lw
    ensemble_metrics["target_f1_hit"] = ensemble_metrics["f1_weighted"] >= target_f1
    all_metrics["ensemble"] = ensemble_metrics

    save_ensemble_meta(
        meta_path,
        {
            "run_id": run_id,
            "bert_checkpoint": str(ckpt),
            "lr_path": str(lr_path),
            "weights": weight_info,
            "thresholds": {
                "bert": fixed_t,
                "lr": lr_metrics["threshold"],
                "ensemble": ens_threshold,
            },
        },
    )

    report_path = reports_dir / f"hybrid_clean_run_{run_id}.json"
    with open(report_path, "w") as f:
        json.dump(_json_safe(all_metrics), f, indent=2)

    md_path = reports_dir / f"integrated_report_{run_id}.md"
    write_hybrid_clean_report(all_metrics, md_path)

    _print_summary(all_metrics, target_f1)
    return all_metrics


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if k not in ("ensemble_probs", "ensemble_preds")}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _print_summary(metrics: dict, target: float) -> None:
    logger.info("=" * 60)
    ens = metrics.get("ensemble", {})
    hit = ens.get("f1_weighted", 0) >= target
    logger.info(
        f"INTEGRATED F1 weighted={ens.get('f1_weighted', 0):.4f} "
        f"{'✅ TARGET' if hit else '⚠️ below'} (target {target})"
    )
    for key in ("transformer", "logistic_regression", "ensemble"):
        m = metrics.get(key, {})
        if m:
            logger.info(
                f"  {m.get('model', key)}: F1w={m.get('f1_weighted')} "
                f"F1tox={m.get('f1_toxic')} gap_pp={m.get('train_test_gap_pp')} "
                f"gap_ok={m.get('gap_ok')}"
            )
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Clean-Signal dual-input hybrid pipeline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-augmentation", action="store_true")
    args = parser.parse_args()
    run_hybrid_clean_pipeline(
        config_path=Path(args.config) if args.config else None,
        skip_augmentation=args.skip_augmentation,
    )


if __name__ == "__main__":
    main()
