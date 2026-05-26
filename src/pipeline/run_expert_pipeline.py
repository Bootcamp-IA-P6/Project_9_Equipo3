"""
Phase 5 Expert pipeline: Toxic-BERT (head-only) + bottleneck LR + hybrid ensemble.

Run from repo root:
    uv sync --extra hf --extra train
    uv run python -m src.pipeline.run_expert_pipeline
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
from datasets import Dataset
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_raw_data
from src.evaluation.expert_report import write_expert_report
from src.evaluation.threshold_tuning import predict_with_threshold, search_best_threshold
from src.features.augmentation import augment_toxic_train
from src.models.hybrid_ensemble import (
    StableLRModel,
    evaluate_ensemble,
    fit_lr_with_gap_control,
    save_ensemble_meta,
    soft_vote_probs,
    tune_ensemble_threshold,
)
from src.models.transformer_trainer import train_transformer_stable
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_hf_dataset(X: pd.Series, y: pd.Series) -> Dataset:
    return Dataset.from_pandas(
        pd.DataFrame({"text": X.values, "label": y.astype(int).values})
    )


def _lr_metrics_with_threshold(
    model: StableLRModel,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    *,
    gap_meta: dict,
) -> dict:
    y_val_arr = np.asarray(y_val).astype(int)
    y_test_arr = np.asarray(y_test).astype(int)
    y_train_arr = np.asarray(y_train).astype(int)

    val_probs = model.predict_proba(X_val)[:, 1]
    threshold, _ = search_best_threshold(y_val_arr, val_probs, metric="f1_toxic")

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    train_preds = predict_with_threshold(train_probs, threshold)
    test_preds = predict_with_threshold(test_probs, threshold)

    f1_toxic_train = float(f1_score(y_train_arr, train_preds, pos_label=1, zero_division=0))
    f1_toxic_test = float(f1_score(y_test_arr, test_preds, pos_label=1, zero_division=0))
    gap_toxic = abs(f1_toxic_train - f1_toxic_test)
    f1_weighted_train = float(
        f1_score(y_train_arr, train_preds, average="weighted", zero_division=0)
    )
    f1_weighted_test = float(
        f1_score(y_test_arr, test_preds, average="weighted", zero_division=0)
    )
    gap_weighted = abs(f1_weighted_train - f1_weighted_test)

    return {
        "model": "LR-TFIDF-expert",
        "C": gap_meta["C"],
        "max_features": int(gap_meta.get("max_features", 250)),
        "min_df": int(gap_meta.get("min_df", 3)),
        "threshold": round(threshold, 4),
        "f1_weighted": round(f1_weighted_test, 4),
        "f1_toxic": round(f1_toxic_test, 4),
        "f1_toxic_train": round(f1_toxic_train, 4),
        "train_test_gap_toxic": round(gap_toxic, 4),
        "train_test_gap_toxic_pp": round(gap_toxic * 100, 2),
        "gap_toxic_ok": gap_toxic < 0.05,
        "f1_train": round(f1_weighted_train, 4),
        "train_test_gap": round(gap_weighted, 4),
        "train_test_gap_pp": round(gap_weighted * 100, 2),
        "roc_auc": round(float(roc_auc_score(y_test_arr, test_probs)), 4),
        "test_probs": test_probs,
        "val_probs": val_probs,
    }


def run_expert_pipeline(*, config_path: Path | None = None) -> dict:
    cfg_path = config_path or (PROJECT_ROOT / "configs" / "expert_training.yaml")
    cfg = yaml.safe_load(open(cfg_path))

    rand = int(cfg["pipeline"]["random_state"])
    test_size = float(cfg["pipeline"]["test_size"])
    val_size = float(cfg["pipeline"]["val_size"])
    target = cfg["data"]["target_binary"]
    raw_path = PROJECT_ROOT / cfg["data"]["raw_path"]

    out_cfg = cfg["output"]
    reports_dir = PROJECT_ROOT / out_cfg["reports_dir"]
    bert_dir = PROJECT_ROOT / out_cfg["transformer_dir"]
    lr_path = PROJECT_ROOT / out_cfg["lr_path"]
    meta_path = PROJECT_ROOT / out_cfg["ensemble_meta_path"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 60)
    logger.info(f"EXPERT PIPELINE (Phase 5) — run={run_id}")
    logger.info("=" * 60)

    df = load_raw_data(raw_path)
    X = df[cfg["data"].get("text_column", "Text")]
    y = df[target].astype(int)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rand, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        random_state=rand,
        stratify=y_trainval,
    )

    cfg_aug = cfg
    cfg_aug.setdefault("augmentation", {})["enabled"] = True
    pivot = cfg_aug.get("augmentation", {}).get("pivot_lang", "de")

    logger.info(f"Augmentation EN→{pivot.upper()}→EN (toxic only)")
    X_train_aug, y_train_aug = augment_toxic_train(X_train, y_train, cfg_aug, seed=rand)

    aug_info = {
        "enabled": True,
        "strategy": "back_translation",
        "pivot_lang": pivot,
        "train_size_before": len(X_train),
        "train_size_after": len(X_train_aug),
        "added_samples": len(X_train_aug) - len(X_train),
    }

    y_test_arr = y_test.astype(int).values
    y_val_arr = y_val.astype(int).values
    max_gap = float(cfg["pipeline"].get("max_train_test_gap", 0.05))

    all_metrics: dict = {"run_id": run_id, "augmentation": aug_info, "config": str(cfg_path)}

    lr_cfg = cfg["logistic_regression"]
    tfidf_cfg = lr_cfg.get("tfidf", {})

    logger.info("LR-TFIDF (max_features=250) + gap search")
    use_orig_gap = lr_cfg.get("gap_search", {}).get("use_original_train_for_gap", True)
    lr_model, lr_gap_meta = fit_lr_with_gap_control(
        X_train_aug,
        y_train_aug,
        X_test,
        y_test,
        lr_cfg,
        tfidf_cfg,
        max_gap=max_gap,
        X_train_gap=X_train if use_orig_gap else None,
        y_train_gap=y_train if use_orig_gap else None,
    )
    lr_model.save(lr_path)
    all_metrics["lr_gap_search"] = lr_gap_meta

    lr_metrics = _lr_metrics_with_threshold(
        lr_model,
        X_train if use_orig_gap else X_train_aug,
        y_train if use_orig_gap else y_train_aug,
        X_val,
        y_val,
        X_test,
        y_test,
        gap_meta=lr_gap_meta,
    )
    lr_probs_test = lr_metrics.pop("test_probs")
    lr_probs_val = lr_metrics.pop("val_probs")
    all_metrics["logistic_regression"] = lr_metrics

    logger.info("Toxic-BERT — head-only fine-tune + val threshold tuning")
    hf_train = _build_hf_dataset(X_train_aug, y_train_aug)
    hf_val = _build_hf_dataset(X_val, y_val)
    hf_test = _build_hf_dataset(X_test, y_test)

    bert_result = train_transformer_stable(
        hf_train,
        hf_val,
        hf_test,
        y_test_arr,
        y_val_arr,
        cfg,
        bert_dir,
        seed=rand,
        model_label="Toxic-BERT-expert",
    )
    bert_metrics = bert_result["metrics"]
    bert_probs_test = bert_result["test_probs"]
    bert_probs_val = bert_result["val_probs"]
    all_metrics["transformer"] = bert_metrics

    ens_cfg = cfg["ensemble"]
    bw = float(ens_cfg.get("bert_weight", 0.7))
    lw = float(ens_cfg.get("lr_weight", 0.3))

    ens_th_cfg = ens_cfg.get("threshold_tuning", {})
    if ens_th_cfg.get("enabled", True):
        ens_threshold, _ = tune_ensemble_threshold(
            bert_probs_val,
            lr_probs_val,
            y_val_arr,
            bert_weight=bw,
            lr_weight=lw,
            metric=ens_th_cfg.get("metric", "f1_toxic"),
        )
    else:
        ens_threshold = 0.5

    ensemble_metrics = evaluate_ensemble(
        bert_probs_test,
        lr_probs_test,
        y_test_arr,
        bert_weight=bw,
        lr_weight=lw,
        model_name="Hybrid-ToxicBERT+LR",
        threshold=ens_threshold,
    )

    ens_probs_train_bert = _bert_probs_on_texts(bert_result, X_train_aug)
    ens_probs_train_lr = lr_model.predict_proba(X_train_aug)[:, 1]
    ens_train = evaluate_ensemble(
        ens_probs_train_bert,
        ens_probs_train_lr,
        y_train_aug.astype(int).values,
        bert_weight=bw,
        lr_weight=lw,
        model_name="Hybrid-train",
        threshold=ens_threshold,
    )
    gap_toxic = abs(ens_train["f1_toxic"] - ensemble_metrics["f1_toxic"])
    ensemble_metrics["f1_toxic_train"] = ens_train["f1_toxic"]
    ensemble_metrics["train_test_gap_toxic"] = round(gap_toxic, 4)
    ensemble_metrics["train_test_gap_toxic_pp"] = round(gap_toxic * 100, 2)
    ensemble_metrics["gap_toxic_ok"] = gap_toxic < 0.05
    ensemble_metrics["f1_train"] = ens_train["f1_weighted"]
    ensemble_metrics["train_test_gap"] = round(
        abs(ens_train["f1_weighted"] - ensemble_metrics["f1_weighted"]), 4
    )
    ensemble_metrics["train_test_gap_pp"] = round(
        ensemble_metrics["train_test_gap"] * 100, 2
    )
    all_metrics["ensemble"] = ensemble_metrics

    save_ensemble_meta(
        meta_path,
        {
            "run_id": run_id,
            "bert_dir": str(bert_dir),
            "lr_path": str(lr_path),
            "ensemble": ens_cfg,
            "bert_threshold": bert_metrics.get("threshold"),
            "lr_threshold": lr_metrics.get("threshold"),
            "ensemble_threshold": ens_threshold,
        },
    )

    summary_rows = [
        {k: v for k, v in bert_metrics.items() if k != "gap_toxic_ok"},
        lr_metrics,
        ensemble_metrics,
    ]
    pd.DataFrame(summary_rows).to_csv(reports_dir / f"expert_summary_{run_id}.csv", index=False)

    report_path = reports_dir / f"expert_run_{run_id}.json"
    with open(report_path, "w") as f:
        json.dump(_json_safe(all_metrics), f, indent=2)

    md_path = reports_dir / f"integrated_report_{run_id}.md"
    write_expert_report(all_metrics, md_path)
    logger.info(f"Expert report: {md_path}")

    _print_expert_targets(all_metrics)
    return all_metrics


def _bert_probs_on_texts(bert_result: dict, X: pd.Series) -> np.ndarray:
    trainer = bert_result["trainer"]
    tokenizer = bert_result["tokenizer"]
    from datasets import Dataset

    ds = Dataset.from_pandas(pd.DataFrame({"text": X.values, "label": [0] * len(X)}))

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tok = ds.map(_tok, batched=True)
    drop_cols = [c for c in tok.column_names if c not in ("input_ids", "attention_mask", "label")]
    if drop_cols:
        tok = tok.remove_columns(drop_cols)
    tok.set_format("torch")
    import torch

    out = trainer.predict(tok)
    return torch.softmax(torch.tensor(out.predictions), dim=1)[:, 1].numpy()


def _json_safe(obj):
    if isinstance(obj, dict):
        return {
            k: _json_safe(v)
            for k, v in obj.items()
            if k not in ("ensemble_probs", "ensemble_preds")
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _print_expert_targets(all_metrics: dict) -> None:
    logger.info("=" * 60)
    for key in ("transformer", "logistic_regression", "ensemble"):
        m = all_metrics.get(key)
        if not m:
            continue
        f1t = m.get("f1_toxic", 0)
        gap = m.get("train_test_gap_toxic", 0.99)
        ok_f1 = "✅" if f1t > 0.75 else "⚠️"
        ok_gap = "✅" if m.get("gap_toxic_ok", gap < 0.05) else "⚠️"
        logger.info(
            f"{m.get('model', key)}: F1-toxic={f1t:.4f} {ok_f1} | "
            f"toxic gap={gap:.4f} {ok_gap} | threshold={m.get('threshold', 0.5)}"
        )
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Phase 5 expert Toxic-BERT + LR pipeline")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    run_expert_pipeline(
        config_path=Path(args.config) if args.config else None,
    )


if __name__ == "__main__":
    main()
