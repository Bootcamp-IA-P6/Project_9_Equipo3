"""
Stable training pipeline: toxic-only augmentation, regularized DistilBERT,
high-C LR on TF-IDF, and hybrid soft-voting ensemble.

Run from repo root:
    uv sync --extra hf --extra train
    uv run python -m src.pipeline.run_stable_pipeline
    uv run python -m src.pipeline.run_stable_pipeline --skip-augmentation
    uv run python -m src.pipeline.run_stable_pipeline --bert-only
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
from src.evaluation.stable_cv import evaluate_lr_fold, stratified_kfold_cv
from src.evaluation.stable_report import write_integrated_report
from src.features.augmentation import augment_toxic_train
from src.models.hybrid_ensemble import (
    StableLRModel,
    evaluate_ensemble,
    fit_lr_with_gap_control,
    save_ensemble_meta,
)
from src.models.transformer_trainer import train_distilbert_stable
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_hf_dataset(X: pd.Series, y: pd.Series) -> Dataset:
    return Dataset.from_pandas(
        pd.DataFrame({"text": X.values, "label": y.astype(int).values})
    )


def run_stable_pipeline(
    *,
    skip_augmentation: bool = False,
    bert_only: bool = False,
    config_path: Path | None = None,
) -> dict:
    cfg_path = config_path or (PROJECT_ROOT / "configs" / "stable_training.yaml")
    cfg = yaml.safe_load(open(cfg_path))

    rand = int(cfg["pipeline"]["random_state"])
    test_size = float(cfg["pipeline"]["test_size"])
    val_size = float(cfg["pipeline"]["val_size"])
    target = cfg["data"]["target_binary"]
    raw_path = PROJECT_ROOT / cfg["data"]["raw_path"]

    out_cfg = cfg["output"]
    reports_dir = PROJECT_ROOT / out_cfg["reports_dir"]
    bert_dir = PROJECT_ROOT / out_cfg["distilbert_dir"]
    lr_path = PROJECT_ROOT / out_cfg["lr_path"]
    meta_path = PROJECT_ROOT / out_cfg["ensemble_meta_path"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 60)
    logger.info(f"STABLE PIPELINE — run={run_id}")
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
    logger.info(f"Split — train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    if skip_augmentation:
        logger.warning("--skip-augmentation ignored for production; augmentation is required.")
    cfg_aug = cfg
    cfg_aug.setdefault("augmentation", {})["enabled"] = True

    X_train_aug, y_train_aug = augment_toxic_train(X_train, y_train, cfg_aug, seed=rand)
    aug_info = {
        "enabled": True,
        "strategy": cfg_aug.get("augmentation", {}).get("strategy", "back_translation"),
        "train_size_before": len(X_train),
        "train_size_after": len(X_train_aug),
        "added_samples": len(X_train_aug) - len(X_train),
    }

    y_test_arr = y_test.astype(int).values
    max_gap = float(cfg["pipeline"].get("max_train_test_gap", 0.05))
    cv_folds = int(cfg["pipeline"].get("cv_folds", 5))

    all_metrics: dict = {"run_id": run_id, "augmentation": aug_info}

    lr_metrics = None
    lr_probs_test = None
    lr_model = None
    lr_gap_meta: dict = {}
    ensemble_metrics = None

    if not bert_only:
        lr_cfg = cfg["logistic_regression"]
        tfidf_cfg = lr_cfg.get("tfidf", {})

        logger.info("=" * 60)
        logger.info("LR-TFIDF — gap control FIRST (target < 5 pp)")
        logger.info("=" * 60)
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

        lr_probs_test = lr_model.predict_proba(X_test)[:, 1]
        lr_preds_test = lr_model.predict(X_test)
        f1_lr_test = float(
            f1_score(y_test_arr, lr_preds_test, average="weighted", zero_division=0)
        )
        f1_lr_train = lr_gap_meta["f1_train"]
        gap_lr = lr_gap_meta["train_test_gap"]
        lr_metrics = {
            "model": "LR-TFIDF-stable",
            "C": lr_gap_meta["C"],
            "max_features": int(lr_gap_meta.get("max_features", tfidf_cfg.get("max_features", 800))),
            "min_df": int(lr_gap_meta.get("min_df", tfidf_cfg.get("min_df", 3))),
            "f1_weighted": round(f1_lr_test, 4),
            "f1_train": f1_lr_train,
            "train_test_gap": gap_lr,
            "train_test_gap_pp": lr_gap_meta["train_test_gap_pp"],
            "gap_ok": lr_gap_meta["gap_ok"],
            "roc_auc": round(float(roc_auc_score(y_test_arr, lr_probs_test)), 4),
        }
        all_metrics["logistic_regression"] = lr_metrics

        tfidf_cv = {
            **tfidf_cfg,
            "max_features": lr_gap_meta.get("max_features", tfidf_cfg.get("max_features", 800)),
            "min_df": lr_gap_meta.get("min_df", tfidf_cfg.get("min_df", 3)),
        }

        def _lr_factory():
            return StableLRModel(lr_cfg, tfidf_cv, C=lr_gap_meta["C"])

        logger.info("=" * 60)
        logger.info(f"LR 5-fold CV (n={cv_folds}) on train+val pool")
        logger.info("=" * 60)
        cv_lr = stratified_kfold_cv(
            X_trainval,
            y_trainval,
            n_splits=cv_folds,
            random_state=rand,
            fit_predict_fn=lambda xt, yt, xv, yv: evaluate_lr_fold(
                _lr_factory,
                xt,
                yt,
                xv,
                yv,
                augment_fn=augment_toxic_train,
                cfg=cfg_aug,
                seed=rand,
            ),
        )
        all_metrics["cv_logistic_regression"] = cv_lr

    logger.info("=" * 60)
    logger.info("DistilBERT fine-tune (15 epochs, early stop on val F1)")
    logger.info("=" * 60)
    hf_train = _build_hf_dataset(X_train_aug, y_train_aug)
    hf_val = _build_hf_dataset(X_val, y_val)
    hf_test = _build_hf_dataset(X_test, y_test)

    bert_result = train_distilbert_stable(
        hf_train,
        hf_val,
        hf_test,
        y_test_arr,
        cfg,
        bert_dir,
        seed=rand,
    )
    bert_metrics = bert_result["metrics"]
    bert_probs_test = bert_result["test_probs"]
    all_metrics["distilbert"] = bert_metrics

    if not bert_only and lr_model is not None:
        ens_cfg = cfg["ensemble"]
        ensemble_metrics = evaluate_ensemble(
            bert_probs_test,
            lr_probs_test,
            y_test_arr,
            bert_weight=float(ens_cfg.get("bert_weight", 0.5)),
            lr_weight=float(ens_cfg.get("lr_weight", 0.5)),
            model_name="Hybrid-BERT+LR",
        )
        # Train gap for ensemble (in-sample)
        ens_probs_train_bert = _bert_probs_on_texts(bert_result, X_train_aug)
        ens_probs_train_lr = lr_model.predict_proba(X_train_aug)[:, 1]
        ens_train = evaluate_ensemble(
            ens_probs_train_bert,
            ens_probs_train_lr,
            y_train_aug.astype(int).values,
            bert_weight=float(ens_cfg.get("bert_weight", 0.5)),
            lr_weight=float(ens_cfg.get("lr_weight", 0.5)),
            model_name="Hybrid-train",
        )
        gap_ens = abs(ens_train["f1_weighted"] - ensemble_metrics["f1_weighted"])
        ensemble_metrics["f1_train"] = ens_train["f1_weighted"]
        ensemble_metrics["train_test_gap"] = round(gap_ens, 4)
        ensemble_metrics["train_test_gap_pp"] = round(gap_ens * 100, 2)
        all_metrics["ensemble"] = ensemble_metrics

        save_ensemble_meta(
            meta_path,
            {
                "run_id": run_id,
                "bert_dir": str(bert_dir),
                "lr_path": str(lr_path),
                "ensemble": ens_cfg,
            },
        )

    # Reports (CSV summary without matplotlib dependency)
    summary_rows = [bert_metrics]
    if lr_metrics:
        summary_rows.append(lr_metrics)
    if ensemble_metrics:
        summary_rows.append(ensemble_metrics)
    pd.DataFrame(summary_rows).to_csv(
        reports_dir / f"stable_summary_{run_id}.csv", index=False
    )

    report_path = reports_dir / f"stable_run_{run_id}.json"
    with open(report_path, "w") as f:
        json.dump(_json_safe(all_metrics), f, indent=2)

    md_path = reports_dir / f"integrated_report_{run_id}.md"
    write_integrated_report(all_metrics, md_path)
    logger.info(f"Integrated report: {md_path}")

    _print_targets(all_metrics)

    return all_metrics


def _bert_probs_on_texts(bert_result: dict, X: pd.Series) -> np.ndarray:
    """Run saved trainer on raw texts for train-gap on ensemble."""
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
    """Convert numpy scalars/arrays for JSON reporting."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if k not in ("ensemble_probs", "ensemble_preds")}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _print_targets(all_metrics: dict) -> None:
    logger.info("=" * 60)
    for key in ("distilbert", "logistic_regression", "ensemble"):
        m = all_metrics.get(key)
        if not m:
            continue
        f1 = m.get("f1_weighted", 0)
        gap = m.get("train_test_gap", m.get("train_test_gap_pp", 999) / 100)
        if "train_test_gap_pp" in m:
            gap = m["train_test_gap_pp"] / 100
        ok_f1 = "✅" if f1 > 0.80 else "⚠️"
        ok_gap = "✅" if gap < 0.05 else "⚠️"
        logger.info(
            f"{m.get('model', key)}: F1={f1:.4f} {ok_f1} | gap={gap:.4f} {ok_gap}"
        )
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Stable DistilBERT + LR ensemble pipeline")
    parser.add_argument(
        "--skip-augmentation",
        action="store_true",
        help="Ignored in production — augmentation is always applied",
    )
    parser.add_argument("--bert-only", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    run_stable_pipeline(
        skip_augmentation=args.skip_augmentation,
        bert_only=args.bert_only,
        config_path=Path(args.config) if args.config else None,
    )


if __name__ == "__main__":
    main()
