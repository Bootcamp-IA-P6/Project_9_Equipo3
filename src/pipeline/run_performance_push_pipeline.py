"""
Final Squeeze — Performance Push (full Toxic-BERT unfreeze, TTA, micro-LR anchor).

    uv run python -m src.pipeline.run_performance_push_pipeline
    uv run python -m src.pipeline.run_performance_push_pipeline --skip-augmentation
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

from src.data.dual_loader import load_dual_track_data
from src.evaluation.hybrid_clean_report import write_hybrid_clean_report
from src.evaluation.threshold_tuning import predict_with_threshold, search_best_threshold
from src.features.metadata_features import DEFAULT_METADATA_COLUMNS
from src.features.text_preprocessor import TextPreprocessor
from src.models.hybrid_ensemble import evaluate_ensemble, save_ensemble_meta, tune_ensemble_threshold
from src.models.metadata_lr import MetadataLRModel, fit_metadata_lr_with_gap_control
from src.models.transformer_trainer import train_transformer_stable
from src.pipeline.run_hybrid_clean_pipeline import (
    _branch_metrics,
    _build_hf_dataset,
    _json_safe,
    _meta_frame,
    augment_dual_track,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_performance_push_pipeline(
    *,
    config_path: Path | None = None,
    skip_augmentation: bool = False,
) -> dict:
    cfg_path = config_path or (PROJECT_ROOT / "configs" / "performance_push_training.yaml")
    cfg = yaml.safe_load(open(cfg_path))

    rand = int(cfg["pipeline"]["random_state"])
    test_size = float(cfg["pipeline"]["test_size"])
    val_size = float(cfg["pipeline"]["val_size"])
    max_gap = float(cfg["pipeline"].get("max_train_test_gap", 0.045))
    target_f1 = float(cfg["pipeline"].get("target_f1_weighted", 0.80))
    target_col = cfg["data"]["target_binary"]
    text_col = cfg["data"].get("text_column", "Text")

    out_cfg = cfg["output"]
    reports_dir = PROJECT_ROOT / out_cfg["reports_dir"]
    bert_dir = PROJECT_ROOT / out_cfg["transformer_dir"]
    lr_path = PROJECT_ROOT / out_cfg["lr_path"]
    meta_path = PROJECT_ROOT / out_cfg["ensemble_meta_path"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_name = cfg.get("pipeline", {}).get("name", "performance_push")
    tr_label = cfg.get("transformer", {}).get("model_label", "Toxic-BERT-push")
    logger.info("=" * 60)
    logger.info(f"{pipeline_name.upper().replace('_', ' ')} — run={run_id}")
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
        write_preprocessed_if_missing=False,
    )

    y = df[target_col].astype(int)
    idx_trainval, idx_test = train_test_split(
        df.index, test_size=test_size, random_state=rand, stratify=y
    )
    y_trainval = y.loc[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_size, random_state=rand, stratify=y_trainval
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
        logger.info("Dual-track augmentation")
        tr["raw"], tr["clean"], tr["meta"], tr["y"] = augment_dual_track(
            tr["raw"], tr["clean"], tr["meta"], tr["y"], cfg, preprocessor, seed=rand
        )
        aug_info = {
            "enabled": True,
            "pivot_lang": cfg["augmentation"].get("pivot_lang", "de"),
            "train_size_after": len(tr["y"]),
        }

    y_test_arr = te["y"].astype(int).values
    y_val_arr = va["y"].astype(int).values
    max_gap_pp = max_gap * 100

    all_metrics: dict = {
        "run_id": run_id,
        "pipeline": pipeline_name,
        "config": str(cfg_path),
        "target_f1_weighted": target_f1,
        "max_train_test_gap_pp": max_gap_pp,
        "augmentation": aug_info,
    }

    # ── LR (clean_text + metadata, C=0.01, max_features=800) ─────────────────
    lr_cfg = cfg["logistic_regression"]
    tfidf_cfg = lr_cfg.get("tfidf", {})
    use_orig_gap = lr_cfg.get("gap_search", {}).get("use_original_train_for_gap", True)

    logger.info("Metadata LR — clean_text anchor, max_features=300, C≈0.01")
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
        model_name="LR-clean+meta-push",
        gap_meta=lr_gap_meta,
    )
    all_metrics["logistic_regression"] = lr_metrics

    # ── Toxic-BERT — full unfreeze, micro-LR, TTA on test ───────────────────
    tr_cfg = cfg.get("transformer", {})
    logger.info(
        f"Training Toxic-BERT — {tr_cfg.get('freeze_mode', 'full')} unfreeze, "
        f"lr={tr_cfg.get('learning_rate', 5e-6)}, gap≤{max_gap_pp:.1f}pp, TTA on test"
    )
    hf_train = _build_hf_dataset(tr["raw"], tr["y"])
    hf_val = _build_hf_dataset(va["raw"], va["y"])
    hf_test = _build_hf_dataset(te["raw"], te["y"])

    bert_result = train_transformer_stable(
        hf_train,
        hf_val,
        hf_test,
        y_test_arr,
        y_val_arr,
        cfg,
        bert_dir,
        seed=rand,
        model_label=tr_label,
    )
    bert_metrics = bert_result["metrics"]
    bert_val_probs = bert_result["val_probs"]
    bert_test_probs = bert_result["test_probs"]
    bert_train_probs = infer_train_probs(bert_result, tr["raw"])
    all_metrics["transformer"] = bert_metrics

    # ── Fixed 0.95 / 0.05 ensemble + val threshold grid 0.30–0.70 ───────────
    ens_cfg = cfg["ensemble"]
    bw = float(ens_cfg.get("bert_weight", 0.95))
    lw = float(ens_cfg.get("lr_weight", 0.05))
    logger.info(f"Fixed ensemble weights — BERT={bw} LR={lw}")

    th_cfg = ens_cfg.get("threshold_tuning", {})
    if th_cfg.get("enabled", True):
        ens_threshold, val_f1_at_t = tune_ensemble_threshold(
            bert_val_probs,
            lr_val_probs,
            y_val_arr,
            bert_weight=bw,
            lr_weight=lw,
            metric=th_cfg.get("metric", "f1_weighted"),
            min_threshold=float(th_cfg.get("min_threshold", 0.30)),
            max_threshold=float(th_cfg.get("max_threshold", 0.70)),
            step=float(th_cfg.get("step", 0.01)),
        )
        logger.info(f"Ensemble threshold={ens_threshold:.2f} val_f1_weighted={val_f1_at_t:.4f}")
    else:
        ens_threshold = 0.5
        val_f1_at_t = None

    ensemble_metrics = evaluate_ensemble(
        bert_test_probs,
        lr_test_probs,
        y_test_arr,
        bert_weight=bw,
        lr_weight=lw,
        model_name=f"{pipeline_name.replace('_', ' ').title()}-Hybrid",
        threshold=ens_threshold,
    )

    from src.models.hybrid_ensemble import soft_vote_probs

    ens_train_probs = soft_vote_probs(bert_train_probs, lr_model.predict_proba(tr["clean"], tr["meta"])[:, 1], bw, lw)
    ens_train_preds = predict_with_threshold(ens_train_probs, ens_threshold)
    y_tr_arr = tr["y"].astype(int).values
    f1_train_ens = float(f1_score(y_tr_arr, ens_train_preds, average="weighted", zero_division=0))
    gap_ens = abs(f1_train_ens - ensemble_metrics["f1_weighted"])
    ensemble_metrics["f1_train"] = round(f1_train_ens, 4)
    ensemble_metrics["train_test_gap"] = round(gap_ens, 4)
    ensemble_metrics["train_test_gap_pp"] = round(gap_ens * 100, 2)
    ensemble_metrics["gap_ok"] = gap_ens <= max_gap
    ensemble_metrics["bert_weight"] = bw
    ensemble_metrics["lr_weight"] = lw
    ensemble_metrics["val_f1_weighted_at_threshold"] = round(val_f1_at_t, 4) if val_f1_at_t else None
    ensemble_metrics["target_f1_hit"] = ensemble_metrics["f1_weighted"] >= target_f1
    all_metrics["ensemble"] = ensemble_metrics

    save_ensemble_meta(
        meta_path,
        {
            "run_id": run_id,
            "bert_dir": str(bert_dir),
            "lr_path": str(lr_path),
            "weights": {"bert": bw, "lr": lw, "fixed": True},
            "thresholds": {
                "bert": bert_metrics.get("threshold"),
                "lr": lr_metrics.get("threshold"),
                "ensemble": ens_threshold,
            },
        },
    )

    report_path = reports_dir / f"{pipeline_name}_run_{run_id}.json"
    with open(report_path, "w") as f:
        json.dump(_json_safe(all_metrics), f, indent=2)

    md_path = reports_dir / f"integrated_report_{run_id}.md"
    write_hybrid_clean_report(all_metrics, md_path)
    logger.info(f"Report: {md_path}")

    _print_summary(all_metrics, target_f1, max_gap_pp)
    return all_metrics


def infer_train_probs(bert_result: dict, X_raw: pd.Series) -> np.ndarray:
    import torch
    from datasets import Dataset

    trainer = bert_result["trainer"]
    tokenizer = bert_result["tokenizer"]
    ds = Dataset.from_pandas(pd.DataFrame({"text": X_raw.values, "label": [0] * len(X_raw)}))

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tok = ds.map(_tok, batched=True)
    drop = [c for c in tok.column_names if c not in ("input_ids", "attention_mask", "label")]
    if drop:
        tok = tok.remove_columns(drop)
    tok.set_format("torch")
    out = trainer.predict(tok)
    return torch.softmax(torch.tensor(out.predictions), dim=1)[:, 1].numpy()


def _print_summary(metrics: dict, target: float, max_gap_pp: float) -> None:
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
            ok = m.get("gap_ok", m.get("train_test_gap_pp", 99) <= max_gap_pp)
            logger.info(
                f"  {m.get('model', key)}: F1w={m.get('f1_weighted')} "
                f"gap_pp={m.get('train_test_gap_pp')} gap_ok={ok}"
            )
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Performance Push pipeline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-augmentation", action="store_true")
    args = parser.parse_args()
    run_performance_push_pipeline(
        config_path=Path(args.config) if args.config else None,
        skip_augmentation=args.skip_augmentation,
    )


if __name__ == "__main__":
    main()
