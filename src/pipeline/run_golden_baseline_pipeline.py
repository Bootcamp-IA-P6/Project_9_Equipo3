"""
Golden Baseline strategy — frozen pretrained BERT, R-Drop squeeze, hybrid safety net.

    uv run python -m src.pipeline.run_golden_baseline_pipeline
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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dual_loader import load_dual_track_data
from src.evaluation.golden_baseline_report import write_golden_baseline_report
from src.evaluation.threshold_tuning import predict_with_threshold
from src.models.hybrid_ensemble import (
    evaluate_ensemble,
    save_ensemble_meta,
    soft_vote_probs,
    tune_ensemble_threshold,
)
from src.models.metadata_lr import MetadataLRModel
from src.models.transformer_trainer import (
    evaluate_pretrained_bert_baseline,
    train_transformer_stable,
)
from src.pipeline.run_hybrid_clean_pipeline import (
    _branch_metrics,
    _build_hf_dataset,
    _json_safe,
    _meta_frame,
)
from src.pipeline.run_performance_push_pipeline import infer_train_probs
from src.utils.logger import get_logger

logger = get_logger(__name__)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "golden_baseline_training.yaml"


def run_golden_baseline_pipeline(*, config_path: Path | None = None) -> dict:
    cfg_path = config_path or DEFAULT_CONFIG
    cfg = yaml.safe_load(open(cfg_path))

    rand = int(cfg["pipeline"]["random_state"])
    test_size = float(cfg["pipeline"]["test_size"])
    val_size = float(cfg["pipeline"]["val_size"])
    max_gap = float(cfg["pipeline"].get("max_train_test_gap", 0.05))
    baseline_gap_target = float(cfg["pipeline"].get("baseline_gap_target", 0.01))
    squeeze_gap_target = float(cfg["pipeline"].get("squeeze_gap_target", 0.049))
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
    logger.info("=" * 60)
    logger.info(f"GOLDEN BASELINE STRATEGY — run={run_id}")
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
    y_train_orig = y.loc[idx_train]
    y_test_arr = te["y"].astype(int).values
    y_val_arr = va["y"].astype(int).values

    hf_train = _build_hf_dataset(tr["raw"], tr["y"])
    hf_val = _build_hf_dataset(va["raw"], va["y"])
    hf_test = _build_hf_dataset(te["raw"], te["y"])

    all_metrics: dict = {
        "run_id": run_id,
        "pipeline": cfg.get("pipeline", {}).get("name", "golden_baseline"),
        "config": str(cfg_path),
        "target_f1_weighted": target_f1,
        "max_train_test_gap_pp": max_gap * 100,
        "baseline_gap_target_pp": baseline_gap_target * 100,
        "squeeze_gap_target_pp": squeeze_gap_target * 100,
        "augmentation": {"enabled": False},
    }

    # ── Step 1: Golden Baseline (frozen pretrained, no training) ─────────────
    logger.info("Step 1 — Golden Baseline (all layers frozen, zero fine-tuning)")
    base_label = cfg.get("baseline", {}).get("model_label", "Golden-Baseline-Toxic-BERT")
    baseline_result = evaluate_pretrained_bert_baseline(
        hf_train,
        hf_val,
        hf_test,
        y_train_orig.values,
        y_test_arr,
        y_val_arr,
        cfg,
        seed=rand,
        model_label=base_label,
    )
    base_m = baseline_result["metrics"]
    base_m["esencial_gap_ok"] = base_m.get("train_test_gap", 1) < baseline_gap_target
    base_m["gap_ok"] = base_m["esencial_gap_ok"]
    all_metrics["golden_baseline"] = base_m
    logger.info(
        f"  Baseline F1w={base_m['f1_weighted']} gap_pp={base_m['train_test_gap_pp']} "
        f"{'✅' if base_m['esencial_gap_ok'] else '⚠️'}"
    )

    # ── Step 2: Performance Squeeze (last 2 layers + R-Drop) ───────────────
    tr_cfg = cfg.get("transformer", {})
    squeeze_label = tr_cfg.get("model_label", "Performance-Squeeze-Toxic-BERT")
    logger.info(
        f"Step 2 — Performance Squeeze (last {tr_cfg.get('train_last_n_layers', 2)} layers, "
        f"R-Drop, lr={tr_cfg.get('learning_rate', 5e-6)}, max_epochs={tr_cfg.get('max_epochs', 15)})"
    )
    squeeze_result = train_transformer_stable(
        hf_train,
        hf_val,
        hf_test,
        y_test_arr,
        y_val_arr,
        cfg,
        bert_dir,
        seed=rand,
        model_label=squeeze_label,
    )
    squeeze_m = squeeze_result["metrics"]
    squeeze_m["squeeze_gap_ok"] = squeeze_m.get("train_test_gap", 1) <= squeeze_gap_target
    squeeze_m["target_f1_hit"] = squeeze_m.get("f1_weighted", 0) >= target_f1
    all_metrics["performance_squeeze"] = squeeze_m

    bert_val_probs = squeeze_result["val_probs"]
    bert_test_probs = squeeze_result["test_probs"]
    bert_train_probs = infer_train_probs(squeeze_result, tr["raw"])

    # ── Step 3: Hybrid Safety Net (C=0.001, 200 features) ────────────────────
    lr_cfg = cfg["logistic_regression"]
    tfidf_cfg = lr_cfg.get("tfidf", {})
    logger.info("Step 3 — Hybrid Safety Net (LR C=0.001, max_features=200)")
    lr_model = MetadataLRModel(lr_cfg, tfidf_cfg)
    lr_model.fit(tr["clean"], tr["meta"], tr["y"])
    lr_model.save(lr_path)

    lr_val_probs = lr_model.predict_proba(va["clean"], va["meta"])[:, 1]
    lr_test_probs = lr_model.predict_proba(te["clean"], te["meta"])[:, 1]
    lr_train_probs = lr_model.predict_proba(tr["clean"], tr["meta"])[:, 1]

    lr_metrics = _branch_metrics(
        tr["y"],
        te["y"],
        va["y"],
        lr_train_probs,
        lr_val_probs,
        lr_test_probs,
        model_name="LR-Safety-Net",
    )
    all_metrics["logistic_regression"] = lr_metrics

    ens_cfg = cfg["ensemble"]
    bw = float(ens_cfg.get("bert_weight", 0.90))
    lw = float(ens_cfg.get("lr_weight", 0.10))
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
    else:
        ens_threshold = 0.5
        val_f1_at_t = None

    hybrid_metrics = evaluate_ensemble(
        bert_test_probs,
        lr_test_probs,
        y_test_arr,
        bert_weight=bw,
        lr_weight=lw,
        model_name="Hybrid-Safety-Net",
        threshold=ens_threshold,
    )
    ens_train_probs = soft_vote_probs(bert_train_probs, lr_train_probs, bw, lw)
    ens_train_preds = predict_with_threshold(ens_train_probs, ens_threshold)
    y_tr_arr = tr["y"].astype(int).values
    f1_train_ens = float(f1_score(y_tr_arr, ens_train_preds, average="weighted", zero_division=0))
    gap_ens = abs(f1_train_ens - hybrid_metrics["f1_weighted"])
    hybrid_metrics["f1_train"] = round(f1_train_ens, 4)
    hybrid_metrics["train_test_gap"] = round(gap_ens, 4)
    hybrid_metrics["train_test_gap_pp"] = round(gap_ens * 100, 2)
    hybrid_metrics["gap_ok"] = gap_ens <= max_gap
    hybrid_metrics["bert_weight"] = bw
    hybrid_metrics["lr_weight"] = lw
    hybrid_metrics["val_f1_weighted_at_threshold"] = (
        round(val_f1_at_t, 4) if val_f1_at_t else None
    )
    hybrid_metrics["target_f1_hit"] = hybrid_metrics["f1_weighted"] >= target_f1
    hybrid_metrics["briefing_compliant"] = hybrid_metrics["gap_ok"] and hybrid_metrics["target_f1_hit"]
    all_metrics["hybrid_safety_net"] = hybrid_metrics

    save_ensemble_meta(
        meta_path,
        {
            "run_id": run_id,
            "bert_dir": str(bert_dir),
            "lr_path": str(lr_path),
            "weights": {"bert": bw, "lr": lw, "fixed": True},
            "thresholds": {
                "bert": squeeze_m.get("threshold"),
                "lr": lr_metrics.get("threshold"),
                "ensemble": ens_threshold,
            },
        },
    )

    report_path = reports_dir / f"golden_baseline_run_{run_id}.json"
    with open(report_path, "w") as f:
        json.dump(_json_safe(all_metrics), f, indent=2)

    md_path = reports_dir / f"integrated_report_{run_id}.md"
    write_golden_baseline_report(all_metrics, md_path)
    logger.info(f"Report: {md_path}")

    _print_summary(all_metrics, target_f1, max_gap * 100, baseline_gap_target * 100)
    return all_metrics


def _print_summary(
    metrics: dict, target: float, max_gap_pp: float, baseline_gap_pp: float
) -> None:
    logger.info("=" * 60)
    base = metrics.get("golden_baseline", {})
    hybrid = metrics.get("hybrid_safety_net", {})
    logger.info(
        f"BASELINE  F1w={base.get('f1_weighted')} gap_pp={base.get('train_test_gap_pp')} "
        f"({'✅ <1%' if base.get('esencial_gap_ok') else '⚠️'})"
    )
    logger.info(
        f"HYBRID    F1w={hybrid.get('f1_weighted')} gap_pp={hybrid.get('train_test_gap_pp')} "
        f"({'✅ F1+gap' if hybrid.get('briefing_compliant') else '⚠️ below target'})"
    )
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Golden Baseline strategy pipeline")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    run_golden_baseline_pipeline(
        config_path=Path(args.config) if args.config else None,
    )


if __name__ == "__main__":
    main()
