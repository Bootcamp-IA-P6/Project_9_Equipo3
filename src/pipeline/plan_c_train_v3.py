"""Plan C v3: v2a fine-tune — lower LR, class weights, F1 early stop (uv run python -m src.pipeline.plan_c_train_v3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments

from src.data.loader import load_processed_data
from src.evaluation.phase1_audit import BINARY_TARGET, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from src.models.plan_c_model import build_distilbert_classifier, count_trainable_parameters
from src.models.plan_c_trainer import ClassWeightedTrainer
from src.pipeline.plan_c_train import CommentDataset, MODEL_NAME, _compute_metrics
from src.pipeline.plan_c_train_v2 import (
    COMPARISON_PATH,
    MIN_TEST_F1_FOR_BALANCED_PASS,
    _balanced_pass,
    _evaluate_train_test_gap,
)
from src.utils.execution_log import log_event
from src.utils.mlflow_helpers import log_overfitting_metrics, setup_mlflow, start_run

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/youtoxic_english_1000.csv"
CONFIG_PATH = ROOT / "configs/config.yaml"
REPORT_PATH = ROOT / "reports/phase5/plan_c_v3_report.json"
LOG_PATH = ROOT / "logs/execution.log"
VARIANT = "v3"


def _load_plan_c_v3_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("plan_c_v3", {})


def _class_weights_tensor(labels: list[int] | np.ndarray) -> torch.Tensor:
    y = np.asarray(labels)
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32)


def run_plan_c_v3(
    *,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    head_dropout: float | None = None,
    frozen_layers: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    early_stopping_patience: int | None = None,
    label_smoothing: float | None = None,
) -> dict[str, Any]:
    """v2a data setup with v3 training: LR 5e-6, balanced loss, val F1 early stop."""
    cfg = _load_plan_c_v3_config()
    learning_rate = learning_rate if learning_rate is not None else float(cfg.get("learning_rate", 5e-6))
    weight_decay = weight_decay if weight_decay is not None else float(cfg.get("weight_decay", 0.01))
    head_dropout = head_dropout if head_dropout is not None else float(cfg.get("head_dropout", 0.4))
    frozen_layers = frozen_layers if frozen_layers is not None else int(cfg.get("frozen_layers", 4))
    epochs = epochs if epochs is not None else int(cfg.get("epochs", 4))
    batch_size = batch_size if batch_size is not None else int(cfg.get("batch_size", 8))
    patience = (
        early_stopping_patience
        if early_stopping_patience is not None
        else int(cfg.get("early_stopping_patience", 2))
    )
    label_smoothing = (
        label_smoothing if label_smoothing is not None else float(cfg.get("label_smoothing", 0.05))
    )

    log_event("Plan C v3 started (v2a setup + class weights + F1 early stop)", phase="5", log_path=LOG_PATH)
    setup_mlflow()

    df, _ = load_processed_data(str(RAW_PATH))
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=df[BINARY_TARGET],
    )
    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    tr_idx, val_idx = train_test_split(
        train_df.index,
        test_size=0.15,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=train_df[BINARY_TARGET],
    )

    train_labels = train_df.loc[tr_idx, BINARY_TARGET].astype(int).tolist()
    class_weights = _class_weights_tensor(train_labels)

    model = build_distilbert_classifier(
        MODEL_NAME,
        frozen_layers=frozen_layers,
        head_dropout=head_dropout,
    )
    trainable, total = count_trainable_parameters(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = CommentDataset(
        train_df.loc[tr_idx, "Text"].astype(str).tolist(),
        train_labels,
        tokenizer,
    )
    val_ds = CommentDataset(
        train_df.loc[val_idx, "Text"].astype(str).tolist(),
        train_df.loc[val_idx, BINARY_TARGET].astype(int).tolist(),
        tokenizer,
    )

    checkpoint_dir = ROOT / "models" / "plan_c_v3_checkpoints"
    model_dir = ROOT / "models" / "plan_c_v3"

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        label_smoothing_factor=label_smoothing,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_toxic",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=DEFAULT_RANDOM_STATE,
    )

    trainer = ClassWeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    run_name = "plan_c_v3"
    with start_run(run_name, tags={"plan": "C", "variant": VARIANT, "model": MODEL_NAME}):
        mlflow.log_param("variant", VARIANT)
        mlflow.log_param("base_model", MODEL_NAME)
        mlflow.log_param("frozen_layers", frozen_layers)
        mlflow.log_param("head_dropout", head_dropout)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("label_smoothing", label_smoothing)
        mlflow.log_param("early_stopping_patience", patience)
        mlflow.log_param("early_stopping_metric", "eval_f1_toxic")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("class_weights", class_weights.tolist())
        mlflow.log_param("augment_backtranslation", False)
        mlflow.log_param("n_train", len(train_df))
        mlflow.log_param("trainable_params", trainable)
        mlflow.log_param("overfitting_rule", "abs_train_test_gap_pp_lt_5")

        trainer.train()

        model_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        mlflow.log_artifacts(str(model_dir), artifact_path="transformer_model")

        metrics = _evaluate_train_test_gap(trainer, tokenizer, train_df, test_df)
        log_overfitting_metrics(metrics)

        status = "PASSED" if metrics["passes_overfitting"] else "FAILED"
        mlflow.set_tag("status", status)
        mlflow.log_metric("test_accuracy", metrics["accuracy"]["test"])
        mlflow.log_metric("test_f1_toxic", metrics["f1_toxic"]["test"])

    result = {
        "plan": "C_v3",
        "variant": VARIANT,
        "description": "v2a setup: freeze 4, LR 5e-6, balanced class weights, F1 early stop (patience 2)",
        "model_dir": str(model_dir.relative_to(ROOT)),
        "base_model": MODEL_NAME,
        "frozen_layers": frozen_layers,
        "head_dropout": head_dropout,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "class_weights": class_weights.tolist(),
        "augment_backtranslation": False,
        "n_train": len(train_df),
        "trainable_params": trainable,
        "passes_overfitting": metrics["passes_overfitting"],
        "balanced_pass": _balanced_pass(
            {"passes_overfitting": metrics["passes_overfitting"], "metrics": metrics}
        ),
        "status": status,
        "metrics": metrics,
        "v2a_reference": {
            "f1_gap_pp": 5.31,
            "note": "v3 targets F1 gap below 5.0pp",
        },
        "mlflow_experiment": "youtube-toxic-detector",
        "mlflow_run": run_name,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _update_comparison_v3(result)

    log_event(
        f"Plan C v3 {status}: test_f1={metrics['f1_toxic']['test']:.3f} "
        f"f1_gap={metrics['f1_toxic']['gap_pct']:.2f}pp",
        phase="5",
        log_path=LOG_PATH,
    )
    return result


def _update_comparison_v3(report: dict[str, Any]) -> None:
    comparison: dict[str, Any] = {}
    if COMPARISON_PATH.exists():
        comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))

    models = [m for m in comparison.get("models", []) if m.get("name") != "plan_c_v3"]
    m = report["metrics"]
    models.append(
        {
            "name": "plan_c_v3",
            "path": report["model_dir"],
            "test_accuracy": m["accuracy"]["test"],
            "test_f1_toxic": m["f1_toxic"]["test"],
            "passes_overfitting": report["passes_overfitting"],
            "accuracy_gap_pct": m["accuracy"]["gap_pct"],
            "f1_gap_pct": m["f1_toxic"]["gap_pct"],
            "status": report["status"],
            "balanced_pass": report.get("balanced_pass", False),
            "mlflow_run": report["mlflow_run"],
        }
    )
    comparison["models"] = models
    comparison["plan_c_v3"] = {
        "path": report["model_dir"],
        "test_f1_toxic": m["f1_toxic"]["test"],
        "f1_gap_pct": m["f1_toxic"]["gap_pct"],
        "passes_overfitting": report["passes_overfitting"],
        "status": report["status"],
    }
    rec = comparison.get("recommendation", {})
    rec["plan_c_v3"] = (
        f"v3: acc={m['accuracy']['test']:.1%} F1={m['f1_toxic']['test']:.1%} "
        f"f1_gap={m['f1_toxic']['gap_pct']:.2f}pp pass={report['passes_overfitting']}"
    )
    comparison["recommendation"] = rec
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")


def main() -> None:
    print(json.dumps(run_plan_c_v3(), indent=2))


if __name__ == "__main__":
    main()
