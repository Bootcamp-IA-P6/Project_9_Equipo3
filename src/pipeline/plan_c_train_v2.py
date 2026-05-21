"""Plan C v2: regularized DistilBERT + MLflow pass/fail (uv run python -m src.pipeline.plan_c_train_v2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

from src.data.loader import load_processed_data
from src.evaluation.overfitting_check import format_gap_report
from src.evaluation.phase1_audit import BINARY_TARGET, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from src.features.plan_c_augmentation import augment_toxic_backtranslation
from src.models.plan_c_model import (
    DEFAULT_MODEL_NAME,
    build_distilbert_classifier,
    count_trainable_parameters,
)
from src.pipeline.plan_c_train import CommentDataset, MODEL_NAME, _compute_metrics
from src.utils.execution_log import log_event
from src.utils.mlflow_helpers import log_overfitting_metrics, setup_mlflow, start_run

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/youtoxic_english_1000.csv"
CONFIG_PATH = ROOT / "configs/config.yaml"
REPORT_PATH = ROOT / "reports/phase5/plan_c_v2_report.json"
COMPARISON_PATH = ROOT / "reports/phase5/comparison.json"
LOG_PATH = ROOT / "logs/execution.log"
MIN_TEST_F1_FOR_BALANCED_PASS = 0.74

Variant = Literal["v2a", "v2b", "v2c"]

VARIANT_SETTINGS: dict[Variant, dict[str, Any]] = {
    "v2a": {
        "frozen_layers": 4,
        "freeze_all_backbone": False,
        "augment": False,
        "description": "Freeze 4/6 layers, head dropout 0.4, weight decay",
    },
    "v2b": {
        "frozen_layers": 6,
        "freeze_all_backbone": True,
        "augment": False,
        "description": "Head-only (full backbone frozen)",
    },
    "v2c": {
        "frozen_layers": 4,
        "freeze_all_backbone": False,
        "augment": True,
        "description": "v2a + toxic back-translation",
    },
}


def _load_plan_c_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("plan_c_v2", {})


def _evaluate_train_test_gap(
    trainer: Trainer,
    tokenizer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    full_train_ds = CommentDataset(
        train_df["Text"].astype(str).tolist(),
        train_df[BINARY_TARGET].astype(int).tolist(),
        tokenizer,
    )
    test_ds = CommentDataset(
        test_df["Text"].astype(str).tolist(),
        test_df[BINARY_TARGET].astype(int).tolist(),
        tokenizer,
    )
    train_out = trainer.predict(full_train_ds)
    test_out = trainer.predict(test_ds)
    y_train = train_df[BINARY_TARGET].values
    y_test = test_df[BINARY_TARGET].values
    train_pred = np.argmax(train_out.predictions, axis=1)
    test_pred = np.argmax(test_out.predictions, axis=1)

    metrics = {
        "accuracy": format_gap_report(
            "accuracy",
            accuracy_score(y_train, train_pred),
            accuracy_score(y_test, test_pred),
        ),
        "f1_toxic": format_gap_report(
            "f1_toxic",
            f1_score(y_train, train_pred, pos_label=1, zero_division=0),
            f1_score(y_test, test_pred, pos_label=1, zero_division=0),
        ),
    }
    metrics["passes_overfitting"] = (
        metrics["accuracy"]["passes"] and metrics["f1_toxic"]["passes"]
    )
    return metrics


def run_plan_c_v2(
    variant: Variant = "v2a",
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    head_dropout: float | None = None,
    early_stopping_patience: int | None = None,
    label_smoothing: float | None = None,
    augment_max_samples: int | None = None,
) -> dict[str, Any]:
    cfg = _load_plan_c_config()
    settings = VARIANT_SETTINGS[variant]
    frozen_layers = int(cfg.get("frozen_layers", settings["frozen_layers"]))
    if settings.get("freeze_all_backbone"):
        frozen_layers = 6

    head_dropout = head_dropout if head_dropout is not None else float(cfg.get("head_dropout", 0.4))
    weight_decay = weight_decay if weight_decay is not None else float(cfg.get("weight_decay", 0.01))
    learning_rate = learning_rate if learning_rate is not None else float(cfg.get("learning_rate", 1e-5))
    epochs = epochs if epochs is not None else int(cfg.get("epochs", 4))
    batch_size = batch_size if batch_size is not None else int(cfg.get("batch_size", 8))
    patience = (
        early_stopping_patience
        if early_stopping_patience is not None
        else int(cfg.get("early_stopping_patience", 3))
    )
    label_smoothing = (
        label_smoothing
        if label_smoothing is not None
        else float(cfg.get("label_smoothing", 0.1))
    )
    use_augment = settings["augment"] or bool(cfg.get("augment", False))

    log_event(f"Plan C v2 ({variant}) started", phase="5", log_path=LOG_PATH)
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

    if use_augment:
        max_aug = augment_max_samples or cfg.get("augment_max_samples")
        train_df = augment_toxic_backtranslation(
            train_df,
            max_samples=int(max_aug) if max_aug else None,
        )

    tr_idx, val_idx = train_test_split(
        train_df.index,
        test_size=0.15,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=train_df[BINARY_TARGET],
    )

    model = build_distilbert_classifier(
        MODEL_NAME,
        frozen_layers=frozen_layers,
        head_dropout=head_dropout,
    )
    trainable, total = count_trainable_parameters(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = CommentDataset(
        train_df.loc[tr_idx, "Text"].astype(str).tolist(),
        train_df.loc[tr_idx, BINARY_TARGET].astype(int).tolist(),
        tokenizer,
    )
    val_ds = CommentDataset(
        train_df.loc[val_idx, "Text"].astype(str).tolist(),
        train_df.loc[val_idx, BINARY_TARGET].astype(int).tolist(),
        tokenizer,
    )

    checkpoint_dir = ROOT / "models" / f"plan_c_v2_{variant}_checkpoints"
    model_dir = ROOT / "models" / f"plan_c_v2_{variant}"

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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        seed=DEFAULT_RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    run_name = f"plan_c_v2_{variant}"
    with start_run(run_name, tags={"plan": "C", "variant": variant, "model": MODEL_NAME}):
        mlflow.log_param("variant", variant)
        mlflow.log_param("base_model", MODEL_NAME)
        mlflow.log_param("frozen_layers", frozen_layers)
        mlflow.log_param("head_dropout", head_dropout)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("label_smoothing", label_smoothing)
        mlflow.log_param("early_stopping_patience", patience)
        mlflow.log_param("augment_backtranslation", use_augment)
        mlflow.log_param("n_train", len(train_df))
        mlflow.log_param("trainable_params", trainable)
        mlflow.log_param("total_params", total)
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
        "plan": "C_v2",
        "variant": variant,
        "description": settings["description"],
        "model_dir": str(model_dir.relative_to(ROOT)),
        "base_model": DEFAULT_MODEL_NAME,
        "frozen_layers": frozen_layers,
        "head_dropout": head_dropout,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "augment_backtranslation": use_augment,
        "n_train": len(train_df),
        "trainable_params": trainable,
        "passes_overfitting": metrics["passes_overfitting"],
        "status": status,
        "metrics": metrics,
        "mlflow_experiment": "youtube-toxic-detector",
        "mlflow_run": run_name,
    }
    log_event(
        f"Plan C v2 {variant} {status}: test_acc={metrics['accuracy']['test']:.3f} "
        f"gap_acc={metrics['accuracy']['gap_pct']:.1f}pp",
        phase="5",
        log_path=LOG_PATH,
    )
    return result


def _balanced_pass(report: dict[str, Any]) -> bool:
    """Passes gap rule and keeps test F1 at or above the balanced threshold."""
    return bool(report["passes_overfitting"]) and report["metrics"]["f1_toxic"]["test"] >= MIN_TEST_F1_FOR_BALANCED_PASS


def _select_best(results: dict[str, Any]) -> dict[str, Any]:
    """Prefer balanced pass; else highest test F1 among v2 runs."""
    balanced = [r for r in results.values() if _balanced_pass(r)]
    if balanced:
        return max(balanced, key=lambda r: r["metrics"]["f1_toxic"]["test"])
    return max(
        results.values(),
        key=lambda r: (
            r["passes_overfitting"],
            r["metrics"]["f1_toxic"]["test"],
        ),
    )


def run_experiment_matrix(
    *,
    include_v2c: bool = True,
    stop_on_pass: bool = True,
) -> dict[str, Any]:
    """Run v2a → v2b → optional v2c; persist combined report."""
    variants: list[Variant] = ["v2a", "v2b"]
    results: dict[str, Any] = {}

    for variant in variants:
        results[variant] = run_plan_c_v2(variant)
        if stop_on_pass and _balanced_pass(results[variant]):
            break

    best = _select_best(results)
    if include_v2c and not _balanced_pass(best):
        v2a = results.get("v2a")
        if v2a and v2a["metrics"]["f1_toxic"]["test"] >= MIN_TEST_F1_FOR_BALANCED_PASS - 0.1:
            results["v2c"] = run_plan_c_v2("v2c")
            best = _select_best(results)

    for rep in results.values():
        rep["balanced_pass"] = _balanced_pass(rep)

    summary = {
        "plan": "C_v2",
        "best_variant": best["variant"],
        "best": best,
        "balanced_pass": _balanced_pass(best),
        "min_test_f1_balanced": MIN_TEST_F1_FOR_BALANCED_PASS,
        "experiments": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _update_comparison(best, results)
    return summary


def _update_comparison(best: dict[str, Any], experiments: dict[str, Any]) -> None:
    comparison: dict[str, Any] = {}
    if COMPARISON_PATH.exists():
        comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))

    models = [m for m in comparison.get("models", []) if not m.get("name", "").startswith("plan_c_v2")]
    for variant, rep in experiments.items():
        models.append(
            {
                "name": f"plan_c_v2_{variant}",
                "path": rep["model_dir"],
                "test_accuracy": rep["metrics"]["accuracy"]["test"],
                "test_f1_toxic": rep["metrics"]["f1_toxic"]["test"],
                "passes_overfitting": rep["passes_overfitting"],
                "accuracy_gap_pct": rep["metrics"]["accuracy"]["gap_pct"],
                "f1_gap_pct": rep["metrics"]["f1_toxic"]["gap_pct"],
                "status": rep["status"],
                "mlflow_run": rep.get("mlflow_run", f"plan_c_v2_{variant}"),
            }
        )
    comparison["models"] = models
    comparison["plan_c_v2_best"] = {
        "variant": best["variant"],
        "path": best["model_dir"],
        "test_accuracy": best["metrics"]["accuracy"]["test"],
        "test_f1_toxic": best["metrics"]["f1_toxic"]["test"],
        "passes_overfitting": best["passes_overfitting"],
        "balanced_pass": _balanced_pass(best),
        "status": best["status"],
    }
    rec = comparison.get("recommendation", {})
    rec["plan_c_v2"] = (
        f"{best['variant']}: acc={best['metrics']['accuracy']['test']:.1%} "
        f"F1={best['metrics']['f1_toxic']['test']:.1%} "
        f"gap_pass={best['passes_overfitting']}"
    )
    comparison["recommendation"] = rec
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan C v2 DistilBERT training")
    parser.add_argument(
        "--variant",
        choices=["v2a", "v2b", "v2c", "all"],
        default="all",
        help="v2a=freeze4, v2b=head-only, v2c=v2a+BT, all=experiment matrix",
    )
    parser.add_argument("--no-v2c", action="store_true", help="Skip v2c even in matrix mode")
    args = parser.parse_args()

    if args.variant == "all":
        out = run_experiment_matrix(include_v2c=not args.no_v2c)
    else:
        out = run_plan_c_v2(args.variant)  # type: ignore[arg-type]
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
