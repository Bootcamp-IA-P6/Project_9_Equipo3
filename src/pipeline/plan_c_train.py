"""Plan C: DistilBERT fine-tuning with MLflow (uv run python -m src.pipeline.plan_c_train)."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.loader import load_processed_data
from src.evaluation.overfitting_check import format_gap_report
from src.evaluation.phase1_audit import BINARY_TARGET, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from src.utils.execution_log import log_event
from src.utils.mlflow_helpers import log_overfitting_metrics, setup_mlflow, start_run

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/youtoxic_english_1000.csv"
MODEL_DIR = ROOT / "models/plan_c_distilbert"
REPORT_PATH = ROOT / "reports/phase5/plan_c_report.json"
LOG_PATH = ROOT / "logs/execution.log"
MODEL_NAME = "distilbert-base-uncased"


class CommentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_toxic": f1_score(labels, preds, pos_label=1, zero_division=0),
    }


def run_plan_c(*, epochs: int = 4, batch_size: int = 8) -> dict:
    log_event("Plan C DistilBERT training started", phase="5", log_path=LOG_PATH)
    setup_mlflow()

    df, _ = load_processed_data(str(RAW_PATH))
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=df[BINARY_TARGET],
    )
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    tr_idx, val_idx = train_test_split(
        train_df.index,
        test_size=0.15,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=train_df[BINARY_TARGET],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

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
    test_ds = CommentDataset(
        test_df["Text"].astype(str).tolist(),
        test_df[BINARY_TARGET].astype(int).tolist(),
        tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=str(ROOT / "models/plan_c_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_toxic",
        greater_is_better=True,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    with start_run("plan_c_distilbert", tags={"plan": "C", "model": MODEL_NAME}):
        mlflow.log_param("base_model", MODEL_NAME)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        trainer.train()

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(MODEL_DIR))
        tokenizer.save_pretrained(str(MODEL_DIR))
        mlflow.log_artifacts(str(MODEL_DIR), artifact_path="transformer_model")

        full_train_ds = CommentDataset(
            train_df["Text"].astype(str).tolist(),
            train_df[BINARY_TARGET].astype(int).tolist(),
            tokenizer,
        )
        train_out = trainer.predict(full_train_ds)
        test_out = trainer.predict(test_ds)
        y_train_true = train_df[BINARY_TARGET].values
        y_test_true = test_df[BINARY_TARGET].values
        train_pred = np.argmax(train_out.predictions, axis=1)
        test_pred = np.argmax(test_out.predictions, axis=1)

        metrics = {
            "accuracy": format_gap_report(
                "accuracy",
                accuracy_score(y_train_true, train_pred),
                accuracy_score(y_test_true, test_pred),
            ),
            "f1_toxic": format_gap_report(
                "f1_toxic",
                f1_score(y_train_true, train_pred, pos_label=1, zero_division=0),
                f1_score(y_test_true, test_pred, pos_label=1, zero_division=0),
            ),
            "passes_overfitting": False,
        }
        metrics["passes_overfitting"] = (
            metrics["accuracy"]["passes"] and metrics["f1_toxic"]["passes"]
        )
        log_overfitting_metrics(metrics)
        mlflow.log_metric("test_accuracy", metrics["accuracy"]["test"])
        mlflow.log_metric("test_f1_toxic", metrics["f1_toxic"]["test"])

    report = {
        "plan": "C",
        "model_dir": str(MODEL_DIR.relative_to(ROOT)),
        "base_model": MODEL_NAME,
        "passes_overfitting": metrics["passes_overfitting"],
        "metrics": metrics,
        "mlflow_experiment": "youtube-toxic-detector",
        "mlflow_run": "plan_c_distilbert",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_event(
        f"Plan C done: test_acc={metrics['accuracy']['test']:.3f} f1={metrics['f1_toxic']['test']:.3f}",
        phase="5",
        log_path=LOG_PATH,
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run_plan_c(), indent=2))
