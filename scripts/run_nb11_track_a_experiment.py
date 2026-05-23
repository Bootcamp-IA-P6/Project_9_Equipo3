#!/usr/bin/env python3
"""Track A experiment: unitary/toxic-distilbert + differential LR + label smoothing.

Does NOT write to models/distilbert_frozen/ or reports/nb11_metrics.json.
Results → models/nb11_experiment_track_a/ and reports/nb11_experiment_track_a.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from datasets import Dataset
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE_F1 = 0.7579
BASELINE_GAP_PP = 3.44
GAP_MAX_PP = 5.0
# Requested id unitary/toxic-distilbert is not on HF; closest Unitary toxic checkpoint:
MODEL_ID = "unitary/toxic-bert"
MODEL_ID_REQUESTED = "unitary/toxic-distilbert"
SAVE_DIR = ROOT / "models" / "nb11_experiment_track_a"
METRICS_PATH = ROOT / "reports" / "nb11_experiment_track_a.json"
LR_PENULT = 5e-6
LR_LAST_HEAD = 2e-5
LABEL_SMOOTHING = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_hf_dataset(X, y) -> Dataset:
    return Dataset.from_pandas(
        pd.DataFrame({"text": X.values, "label": y.astype(int).values})
    )


def tokenize_dataset(dataset, tokenizer, max_len: int = 128) -> Dataset:
    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    out = dataset.map(_tokenize, batched=True)
    out = out.remove_columns(["text"])
    out = out.rename_column("label", "labels")
    out.set_format("torch")
    return out


def _encoder_layers(model):
    if hasattr(model, "distilbert"):
        return list(model.distilbert.transformer.layer)
    if hasattr(model, "bert"):
        return list(model.bert.encoder.layer)
    return list(model.base_model.transformer.layer)


def hard_freeze_differential(model) -> None:
    """Freeze all except penultimate block, last block, and classification head."""
    backbone = model.distilbert if hasattr(model, "distilbert") else model.bert
    for param in backbone.parameters():
        param.requires_grad = False
    layers = _encoder_layers(model)
    for param in layers[-2].parameters():
        param.requires_grad = True
    for param in layers[-1].parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        if "classifier" in name or "pre_classifier" in name:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.1f}%)")


class LabelSmoothingTrainer(Trainer):
    def __init__(self, *args, label_smoothing: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def create_optimizer(self):
        layers = _encoder_layers(self.model)
        penult, last = [], []
        head = []
        for p in layers[-2].parameters():
            if p.requires_grad:
                penult.append(p)
        for p in layers[-1].parameters():
            if p.requires_grad:
                last.append(p)
        for name, p in self.model.named_parameters():
            if p.requires_grad and ("classifier" in name or "pre_classifier" in name):
                head.append(p)
        groups = [
            {"params": penult, "lr": LR_PENULT},
            {"params": last + head, "lr": LR_LAST_HEAD},
        ]
        self.optimizer = torch.optim.AdamW(
            groups, weight_decay=self.args.weight_decay
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(
            outputs.logits,
            labels,
            label_smoothing=self.label_smoothing,
        )
        return (loss, outputs) if return_outputs else loss


class GapEarlyStopCallback(TrainerCallback):
    """Stop training when train–test gap exceeds GAP_MAX_PP (percentage points)."""

    def __init__(self, trainer_ref, tok_test, y_test_arr):
        self.trainer_ref = trainer_ref
        self.tok_test = tok_test
        self.y_test = y_test_arr
        self.history: list[dict] = []

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.trainer_ref[0]
        train_out = trainer.predict(trainer.train_dataset)
        f1_tr = f1_score(
            train_out.label_ids,
            np.argmax(train_out.predictions, axis=1),
            average="weighted",
        )
        test_out = trainer.predict(self.tok_test)
        f1_te = f1_score(
            self.y_test,
            np.argmax(test_out.predictions, axis=1),
            average="weighted",
        )
        gap_pp = abs(f1_tr - f1_te) * 100
        row = {
            "epoch": int(state.epoch),
            "f1_train": float(f1_tr),
            "f1_test": float(f1_te),
            "gap_pp": float(gap_pp),
        }
        self.history.append(row)
        print(
            f"  [gap monitor] epoch {row['epoch']}: "
            f"train F1={f1_tr:.4f}, test F1={f1_te:.4f}, gap={gap_pp:.2f} pp"
        )
        if gap_pp > GAP_MAX_PP:
            print(f"  [gap early-stop] gap {gap_pp:.2f} pp > {GAP_MAX_PP} — stopping.")
            control.should_training_stop = True


def classify_result(f1_test: float, gap_pp: float) -> str:
    gap_ok = gap_pp < GAP_MAX_PP
    if f1_test > BASELINE_F1 and gap_ok:
        return "IMPROVED"
    if f1_test >= BASELINE_F1 - 0.005 and gap_ok:
        return "SIMILAR"
    return "WORSE"


def main() -> None:
    with open(ROOT / "configs" / "pipeline.yaml") as f:
        pipe_cfg = yaml.safe_load(f)
    target = pipe_cfg["data"]["target_binary"]
    text_col = pipe_cfg["data"]["text_column"]
    rand = pipe_cfg["pipeline"]["random_state"]
    test_size = pipe_cfg["pipeline"]["test_size"]

    data_path = ROOT / "data" / "processed" / "v2" / "comments_preprocessed.csv"
    df = pd.read_csv(data_path)
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df[text_col] != ""].copy()
    df[target] = df[target].astype(int)
    X, y = df[text_col], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rand, stratify=y
    )

    set_seed(rand)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tok_train = tokenize_dataset(build_hf_dataset(X_train, y_train), tokenizer)
    tok_test = tokenize_dataset(build_hf_dataset(X_test, y_test), tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, ignore_mismatched_sizes=True
    )
    hard_freeze_differential(model)

    epochs = 10 if not torch.cuda.is_available() else 8
    training_args = TrainingArguments(
        output_dir=str(SAVE_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        warmup_ratio=0.1,
        logging_steps=25,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=rand,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "f1_weighted": f1_score(labels, preds, average="weighted"),
            "f1_toxic": f1_score(labels, preds, pos_label=1),
        }

    trainer_holder: list[Trainer | None] = [None]
    trainer = LabelSmoothingTrainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_test,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        label_smoothing=LABEL_SMOOTHING,
    )
    trainer_holder[0] = trainer
    gap_cb = GapEarlyStopCallback(trainer_holder, tok_test, y_test.values)
    trainer.add_callback(gap_cb)

    print(f"Training {MODEL_ID} (up to {epochs} epochs)...")
    trainer.train()

    test_out = trainer.predict(tok_test)
    preds = np.argmax(test_out.predictions, axis=1)
    probs = torch.softmax(torch.tensor(test_out.predictions), dim=1)[:, 1].numpy()
    f1_test = float(f1_score(y_test, preds, average="weighted"))
    train_out = trainer.predict(tok_train)
    f1_train = float(
        f1_score(
            train_out.label_ids,
            np.argmax(train_out.predictions, axis=1),
            average="weighted",
        )
    )
    gap_pp = abs(f1_train - f1_test) * 100
    status = classify_result(f1_test, gap_pp)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if status != "WORSE":
        trainer.save_model(str(SAVE_DIR))
        tokenizer.save_pretrained(str(SAVE_DIR))

    payload = {
        "strategy": (
            f"{MODEL_ID} (substitute for {MODEL_ID_REQUESTED}) | "
            f"diff-LR | label_smooth={LABEL_SMOOTHING}"
        ),
        "model_id_requested": MODEL_ID_REQUESTED,
        "model_id_used": MODEL_ID,
        "experiment": "track_a_nb11",
        "baseline": {"f1_test": BASELINE_F1, "gap_pp": BASELINE_GAP_PP},
        "result": {
            "f1_train": round(f1_train, 4),
            "f1_test": round(f1_test, 4),
            "train_test_gap_pp": round(gap_pp, 2),
            "gap_ok": gap_pp < GAP_MAX_PP,
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "epoch_history": gap_cb.history,
        },
        "status": status,
        "model_path": str(SAVE_DIR.relative_to(ROOT)) if status != "WORSE" else None,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2))

    if status == "WORSE":
        import shutil

        for path in (SAVE_DIR, SAVE_DIR / "checkpoints"):
            if path.exists():
                shutil.rmtree(path)
        print("Discarded experiment artifacts (WORSE).")

    print("\n--- Experiment summary ---")
    print(f"Strategy: {payload['strategy']}")
    print(f"F1: {BASELINE_F1:.4f} -> {f1_test:.4f}")
    print(f"Gap: {BASELINE_GAP_PP:.2f} -> {gap_pp:.2f} pp")
    print(f"Status: {status}")
    print(f"Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
