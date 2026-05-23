#!/usr/bin/env python3
"""Train NB 11: unitary/toxic-bert partial freeze + differential LR + label smoothing."""
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
SAVE_DIR = ROOT / "models" / "toxic_bert_frozen"
METRICS_PATH = ROOT / "reports" / "nb11_metrics.json"
MODEL_ID = "unitary/toxic-bert"
GAP_MAX_PP = 5.0
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
    if hasattr(model, "bert"):
        return list(model.bert.encoder.layer)
    if hasattr(model, "distilbert"):
        return list(model.distilbert.transformer.layer)
    return list(model.base_model.transformer.layer)


def hard_freeze_differential(model) -> None:
    backbone = model.bert if hasattr(model, "bert") else model.distilbert
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
        penult, last_head = [], []
        for p in layers[-2].parameters():
            if p.requires_grad:
                penult.append(p)
        for p in layers[-1].parameters():
            if p.requires_grad:
                last_head.append(p)
        for name, p in self.model.named_parameters():
            if p.requires_grad and ("classifier" in name or "pre_classifier" in name):
                last_head.append(p)
        self.optimizer = torch.optim.AdamW(
            [{"params": penult, "lr": LR_PENULT}, {"params": last_head, "lr": LR_LAST_HEAD}],
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(
            outputs.logits, labels, label_smoothing=self.label_smoothing
        )
        return (loss, outputs) if return_outputs else loss


class GapEarlyStopCallback(TrainerCallback):
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
        self.history.append(
            {
                "epoch": int(state.epoch),
                "f1_train": float(f1_tr),
                "f1_test": float(f1_te),
                "gap_pp": float(gap_pp),
            }
        )
        print(
            f"  [gap] epoch {int(state.epoch)}: "
            f"train F1={f1_tr:.4f}, test F1={f1_te:.4f}, gap={gap_pp:.2f} pp"
        )
        if gap_pp > GAP_MAX_PP:
            print(f"  [gap early-stop] {gap_pp:.2f} pp > {GAP_MAX_PP}")
            control.should_training_stop = True


def main() -> None:
    with open(ROOT / "configs" / "pipeline.yaml") as f:
        pipe_cfg = yaml.safe_load(f)
    target = pipe_cfg["data"]["target_binary"]
    text_col = pipe_cfg["data"]["text_column"]
    rand = pipe_cfg["pipeline"]["random_state"]
    test_size = pipe_cfg["pipeline"]["test_size"]

    df = pd.read_csv(ROOT / "data/processed/v2/comments_preprocessed.csv")
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df[text_col] != ""].copy()
    X, y = df[text_col], df[target].astype(int)
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

    print(f"Training {MODEL_ID}...")
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

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))

    metrics = {
        "f1_train": round(f1_train, 4),
        "f1_test": round(f1_test, 4),
        "train_test_gap_pp": round(gap_pp, 2),
        "gap_ok": gap_pp < GAP_MAX_PP,
        "f1_test_ok": f1_test >= 0.70,
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "epoch_history": gap_cb.history,
    }
    payload = {
        "notebook": "11_distilbert_frozen_v2",
        "model_id": MODEL_ID,
        "model_path": str(SAVE_DIR.relative_to(ROOT)),
        "freeze": "penultimate_and_last_encoder_blocks_and_head",
        "lr_penultimate": LR_PENULT,
        "lr_last_and_head": LR_LAST_HEAD,
        "label_smoothing": LABEL_SMOOTHING,
        "weight_decay": 0.01,
        "gap_early_stop_pp": GAP_MAX_PP,
        "metrics": metrics,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()
