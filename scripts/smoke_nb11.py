#!/usr/bin/env python3
"""Minimal DistilBERT hard-freeze smoke (2 epochs, CPU)."""
from pathlib import Path
import random
import yaml
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "configs/pipeline.yaml") as f:
    pipe_cfg = yaml.safe_load(f)
TARGET = pipe_cfg["data"]["target_binary"]
RAND = pipe_cfg["pipeline"]["random_state"]
TEST_SIZE = pipe_cfg["pipeline"]["test_size"]
GAP_MAX = 5.0
F1_MIN = 0.70

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "distilbert-base-uncased"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_hf(X, y):
    return Dataset.from_pandas(
        pd.DataFrame({"text": X.values, "label": y.astype(int).values})
    )


def tokenize(ds, tokenizer, max_len=128):
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    out = ds.map(_tok, batched=True)
    return out.remove_columns(["text"]).rename_column("label", "labels")


def freeze_distilbert(model):
    for p in model.base_model.parameters():
        p.requires_grad = False
    layers = list(model.base_model.transformer.layer)
    for p in layers[-1].parameters():
        p.requires_grad = True
    for name, p in model.named_parameters():
        if "classifier" in name or "pre_classifier" in name:
            p.requires_grad = True


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"f1_weighted": f1_score(labels, preds, average="weighted")}


df = pd.read_csv(ROOT / "data/processed/v2/comments_preprocessed.csv")
df["Text"] = df["Text"].fillna("").astype(str).str.strip()
df = df[df["Text"] != ""].copy()
X, y = df["Text"], df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RAND, stratify=y
)

set_seed(RAND)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
hf_train = tokenize(build_hf(X_train, y_train), tokenizer)
hf_test = tokenize(build_hf(X_test, y_test), tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=2, ignore_mismatched_sizes=True
)
model.config.hidden_dropout_prob = 0.5
model.config.attention_probs_dropout_prob = 0.5
freeze_distilbert(model)
model.to(device)

args = TrainingArguments(
    output_dir=str(ROOT / "models" / "_smoke_distilbert"),
    learning_rate=2e-5,
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    seed=RAND,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=hf_train,
    eval_dataset=hf_test,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)
trainer.train()
out = trainer.predict(hf_test)
preds = np.argmax(out.predictions, axis=1)
f1_te = f1_score(y_test, preds, average="weighted")
pred_tr = trainer.predict(tokenize(build_hf(X_train, y_train), tokenizer)).predictions
f1_tr = f1_score(y_train, np.argmax(pred_tr, axis=1), average="weighted")
gap = abs(f1_tr - f1_te) * 100
print(f"f1_train={f1_tr:.4f} f1_test={f1_te:.4f} gap={gap:.2f}pp")
print(f"pass gap={gap<GAP_MAX} f1={f1_te>=F1_MIN}")
