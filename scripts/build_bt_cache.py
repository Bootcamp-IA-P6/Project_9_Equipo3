#!/usr/bin/env python3
"""Build back-translation cache for notebook 12 (toxic train only)."""
import json
import random
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "configs/pipeline.yaml") as f:
    pipe_cfg = yaml.safe_load(f)
TARGET = pipe_cfg["data"]["target_binary"]
RAND = pipe_cfg["pipeline"]["random_state"]
TEST_SIZE = pipe_cfg["pipeline"]["test_size"]

from deep_translator import GoogleTranslator

df = pd.read_csv(ROOT / "data/processed/v2/comments_preprocessed.csv")
df["clean_text"] = df["clean_text"].fillna("").astype(str)
X, y = df["clean_text"], df[TARGET]
X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RAND, stratify=y)

mask = y_train.astype(bool)
X_toxic = X_train[mask].tolist()
y_toxic = y_train[mask].tolist()

to_es = GoogleTranslator(source="en", target="es")
to_en = GoogleTranslator(source="es", target="en")
texts_bt, labels_bt = [], []
random.seed(RAND)

for i, (text, label) in enumerate(zip(X_toxic, y_toxic)):
    if len(text.split()) < 3:
        continue
    try:
        text_short = " ".join(text.split()[:60])
        es = to_es.translate(text_short)
        back = to_en.translate(es)
        if back and back.strip() != text_short.strip():
            texts_bt.append(back.strip())
            labels_bt.append(bool(label))
        if i % 30 == 0 and i > 0:
            time.sleep(0.5)
            print(f"  {i}/{len(X_toxic)} -> {len(texts_bt)} ok")
    except Exception as exc:
        print(f"skip {i}: {exc}")

out = ROOT / "data/processed/v2/backtranslation_toxic_train.json"
out.write_text(json.dumps({"texts": texts_bt, "labels": labels_bt}, indent=2))
print(f"Wrote {len(texts_bt)} samples to {out}")
