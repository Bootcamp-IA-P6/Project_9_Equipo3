#!/usr/bin/env python3
"""Download real HF weights into models/finetuned_hf (no Git LFS required).

The repo may only contain a Git LFS pointer for model.safetensors (~134 bytes).
This script saves a compatible DistilBERT toxic classifier from Hugging Face Hub
so "Fine-tuned (local HF)" can load offline after one download.

Run from repo root:
  uv sync --extra hf
  uv run python scripts/materialize_finetuned_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "models" / "finetuned_hf"
# Same architecture family as notebook 08 (DistilBERT sequence classification)
HUB_ID = "martin-ha/toxic-comment-model"


def main() -> int:
    weights = OUT_DIR / "model.safetensors"
    if weights.is_file() and weights.stat().st_size > 1_000_000:
        print(f"OK: {weights} already exists ({weights.stat().st_size // 1_000_000} MB)")
        return 0

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("Install HF deps first: uv sync --extra hf", file=sys.stderr)
        return 1

    print(f"Downloading {HUB_ID} into {OUT_DIR} …")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(HUB_ID)
    tokenizer = AutoTokenizer.from_pretrained(HUB_ID)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    meta = OUT_DIR / "model_metadata.json"
    if not meta.exists():
        meta.write_text(
            '{"model_name":"DistilBERT (materialized from Hub)","note":"Run notebook 08 to replace with team weights"}\n',
            encoding="utf-8",
        )

    size_mb = weights.stat().st_size // 1_000_000 if weights.is_file() else 0
    print(f"Done. {weights} ({size_mb} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
