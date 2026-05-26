"""Inference for Notebook 14 meta-feature stacking (frozen CLS + metadata + LR)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.features.metadata_features import extract_metadata_features

MODEL_ID = "unitary/toxic-bert"
_EMOJI_PAT = re.compile(
    "["
    "\U0001f300-\U0001f9ff"
    "\U0001f600-\U0001f64f"
    "]+",
    flags=re.UNICODE,
)


def _extended_meta_frame(text: str) -> pd.DataFrame:
    df = pd.DataFrame({"Text": [text]})
    base = extract_metadata_features(df, text_column="Text")
    length = max(len(text), 1)
    base = base.copy()
    base["emoji_count"] = len(_EMOJI_PAT.findall(text))
    base["punctuation_density"] = len(re.findall(r"[^\w\s]", text)) / length
    return base.astype(float)


class MetaStackPredictor:
    """Load production bundle and score a single comment."""

    def __init__(
        self,
        bundle_path: Path,
        *,
        manifest_path: Path | None = None,
        frozen_model_id: str = MODEL_ID,
    ) -> None:
        self.bundle_path = bundle_path
        self.frozen_model_id = frozen_model_id
        self.manifest: dict[str, Any] = {}
        if manifest_path and manifest_path.is_file():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        bundle = joblib.load(bundle_path)
        self.scaler = bundle["scaler"]
        self.clf = bundle["clf"]
        self.meta_columns: list[str] = list(bundle.get("meta_columns", []))
        self.default_threshold = float(self.manifest.get("threshold", 0.381))

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(frozen_model_id)
        self._bert = AutoModelForSequenceClassification.from_pretrained(frozen_model_id)
        for p in self._bert.parameters():
            p.requires_grad = False
        self._bert.eval()
        self._bert.to(self._device)

    def _cls_vector(self, text: str) -> np.ndarray:
        with torch.no_grad():
            enc = self._tokenizer(
                [text],
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            cls = self._bert.bert(**enc).last_hidden_state[:, 0, :].cpu().numpy()
        return cls

    def _feature_row(self, text: str) -> np.ndarray:
        meta = _extended_meta_frame(text)
        if self.meta_columns:
            meta = meta.reindex(columns=self.meta_columns, fill_value=0.0)
        cls = self._cls_vector(text)
        return np.hstack([cls, meta.values.astype(float)])

    def predict_proba(self, text: str) -> float:
        row = self._feature_row(text)
        scaled = self.scaler.transform(row)
        return float(self.clf.predict_proba(scaled)[0][1])

    def predict(self, text: str, *, threshold: float | None = None) -> dict[str, Any]:
        if not text or not text.strip():
            return {
                "is_toxic": False,
                "probability": 0.0,
                "labels": [],
                "recommended_threshold": self.default_threshold,
            }
        proba = self.predict_proba(text)
        thresh = self.default_threshold if threshold is None else threshold
        tox = proba >= thresh
        return {
            "is_toxic": tox,
            "probability": proba,
            "labels": ["Offensive content"] if tox else [],
            "recommended_threshold": self.default_threshold,
        }
