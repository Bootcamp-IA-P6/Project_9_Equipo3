"""Centralized toxicity prediction service."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Optional

import joblib

from src.service.model_catalog import load_model_catalog

AVAILABLE_MODELS: dict[str, dict[str, Any]] = load_model_catalog()

_HF_DEPS_MSG = "Install HF deps: uv sync --extra hf"


def hf_deps_available() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def check_model_availability(name: str, project_root: Path | None = None) -> tuple[bool, str | None]:
    """Return (available, reason) for a catalog model name."""
    cfg = AVAILABLE_MODELS.get(name)
    if not cfg:
        return False, "Unknown model"

    root = project_root or Path.cwd()
    model_type = cfg.get("type", "local")

    if model_type == "local":
        models_dir = root / "models"
        if any((models_dir / n).exists() for n in (
            "final_model.joblib",
            "lr_tuned.joblib",
            "lr_baseline.joblib",
            "best_ensemble.joblib",
        )):
            return True, None
        return False, f"No model in {models_dir}"

    if model_type == "hf_local":
        if not hf_deps_available():
            return False, _HF_DEPS_MSG
        path = root / cfg["model_path"]
        if not path.exists():
            return False, f"Model not found at {path}."
        return True, None

    if model_type == "hf_remote":
        if not hf_deps_available():
            return False, _HF_DEPS_MSG
        return True, None

    return False, f"Unsupported model type: {model_type}"

HF_LABEL_MAP = {
    "toxic": "Toxic",
    "severe_toxic": "Severely offensive",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity hate",
    "label_1": "Toxic",
}

_KEYWORD_LABELS = {
    "Insult": ["idiot", "stupid", "dumb", "fool", "moron", "loser"],
    "Identity hate": ["thug", "racist", "race", "criminal"],
    "Threat": ["kill", "shoot", "die", "dead", "hurt", "attack"],
    "Obscene": ["fuck", "shit", "ass", "bitch", "cunt", "bastard"],
    "Aggression": ["hate", "despise", "disgusting", "pathetic", "worthless"],
}


def _labels_from_keywords(text: str, probability: float) -> list[str]:
    t = text.lower()
    found = [lbl for lbl, kws in _KEYWORD_LABELS.items() if any(k in t for k in kws)]
    return found if found else (["Offensive content"] if probability >= 0.5 else [])


class _FallbackPreprocessor:
    _SW = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "is", "it", "this", "that", "are", "was", "be", "have",
        "has", "he", "she", "they", "we", "you", "i", "not", "do", "did",
        "will", "can", "would", "should", "could", "from", "by", "as", "if",
    }

    def transform(self, text: str) -> str:
        t = re.sub(r"http\S+|www\.\S+|@\w+", " ", str(text).lower())
        t = re.sub(r"[^\x00-\x7F]+", " ", t)
        t = re.sub(r"[^a-z\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return " ".join(w for w in t.split() if w not in self._SW and len(w) > 2)


class ModelService:
    def __init__(self, model_name: str, project_root: Optional[Path] = None):
        self.model_name = model_name
        self.cfg = AVAILABLE_MODELS.get(model_name) or next(iter(AVAILABLE_MODELS.values()))
        self.project_root = project_root or Path.cwd()
        self._model = None
        self._preprocessor = None

    def _get_model(self):
        if self._model is None:
            t = self.cfg["type"]
            if t == "local":
                self._load_local()
            elif t == "hf_remote":
                self._load_hf(self.cfg["model_id"])
            elif t == "hf_local":
                path = self.project_root / self.cfg["model_path"]
                if not path.exists():
                    raise FileNotFoundError(f"Model not found at {path}.")
                self._load_hf(str(path))
        return self._model

    def _load_local(self) -> None:
        for name in ("final_model.joblib", "lr_tuned.joblib", "lr_baseline.joblib", "best_ensemble.joblib"):
            p = self.project_root / "models" / name
            if p.exists():
                self._model = joblib.load(p)
                break
        if self._model is None:
            raise FileNotFoundError(f"No model in {self.project_root / 'models'}")
        try:
            sys.path.insert(0, str(self.project_root))
            from src.features.text_preprocessor import TextPreprocessor

            self._preprocessor = TextPreprocessor(
                config_path=str(self.project_root / "configs" / "features.yaml")
            )
        except Exception:
            self._preprocessor = _FallbackPreprocessor()

    def _load_hf(self, model_id_or_path: str) -> None:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError("Install HF deps: uv sync --extra hf") from exc
        self._model = hf_pipeline(
            "text-classification",
            model=model_id_or_path,
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {"is_toxic": False, "probability": 0.0, "labels": [], "model_used": self.model_name}
        try:
            model = self._get_model()
            if self.cfg["type"] == "local":
                return self._pred_local(text, model)
            return self._pred_hf(text, model)
        except Exception as e:
            return {
                "is_toxic": False,
                "probability": 0.0,
                "labels": [],
                "model_used": self.model_name,
                "error": str(e),
            }

    def _pred_local(self, text: str, model) -> dict:
        clean = self._preprocessor.transform(text) or text
        proba = float(model.predict_proba([clean])[0][1])
        tox = proba >= 0.5
        return {
            "is_toxic": tox,
            "probability": proba,
            "labels": _labels_from_keywords(text, proba) if tox else [],
            "model_used": self.model_name,
        }

    def _pred_hf(self, text: str, pipeline_fn) -> dict:
        raw = pipeline_fn(text[:512])
        smap = {s["label"].lower(): s["score"] for s in (raw[0] if isinstance(raw[0], list) else raw)}
        proba = 0.0
        for key in ("label_1", "toxic", "toxic_1"):
            if key in smap:
                proba = smap[key]
                break
        else:
            neg = {"label_0", "non_toxic", "not_toxic", "not toxic"}
            vals = [v for k, v in smap.items() if k not in neg]
            proba = max(vals) if vals else 0.0
        tox = proba >= 0.5
        labels: list[str] = []
        if tox:
            for k, v in smap.items():
                if k not in ("label_0", "non_toxic") and v >= 0.35:
                    friendly = HF_LABEL_MAP.get(k, k.replace("_", " ").title())
                    labels.append(friendly)
            if not labels:
                labels = ["Offensive content"]
        return {"is_toxic": tox, "probability": proba, "labels": labels, "model_used": self.model_name}

    @staticmethod
    def get_available_models() -> dict:
        return AVAILABLE_MODELS

    def get_model_info(self) -> dict:
        return self.cfg
