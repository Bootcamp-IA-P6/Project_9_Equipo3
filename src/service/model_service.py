"""
src/services/model_service.py

Servicio centralizado de predicción de toxicidad.

Modelos soportados:
  local      → models/final_model.joblib  (LR + TF-IDF, instantáneo)
  hf_remote  → HuggingFace Hub (requiere internet + transformers)
  hf_local   → modelo HF fine-tuneado localmente (notebook 08)

Instalación para modelos HF:
  pip install transformers torch sentencepiece accelerate
"""

import re
import yaml
import joblib
from pathlib import Path
from typing import Optional

# ─── Catálogo de modelos ──────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "LR + TF-IDF (local)": {
        "type"       : "local",
        "icon"       : "⚡",
        "description": "Modelo del proyecto. Sin GPU, instantáneo.",
        "speed"      : "< 50ms",
        "accuracy"   : "F1 0.76",
        "requires"   : "Solo joblib",
    },
    "DistilBERT Toxicity": {
        "type"       : "hf_remote",
        "icon"       : "🤖",
        "model_id"   : "martin-ha/toxic-comment-model",
        "description": "DistilBERT fine-tuned en comentarios tóxicos.",
        "speed"      : "~200ms CPU",
        "accuracy"   : "F1 0.85",
        "requires"   : "transformers torch",
    },
    "toxic-bert (multilabel)": {
        "type"       : "hf_remote",
        "icon"       : "🧠",
        "model_id"   : "unitary/toxic-bert",
        "description": "BERT multi-label (Jigsaw). Detecta 6 categorías.",
        "speed"      : "~400ms CPU",
        "accuracy"   : "F1 0.88",
        "requires"   : "transformers torch",
    },
    "RoBERTa Toxicity": {
        "type"       : "hf_remote",
        "icon"       : "🔬",
        "model_id"   : "s-nlp/roberta_toxicity_classifier",
        "description": "RoBERTa fine-tuned para toxicidad general.",
        "speed"      : "~350ms CPU",
        "accuracy"   : "F1 0.87",
        "requires"   : "transformers torch",
    },
    "Modelo fine-tuneado (local)": {
        "type"       : "hf_local",
        "icon"       : "✨",
        "model_path" : "models/finetuned_hf",
        "description": "Tu modelo fine-tuneado en el notebook 08.",
        "speed"      : "Depende del hardware",
        "accuracy"   : "A evaluar",
        "requires"   : "transformers torch",
    },
}

HF_LABEL_MAP = {
    "toxic": "Tóxico", "severe_toxic": "Muy ofensivo",
    "obscene": "Obsceno", "threat": "Amenaza",
    "insult": "Insulto", "identity_hate": "Odio racial",
    "label_1": "Tóxico",
}

_KEYWORD_LABELS = {
    "Insulto"    : ["idiot","stupid","dumb","fool","moron","loser"],
    "Odio racial": ["thug","racist","race","criminal"],
    "Amenaza"    : ["kill","shoot","die","dead","hurt","attack"],
    "Obsceno"    : ["fuck","shit","ass","bitch","cunt","bastard"],
    "Agresividad": ["hate","despise","disgusting","pathetic","worthless"],
}


def _labels_from_keywords(text: str, probability: float) -> list:
    t = text.lower()
    found = [lbl for lbl, kws in _KEYWORD_LABELS.items() if any(k in t for k in kws)]
    return found if found else (["Contenido ofensivo"] if probability >= 0.5 else [])


class _FallbackPreprocessor:
    _SW = {"the","a","an","and","or","but","in","on","at","to","for",
           "of","with","is","it","this","that","are","was","be","have",
           "has","he","she","they","we","you","i","not","do","did",
           "will","can","would","should","could","from","by","as","if"}
    def transform(self, text):
        t = re.sub(r"http\S+|www\.\S+|@\w+", " ", str(text).lower())
        t = re.sub(r"[^\x00-\x7F]+", " ", t)
        t = re.sub(r"[^a-z\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return " ".join(w for w in t.split() if w not in self._SW and len(w) > 2)


class ModelService:
    def __init__(self, model_name: str, project_root: Optional[Path] = None):
        self.model_name   = model_name
        self.cfg          = AVAILABLE_MODELS.get(model_name) or list(AVAILABLE_MODELS.values())[0]
        self.project_root = project_root or Path.cwd()
        self._model       = None
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
                    raise FileNotFoundError(
                        f"Modelo no encontrado en {path}. Ejecuta el notebook 08 primero."
                    )
                self._load_hf(str(path))
        return self._model

    def _load_local(self):
        for name in ["final_model.joblib","lr_tuned.joblib",
                     "lr_baseline.joblib","best_ensemble.joblib"]:
            p = self.project_root / "models" / name
            if p.exists():
                self._model = joblib.load(p)
                break
        if self._model is None:
            raise FileNotFoundError(f"No hay modelo en {self.project_root / 'models'}")
        try:
            import sys; sys.path.insert(0, str(self.project_root))
            from src.features.text_preprocessor import TextPreprocessor
            self._preprocessor = TextPreprocessor(
                config_path=str(self.project_root / "configs" / "features.yaml")
            )
        except Exception:
            self._preprocessor = _FallbackPreprocessor()

    def _load_hf(self, model_id_or_path: str):
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError("Instala: pip install transformers torch sentencepiece")
        self._model = hf_pipeline(
            "text-classification", model=model_id_or_path,
            return_all_scores=True, truncation=True, max_length=512,
        )

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {"is_toxic": False, "probability": 0.0,
                    "labels": [], "model_used": self.model_name}
        try:
            model = self._get_model()
            if self.cfg["type"] == "local":
                return self._pred_local(text, model)
            return self._pred_hf(text, model)
        except Exception as e:
            return {"is_toxic": False, "probability": 0.0,
                    "labels": [], "model_used": self.model_name, "error": str(e)}

    def _pred_local(self, text, model):
        clean = self._preprocessor.transform(text) or text
        proba = float(model.predict_proba([clean])[0][1])
        tox   = proba >= 0.5
        return {"is_toxic": tox, "probability": proba,
                "labels": _labels_from_keywords(text, proba) if tox else [],
                "model_used": self.model_name}

    def _pred_hf(self, text, pipeline_fn):
        raw   = pipeline_fn(text[:512])
        smap  = {s["label"].lower(): s["score"] for s in (raw[0] if isinstance(raw[0], list) else raw)}
        for key in ("label_1","toxic","toxic_1"):
            if key in smap:
                proba = smap[key]; break
        else:
            neg  = {"label_0","non_toxic","not_toxic","not toxic"}
            vals = [v for k,v in smap.items() if k not in neg]
            proba = max(vals) if vals else 0.0
        tox = proba >= 0.5
        labels = []
        if tox:
            for k,v in smap.items():
                if k not in ("label_0","non_toxic") and v >= 0.35:
                    friendly = HF_LABEL_MAP.get(k, k.replace("_"," ").title())
                    if "no tóxico" not in friendly.lower():
                        labels.append(friendly)
            if not labels:
                labels = ["Contenido ofensivo"]
        return {"is_toxic": tox, "probability": proba,
                "labels": labels, "model_used": self.model_name}

    @staticmethod
    def get_available_models(): return AVAILABLE_MODELS
    def get_model_info(self):   return self.cfg