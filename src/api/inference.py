"""Load model bundle and run inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from scipy.sparse import csr_matrix, hstack

from src.features.plan_a_dataset import ROOT_FEATURES
from src.features.phase2_text_pipeline import process_text
from src.features.preprocessing import extract_advanced_features
from src.models.phase3_baseline import load_model_bundle

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/config.yaml"

_bundle: dict | None = None


def _load_config() -> dict:
    with DEFAULT_CONFIG.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_bundle() -> dict:
    global _bundle
    if _bundle is None:
        cfg = _load_config()
        path = ROOT / cfg["model"]["path"]
        if not path.exists() and cfg["model"].get("fallback_path"):
            path = ROOT / cfg["model"]["fallback_path"]
        _bundle = load_model_bundle(path)
        _bundle.setdefault("borderline_low", cfg["model"].get("borderline_low", 0.4))
        _bundle.setdefault("borderline_high", cfg["model"].get("borderline_high", 0.6))
    return _bundle


def _vectorize_plan_a(text: str, bundle: dict):
    _, processed = process_text(text)
    adv = extract_advanced_features(pd.DataFrame({"Text": [text]}))
    X_tfidf = bundle["vectorizer"].transform([processed])
    X_num = bundle["scaler"].transform(adv[ROOT_FEATURES].values)
    return hstack([X_tfidf, csr_matrix(X_num)])


def predict_comment(text: str) -> dict:
    """Return toxic score, label, and UI color band."""
    bundle = get_bundle()
    if bundle.get("bundle_type") == "plan_a_hybrid":
        X = _vectorize_plan_a(text, bundle)
    else:
        _, processed = process_text(text)
        X = bundle["vectorizer"].transform([processed])

    clf = bundle["classifier"]
    if hasattr(clf, "predict_proba"):
        proba = float(clf.predict_proba(X)[0, 1])
    else:
        proba = float(clf.predict(X)[0])

    low = bundle["borderline_low"]
    high = bundle["borderline_high"]

    if proba >= high:
        label = "Toxic"
        color = "red"
    elif proba <= low:
        label = "Safe"
        color = "green"
    else:
        label = "Toxic" if proba >= 0.5 else "Safe"
        color = "yellow"

    return {
        "toxic_score": round(proba * 100, 2),
        "label": label,
        "status_color": color,
        "mode": bundle.get("mode", "binary"),
        "version": bundle.get("version", "phase3-esencial"),
        "model_name": bundle.get("model_name", "unknown"),
    }
