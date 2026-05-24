"""Load inference model catalog from configs/model_catalog.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "configs" / "model_catalog.yaml"

_DEFAULT_CATALOG: dict[str, dict[str, Any]] = {
    "LR + TF-IDF (local)": {
        "type": "local",
        "icon": "⚡",
        "description": "Project baseline.",
        "speed": "< 50ms",
        "accuracy": "F1 0.76",
        "requires": "joblib only",
    },
}


def load_model_catalog() -> dict[str, dict[str, Any]]:
    if not CATALOG_PATH.exists():
        return dict(_DEFAULT_CATALOG)
    with CATALOG_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or not data:
        return dict(_DEFAULT_CATALOG)
    return data
