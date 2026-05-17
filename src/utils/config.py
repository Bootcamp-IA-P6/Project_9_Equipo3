from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    return ROOT
