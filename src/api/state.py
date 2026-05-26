"""Application state shared across routes."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_state: dict = {
    "service": None,
    "model_name": None,
    "startup_time": None,
    "predictions_served": 0,
}


def get_state() -> dict:
    return _state
