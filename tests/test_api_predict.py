"""Unit tests for FastAPI predict endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "phase4_best.joblib"
PHASE3 = ROOT / "models" / "phase3_baseline.joblib"


@pytest.fixture(scope="module")
def client():
    if not MODEL.exists() and not PHASE3.exists():
        pytest.skip("No trained model found")
    from src.api.main import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_safe_comment(client):
    r = client.post("/predict", json={"text": "Great tutorial, very helpful!"})
    assert r.status_code == 200
    data = r.json()
    assert "toxic_score" in data
    assert data["label"] in ("Safe", "Toxic")
    assert data["mode"] == "binary"


def test_predict_empty_rejected(client):
    r = client.post("/predict", json={"text": "   "})
    assert r.status_code == 400
