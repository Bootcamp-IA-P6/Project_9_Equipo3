"""Tests del endpoint POST /predict."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main

PREDICT_RESPONSE_KEYS = {
    "text",
    "is_toxic",
    "probability",
    "labels",
    "model_used",
    "latency_ms",
}


@pytest.fixture
def client():
    mock_service = MagicMock()
    mock_service.predict.return_value = {
        "is_toxic": False,
        "probability": 0.12,
        "labels": [],
        "model_used": "LR + TF-IDF (local)",
    }

    with TestClient(api_main.app) as test_client:
        api_main._state["service"] = mock_service
        api_main._state["model_name"] = "LR + TF-IDF (local)"
        api_main._state["predictions_served"] = 0
        yield test_client

    api_main._state["service"] = None
    api_main._state["model_name"] = None


def test_predict_returns_correct_structure(client: TestClient):
    response = client.post(
        "/predict",
        json={"text": "This is a sample comment", "threshold": 0.5},
    )

    assert response.status_code == 200
    data = response.json()
    assert PREDICT_RESPONSE_KEYS <= set(data.keys())
    assert data["text"] == "This is a sample comment"
    assert isinstance(data["is_toxic"], bool)
    assert 0.0 <= data["probability"] <= 1.0
    assert isinstance(data["labels"], list)
    assert isinstance(data["model_used"], str)
    assert isinstance(data["latency_ms"], (int, float))


def test_predict_rejects_empty_text(client: TestClient):
    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422
