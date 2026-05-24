"""Tests for POST /predict."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api.state import get_state

PREDICT_RESPONSE_KEYS = {
    "text",
    "is_toxic",
    "probability",
    "status",
    "mode",
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
        state = get_state()
        state["service"] = mock_service
        state["model_name"] = "LR + TF-IDF (local)"
        state["predictions_served"] = 0
        state["startup_time"] = 0.0
        yield test_client

    state = get_state()
    state["service"] = None
    state["model_name"] = None


def test_predict_returns_correct_structure(client: TestClient):
    response = client.post(
        "/predict",
        json={"text": "This is a sample comment", "threshold": 0.5},
    )

    assert response.status_code == 200
    data = response.json()
    assert PREDICT_RESPONSE_KEYS <= set(data.keys())
    assert data["text"] == "This is a sample comment"
    assert data["status"] == "Safe"
    assert data["mode"] == "binary"
    assert isinstance(data["is_toxic"], bool)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_rejects_empty_text(client: TestClient):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_health_includes_project_name(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["project"] == "youtube_hate_detector"


def test_predict_video_demo_comments_differ_by_url(client: TestClient, monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)

    r1 = client.post(
        "/predict-video",
        json={
            "url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "max_comments": 5,
            "threshold": 0.5,
        },
    )
    r2 = client.post(
        "/predict-video",
        json={
            "url": "https://www.youtube.com/watch?v=IEEhzQoKtQU",
            "max_comments": 5,
            "threshold": 0.5,
        },
    )

    assert r1.status_code == 200
    assert r2.status_code == 200
    data1 = r1.json()
    data2 = r2.json()
    assert data1["source"] == "demo"
    assert data2["source"] == "demo"
    assert data1["results"][0]["text"] != data2["results"][0]["text"]


def test_finetuned_local_reports_lfs_when_pointer_only():
    from src.api.state import PROJECT_ROOT
    from src.service.model_service import check_model_availability

    weights = PROJECT_ROOT / "models" / "finetuned_hf" / "model.safetensors"
    if not weights.is_file() or weights.stat().st_size >= 4096:
        pytest.skip("finetuned_hf weights present or missing — LFS pointer test N/A")

    ok, reason = check_model_availability("Fine-tuned (local HF)", PROJECT_ROOT)
    assert ok is False
    assert reason is not None
    assert "materialize" in reason.lower() or "lfs" in reason.lower()


def test_select_model_via_post(client: TestClient):
    response = client.post(
        "/models/select",
        json={"model_name": "LR + TF-IDF (local)"},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "LR + TF-IDF (local)"


def test_models_status_lists_catalog(client: TestClient):
    response = client.get("/models/status")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 1
    names = {m["name"] for m in data["models"]}
    assert "LR + TF-IDF (local)" in names


def test_predict_video_comments_disabled_raises_422(client: TestClient, monkeypatch):
    from src.api.youtube import CommentsFetchError

    monkeypatch.setenv("YOUTUBE_API_KEY", "fake-key")

    def _raise_disabled(*_args, **_kwargs):
        raise CommentsFetchError("Comments are disabled on this video")

    monkeypatch.setattr("src.api.routes.predict.fetch_comments", _raise_disabled)

    response = client.post(
        "/predict-video",
        json={
            "url": "https://www.youtube.com/watch?v=disabled123",
            "max_comments": 5,
            "threshold": 0.5,
        },
    )
    assert response.status_code == 422
    assert "disabled" in response.json()["detail"].lower()
