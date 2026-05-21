import sys
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

# Let the test import "src.*" from the project root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.api import main
from src.api.main import app

client = TestClient(app)


def test_api_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert "service" in json_data
    assert json_data["service"] == "YouTube Toxic Comment Detector"


def test_api_health_endpoint():
    # Swap in a fake model so /health believes one is loaded
    main._pipeline = MagicMock()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_api_predict_toxic():
    # Fake model that always reports "toxic"
    mock_pipeline = MagicMock()
    # predict_proba returns [[chance not toxic, chance toxic]]
    import numpy as np
    mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.9]])
    main._pipeline = mock_pipeline

    response = client.post("/predict", json={"text": "this is a hateful comment!"})
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["is_toxic"] is True
    assert json_data["toxic_probability"] == 0.9
    assert json_data["label"] == "toxic"


def test_api_predict_non_toxic():
    mock_pipeline = MagicMock()
    import numpy as np
    mock_pipeline.predict_proba.return_value = np.array([[0.85, 0.15]])
    main._pipeline = mock_pipeline

    response = client.post("/predict", json={"text": "hello, beautiful world!"})
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["is_toxic"] is False
    assert json_data["toxic_probability"] == 0.15
    assert json_data["label"] == "non_toxic"


def test_api_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    # Empty text breaks the min_length=1 rule, so FastAPI answers 422
    assert response.status_code == 422
