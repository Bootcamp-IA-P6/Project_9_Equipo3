from contextlib import asynccontextmanager
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.config import load_config, project_root

_pipeline = None
_config = None


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Comment text to classify")


class PredictResponse(BaseModel):
    is_toxic: bool
    toxic_probability: float
    label: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _config
    _config = load_config()
    model_path = project_root() / _config["paths"]["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python -m src.pipeline.train"
        )
    _pipeline = joblib.load(model_path)
    yield


app = FastAPI(
    title="YouTube Toxic Comment Detector",
    description="Essential-level API: predicts IsToxic from comment text.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "YouTube Toxic Comment Detector",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "example": {"text": "great video, thanks for sharing"},
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    proba = float(_pipeline.predict_proba([text])[0, 1])
    is_toxic = proba >= 0.5

    return PredictResponse(
        is_toxic=is_toxic,
        toxic_probability=round(proba, 4),
        label="toxic" if is_toxic else "non_toxic",
    )
