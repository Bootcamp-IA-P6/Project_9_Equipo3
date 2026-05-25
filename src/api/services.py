"""Prediction helpers used by route handlers."""

from __future__ import annotations

import time

from fastapi import HTTPException

from src.api.schemas import PredictResponse
from src.api.state import get_state
from src.service.model_service import ModelService


def get_service() -> ModelService:
    state = get_state()
    if state["service"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again shortly.")
    return state["service"]


def to_predict_response(text: str, result: dict, latency_ms: float, threshold: float) -> PredictResponse:
    proba = float(result["probability"])
    is_toxic = proba >= threshold
    labels = result.get("labels", []) if is_toxic else []
    return PredictResponse(
        text=text,
        is_toxic=is_toxic,
        probability=round(proba, 4),
        status="Toxic" if is_toxic else "Safe",
        mode="binary",
        labels=labels,
        model_used=result.get("model_used", ""),
        latency_ms=latency_ms,
    )


def predict_single(text: str, threshold: float) -> PredictResponse:
    state = get_state()
    t0 = time.perf_counter()
    result = get_service().predict(text)
    ms = round((time.perf_counter() - t0) * 1000, 2)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    state["predictions_served"] = state.get("predictions_served", 0) + 1
    return to_predict_response(text, result, ms, threshold)
