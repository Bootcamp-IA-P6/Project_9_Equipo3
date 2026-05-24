import time

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    VideoRequest,
    VideoResponse,
)
from src.api.services import predict_single, to_predict_response
from src.api.state import get_state
from src.api.youtube import CommentsFetchError, fetch_comments
router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    return predict_single(request.text, request.threshold)


@router.post("/predict-batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    t0 = time.perf_counter()
    results: list[PredictResponse] = []
    for text in request.texts:
        if not text.strip():
            continue
        results.append(predict_single(text.strip(), request.threshold))
    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    toxic_count = sum(1 for r in results if r.is_toxic)
    return BatchPredictResponse(
        results=results,
        total=len(results),
        toxic_count=toxic_count,
        latency_ms=total_ms,
    )


@router.post("/predict-video", response_model=VideoResponse)
async def predict_video(request: VideoRequest):
    try:
        comments, source = fetch_comments(request.url, request.max_comments)
    except CommentsFetchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to fetch comments: {exc}") from exc

    if not comments:
        raise HTTPException(status_code=404, detail="No comments found for this video")

    t0 = time.perf_counter()
    results: list[PredictResponse] = []
    service = get_state()["service"]
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    for text in comments:
        if not text.strip():
            continue
        raw = service.predict(text)
        results.append(to_predict_response(text, raw, 0.0, request.threshold))

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    toxic_count = sum(1 for r in results if r.is_toxic)
    get_state()["predictions_served"] = get_state().get("predictions_served", 0) + len(results)

    return VideoResponse(
        video_url=request.url,
        total_fetched=len(results),
        toxic_count=toxic_count,
        toxic_rate=round(toxic_count / len(results), 4) if results else 0.0,
        results=results,
        source=source,
    )
