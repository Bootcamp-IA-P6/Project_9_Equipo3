import time

from fastapi import APIRouter, HTTPException, Query

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
from src.api.youtube import CommentsFetchError, extract_video_id, fetch_comments
from src.db.supabase_client import list_predictions, save_prediction

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    response = predict_single(request.text, request.threshold)
    if request.persist:
        save_prediction(
            text=request.text,
            result=response,
            source="user_comment" if request.author else "api_direct",
            video_id=request.video_id,
            author=request.author,
            threshold=request.threshold,
            latency_ms=response.latency_ms,
        )
    return response


@router.post("/predict-batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    t0 = time.perf_counter()
    results: list[PredictResponse] = []
    for text in request.texts:
        if not text.strip():
            continue
        single = predict_single(text.strip(), request.threshold)
        results.append(single)
        save_prediction(
            text=text.strip(),
            result=single,
            source="api_direct",
            threshold=request.threshold,
            latency_ms=single.latency_ms,
        )
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

    video_id = extract_video_id(request.url)

    t0 = time.perf_counter()
    results: list[PredictResponse] = []
    service = get_state()["service"]
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    for text in comments:
        if not text.strip():
            continue
        raw = service.predict(text)
        response = to_predict_response(text, raw, 0.0, request.threshold)
        results.append(response)
        save_prediction(
            text=text,
            result=response,
            source="video_fetch",
            video_id=video_id,
            video_url=request.url,
            threshold=request.threshold,
            latency_ms=response.latency_ms,
        )

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


@router.get("/predictions")
async def get_predictions(
    video_id: str | None = Query(default=None),
    source: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    rows = list_predictions(video_id=video_id, source=source, limit=limit)
    return {"predictions": rows, "total": len(rows)}
