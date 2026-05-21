"""FastAPI app: /predict, /health, optional React static (AGENTS.md Phase 3)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.inference import predict_comment
from src.api.schemas import PredictRequest, PredictResponse
from src.utils.execution_log import log_event

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = ROOT / "frontend" / "dist"
LOG_PATH = ROOT / "logs" / "execution.log"

app = FastAPI(
    title="YouTube Toxic Comment Detector",
    version="phase3-esencial",
    description="Binary Safe vs Toxic inference API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "binary"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Comment text is empty")
    result = predict_comment(text)
    log_event(f"predict label={result['label']} score={result['toxic_score']}", phase="3")
    return PredictResponse(**result)


@app.post("/scrape")
def scrape_comments(body: PredictRequest):
    """Fetch comments for a YouTube video URL (requires YOUTUBE_API_KEY)."""
    from src.data.youtube_scraper import YouTubeScraperError, extract_video_id, fetch_comments_for_video

    url = body.text.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Video URL is required in text field")
    try:
        video_id = extract_video_id(url)
        comments = fetch_comments_for_video(video_id, max_results=20)
        return {"video_id": video_id, "count": len(comments), "comments": comments}
    except YouTubeScraperError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
