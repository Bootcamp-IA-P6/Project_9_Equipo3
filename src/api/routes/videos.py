from fastapi import APIRouter

from src.api.schemas import SuggestedVideo, SuggestedVideosResponse
from src.api.youtube import fetch_video_metadata, load_suggested_config

router = APIRouter(tags=["Videos"])


@router.get("/videos/suggested", response_model=SuggestedVideosResponse)
async def suggested_videos():
    cfg = load_suggested_config()
    max_comments = int(cfg.get("max_comments", 15))
    entries = cfg.get("videos") or []
    ids = [e["id"] if isinstance(e, dict) else str(e) for e in entries]
    meta = fetch_video_metadata(ids)
    videos = [SuggestedVideo(**m) for m in meta]
    return SuggestedVideosResponse(videos=videos, max_comments=max_comments)
