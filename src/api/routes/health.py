import time

from fastapi import APIRouter

from src.api.state import get_state

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    state = get_state()
    service = state["service"]
    uptime = 0.0
    if state["startup_time"]:
        uptime = round(time.time() - state["startup_time"], 1)
    return {
        "status": "ok" if service else "loading",
        "model": state["model_name"],
        "uptime_s": uptime,
        "project": "youtube_hate_detector",
    }
