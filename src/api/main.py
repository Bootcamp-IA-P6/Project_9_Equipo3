"""
youtube_hate_detector API

Run: uv run uvicorn src.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from src.api.routes import health, models, predict, videos
from src.api.state import PROJECT_ROOT, get_state
from src.service.model_service import AVAILABLE_MODELS, ModelService, check_model_availability
from src.utils.logger import get_logger

logger = get_logger(__name__)

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = get_state()
    model_name = os.getenv("MODEL_NAME", next(iter(AVAILABLE_MODELS.keys())))
    available, reason = check_model_availability(model_name, PROJECT_ROOT)
    if not available:
        fallback = next(iter(AVAILABLE_MODELS.keys()))
        logger.warning(
            "MODEL_NAME '%s' unavailable (%s) — using '%s'",
            model_name,
            reason,
            fallback,
        )
        model_name = fallback
    logger.info("Starting youtube_hate_detector API — model: %s", model_name)
    state["service"] = ModelService(model_name, PROJECT_ROOT)
    state["model_name"] = model_name
    state["startup_time"] = time.time()
    state["predictions_served"] = 0

    try:
        state["service"].predict("warmup")
        logger.info("Model warm-up complete")
    except Exception as exc:
        logger.warning("Warm-up failed (non-critical): %s", exc)

    yield

    state["service"] = None
    logger.info("API shutdown")


app = FastAPI(
    title="youtube_hate_detector API",
    description="Toxic comment detection for YouTube-style moderation demos",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(predict.router)
app.include_router(videos.router)


_API_PATH_ROOTS = frozenset(
    {"models", "model", "videos", "predict", "health", "docs", "redoc", "openapi"}
)


def _is_api_spa_path(full_path: str) -> bool:
    root = full_path.split("/")[0] if full_path else ""
    return root in _API_PATH_ROOTS


def _mount_frontend() -> None:
    if not FRONTEND_DIST.is_dir():
        return
    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if _is_api_spa_path(full_path):
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Not found")
        index = FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(index)
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Not found")


_mount_frontend()
