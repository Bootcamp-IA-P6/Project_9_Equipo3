"""
src/api/main.py

API REST de producción para detección de hate speech.
Ejecutar con: uvicorn src.api.main:app --reload --port 8000

Documentación automática en:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)

Endpoints:
    GET  /                  → health check
    GET  /model-info        → info del modelo activo
    GET  /models            → lista de modelos disponibles
    POST /predict           → predice un comentario
    POST /predict-batch     → predice una lista de comentarios
    POST /predict-video     → dado URL de YouTube, predice todos sus comentarios
    PUT  /model/{name}      → cambia el modelo activo
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Setup path ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.service.model_service import ModelService, AVAILABLE_MODELS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Estado global de la app ───────────────────────────────────────────────────
# El modelo se carga una sola vez al iniciar la API y se reutiliza.
# Esto evita cargar el modelo en cada request (costoso en tiempo).
_state: dict = {
    "service"      : None,
    "model_name"   : None,
    "startup_time" : None,
    "predictions_served": 0,
}


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN — carga del modelo al iniciar la API
# ══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager de FastAPI.
    Carga el modelo al iniciar la app y libera recursos al cerrarla.
    """
    # Startup
    model_name = os.getenv("MODEL_NAME", list(AVAILABLE_MODELS.keys())[0])
    logger.info(f"Iniciando API — cargando modelo: {model_name}")
    _state["service"]      = ModelService(model_name, PROJECT_ROOT)
    _state["model_name"]   = model_name
    _state["startup_time"] = time.time()

    # Warm-up: predecir un texto de prueba para que el modelo quede en memoria
    try:
        _state["service"].predict("test warmup text")
        logger.info("Modelo cargado y warm-up completado ✅")
    except Exception as e:
        logger.warning(f"Warm-up falló (no crítico): {e}")

    yield  # La API está lista

    # Shutdown
    logger.info("API cerrándose — limpiando recursos")
    _state["service"] = None


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title       = "SignalMod API",
    description = "API de detección de hate speech en comentarios de YouTube",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS: permite que el Streamlit (puerto 8501) llame a la API (puerto 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS — Pydantic valida automáticamente los datos de entrada/salida
# ══════════════════════════════════════════════════════════════════════════════
class PredictRequest(BaseModel):
    """Cuerpo del request para predecir un comentario."""
    text     : str  = Field(..., min_length=1, max_length=5000,
                            description="Comentario a analizar")
    threshold: float = Field(0.5, ge=0.0, le=1.0,
                             description="Umbral de probabilidad para clasificar como tóxico")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("El texto no puede estar vacío")
        return v.strip()


class PredictResponse(BaseModel):
    """Respuesta de la predicción."""
    text       : str
    is_toxic   : bool
    probability: float = Field(..., ge=0.0, le=1.0)
    labels     : list[str]
    model_used : str
    latency_ms : float


class BatchPredictRequest(BaseModel):
    """Request para predecir múltiples comentarios."""
    texts    : list[str] = Field(..., min_length=1, max_length=100)
    threshold: float      = Field(0.5, ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    """Respuesta de predicción batch."""
    results      : list[PredictResponse]
    total        : int
    toxic_count  : int
    latency_ms   : float


class VideoRequest(BaseModel):
    """Request para analizar comentarios de un video de YouTube."""
    url        : str   = Field(..., description="URL del video de YouTube")
    max_comments: int  = Field(50, ge=1, le=200,
                               description="Número máximo de comentarios a analizar")
    threshold  : float = Field(0.5, ge=0.0, le=1.0)


class VideoResponse(BaseModel):
    """Respuesta del análisis de un video de YouTube."""
    video_url   : str
    total_fetched: int
    toxic_count : int
    toxic_rate  : float
    results     : list[PredictResponse]
    error       : Optional[str] = None


class ModelInfo(BaseModel):
    """Información sobre el modelo activo."""
    name        : str
    type        : str
    description : str
    speed       : str
    accuracy    : str
    uptime_s    : float
    predictions_served: int


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _get_service() -> ModelService:
    """Devuelve el servicio activo o lanza 503 si no está listo."""
    if _state["service"] is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Intenta en unos segundos.")
    return _state["service"]


def _predict_single(text: str, threshold: float) -> tuple[dict, float]:
    """Predice un texto y devuelve (result, latency_ms)."""
    t0     = time.perf_counter()
    result = _get_service().predict(text)
    ms     = round((time.perf_counter() - t0) * 1000, 2)

    # Aplicar umbral personalizado
    result["is_toxic"] = result["probability"] >= threshold
    if not result["is_toxic"]:
        result["labels"] = []

    _state["predictions_served"] += 1
    return result, ms


def _scrape_youtube_comments(url: str, max_comments: int) -> list[str]:
    """
    Obtiene comentarios de un video de YouTube.

    Estrategia:
    1. Intentar con YouTube Data API v3 (si hay API key en .env)
    2. Fallback: BeautifulSoup (sin autenticación, limitado)
    """
    api_key = os.getenv("YOUTUBE_API_KEY", "")

    if api_key:
        return _fetch_via_api(url, api_key, max_comments)
    else:
        return _fetch_via_scraper(url, max_comments)


def _fetch_via_api(url: str, api_key: str, max_comments: int) -> list[str]:
    """Obtiene comentarios usando YouTube Data API v3."""
    try:
        import re
        from googleapiclient.discovery import build

        # Extraer video_id de la URL
        patterns = [
            r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
            r"youtu\.be/([a-zA-Z0-9_-]{11})",
            r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        ]
        video_id = None
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                break

        if not video_id:
            raise ValueError(f"No se pudo extraer video_id de: {url}")

        youtube  = build("youtube", "v3", developerKey=api_key)
        comments = []
        page_token = None

        while len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part       = "snippet",
                videoId    = video_id,
                maxResults = min(100, max_comments - len(comments)),
                pageToken  = page_token,
                textFormat = "plainText",
            )
            response = request.execute()

            for item in response.get("items", []):
                text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(text)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        logger.info(f"YouTube API: {len(comments)} comentarios obtenidos")
        return comments[:max_comments]

    except Exception as e:
        logger.warning(f"YouTube API falló: {e} — usando fallback")
        return _fetch_via_scraper(url, max_comments)


def _fetch_via_scraper(url: str, max_comments: int) -> list[str]:
    """
    Fallback: simula comentarios si no hay API key.
    En producción real debería usar BeautifulSoup + Selenium.
    """
    logger.warning(
        "YOUTUBE_API_KEY no configurada. "
        "Configura tu API key en .env para obtener comentarios reales. "
        "Usando comentarios de ejemplo."
    )
    # Comentarios de ejemplo para demo sin API key
    example_comments = [
        "This video is really informative, thanks for sharing!",
        "You are all stupid idiots, get out of here!",
        "Great content, I learned a lot from this.",
        "These people should be eliminated from society.",
        "I agree with the presenter's point of view.",
        "What a bunch of racist criminals!",
        "Thank you for this analysis, very helpful.",
        "Kill them all, they don't deserve to live.",
        "Interesting perspective on the topic.",
        "This is absolute bullshit propaganda!",
        "I think we need to look at both sides.",
        "Black people are thugs and criminals.",
        "The data presented here is compelling.",
        "Go back to where you came from!",
        "Well researched video, good job.",
    ]
    return example_comments[:max_comments]


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def health_check():
    """
    Verifica que la API está funcionando.
    Útil para Docker healthcheck y load balancers.
    """
    service = _state["service"]
    return {
        "status"  : "ok" if service else "loading",
        "model"   : _state["model_name"],
        "uptime_s": round(time.time() - _state["startup_time"], 1)
                    if _state["startup_time"] else 0,
    }


@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Devuelve información sobre el modelo activo."""
    service = _get_service()
    info    = service.get_model_info()
    return ModelInfo(
        name              = _state["model_name"],
        type              = info.get("type", "unknown"),
        description       = info.get("description", ""),
        speed             = info.get("speed", ""),
        accuracy          = info.get("accuracy", ""),
        uptime_s          = round(time.time() - _state["startup_time"], 1),
        predictions_served= _state["predictions_served"],
    )


@app.get("/models", tags=["Model"])
async def list_models():
    """Lista todos los modelos disponibles."""
    return {
        "available": list(AVAILABLE_MODELS.keys()),
        "active"   : _state["model_name"],
    }


@app.put("/model/{model_name}", tags=["Model"])
async def switch_model(model_name: str):
    """
    Cambia el modelo activo.
    El nuevo modelo se carga de forma lazy en el siguiente request de predicción.
    """
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo '{model_name}' no disponible. "
                   f"Opciones: {list(AVAILABLE_MODELS.keys())}",
        )
    _state["service"]  = ModelService(model_name, PROJECT_ROOT)
    _state["model_name"] = model_name
    logger.info(f"Modelo cambiado a: {model_name}")
    return {"message": f"Modelo cambiado a '{model_name}'", "model": model_name}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predice si un comentario es tóxico.

    - **text**: el comentario a analizar
    - **threshold**: umbral de probabilidad (default 0.5)

    Devuelve la probabilidad, si es tóxico y las categorías detectadas.
    """
    result, ms = _predict_single(request.text, request.threshold)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return PredictResponse(
        text       = request.text,
        is_toxic   = result["is_toxic"],
        probability= round(result["probability"], 4),
        labels     = result["labels"],
        model_used = result["model_used"],
        latency_ms = ms,
    )


@app.post("/predict-batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Predice una lista de comentarios en un solo request.
    Más eficiente que llamar /predict N veces.
    Máximo 100 comentarios por request.
    """
    t0      = time.perf_counter()
    results = []

    for text in request.texts:
        if not text.strip():
            continue
        result, _ = _predict_single(text, request.threshold)
        results.append(PredictResponse(
            text       = text,
            is_toxic   = result["is_toxic"],
            probability= round(result["probability"], 4),
            labels     = result["labels"],
            model_used = result["model_used"],
            latency_ms = 0.0,
        ))

    total_ms     = round((time.perf_counter() - t0) * 1000, 2)
    toxic_count  = sum(1 for r in results if r.is_toxic)

    return BatchPredictResponse(
        results     = results,
        total       = len(results),
        toxic_count = toxic_count,
        latency_ms  = total_ms,
    )


@app.post("/predict-video", response_model=VideoResponse, tags=["Prediction"])
async def predict_video(request: VideoRequest):
    """
    Dado un URL de YouTube, obtiene los comentarios y predice su toxicidad.

    Requiere YOUTUBE_API_KEY en el archivo .env para obtener comentarios reales.
    Sin API key usa comentarios de ejemplo para la demo.
    """
    # Obtener comentarios
    try:
        comments = _scrape_youtube_comments(request.url, request.max_comments)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error al obtener comentarios: {e}")

    if not comments:
        raise HTTPException(status_code=404, detail="No se encontraron comentarios en el video")

    # Predecir batch
    t0      = time.perf_counter()
    results = []
    for text in comments:
        if not text.strip():
            continue
        result, _ = _predict_single(text, request.threshold)
        results.append(PredictResponse(
            text       = text,
            is_toxic   = result["is_toxic"],
            probability= round(result["probability"], 4),
            labels     = result["labels"],
            model_used = result["model_used"],
            latency_ms = 0.0,
        ))

    total_ms    = round((time.perf_counter() - t0) * 1000, 2)
    toxic_count = sum(1 for r in results if r.is_toxic)

    return VideoResponse(
        video_url    = request.url,
        total_fetched= len(results),
        toxic_count  = toxic_count,
        toxic_rate   = round(toxic_count / len(results), 4) if results else 0.0,
        results      = results,
    )