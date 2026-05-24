"""Pydantic request/response models for the API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    threshold: float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class PredictResponse(BaseModel):
    text: str
    is_toxic: bool
    probability: float = Field(..., ge=0.0, le=1.0)
    status: Literal["Safe", "Toxic"]
    mode: Literal["binary"] = "binary"
    labels: list[str]
    model_used: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total: int
    toxic_count: int
    latency_ms: float


class VideoRequest(BaseModel):
    url: str
    max_comments: int = Field(50, ge=1, le=200)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class VideoResponse(BaseModel):
    video_url: str
    total_fetched: int
    toxic_count: int
    toxic_rate: float
    results: list[PredictResponse]
    source: Literal["youtube", "demo", "unavailable"] = "demo"
    reason: Optional[str] = None
    error: Optional[str] = None


class ModelStatusEntry(BaseModel):
    name: str
    available: bool
    reason: Optional[str] = None
    type: str = "unknown"


class ModelsStatusResponse(BaseModel):
    models: list[ModelStatusEntry]
    active: str


class ModelInfo(BaseModel):
    name: str
    type: str
    description: str
    speed: str
    accuracy: str
    uptime_s: float
    predictions_served: int


class SuggestedVideo(BaseModel):
    id: str
    title: str
    channel_title: str
    thumbnail_url: str
    watch_url: str
    embeddable: bool = True


class SuggestedVideosResponse(BaseModel):
    videos: list[SuggestedVideo]
    max_comments: int
