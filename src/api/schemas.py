"""FastAPI request/response schemas."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class PredictResponse(BaseModel):
    toxic_score: float = Field(..., description="Toxic probability 0-100")
    label: str = Field(..., description="Safe or Toxic")
    status_color: str = Field(..., description="green, yellow, or red")
    mode: str = "binary"
    version: str = "phase3-esencial"
    model_name: str
