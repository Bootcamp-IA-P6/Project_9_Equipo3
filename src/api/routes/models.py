import time

from fastapi import APIRouter, HTTPException

from src.api.schemas import ModelInfo, ModelStatusEntry, ModelsStatusResponse
from src.api.services import get_service
from src.api.state import PROJECT_ROOT, get_state
from src.service.model_service import AVAILABLE_MODELS, ModelService, check_model_availability

router = APIRouter(tags=["Model"])


@router.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    service = get_service()
    info = service.get_model_info()
    state = get_state()
    uptime = round(time.time() - state["startup_time"], 1) if state["startup_time"] else 0.0
    return ModelInfo(
        name=state["model_name"],
        type=info.get("type", "unknown"),
        description=info.get("description", ""),
        speed=info.get("speed", ""),
        accuracy=info.get("accuracy", ""),
        uptime_s=uptime,
        predictions_served=state.get("predictions_served", 0),
    )


@router.get("/models/status", response_model=ModelsStatusResponse)
async def models_status():
    state = get_state()
    entries: list[ModelStatusEntry] = []
    for name, cfg in AVAILABLE_MODELS.items():
        available, reason = check_model_availability(name, PROJECT_ROOT)
        entries.append(
            ModelStatusEntry(
                name=name,
                available=available,
                reason=reason,
                type=cfg.get("type", "unknown"),
            )
        )
    return ModelsStatusResponse(models=entries, active=state["model_name"] or "")


@router.get("/models")
async def list_models():
    state = get_state()
    return {"available": list(AVAILABLE_MODELS.keys()), "active": state["model_name"]}


@router.put("/model/{model_name}")
async def switch_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Options: {list(AVAILABLE_MODELS.keys())}",
        )

    available, reason = check_model_availability(model_name, PROJECT_ROOT)
    if not available:
        raise HTTPException(status_code=400, detail=reason or "Model unavailable")

    state = get_state()
    prev_service = state["service"]
    prev_name = state["model_name"]

    new_service = ModelService(model_name, PROJECT_ROOT)
    warmup = new_service.predict("warmup")
    if warmup.get("error"):
        state["service"] = prev_service
        state["model_name"] = prev_name
        raise HTTPException(status_code=400, detail=str(warmup["error"]))

    state["service"] = new_service
    state["model_name"] = model_name
    return {"message": f"Active model set to '{model_name}'", "model": model_name}
