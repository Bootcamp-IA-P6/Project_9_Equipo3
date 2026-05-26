"""Optional Supabase persistence for predictions.

The API works fine without credentials — all functions degrade
gracefully when ``SUPABASE_URL`` / ``SUPABASE_KEY`` are missing.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover - dep listed in pyproject
    Client = None  # type: ignore[assignment,misc]
    create_client = None  # type: ignore[assignment]


_TABLE = "predictions"


@lru_cache(maxsize=1)
def get_client() -> "Client | None":
    """Return a cached Supabase client, or ``None`` if not configured."""
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        return None
    if create_client is None:
        logger.warning("supabase package not available; persistence disabled")
        return None
    try:
        client = create_client(url, key)
        logger.info("Supabase client initialized")
        return client
    except Exception as exc:  # pragma: no cover - network/config errors
        logger.warning("Failed to initialize Supabase client: %s", exc)
        return None


def save_prediction(
    text: str,
    result: Any,
    source: str,
    video_id: str | None = None,
    video_url: str | None = None,
    threshold: float | None = None,
    latency_ms: float | None = None,
) -> None:
    """Persist a single prediction, silently no-op when DB is not configured.

    ``result`` may be a Pydantic ``PredictResponse`` or a dict with the same
    fields (``probability``, ``is_toxic``, ``labels``, ``model_used``,
    ``latency_ms``).
    """
    client = get_client()
    if client is None:
        return

    try:
        if hasattr(result, "model_dump"):
            data = result.model_dump()
        elif isinstance(result, dict):
            data = result
        else:
            data = {
                "probability": getattr(result, "probability", None),
                "is_toxic": getattr(result, "is_toxic", None),
                "labels": getattr(result, "labels", []),
                "model_used": getattr(result, "model_used", ""),
                "latency_ms": getattr(result, "latency_ms", None),
            }

        row = {
            "text": text,
            "video_id": video_id,
            "video_url": video_url,
            "probability": data.get("probability"),
            "is_toxic": data.get("is_toxic"),
            "labels": data.get("labels", []) or [],
            "model_used": data.get("model_used", ""),
            "threshold": threshold,
            "latency_ms": latency_ms if latency_ms is not None else data.get("latency_ms"),
            "source": source,
        }
        client.table(_TABLE).insert(row).execute()
    except Exception as exc:
        logger.warning("save_prediction failed (non-critical): %s", exc)


def list_predictions(
    video_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Return latest predictions ordered by ``created_at`` desc.

    Returns ``[]`` when the client is not configured.
    """
    client = get_client()
    if client is None:
        return []

    try:
        query = client.table(_TABLE).select("*").order("created_at", desc=True)
        if video_id:
            query = query.eq("video_id", video_id)
        query = query.limit(max(1, min(limit, 200)))
        response = query.execute()
        return list(getattr(response, "data", []) or [])
    except Exception as exc:
        logger.warning("list_predictions failed: %s", exc)
        return []
