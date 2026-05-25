"""YouTube comment fetch and suggested-video metadata."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUGGESTED_CONFIG = PROJECT_ROOT / "configs" / "suggested_videos.yaml"

_VIDEO_ID_PATTERNS = (
    r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
    r"youtu\.be/([a-zA-Z0-9_-]{11})",
    r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
)


class CommentsFetchError(Exception):
    """Raised when comments cannot be fetched and demo fallback must not be used."""


def extract_video_id(url: str) -> str | None:
    for pattern in _VIDEO_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def load_suggested_config() -> dict[str, Any]:
    if not SUGGESTED_CONFIG.exists():
        return {"max_comments": 15, "videos": [{"id": "jNQXAC9IVRw"}]}
    with SUGGESTED_CONFIG.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_youtube_error(exc: Exception) -> str:
    err_text = str(exc)
    if "commentsDisabled" in err_text:
        return "Comments are disabled on this video"
    if "disabled comments" in err_text.lower():
        return "Comments are disabled on this video"
    if "quota" in err_text.lower():
        return "YouTube API quota exceeded"
    try:
        from googleapiclient.errors import HttpError

        if isinstance(exc, HttpError):
            for detail in getattr(exc, "error_details", []) or []:
                reason = detail.get("reason") if isinstance(detail, dict) else None
                if reason == "commentsDisabled":
                    return "Comments are disabled on this video"
    except ImportError:
        pass
    return err_text


def fetch_comments(url: str, max_comments: int) -> tuple[list[str], str]:
    video_id = extract_video_id(url) or "unknown"
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if api_key:
        return _fetch_via_api(url, api_key, max_comments, video_id)
    return _demo_comments(video_id, max_comments), "demo"


def _fetch_via_api(
    url: str, api_key: str, max_comments: int, video_id: str
) -> tuple[list[str], str]:
    from googleapiclient.discovery import build

    if video_id == "unknown":
        raise CommentsFetchError(f"Could not parse video id from: {url}")

    youtube = build("youtube", "v3", developerKey=api_key)
    comments: list[str] = []
    page_token = None

    try:
        while len(comments) < max_comments:
            response = (
                youtube.commentThreads()
                .list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, max_comments - len(comments)),
                    pageToken=page_token,
                    textFormat="plainText",
                    order="time",  # newest first
                )
                .execute()
            )
            for item in response.get("items", []):
                text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(text)
            page_token = response.get("nextPageToken")
            if not page_token:
                break
    except Exception as exc:
        message = _parse_youtube_error(exc)
        logger.warning("YouTube API failed for %s: %s", video_id, message)
        raise CommentsFetchError(message) from exc

    if not comments:
        raise CommentsFetchError("No comments found for this video")

    logger.info("YouTube API: fetched %s comments for %s", len(comments), video_id)
    return comments[:max_comments], "youtube"


def fetch_video_metadata(video_ids: list[str]) -> list[dict[str, Any]]:
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key or not video_ids:
        return [_placeholder_meta(vid) for vid in video_ids]

    try:
        from googleapiclient.discovery import build

        youtube = build("youtube", "v3", developerKey=api_key)
        response = (
            youtube.videos()
            .list(part="snippet,status", id=",".join(video_ids))
            .execute()
        )
        by_id: dict[str, dict[str, Any]] = {}
        for item in response.get("items", []):
            vid = item["id"]
            snip = item["snippet"]
            status = item.get("status", {})
            thumbs = snip.get("thumbnails", {})
            thumb = thumbs.get("medium") or thumbs.get("default") or {}
            embeddable = status.get("embeddable", True)
            by_id[vid] = {
                "id": vid,
                "title": snip.get("title", vid),
                "channel_title": snip.get("channelTitle", "Unknown"),
                "thumbnail_url": thumb.get("url", f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg"),
                "watch_url": f"https://www.youtube.com/watch?v={vid}",
                "embeddable": bool(embeddable),
            }
        return [by_id.get(vid, _placeholder_meta(vid)) for vid in video_ids]
    except Exception as exc:
        logger.warning("YouTube metadata failed: %s", exc)
        return [_placeholder_meta(vid) for vid in video_ids]


def _placeholder_meta(video_id: str) -> dict[str, Any]:
    return {
        "id": video_id,
        "title": f"Video {video_id}",
        "channel_title": "YouTube",
        "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
        "watch_url": f"https://www.youtube.com/watch?v={video_id}",
        "embeddable": True,
    }


def _demo_comments(video_id: str, max_comments: int) -> list[str]:
    import random

    logger.info("Using demo comments for %s (no YOUTUBE_API_KEY)", video_id)
    examples = [
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
        "Well researched video, good job.",
        "Go back to where you came from!",
        "The data presented here is compelling.",
    ]
    rng = random.Random(video_id)
    rotated = examples[:]
    rng.shuffle(rotated)
    return rotated[:max_comments]
