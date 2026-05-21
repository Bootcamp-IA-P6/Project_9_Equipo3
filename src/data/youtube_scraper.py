"""YouTube comment scraper (Phase 4 — requires YOUTUBE_API_KEY in .env)."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class YouTubeScraperError(Exception):
    pass


def get_api_key() -> str:
    key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not key:
        raise YouTubeScraperError(
            "YOUTUBE_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return key


def fetch_comments_for_video(
    video_id: str,
    *,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """
    Fetch top-level comments for a video via YouTube Data API v3.

    Returns list of dicts: comment_id, text, author, published_at.
    """
    try:
        from googleapiclient.discovery import build
    except ImportError as e:
        raise YouTubeScraperError(
            "Install google-api-python-client: uv add google-api-python-client"
        ) from e

    api_key = get_api_key()
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=min(max_results, 100),
        textFormat="plainText",
    )
    response = request.execute()
    comments = []
    for item in response.get("items", []):
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        comments.append(
            {
                "comment_id": item["id"],
                "text": snippet.get("textDisplay", ""),
                "author": snippet.get("authorDisplayName", ""),
                "published_at": snippet.get("publishedAt", ""),
            }
        )
    return comments


def extract_video_id(url: str) -> str:
    """Parse YouTube video ID from common URL formats."""
    import re

    patterns = [
        r"(?:v=|/embed/|/shorts/)([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    if len(url) == 11 and url.isalnum():
        return url
    raise YouTubeScraperError(f"Could not parse video ID from: {url}")
