from __future__ import annotations

from typing import Literal, Optional

from youtube_insights.config import Settings
from youtube_insights.schemas import AnalyzeVideosRequest, AnalyzeVideosResponse
from youtube_insights.service import YouTubeInsightService


def analyze_youtube_videos(
    description: str,
    video_ids: list[str],
    max_comments: Optional[int] = None,
    settings: Optional[Settings] = None,
    mode: Optional[Literal["fast", "full"]] = None,
) -> AnalyzeVideosResponse:
    resolved_settings = settings or Settings.from_env()
    service = YouTubeInsightService(resolved_settings)
    request = AnalyzeVideosRequest(
        description=description,
        video_ids=video_ids,
        max_comments=max_comments,
        mode=mode,
    )
    return service.analyze(request)
