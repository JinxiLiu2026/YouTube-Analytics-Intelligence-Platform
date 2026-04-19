from __future__ import annotations

from fastapi import FastAPI, HTTPException

from youtube_insights.config import Settings
from youtube_insights.schemas import AnalyzeVideosRequest, AnalyzeVideosResponse
from youtube_insights.service import YouTubeInsightService


app = FastAPI(title="YouTube Insight Module", version="0.1.0")


def _build_service() -> YouTubeInsightService:
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return YouTubeInsightService(settings)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeVideosResponse)
def analyze_videos(request: AnalyzeVideosRequest) -> AnalyzeVideosResponse:
    service = _build_service()
    return service.analyze(request)
