from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class AnalyzeVideosRequest(BaseModel):
    description: str = Field(min_length=1)
    video_ids: list[str] = Field(min_length=1, max_length=5)
    max_comments: Optional[int] = Field(default=None, ge=0, le=100)
    mode: Optional[Literal["fast", "full"]] = None

    @field_validator("video_ids")
    @classmethod
    def dedupe_video_ids(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in value:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)
        if not deduped:
            raise ValueError("At least one valid video ID is required.")
        if len(deduped) > 5:
            raise ValueError("At most five unique video IDs are supported.")
        return deduped


class VideoStats(BaseModel):
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None


class TranscriptSegment(BaseModel):
    start: float
    duration: float
    end: float
    text: str


class TranscriptChunk(BaseModel):
    start: float
    end: float
    text: str
    segment_count: int = 0


class CommentSnippet(BaseModel):
    author: str
    text: str
    like_count: Optional[int] = None
    published_at: Optional[str] = None


class VideoMetadata(BaseModel):
    video_id: str
    title: Optional[str] = None
    channel_title: Optional[str] = None
    published_at: Optional[str] = None
    description: Optional[str] = None
    duration: Optional[str] = None
    stats: VideoStats = Field(default_factory=VideoStats)


class VideoContext(BaseModel):
    metadata: VideoMetadata
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    transcript_chunks: list[TranscriptChunk] = Field(default_factory=list)
    comments: list[CommentSnippet] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HighlightMoment(BaseModel):
    start: float
    end: float
    headline: str
    rationale: str
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    related_user_goal: Optional[str] = None


class WindowCandidateMoments(BaseModel):
    moments: list[HighlightMoment] = Field(default_factory=list)


class VideoAnalysis(BaseModel):
    video_id: str
    title: Optional[str] = None
    channel_title: Optional[str] = None
    stats: VideoStats = Field(default_factory=VideoStats)
    transcript_found: bool
    warnings: list[str] = Field(default_factory=list)
    highlight_moments: list[HighlightMoment] = Field(default_factory=list)
    creator_takeaways: list[str] = Field(default_factory=list)
    comments_considered: list[CommentSnippet] = Field(default_factory=list)
    transcript_chunks_analyzed: int = 0


class GroundedSuggestion(BaseModel):
    tip: str
    supporting_pattern: str
    source_title: Optional[str] = None


class CrossVideoSummary(BaseModel):
    recurring_patterns: list[str] = Field(default_factory=list)
    content_suggestions: list[str] = Field(default_factory=list)
    positioning_advice: list[str] = Field(default_factory=list)
    grounded_suggestions: list[GroundedSuggestion] = Field(default_factory=list)
    overall_summary: str


class AnalyzeVideosResponse(BaseModel):
    description: str
    videos: list[VideoAnalysis]
    cross_video_summary: CrossVideoSummary
    debug: dict[str, Any] = Field(default_factory=dict)


class VideoFinalAnalysis(BaseModel):
    highlight_moments: list[HighlightMoment] = Field(default_factory=list)
    creator_takeaways: list[str] = Field(default_factory=list)
    analyzed_chunk_count: int = 0
