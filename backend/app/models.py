from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = ""
    content: str = ""
    tags: list[str] = Field(default_factory=list)
    country: Optional[str] = Field(default=None, description="Optional two-letter country code.")
    top_k: int = Field(default=5, ge=1, le=10)


class UIAnalyzeRequest(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    country: Optional[str] = Field(default="US", description="Optional two-letter country code.")
    input_mode: Literal["draft", "rough_idea"] = "draft"
    selected_sections: list[str] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1, le=5)
    mode: Literal["fast", "full"] = "fast"
    max_comments: int = Field(default=20, ge=0, le=100)


class SimilarVideo(BaseModel):
    video_id: str
    country: str
    title: str
    channel_title: str
    category_id: str
    publish_time: str
    trending_date: str
    tags: list[str]
    description_preview: str
    views: int
    likes: int
    comments: int
    engagement_rate: float
    similarity_score: float
    tag_score: float
    text_score: float
    matched_terms: list[str]
    matched_tags: list[str]


class ImprovementSuggestion(BaseModel):
    type: Literal["title", "description", "tags", "positioning"]
    message: str


class LLMRewriteResult(BaseModel):
    enabled: bool
    model: str
    summary: str
    why_this_works: list[str]
    title_suggestions: list[str]
    description_suggestion: str
    tag_suggestions: list[str]


class AnalyzeResponse(BaseModel):
    query_text: str
    extracted_keywords: list[str]
    analysis_mode: str
    similar_videos: list[SimilarVideo]
    pattern_summary: list[str]
    suggestions: list[ImprovementSuggestion]
    draft_titles: list[str]
    draft_tags: list[str]
    llm_rewrite: LLMRewriteResult


class HealthResponse(BaseModel):
    status: str
    videos_loaded: int
    countries: list[str]
