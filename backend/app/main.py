from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parents[2]
TEAM_PROJECT_DIR = ROOT_DIR / "New project"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TEAM_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(TEAM_PROJECT_DIR))

from youtube_insights.config import Settings
from youtube_insights.facade import analyze_youtube_videos

from .dimension_vectors import DimensionVectorEncoder
from .frontend_payload import build_frontend_payload
from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    ImprovementSuggestion,
    SimilarVideo,
    UIAnalyzeRequest,
)
from .llm_suggestions import generate_llm_rewrite
from .processed_vector_index import ProcessedVectorIndex
from .rough_idea_query import rewrite_rough_idea
from .search import get_index
from .suggestions import build_draft_tags, build_draft_titles, build_suggestions, summarize_patterns


app = FastAPI(title="Trend Rewrite API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROCESSED_DIR = ROOT_DIR / "processed"
ROOT_CONFIG_PATH = ROOT_DIR / "config.json"


@lru_cache(maxsize=1)
def _get_encoder() -> DimensionVectorEncoder:
    return DimensionVectorEncoder()


@lru_cache(maxsize=8)
def _get_processed_index(country: str) -> ProcessedVectorIndex:
    normalized = (country or "US").upper()
    path = PROCESSED_DIR / f"{normalized}_Trending.processed.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return ProcessedVectorIndex.load(path)


@lru_cache(maxsize=1)
def _get_insight_settings() -> Settings:
    return Settings.load(config_path=str(ROOT_CONFIG_PATH))


def _build_user_goal(title: str, description: str, tags: list[str]) -> str:
    parts = [f"Title: {title}"]
    if description.strip():
        parts.append(f"Description: {description.strip()}")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    return "\n".join(parts)


def _log(message: str) -> None:
    print(f"[ui/analyze] {message}", flush=True)


def _print_frontend_payload_preview(frontend_payload: dict) -> None:
    modules = frontend_payload.get("modules", {})
    metadata = modules.get("metadata_rewrite", {})
    creation_tips = modules.get("creation_tips", {})
    matched_videos = modules.get("matched_videos", [])
    meta = frontend_payload.get("meta", {})

    _log("=== Frontend Payload Preview ===")
    _log(f"Country: {frontend_payload.get('country', '')}")
    _log(
        "Retrieval meta: "
        f"coarse_pool_size={meta.get('coarse_pool_size', 0)} "
        f"relevant_pool_size={meta.get('relevant_pool_size', 0)} "
        f"selected_video_count={meta.get('selected_video_count', 0)}"
    )

    _log("=== Module 1: Metadata Rewrite ===")
    _log(f"Suggested title: {metadata.get('suggested_title', '')}")
    _log(f"Suggested tags: {', '.join(metadata.get('suggested_tags') or [])}")
    _log(f"Suggested description: {metadata.get('suggested_description', '')}")
    _log(f"Based on: {metadata.get('based_on', '')}")
    why_this_works = metadata.get("why_this_works") or []
    if why_this_works:
        for item in why_this_works:
            _log(f"Why this works: {item}")

    _log("=== Module 2: Creation Tips ===")
    _log(creation_tips.get("intro", ""))
    for index, item in enumerate(creation_tips.get("tips") or [], start=1):
        _log(f"Tip {index}: {item.get('tip', '')}")
        _log(f"Based on: {item.get('based_on', '')}")

    _log("=== Module 3: Matched Videos ===")
    if not matched_videos:
        _log("(none)")
        return

    for index, item in enumerate(matched_videos, start=1):
        _log(
            f"{index}. {item.get('title', '')} | channel={item.get('channel_title', '')} "
            f"| views={item.get('views', 0)} | similarity_score={item.get('similarity_score', 0)}"
        )
        _log(f"   key_summary={item.get('key_summary', '')}")
        _log(f"   key_takeaways={item.get('key_takeaways', '')}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    index = get_index()
    return HealthResponse(
        status="ok",
        videos_loaded=len(index.videos),
        countries=sorted(index.countries),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    index = get_index()
    query_text, extracted_keywords, matches = index.search(
        title=request.title,
        description=request.description,
        content=request.content,
        tags=request.tags,
        country=request.country,
        top_k=request.top_k,
    )

    records = [match["record"] for match in matches]
    similar_videos = [
        SimilarVideo(
            video_id=record.video_id,
            country=record.country,
            title=record.title,
            channel_title=record.channel_title,
            category_id=record.category_id,
            publish_time=record.publish_time,
            trending_date=record.trending_date,
            tags=record.tags[:10],
            description_preview=record.description[:240],
            views=record.views,
            likes=record.likes,
            comments=record.comments,
            engagement_rate=record.engagement_rate,
            similarity_score=match["score"],
            tag_score=match["tag_score"],
            text_score=match["text_score"],
            matched_terms=match["matched_terms"],
            matched_tags=match["matched_tags"],
        )
        for match in matches
        for record in [match["record"]]
    ]

    pattern_summary = summarize_patterns(records)
    suggestions = [
        ImprovementSuggestion(**item)
        for item in build_suggestions(request.title, request.description, request.content, request.tags, records)
    ]
    draft_titles = build_draft_titles(request.title, records)
    draft_tags = build_draft_tags(request.tags, records)
    llm_rewrite = generate_llm_rewrite(
        user_title=request.title,
        user_description=request.description,
        user_content=request.content,
        user_tags=request.tags,
        similar_videos=similar_videos,
        draft_titles=draft_titles,
        draft_tags=draft_tags,
    )

    return AnalyzeResponse(
        query_text=query_text,
        extracted_keywords=extracted_keywords,
        analysis_mode="tag_first_text_then_views_rerank",
        similar_videos=similar_videos,
        pattern_summary=pattern_summary,
        suggestions=suggestions,
        draft_titles=draft_titles,
        draft_tags=draft_tags,
        llm_rewrite=llm_rewrite,
    )


@app.post("/ui/analyze")
def ui_analyze(request: UIAnalyzeRequest) -> dict:
    started_at = perf_counter()
    country = (request.country or "US").upper()
    enabled_features = [
        str(feature).strip().lower()
        for feature in request.selected_sections
        if str(feature).strip().lower() in {"topic", "format", "context", "style"}
    ]
    tags = [str(tag).strip() for tag in request.tags if str(tag).strip()]
    raw_title = request.title.strip()
    raw_description = request.description.strip()
    input_mode = request.input_mode

    if input_mode == "rough_idea":
        idea_query = rewrite_rough_idea(raw_title)
        _log(
            f"Rough idea rewrite | llm_used={idea_query.used_llm} model={idea_query.model}"
        )
        _log(f"Original idea: {idea_query.original_idea}")
        _log(f"Generated search title: {idea_query.search_title}")
        _log(f"Generated search tags: {', '.join(idea_query.search_tags)}")
        query = {
            "title": idea_query.search_title,
            "description": "",
            "content": "",
            "tags": idea_query.search_tags,
        }
    else:
        query = {
            "title": raw_title,
            "description": raw_description,
            "content": "",
            "tags": tags,
        }
    _log(
        f"Request received | country={country} top_k={request.top_k} mode={request.mode} input_mode={input_mode} "
        f"selected_sections={enabled_features or ['topic', 'format', 'context', 'style']}"
    )
    _log(f"Title: {query['title']}")
    _log(f"Tags: {', '.join(query['tags'])}")

    step_started = perf_counter()
    _log("Loading processed index...")
    index = _get_processed_index(country)
    _log(f"Index loaded | rows={index.size} | took={perf_counter() - step_started:.2f}s")

    step_started = perf_counter()
    _log("Encoding query into vectors...")
    encoder = _get_encoder()
    query_signature = encoder.encode_video(query["title"], query["tags"])
    _log(f"Query encoded | took={perf_counter() - step_started:.2f}s")

    step_started = perf_counter()
    _log("Running vector search...")
    search_result = index.search(
        query_signature=query_signature,
        enabled_features=enabled_features,
        top_k=request.top_k,
        topic_percent=0.10,
        topic_hard_cap=1000,
        topic_min_keep=50,
        absolute_threshold=0.60,
        relative_ratio=0.85,
    )
    _log(
        "Vector search complete | "
        f"coarse_pool_size={search_result['coarse_pool_size']} "
        f"relevant_pool_size={search_result['relevant_pool_size']} "
        f"selected={len(search_result['selected_results'])} "
        f"| took={perf_counter() - step_started:.2f}s"
    )

    selected_results = search_result["selected_results"]
    selected_rows = [index.fetch_metadata(result.row_index) for result in selected_results]
    if selected_rows:
        _log(
            "Selected video IDs: "
            + ", ".join(str(row.get("video_id", "")) for row in selected_rows)
        )
    else:
        _log("No videos passed the relevance gate.")

    analysis_response: dict = {
        "description": _build_user_goal(query["title"], query["description"], query["tags"]),
        "videos": [],
        "cross_video_summary": {
            "recurring_patterns": [],
            "content_suggestions": [],
            "positioning_advice": [],
            "grounded_suggestions": [],
            "overall_summary": "",
        },
        "debug": {},
    }
    if selected_rows:
        step_started = perf_counter()
        _log("Running transcript/content insight analysis...")
        settings = _get_insight_settings()
        analysis_response = analyze_youtube_videos(
            description=_build_user_goal(query["title"], query["description"], query["tags"]),
            video_ids=[row["video_id"] for row in selected_rows],
            max_comments=request.max_comments,
            settings=settings,
            mode=request.mode,
        ).model_dump(mode="json")
        _log(f"Insight analysis complete | took={perf_counter() - step_started:.2f}s")

    retrieval = {
        "coarse_pool_size": search_result["coarse_pool_size"],
        "relevant_pool_size": search_result["relevant_pool_size"],
        "selected_videos": [
            {
                "video_id": row["video_id"],
                "title": row["title"],
                "channel_title": row["channel_title"],
                "views": row["views"],
                "score": result.score,
            }
            for row, result in zip(selected_rows, selected_results)
        ],
    }

    step_started = perf_counter()
    _log("Building frontend payload...")
    frontend_payload = build_frontend_payload(
        query=query,
        country=country,
        retrieval=retrieval,
        selected_rows=selected_rows,
        selected_results=selected_results,
        insight_analysis=analysis_response,
    )
    _log(f"Frontend payload built | took={perf_counter() - step_started:.2f}s")
    _print_frontend_payload_preview(frontend_payload)
    _log(f"Request complete | total={perf_counter() - started_at:.2f}s")

    return {
        "ok": True,
        "frontend_payload": frontend_payload,
    }
