from __future__ import annotations

import re
from typing import Any

from .llm_suggestions import generate_llm_rewrite
from .models import SimilarVideo
from .processed_vector_index import ProcessedVectorSearchResult
from .suggestions import build_draft_tags, build_draft_titles


def _truncate(text: str, limit: int = 220) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _keyword_tokens(text: str) -> set[str]:
    tokens = []
    current = []
    for char in str(text).lower():
        if char.isalnum():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return {token for token in tokens if len(token) >= 4}


def _clean_optional_text(value: Any) -> str:
    cleaned = " ".join(str(value or "").split()).strip()
    if cleaned.lower() in {"none", "null", "n/a"}:
        return ""
    return cleaned


def _pick_source_video_title(pattern: str, videos: list[dict[str, Any]]) -> str:
    pattern_tokens = _keyword_tokens(pattern)
    if not pattern_tokens:
        return ""

    best_title = ""
    best_score = -1

    for video in videos:
        title = _clean_optional_text(video.get("title", ""))
        if not title:
            continue

        candidate_parts = [title]
        for takeaway in video.get("creator_takeaways") or []:
            cleaned = _clean_optional_text(takeaway)
            if cleaned:
                candidate_parts.append(cleaned)

        for moment in video.get("highlight_moments") or []:
            headline = _clean_optional_text(moment.get("headline", ""))
            rationale = _clean_optional_text(moment.get("rationale", ""))
            if headline:
                candidate_parts.append(headline)
            if rationale:
                candidate_parts.append(rationale)

        candidate_tokens = _keyword_tokens(" ".join(candidate_parts))
        score = len(pattern_tokens & candidate_tokens)
        if score > best_score:
            best_score = score
            best_title = title

    return best_title if best_score > 0 else ""


def _resolve_source_title(raw_source_title: Any, insight_videos: list[dict[str, Any]]) -> str:
    cleaned = _clean_optional_text(raw_source_title)
    if cleaned and "video_id" not in cleaned.lower() and not re.fullmatch(r"[A-Za-z0-9_-]{8,}", cleaned):
        return cleaned

    video_id_match = re.search(r"video_id[\s:_-]*([A-Za-z0-9_-]{6,})", str(raw_source_title or ""), re.I)
    if not video_id_match:
        video_id_match = re.search(r"\b([A-Za-z0-9_-]{10,})\b", str(raw_source_title or ""))

    candidate_video_id = video_id_match.group(1) if video_id_match else ""
    if not candidate_video_id:
        return ""

    for video in insight_videos:
        if str(video.get("video_id", "")).strip() == candidate_video_id:
            return _clean_optional_text(video.get("title", ""))
    return ""


def _format_tip_basis(supporting_pattern: str, insight_videos: list[dict[str, Any]]) -> str:
    cleaned_pattern = " ".join(str(supporting_pattern).split()).strip()
    if not cleaned_pattern:
        return ""

    source_title = _pick_source_video_title(cleaned_pattern, insight_videos)
    if source_title:
        return f'From "{source_title}": {cleaned_pattern}'
    return cleaned_pattern


def _match_pattern_for_tip(tip: str, recurring_patterns: list[str], used: set[int]) -> str:
    tip_tokens = _keyword_tokens(tip)
    best_index = None
    best_score = -1

    for index, pattern in enumerate(recurring_patterns):
        if index in used:
            continue
        pattern_tokens = _keyword_tokens(pattern)
        overlap = len(tip_tokens & pattern_tokens)
        if overlap > best_score:
            best_score = overlap
            best_index = index

    if best_index is None and recurring_patterns:
        best_index = 0

    if best_index is None:
        return ""

    used.add(best_index)
    return recurring_patterns[best_index]


def _pattern_to_basis_sentence(pattern: str) -> str:
    cleaned = " ".join(str(pattern).split()).strip()
    if not cleaned:
        return ""
    if cleaned[-1] in ".!?":
        cleaned = cleaned[:-1]
    return f"Matched videos repeatedly showed this pattern: {cleaned}."


def _short_reference_clause(reference_titles: list[str]) -> str:
    references = [title.strip() for title in reference_titles if str(title).strip()][:2]
    if not references:
        return "This rewrite is based on the closest matched videos."
    if len(references) == 1:
        return f'This rewrite is based on the matched video "{references[0]}".'
    return f'This rewrite is based on the matched videos "{references[0]}" and "{references[1]}".'


def _clean_bullets(items: list[str], limit: int = 3, item_limit: int = 120) -> list[str]:
    bullets = []
    for item in items:
        cleaned = " ".join(str(item).split()).strip()
        if not cleaned:
            continue
        if cleaned[-1] not in ".!?":
            cleaned += "."
        bullets.append(_truncate(cleaned, limit=item_limit))
        if len(bullets) >= limit:
            break
    return bullets


def _row_to_similar_video(row: dict[str, Any], score: dict[str, float]) -> SimilarVideo:
    return SimilarVideo(
        video_id=str(row.get("video_id", "")),
        country=str(row.get("country", "")),
        title=str(row.get("title", "")),
        channel_title=str(row.get("channel_title", "")),
        category_id="",
        publish_time="",
        trending_date="",
        tags=list(row.get("tags") or []),
        description_preview=_truncate(str(row.get("description", "")), limit=280),
        views=int(row.get("views", 0) or 0),
        likes=0,
        comments=0,
        engagement_rate=0.0,
        similarity_score=float(score.get("total", 0.0)),
        tag_score=0.0,
        text_score=0.0,
        matched_terms=[],
        matched_tags=[],
    )


def _build_metadata_rewrite(
    query: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    selected_results: list[ProcessedVectorSearchResult],
) -> dict[str, Any]:
    similar_videos = [
        _row_to_similar_video(row, result.score)
        for row, result in zip(selected_rows, selected_results)
    ]
    reference_titles = [video.title for video in similar_videos[:5] if video.title]
    draft_titles = build_draft_titles(str(query.get("title", "")), [])
    draft_tags = list(query.get("tags") or [])

    if similar_videos:
        draft_titles = build_draft_titles(str(query.get("title", "")), similar_videos)
        draft_tags = build_draft_tags(list(query.get("tags") or []), similar_videos)

    rewrite = generate_llm_rewrite(
        user_title=str(query.get("title", "")),
        user_description=str(query.get("description", "")),
        user_content=str(query.get("content", "")),
        user_tags=list(query.get("tags") or []),
        similar_videos=similar_videos,
        draft_titles=draft_titles,
        draft_tags=draft_tags,
    )

    if similar_videos:
        based_on = _short_reference_clause(reference_titles)
    else:
        based_on = "No close matched videos were found, so these are general AI suggestions based on your draft."

    return {
        "suggested_title": " / ".join(
            [item for item in rewrite.title_suggestions if str(item).strip()]
        ) or str(query.get("title", "")),
        "suggested_tags": rewrite.tag_suggestions,
        "suggested_description": str(rewrite.description_suggestion).strip(),
        "based_on": based_on,
        "why_this_works": _clean_bullets(rewrite.why_this_works, limit=3, item_limit=120),
        "model": rewrite.model,
        "llm_enabled": rewrite.enabled,
    }


def _build_creation_tips(insight_analysis: dict[str, Any]) -> dict[str, Any]:
    cross = insight_analysis.get("cross_video_summary") or {}
    insight_videos = insight_analysis.get("videos") or []
    grounded_suggestions = cross.get("grounded_suggestions") or []
    content_suggestions = [str(item).strip() for item in cross.get("content_suggestions") or [] if str(item).strip()]
    positioning_advice = [str(item).strip() for item in cross.get("positioning_advice") or [] if str(item).strip()]
    recurring_patterns = [str(item).strip() for item in cross.get("recurring_patterns") or [] if str(item).strip()]

    if grounded_suggestions:
        tip_items = []
        for item in grounded_suggestions[:5]:
            tip = " ".join(str(item.get("tip", "")).split()).strip()
            supporting_pattern = " ".join(
                str(item.get("supporting_pattern", "")).split()
            ).strip()
            source_title = _resolve_source_title(item.get("source_title", ""), insight_videos)
            if not tip:
                continue
            tip_items.append(
                {
                    "tip": tip,
                    "based_on": (
                        f'From matched video "{source_title}": {supporting_pattern}'
                        if source_title
                        else _format_tip_basis(supporting_pattern, insight_videos)
                    ),
                }
            )
        return {
            "intro": "These tips are grounded in repeated patterns from the matched videos.",
            "tips": tip_items,
        }

    tip_items: list[dict[str, str]] = []
    used_pattern_indices: set[int] = set()
    for item in content_suggestions + positioning_advice:
        if not item:
            continue
        if any(existing["tip"] == item for existing in tip_items):
            continue
        tip_items.append(
            {
                "tip": item,
                "based_on": _pattern_to_basis_sentence(
                    _match_pattern_for_tip(item, recurring_patterns, used_pattern_indices)
                ),
            }
        )
        if len(tip_items) >= 5:
            break

    return {
        "intro": "These tips are grounded in repeated patterns from the matched videos.",
        "tips": tip_items,
    }


def _build_matched_videos(
    selected_rows: list[dict[str, Any]],
    selected_results: list[ProcessedVectorSearchResult],
    insight_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    analysis_by_video_id = {
        str(video.get("video_id", "")): video
        for video in insight_analysis.get("videos") or []
    }

    cards = []
    for row, result in zip(selected_rows, selected_results):
        video_id = str(row.get("video_id", ""))
        analysis = analysis_by_video_id.get(video_id, {})
        takeaways = [str(item).strip() for item in analysis.get("creator_takeaways") or [] if str(item).strip()]
        moments = analysis.get("highlight_moments") or []

        summary_parts = []
        if moments:
            first_moment = moments[0]
            headline = str(first_moment.get("headline") or "").strip()
            rationale = str(first_moment.get("rationale") or "").strip()
            if headline:
                summary_parts.append(headline)
            if rationale:
                summary_parts.append(rationale)

        if not summary_parts and takeaways:
            summary_parts.append(takeaways[0])

        learn_text = ""
        if takeaways:
            learn_text = takeaways[0]
        elif summary_parts:
            learn_text = " ".join(summary_parts)

        style_text = ""
        if len(takeaways) >= 2:
            style_text = takeaways[1]
        elif moments:
            style_text = str(moments[0].get("headline") or "").strip()

        cards.append(
            {
                "video_id": video_id,
                "title": str(row.get("title", "")),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "channel_title": str(row.get("channel_title", "")),
                "views": int(row.get("views", 0) or 0),
                "similarity_score": round(float(result.score.get("total", 0.0)), 4),
                "key_summary": style_text,
                "key_takeaways": learn_text,
            }
        )
    return cards


def build_frontend_payload(
    query: dict[str, Any],
    country: str,
    retrieval: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    selected_results: list[ProcessedVectorSearchResult],
    insight_analysis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "query": query,
        "country": country,
        "modules": {
            "metadata_rewrite": _build_metadata_rewrite(
                query=query,
                selected_rows=selected_rows,
                selected_results=selected_results,
            ),
            "creation_tips": _build_creation_tips(insight_analysis=insight_analysis),
            "matched_videos": _build_matched_videos(
                selected_rows=selected_rows,
                selected_results=selected_results,
                insight_analysis=insight_analysis,
            ),
        },
        "meta": {
            "coarse_pool_size": retrieval.get("coarse_pool_size", 0),
            "relevant_pool_size": retrieval.get("relevant_pool_size", 0),
            "selected_video_count": len(selected_rows),
        },
    }
