from __future__ import annotations

import json
from typing import Any

from .config import OPENAI_CONFIG
from .models import LLMRewriteResult, SimilarVideo
from .openai_client import get_openai_client


def build_llm_fallback(similar_videos: list[SimilarVideo], draft_titles: list[str], draft_tags: list[str]) -> LLMRewriteResult:
    return LLMRewriteResult(
        enabled=False,
        model=OPENAI_CONFIG.rewrite_model,
        summary="LLM rewrite is unavailable until OPENAI_API_KEY is set. Showing heuristic drafts only.",
        why_this_works=[
            "Emphasizes the main topic and challenge angle more clearly.",
            "Makes the unusual capsule-hotel hook easier to understand at a glance.",
            "Uses wording that is closer to similar travel and experience-led videos.",
        ],
        title_suggestions=draft_titles[:3],
        description_suggestion=(
            "Add a short 2-3 sentence setup that clearly states the topic, what happens in the video, "
            "and why a viewer should care."
        ),
        tag_suggestions=draft_tags[:10],
    )


def build_grounded_rewrite_prompt(
    user_title: str,
    user_description: str,
    user_content: str,
    user_tags: list[str],
    similar_videos: list[SimilarVideo],
) -> tuple[str, str]:
    system_prompt = (
        "You are a YouTube metadata strategist. "
        "Given a user's draft metadata and 5 similar trending videos, propose better metadata. "
        "Use only title, description, and tags from the references. "
        "Do not mention comments, views, or engagement stats. "
        "Return strict JSON with keys: summary, why_this_works, title_suggestions, description_suggestion, tag_suggestions. "
        "summary must be a single short sentence under 18 words. "
        "why_this_works must be an array of exactly 3 complete bullet-style sentences. "
        "Each why_this_works item must be under 14 words, concrete, readable, and not mention reference numbers. "
        "title_suggestions must be an array of exactly 3 strings. "
        "description_suggestion must be one short paragraph under 70 words. "
        "tag_suggestions must be an array of 6 to 10 strings. "
        "Avoid filler, marketing language, and generic advice. "
        "Base suggestions primarily on repeated patterns visible in the references. "
        "If the references do not clearly support a rewrite direction, stay conservative and keep edits light. "
        "Do not use generic YouTube best practices unless they are also visible in the references. "
        "Do not invent reference patterns that are not visible in the provided titles, descriptions, and tags."
    )

    reference_blocks = []
    for index, video in enumerate(similar_videos[:5], start=1):
        reference_blocks.append(
            "\n".join(
                [
                    f"Reference {index}:",
                    f"Title: {video.title}",
                    f"Description: {video.description_preview}",
                    f"Tags: {', '.join(video.tags)}",
                ]
            )
        )

    user_block = "\n".join(
        [
            "User Draft:",
            f"Title: {user_title}",
            f"Description: {user_description}",
            f"Content Summary: {user_content}",
            f"Tags: {', '.join(user_tags)}",
            "",
            "Similar Trending References:",
            "\n\n".join(reference_blocks),
        ]
    )
    return system_prompt, user_block


def build_general_rewrite_prompt(
    user_title: str,
    user_description: str,
    user_content: str,
    user_tags: list[str],
) -> tuple[str, str]:
    system_prompt = (
        "You are a YouTube metadata strategist. "
        "No closely matched reference videos were found, so you must provide general AI-guided suggestions "
        "based only on the user's draft and broad content strategy knowledge. "
        "Do not claim to use reference videos, matched trends, or database evidence. "
        "Focus on clearer phrasing, stronger topic visibility, better searchability, and more clickable wording. "
        "Return strict JSON with keys: summary, why_this_works, title_suggestions, description_suggestion, tag_suggestions. "
        "summary must be a single short sentence under 18 words. "
        "why_this_works must be an array of exactly 3 complete bullet-style sentences. "
        "Each why_this_works item must be under 14 words, concrete, and readable. "
        "title_suggestions must be an array of exactly 3 strings. "
        "description_suggestion must be one short paragraph under 70 words. "
        "tag_suggestions must be an array of 6 to 10 strings. "
        "Avoid filler and avoid pretending these are reference-grounded recommendations."
    )

    user_block = "\n".join(
        [
            "User Draft:",
            f"Title: {user_title}",
            f"Description: {user_description}",
            f"Content Summary: {user_content}",
            f"Tags: {', '.join(user_tags)}",
        ]
    )
    return system_prompt, user_block


def parse_rewrite_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        return json.loads(payload)
    if isinstance(payload, dict):
        return payload
    raise ValueError("Unsupported LLM payload format.")


def generate_llm_rewrite(
    user_title: str,
    user_description: str,
    user_content: str,
    user_tags: list[str],
    similar_videos: list[SimilarVideo],
    draft_titles: list[str],
    draft_tags: list[str],
) -> LLMRewriteResult:
    if not OPENAI_CONFIG.api_key:
        return build_llm_fallback(similar_videos, draft_titles, draft_tags)

    if similar_videos:
        system_prompt, user_prompt = build_grounded_rewrite_prompt(
            user_title=user_title,
            user_description=user_description,
            user_content=user_content,
            user_tags=user_tags,
            similar_videos=similar_videos,
        )
    else:
        system_prompt, user_prompt = build_general_rewrite_prompt(
            user_title=user_title,
            user_description=user_description,
            user_content=user_content,
            user_tags=user_tags,
        )

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_CONFIG.rewrite_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )

        content = response.choices[0].message.content or "{}"
        data = parse_rewrite_payload(content)
    except Exception as exc:
        fallback = build_llm_fallback(similar_videos, draft_titles, draft_tags)
        fallback.summary = f"LLM rewrite failed, so heuristic drafts are shown instead. Error: {exc}"
        return fallback

    title_suggestions = data.get("title_suggestions") or draft_titles[:3]
    why_this_works = data.get("why_this_works") or []
    description_suggestion = data.get("description_suggestion") or ""
    tag_suggestions = data.get("tag_suggestions") or draft_tags[:10]
    summary = data.get("summary") or (
        "Generated from similar matched videos." if similar_videos else "Generated from your draft with general AI guidance."
    )

    return LLMRewriteResult(
        enabled=True,
        model=OPENAI_CONFIG.rewrite_model,
        summary=summary,
        why_this_works=[str(item).strip() for item in why_this_works if str(item).strip()][:3],
        title_suggestions=[str(item).strip() for item in title_suggestions][:3],
        description_suggestion=str(description_suggestion).strip(),
        tag_suggestions=[str(item).strip() for item in tag_suggestions if str(item).strip()][:10],
    )
