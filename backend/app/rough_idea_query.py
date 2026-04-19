from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .config import OPENAI_CONFIG
from .openai_client import get_openai_client


@dataclass
class RoughIdeaQuery:
    original_idea: str
    search_title: str
    search_tags: list[str]
    used_llm: bool
    model: str


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _dedupe_keep_order(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _clean_text(item).strip(",")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def _fallback_tags(idea: str) -> list[str]:
    text = _clean_text(idea)
    lowered = text.lower()
    fragments = re.split(r"[,.!?;:]|\band\b|\bwith\b|\babout\b", lowered)
    candidates: list[str] = []
    for fragment in fragments:
        cleaned = _clean_text(fragment)
        if len(cleaned) < 4:
            continue
        if cleaned.startswith("i want to make a video"):
            cleaned = cleaned.replace("i want to make a video", "", 1).strip()
        elif cleaned.startswith("i want to make"):
            cleaned = cleaned.replace("i want to make", "", 1).strip()
        elif cleaned.startswith("video about"):
            cleaned = cleaned.replace("video about", "", 1).strip()
        if len(cleaned) < 4:
            continue
        candidates.append(cleaned)

    compressed = []
    for item in candidates:
        words = item.split()
        if len(words) > 4:
            compressed.append(" ".join(words[:4]))
        else:
            compressed.append(item)

    return _dedupe_keep_order(compressed)[:8]


def _fallback_title(idea: str) -> str:
    cleaned = _clean_text(idea)
    cleaned = re.sub(r"^(i want to make a video about|i want to make a video on|i want to make|video about)\s+", "", cleaned, flags=re.I)
    cleaned = cleaned[:120].strip(" .")
    if not cleaned:
        return _clean_text(idea)[:120]
    return cleaned[0].upper() + cleaned[1:]


def _parse_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        return json.loads(payload)
    if isinstance(payload, dict):
        return payload
    raise ValueError("Unsupported rough-idea payload format.")


def _build_prompt(idea: str) -> tuple[str, str]:
    system_prompt = (
        "You convert a rough YouTube video idea into a better retrieval query. "
        "Return strict JSON with keys: search_title, search_tags. "
        "search_title must be one concise YouTube-style search title under 80 characters. "
        "search_tags must be an array of 5 to 8 short tag-like phrases. "
        "Tags should be broad, simple search phrases, not full sentences. "
        "Prefer general topic tags over highly specific or decorative phrasing. "
        "Think like a dataset retriever: use tags that could appear across many related videos. "
        "Avoid overly unique wording, emotional hooks, or long descriptive phrases. "
        "Each tag should usually be 1 to 3 words, and should stay as general as possible while remaining on-topic. "
        "Favor broad tags such as location, category, format, and challenge type. "
        "If the idea is very specific, still step back and output the broader searchable labels behind it. "
        "Good tags are things like 'japan travel', 'tokyo vlog', 'capsule hotel', 'travel challenge', 'hotel review'. "
        "Avoid tags like 'weirdest capsule hotel ever', 'strangest midnight vending machine dinner', or other one-off details. "
        "Keep the user's original topic, location, challenge, and niche intent. "
        "Do not add unrelated trends, celebrities, or genres."
    )
    user_prompt = (
        "Rewrite this rough video idea into a retrieval query.\n"
        "Important: make the tags broader than the original wording if needed.\n"
        "The tags are for retrieval, not for polished publishing.\n"
        f"Idea: {idea}"
    )
    return system_prompt, user_prompt


def rewrite_rough_idea(idea: str) -> RoughIdeaQuery:
    cleaned_idea = _clean_text(idea)
    fallback_title = _fallback_title(cleaned_idea)
    fallback_tags = _fallback_tags(cleaned_idea)

    if not OPENAI_CONFIG.api_key:
        return RoughIdeaQuery(
            original_idea=cleaned_idea,
            search_title=fallback_title,
            search_tags=fallback_tags,
            used_llm=False,
            model=OPENAI_CONFIG.suggestion_model,
        )

    system_prompt, user_prompt = _build_prompt(cleaned_idea)

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_CONFIG.suggestion_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or "{}"
        data = _parse_payload(content)
        search_title = _clean_text(data.get("search_title") or fallback_title)
        search_tags = _dedupe_keep_order([str(item) for item in data.get("search_tags") or []])[:8]
        if not search_title:
            search_title = fallback_title
        if not search_tags:
            search_tags = fallback_tags
        return RoughIdeaQuery(
            original_idea=cleaned_idea,
            search_title=search_title,
            search_tags=search_tags,
            used_llm=True,
            model=OPENAI_CONFIG.suggestion_model,
        )
    except Exception:
        return RoughIdeaQuery(
            original_idea=cleaned_idea,
            search_title=fallback_title,
            search_tags=fallback_tags,
            used_llm=False,
            model=OPENAI_CONFIG.suggestion_model,
        )
