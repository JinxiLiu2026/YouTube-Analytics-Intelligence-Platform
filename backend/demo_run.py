from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.main import analyze
from backend.app.models import AnalyzeRequest


def main() -> None:
    demo_request = AnalyzeRequest(
        title="I Tried Surviving 24 Hours in Tokyo's Weirdest Capsule Hotel",
        description=(
            "I booked one of the strangest capsule hotels in Tokyo and documented the full experience, "
            "from check-in to the midnight vending machine dinner."
        ),
        content=(
            "This video follows a solo travel challenge in Japan. I compare the tiny room setup, shared bath, "
            "sleep quality, food options, neighborhood vibe, and whether the experience is actually worth the money. "
            "The tone is curious, funny, and a little chaotic."
        ),
        tags=["tokyo travel", "capsule hotel", "japan vlog", "24 hour challenge"],
        country="US",
        top_k=5,
    )

    response = analyze(demo_request)

    print("=== Demo Input ===")
    print("Title:", demo_request.title)
    print("Description:", demo_request.description)
    print("Content:", demo_request.content)
    print("Tags:", ", ".join(demo_request.tags))
    print()

    print("=== Retrieval Summary ===")
    print("Mode:", response.analysis_mode)
    print("Keywords:", ", ".join(response.extracted_keywords[:10]))
    print()

    print("=== Top Matches ===")
    for index, video in enumerate(response.similar_videos, start=1):
        print(f"{index}. [{video.country}] {video.title}")
        print(f"   channel={video.channel_title}")
        print(
            f"   score={video.similarity_score} tag_score={video.tag_score} "
            f"text_score={video.text_score} views={video.views} likes={video.likes} comments={video.comments}"
        )
        print(f"   matched_terms={', '.join(video.matched_terms)}")
        print(f"   matched_tags={', '.join(video.matched_tags)}")
        print(f"   tags={', '.join(video.tags[:6])}")
        print()

    print("=== Pattern Summary ===")
    for item in response.pattern_summary:
        print("-", item)
    print()

    print("=== Suggestions ===")
    for item in response.suggestions:
        print(f"- [{item.type}] {item.message}")
    print()

    print("=== LLM Rewrite ===")
    print("Enabled:", response.llm_rewrite.enabled)
    print("Model:", response.llm_rewrite.model)
    print("Summary:", response.llm_rewrite.summary)
    print("Why This Works:")
    for item in response.llm_rewrite.why_this_works:
        print("-", item)
    print("Title Suggestions:")
    for title in response.llm_rewrite.title_suggestions:
        print("-", title)
    print("Description Suggestion:")
    print(response.llm_rewrite.description_suggestion)
    print("Tag Suggestions:")
    print(", ".join(response.llm_rewrite.tag_suggestions))
    print()

    print("=== Draft Titles ===")
    for title in response.draft_titles:
        print("-", title)
    print()

    print("=== Draft Tags ===")
    print(", ".join(response.draft_tags))


if __name__ == "__main__":
    main()
