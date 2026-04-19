from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.dimension_vectors import DimensionVectorEncoder, signature_similarity
from backend.app.llm_suggestions import generate_llm_rewrite
from backend.app.models import SimilarVideo


def to_similar_video(example: dict, score_bundle: dict[str, float], rank: int) -> SimilarVideo:
    return SimilarVideo(
        video_id=f"demo-{rank}",
        country="DEMO",
        title=example["title"],
        channel_title=example.get("channel_title", "Demo Channel"),
        category_id="demo",
        publish_time="",
        trending_date="",
        tags=example["tags"],
        description_preview=example.get("description", ""),
        views=example.get("views", 0),
        likes=0,
        comments=0,
        engagement_rate=0.0,
        similarity_score=score_bundle["total"],
        tag_score=0.0,
        text_score=0.0,
        matched_terms=[],
        matched_tags=[],
    )


def main() -> None:
    query = {
        "title": "I Tried Surviving 24 Hours in Tokyo's Weirdest Capsule Hotel",
        "tags": ["tokyo travel", "capsule hotel", "japan vlog", "24 hour challenge"],
        "description": (
            "A solo travel challenge in Japan focused on a bizarre capsule hotel stay, "
            "tiny room experience, and whether it is worth the money."
        ),
        "content": (
            "The video is a travel vlog with a challenge angle. It focuses on staying in an unusual "
            "capsule hotel in Tokyo, showing the room, food, neighborhood, and the overall experience."
        ),
    }

    candidates = [
        {
            "title": "24 Hours in Osaka's Tiny Capsule Hotel Vlog",
            "tags": ["osaka", "travel", "capsule hotel", "japan", "vlog", "hotel challenge"],
            "description": "A travel vlog about spending 24 hours inside a tiny capsule hotel in Osaka.",
            "channel_title": "Travel Demo 1",
            "views": 180000,
        },
        {
            "title": "Official Tokyo Music Video Teaser",
            "tags": ["tokyo", "japan", "music video", "official teaser"],
            "description": "A teaser for a music video filmed in Tokyo.",
            "channel_title": "Music Demo",
            "views": 750000,
        },
        {
            "title": "My Cozy Kyoto Travel Vlog and Cafe Diary",
            "tags": ["kyoto vlog", "japan travel", "cafe diary", "cozy vlog"],
            "description": "A calm Kyoto travel vlog focused on cafes, journaling, and slow travel.",
            "channel_title": "Travel Demo 2",
            "views": 220000,
        },
        {
            "title": "Celebrity Interview Backstage at Tokyo Film Festival",
            "tags": ["celebrity interview", "backstage", "tokyo", "film festival"],
            "description": "A red carpet backstage interview from a Tokyo film event.",
            "channel_title": "Interview Demo",
            "views": 140000,
        },
    ]

    encoder = DimensionVectorEncoder()
    query_signature = encoder.encode_video(query["title"], query["tags"])

    scored_candidates = []
    for candidate in candidates:
        candidate_signature = encoder.encode_video(candidate["title"], candidate["tags"])
        sim = signature_similarity(query_signature, candidate_signature)
        scored_candidates.append((candidate, sim))

    scored_candidates.sort(key=lambda item: item[1]["total"], reverse=True)
    top_two = scored_candidates[:2]

    print("=== Rewrite Similarity Demo ===\n")
    print("Query Title:", query["title"])
    print("Query Tags:", ", ".join(query["tags"]))
    print()

    print("=== Ranked Candidates ===")
    for rank, (candidate, sim) in enumerate(scored_candidates, start=1):
        print(f"{rank}. {candidate['title']}")
        print(
            f"   topic={sim['topic']:.4f} format={sim['format']:.4f} "
            f"context={sim['context']:.4f} style={sim['style']:.4f} total={sim['total']:.4f}"
        )
        print(f"   tags={', '.join(candidate['tags'])}")
        print()

    similar_videos = [
        to_similar_video(candidate, sim, rank)
        for rank, (candidate, sim) in enumerate(top_two, start=1)
    ]

    llm_rewrite = generate_llm_rewrite(
        user_title=query["title"],
        user_description=query["description"],
        user_content=query["content"],
        user_tags=query["tags"],
        similar_videos=similar_videos,
        draft_titles=[
            query["title"],
            "24 Hours in Tokyo's Weirdest Capsule Hotel",
            "Inside Tokyo's Weirdest Capsule Hotel for 24 Hours",
        ],
        draft_tags=list(query["tags"]),
    )

    print("=== Selected Top 2 References ===")
    for index, video in enumerate(similar_videos, start=1):
        print(f"{index}. {video.title}")
        print(f"   tags={', '.join(video.tags)}")
        print()

    print("=== LLM Rewrite Output ===")
    print("Enabled:", llm_rewrite.enabled)
    print("Model:", llm_rewrite.model)
    print("Summary:", llm_rewrite.summary)
    print("Why This Works:")
    for idea in llm_rewrite.why_this_works:
        print("-", idea)
    print("Title Suggestions:")
    for title in llm_rewrite.title_suggestions:
        print("-", title)
    print("Description Suggestion:")
    print(llm_rewrite.description_suggestion)
    print("Tag Suggestions:")
    print(", ".join(llm_rewrite.tag_suggestions))


if __name__ == "__main__":
    main()
