from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.search import get_index


def print_video_block(label: str, items: list[dict]) -> None:
    print(label)
    for rank, item in enumerate(items, start=1):
        record = item["record"]
        print(f"{rank}. [{record.country}] {record.title}")
        print(
            f"   final_score={item['score']} tag_score={item['tag_score']} "
            f"text_score={item['text_score']} views={record.views}"
        )
        print(f"   matched_tags={', '.join(item['matched_tags'])}")
        print(f"   matched_terms={', '.join(item['matched_terms'])}")
        print(f"   tags={', '.join(record.tags)}")
        print()


def main() -> None:
    sample = {
        "title": "I Tried Surviving 24 Hours in Tokyo's Weirdest Capsule Hotel",
        "description": (
            "I booked one of the strangest capsule hotels in Tokyo and documented the full experience, "
            "from check-in to the midnight vending machine dinner."
        ),
        "tags": ["tokyo travel", "capsule hotel", "japan vlog", "24 hour challenge"],
        "country": "US",
        "top_k": 5,
    }

    index = get_index()
    query_text, keywords, top_by_score, candidate_pool, top_by_views = index.search_debug(
        title=sample["title"],
        description=sample["description"],
        tags=sample["tags"],
        country=sample["country"],
        top_k=sample["top_k"],
    )

    print("=== Retrieval Debug Demo ===")
    print("Title:", sample["title"])
    print("Description:", sample["description"])
    print("Tags:", ", ".join(sample["tags"]))
    print("Country:", sample["country"])
    print("Query Text:")
    print(query_text)
    print()
    print("Keywords:", ", ".join(keywords[:12]))
    print("Actual candidate pool length:", len(candidate_pool))
    print()

    print_video_block("=== Top 5 By Similarity Score ===", top_by_score)
    print_video_block("=== Top 5 By Views From Candidate Pool ===", top_by_views)


if __name__ == "__main__":
    main()
