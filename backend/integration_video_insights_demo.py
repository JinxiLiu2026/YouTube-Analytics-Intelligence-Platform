from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TEAM_PROJECT_DIR = ROOT_DIR / "New project"
if str(TEAM_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(TEAM_PROJECT_DIR))

from backend.app.dimension_vectors import DimensionVectorEncoder
from backend.app.frontend_payload import build_frontend_payload
from backend.app.processed_vector_index import ProcessedVectorIndex
from youtube_insights.config import Settings
from youtube_insights.facade import analyze_youtube_videos


PROCESSED_DIR = ROOT_DIR / "processed"
ROOT_CONFIG_PATH = ROOT_DIR / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve top video IDs from processed vectors and pass them into the teammate insight module."
    )
    parser.add_argument(
        "--country",
        type=str,
        default="US",
        help="Country code for the retrieval index, such as US or JP.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many reference video IDs to pass into the insight module.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["fast", "full"],
        help="Analysis mode for the teammate insight module.",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=20,
        help="Max public comments to request per video in the teammate module.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "integration_video_insights_demo_output.json",
        help="Where to save the combined analysis result JSON.",
    )
    return parser.parse_args()


def build_user_goal(title: str, description: str, tags: list[str]) -> str:
    parts = [f"Title: {title}"]
    if description.strip():
        parts.append(f"Description: {description.strip()}")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    return "\n".join(parts)


def print_frontend_payload(frontend_payload: dict) -> None:
    modules = frontend_payload.get("modules", {})
    metadata = modules.get("metadata_rewrite", {})
    creation_tips = modules.get("creation_tips", {})
    matched_videos = modules.get("matched_videos", [])
    meta = frontend_payload.get("meta", {})

    print("\n=== Frontend Payload Preview ===")
    print(f"Country: {frontend_payload.get('country', '')}")
    print(
        "Retrieval meta: "
        f"coarse_pool_size={meta.get('coarse_pool_size', 0)} "
        f"relevant_pool_size={meta.get('relevant_pool_size', 0)} "
        f"selected_video_count={meta.get('selected_video_count', 0)}"
    )

    print("\n=== Module 1: Metadata Rewrite ===")
    print("Suggested title:")
    print(metadata.get("suggested_title", ""))

    print("\nSuggested tags:")
    print(", ".join(metadata.get("suggested_tags") or []))

    print("\nSuggested description:")
    print(metadata.get("suggested_description", ""))

    print("\nBased on:")
    print(metadata.get("based_on", ""))

    print("\nWhy this works:")
    why_this_works = metadata.get("why_this_works") or []
    if why_this_works:
        for item in why_this_works:
            print(f"- {item}")
    else:
        print("(none)")

    print(
        "\nRewrite model info: "
        f"model={metadata.get('model', '')} "
        f"llm_enabled={metadata.get('llm_enabled', False)}"
    )

    print("\n=== Module 2: Creation Tips ===")
    print(creation_tips.get("intro", ""))
    tips = creation_tips.get("tips") or []
    if tips:
        for index, item in enumerate(tips, start=1):
            print(f"\n{index}. Tip: {item.get('tip', '')}")
            print(
                "   Based on: "
                f"{item.get('based_on', '') or '(no direct recurring pattern attached)'}"
            )
    else:
        print("(none)")

    print("\n=== Module 3: Matched Videos ===")
    if not matched_videos:
        print("(none)")
        return

    for index, item in enumerate(matched_videos, start=1):
        print(f"\n{index}. {item.get('title', '')}")
        print(f"   url={item.get('url', '')}")
        print(f"   channel={item.get('channel_title', '')}")
        print(
            f"   views={item.get('views', 0)} "
            f"similarity_score={item.get('similarity_score', 0)}"
        )
        print(f"   key_summary={item.get('key_summary', '')}")
        print(f"   key_takeaways={item.get('key_takeaways', '')}")


def main() -> None:
    args = parse_args()

    query = {
        "title": "I Tried Surviving 24 Hours in Tokyo's Weirdest Capsule Hotel",
        "description": (
            "I booked one of the strangest capsule hotels in Tokyo and documented the full experience, "
            "from check-in to the midnight vending machine dinner."
        ),
        "tags": ["tokyo travel", "capsule hotel", "japan vlog", "24 hour challenge"],
    }

    index_path = PROCESSED_DIR / f"{args.country.upper()}_Trending.processed.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Processed file not found: {index_path}")
    if not ROOT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Root config file not found: {ROOT_CONFIG_PATH}")

    print("Loading processed vector index...")
    index = ProcessedVectorIndex.load(index_path)
    print(f"Rows loaded: {index.size}")

    encoder = DimensionVectorEncoder()
    query_signature = encoder.encode_video(query["title"], query["tags"])
    search_result = index.search(
        query_signature=query_signature,
        top_k=args.top_k,
        topic_percent=0.10,
        topic_hard_cap=1000,
        topic_min_keep=50,
        absolute_threshold=0.60,
        relative_ratio=0.85,
    )

    selected_results = search_result["selected_results"]
    if not selected_results:
        print("No reference videos passed the relevance gate, so integration analysis was skipped.")
        return

    selected_rows = [index.fetch_metadata(result.row_index) for result in selected_results]
    video_ids = [row["video_id"] for row in selected_rows]
    print("Selected video IDs:", ", ".join(video_ids))

    user_goal = build_user_goal(query["title"], query["description"], query["tags"])
    settings = Settings.load(config_path=str(ROOT_CONFIG_PATH))
    analysis_response = analyze_youtube_videos(
        description=user_goal,
        video_ids=video_ids,
        max_comments=args.max_comments,
        settings=settings,
        mode=args.mode,
    ).model_dump(mode="json")

    payload = {
        "query": query,
        "country": args.country.upper(),
        "retrieval": {
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
        },
        "insight_analysis": analysis_response,
    }
    payload["frontend_payload"] = build_frontend_payload(
        query=query,
        country=args.country.upper(),
        retrieval=payload["retrieval"],
        selected_rows=selected_rows,
        selected_results=selected_results,
        insight_analysis=analysis_response,
    )

    print_frontend_payload(payload["frontend_payload"])

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote integration output: {args.output}")


if __name__ == "__main__":
    main()
