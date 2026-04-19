from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.dimension_vectors import DimensionVectorEncoder
from backend.app.processed_vector_index import ProcessedVectorIndex


PROCESSED_DIR = ROOT_DIR / "processed"


def print_ranked(label: str, items: list[dict]) -> None:
    print(label)
    if not items:
        print("(no results)")
        print()
        return

    for index, item in enumerate(items, start=1):
        row = item["row"]
        score = item["score"]
        print(f"{index}. [{row['country']}] {row['title']}")
        print(
            f"   total={score['total']:.4f} topic={score['topic']:.4f} "
            f"format={score['format']:.4f} context={score['context']:.4f} style={score['style']:.4f}"
        )
        print(f"   views={row['views']} channel={row['channel_title']}")
        print(f"   tags={', '.join(row['tags'])}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vector search over one processed country index in memory."
    )
    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="Country code such as US or JP.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many final reference videos to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    query = {
        "title": "I Tried Surviving 24 Hours in Tokyo's Weirdest Capsule Hotel",
        "tags": ["tokyo travel", "capsule hotel", "japan vlog", "24 hour challenge"],
    }

    index_path = PROCESSED_DIR / f"{args.country.upper()}_Trending.processed.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Processed file not found: {index_path}")

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

    top_by_similarity = [
        {
            "row": index.fetch_metadata(result.row_index),
            "score": result.score,
        }
        for result in search_result["top_results"]
    ]
    top_by_views = [
        {
            "row": index.fetch_metadata(result.row_index),
            "score": result.score,
        }
        for result in search_result["selected_results"]
    ]

    print()
    print("=== Processed Vector Search Demo ===")
    print("Country filter:", args.country.upper())
    print("Query title:", query["title"])
    print("Query tags:", ", ".join(query["tags"]))
    print("Stage 1 coarse pool size:", search_result["coarse_pool_size"])
    print("Stage 2 relevant pool size:", search_result["relevant_pool_size"])
    print()

    print_ranked("=== Top Results By Final Similarity ===", top_by_similarity)
    print_ranked("=== Final Reference Videos (Views Within Relevance Gate) ===", top_by_views)


if __name__ == "__main__":
    main()
