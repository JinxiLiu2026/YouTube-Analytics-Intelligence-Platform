from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.dimension_vectors import DimensionVectorEncoder
from backend.app.processed_vector_index import ProcessedVectorIndex


PROCESSED_DIR = ROOT_DIR / "processed"


def print_results(query_name: str, query: dict, search_result: dict, index: ProcessedVectorIndex, top_k: int) -> None:
    print(f"=== {query_name} ===")
    print("Title:", query["title"])
    print("Tags:", ", ".join(query["tags"]))
    print("Stage 1 coarse pool size:", search_result["coarse_pool_size"])
    print("Stage 2 relevant pool size:", search_result["relevant_pool_size"])
    print()

    print("Top By Similarity:")
    for rank, result in enumerate(search_result["top_results"][:top_k], start=1):
        row = index.fetch_metadata(result.row_index)
        score = result.score
        print(f"{rank}. [{row['country']}] {row['title']}")
        print(
            f"   total={score['total']:.4f} topic={score['topic']:.4f} "
            f"format={score['format']:.4f} context={score['context']:.4f} style={score['style']:.4f}"
        )
        print(f"   views={row['views']} channel={row['channel_title']}")
        print(f"   tags={', '.join(row['tags'][:12])}")
        print()

    print("Final Reference Videos:")
    if not search_result["selected_results"]:
        print("(no results)")
    else:
        for rank, result in enumerate(search_result["selected_results"], start=1):
            row = index.fetch_metadata(result.row_index)
            score = result.score
            print(f"{rank}. [{row['country']}] {row['title']}")
            print(
                f"   total={score['total']:.4f} topic={score['topic']:.4f} "
                f"format={score['format']:.4f} context={score['context']:.4f} style={score['style']:.4f}"
            )
            print(f"   views={row['views']} channel={row['channel_title']}")
            print()


def main() -> None:
    country = "US"
    top_k = 5
    index_path = PROCESSED_DIR / f"{country}_Trending.processed.jsonl"

    queries = [
        (
            "Capsule Hotel Query",
            {
                "title": "I Tried Surviving 24 Hours in Tokyo's Weirdest Capsule Hotel",
                "tags": ["tokyo travel", "capsule hotel", "japan vlog", "24 hour challenge"],
            },
        ),
        (
            "Trailer Query",
            {
                "title": "Mortal Kombat II Official Trailer Reaction",
                "tags": ["mortal kombat", "official trailer", "movie trailer", "reaction"],
            },
        ),
        (
            "Travel Vlog Query",
            {
                "title": "Tokyo Travel Vlog: 3 Days of Food, Hotels, and City Walks",
                "tags": ["tokyo travel", "travel vlog", "japan trip", "hotel review"],
            },
        ),
    ]

    t0 = time.perf_counter()
    index = ProcessedVectorIndex.load(index_path)
    load_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    encoder = DimensionVectorEncoder()
    encoder_elapsed = time.perf_counter() - t1

    print("=== Multi Query Vector Demo ===")
    print("Country:", country)
    print(f"Rows loaded: {index.size}")
    print(f"Index load time: {load_elapsed:.3f}s")
    print(f"Encoder init time: {encoder_elapsed:.3f}s")
    print()

    for query_name, query in queries:
        q0 = time.perf_counter()
        query_signature = encoder.encode_video(query["title"], query["tags"])
        encode_elapsed = time.perf_counter() - q0

        q1 = time.perf_counter()
        search_result = index.search(
            query_signature=query_signature,
            top_k=top_k,
            topic_percent=0.10,
            topic_hard_cap=1000,
            topic_min_keep=50,
            absolute_threshold=0.60,
            relative_ratio=0.85,
        )
        search_elapsed = time.perf_counter() - q1

        print(f"Query encode time: {encode_elapsed:.3f}s")
        print(f"Query search time: {search_elapsed:.3f}s")
        print()
        print_results(query_name, query, search_result, index, top_k)
        print("---")
        print()


if __name__ == "__main__":
    main()
