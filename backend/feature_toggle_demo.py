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

FEATURE_SETS = [
    ("All Features", ["topic", "format", "context", "style"]),
    ("Topic Only", ["topic"]),
    ("Topic + Context", ["topic", "context"]),
    ("Format + Style", ["format", "style"]),
    ("Context Only", ["context"]),
]

QUERY_PRESETS = {
    "capsule_hotel": {
        "title": "I Tried Surviving 24 Hours in Tokyo's Weirdest Capsule Hotel",
        "tags": ["tokyo travel", "capsule hotel", "japan vlog", "24 hour challenge"],
    },
    "tokyo_budget": {
        "title": "I Spent a Week Living Alone in Tokyo and Tracked Every Dollar",
        "tags": ["tokyo vlog", "solo travel", "japan budget", "week in my life", "travel diary"],
    },
    "trailer_reaction": {
        "title": "Mortal Kombat II Official Trailer Reaction",
        "tags": ["mortal kombat", "official trailer", "movie trailer", "reaction"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare retrieval results across different enabled feature combinations."
    )
    parser.add_argument(
        "--country",
        type=str,
        default="US",
        help="Country code such as US or JP.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="capsule_hotel",
        choices=sorted(QUERY_PRESETS.keys()),
        help="Which built-in query preset to run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many rows to print for each feature set.",
    )
    return parser.parse_args()


def print_results(
    label: str,
    enabled_features: list[str],
    search_result: dict,
    index: ProcessedVectorIndex,
    top_k: int,
) -> None:
    print(f"=== {label} ===")
    print(f"Enabled features: {', '.join(enabled_features)}")
    print(
        "Pools: "
        f"coarse={search_result['coarse_pool_size']} "
        f"relevant={search_result['relevant_pool_size']}"
    )
    print()

    print("Top by Similarity:")
    for rank, result in enumerate(search_result["top_results"][:top_k], start=1):
        row = index.fetch_metadata(result.row_index)
        score = result.score
        print(f"{rank}. [{row['country']}] {row['title']}")
        print(
            f"   total={score['total']:.4f} topic={score['topic']:.4f} "
            f"format={score['format']:.4f} context={score['context']:.4f} style={score['style']:.4f}"
        )
        print(f"   views={row['views']} channel={row['channel_title']}")
        print(f"   tags={', '.join(row['tags'][:10])}")
        print()

    print("Final Reference Videos:")
    if not search_result["selected_results"]:
        print("(no results)")
    else:
        for rank, result in enumerate(search_result["selected_results"][:top_k], start=1):
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
    args = parse_args()
    query = QUERY_PRESETS[args.preset]
    index_path = PROCESSED_DIR / f"{args.country.upper()}_Trending.processed.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Processed file not found: {index_path}")

    print("=== Feature Toggle Demo ===")
    print("Country:", args.country.upper())
    print("Preset:", args.preset)
    print("Title:", query["title"])
    print("Tags:", ", ".join(query["tags"]))
    print()

    print("Loading processed vector index...")
    index = ProcessedVectorIndex.load(index_path)
    print(f"Rows loaded: {index.size}")
    print()

    encoder = DimensionVectorEncoder()
    query_signature = encoder.encode_video(query["title"], query["tags"])

    for label, enabled_features in FEATURE_SETS:
        search_result = index.search(
            query_signature=query_signature,
            enabled_features=enabled_features,
            top_k=args.top_k,
            topic_percent=0.10,
            topic_hard_cap=1000,
            topic_min_keep=50,
            absolute_threshold=0.60,
            relative_ratio=0.85,
        )
        print_results(label, enabled_features, search_result, index, args.top_k)
        print("---")
        print()


if __name__ == "__main__":
    main()
