from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.dimension_vectors import (
    CONTEXT_SEEDS,
    FORMAT_SEEDS,
    STYLE_SEEDS,
    build_coarse_candidates,
    build_phrase_units,
    cosine_similarity,
    l2_normalize,
    mean_vector,
    rerank_candidates,
    select_reference_videos,
    softmax,
    weighted_average,
)
from backend.app.openai_client import get_openai_client
from backend.app.config import OPENAI_CONFIG


DATASET_PATH = ROOT_DIR / "archive" / "US_Trending.csv"
SUBSET_SIZE = 200


def normalize_tags(raw_tags: str) -> list[str]:
    if not raw_tags or raw_tags == "[none]":
        return []
    return [" ".join(tag.strip().split()) for tag in raw_tags.split("|") if tag.strip()]


def chunked(seq: list[str], size: int) -> list[list[str]]:
    return [seq[index:index + size] for index in range(0, len(seq), size)]


def load_subset(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "video_id": row.get("video_id", ""),
                    "title": row.get("title", ""),
                    "tags": normalize_tags(row.get("tags", "")),
                    "views": int(row.get("views", "0") or 0),
                    "channel_title": row.get("channel_title", ""),
                    "description": (row.get("description") or "").strip(),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def build_synthetic_rows() -> list[dict]:
    return [
        {
            "video_id": "synthetic-1",
            "title": "24 Hours in Osaka's Tiny Capsule Hotel Vlog",
            "tags": ["osaka", "travel", "capsule hotel", "japan", "vlog", "hotel challenge"],
            "views": 180000,
            "channel_title": "Synthetic Travel 1",
            "description": "A travel vlog about spending 24 hours inside a tiny capsule hotel in Osaka.",
        },
        {
            "video_id": "synthetic-2",
            "title": "My Cozy Kyoto Travel Vlog and Cafe Diary",
            "tags": ["kyoto vlog", "japan travel", "cafe diary", "cozy vlog"],
            "views": 220000,
            "channel_title": "Synthetic Travel 2",
            "description": "A calm Kyoto travel vlog focused on cafes, journaling, and slow travel.",
        },
        {
            "video_id": "synthetic-3",
            "title": "I Stayed in Tokyo's Weirdest Budget Capsule Hotel",
            "tags": ["tokyo hotel", "capsule hotel", "budget travel", "japan vlog", "weird hotel"],
            "views": 145000,
            "channel_title": "Synthetic Travel 3",
            "description": "A quirky capsule hotel travel experience in Tokyo with room tour and local food.",
        },
        {
            "video_id": "synthetic-4",
            "title": "Tokyo Night Food Tour Travel Vlog",
            "tags": ["tokyo food", "travel vlog", "japan nightlife", "street food"],
            "views": 260000,
            "channel_title": "Synthetic Travel 4",
            "description": "A Japan travel vlog focused on late-night food spots in Tokyo.",
        },
    ]


def build_phrase_embedding_map(phrases: list[str]) -> dict[str, list[float]]:
    client = get_openai_client()
    mapping: dict[str, list[float]] = {}
    unique_phrases = []
    seen = set()
    for phrase in phrases:
        key = phrase.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_phrases.append(key)

    for batch in chunked(unique_phrases, 200):
        response = client.embeddings.create(model=OPENAI_CONFIG.embedding_model, input=batch)
        for phrase, item in zip(batch, response.data):
            mapping[phrase] = l2_normalize(item.embedding)
    return mapping


def build_prototypes(embedding_map: dict[str, list[float]]) -> dict[str, list[float]]:
    return {
        "format": l2_normalize(mean_vector([embedding_map[seed] for seed in FORMAT_SEEDS])),
        "context": l2_normalize(mean_vector([embedding_map[seed] for seed in CONTEXT_SEEDS])),
        "style": l2_normalize(mean_vector([embedding_map[seed] for seed in STYLE_SEEDS])),
    }


def encode_signature(phrase_units: list[str], embedding_map: dict[str, list[float]], prototypes: dict[str, list[float]]) -> dict:
    if not phrase_units:
        return {
            "topic_vector": [],
            "format_vector": [],
            "context_vector": [],
            "style_vector": [],
        }

    phrase_embeddings = [embedding_map[phrase] for phrase in phrase_units]
    title_phrase_count = 1 if phrase_units else 0

    format_weights = []
    context_weights = []
    style_weights = []

    for idx, embedding in enumerate(phrase_embeddings):
        raw_format = max(cosine_similarity(embedding, prototypes["format"]), 0.0)
        raw_context = max(cosine_similarity(embedding, prototypes["context"]), 0.0)
        raw_style = max(cosine_similarity(embedding, prototypes["style"]), 0.0)

        if raw_format == 0 and raw_context == 0 and raw_style == 0:
            format_weight = context_weight = style_weight = 1.0 / 3.0
        else:
            format_weight, context_weight, style_weight = softmax(
                [raw_format, raw_context, raw_style],
                temperature=6.0,
            )

        source_weight = 2.0 if idx < title_phrase_count else 1.0
        format_weights.append(format_weight * source_weight)
        context_weights.append(context_weight * source_weight)
        style_weights.append(style_weight * source_weight)

    return {
        "topic_vector": l2_normalize(mean_vector(phrase_embeddings)),
        "format_vector": l2_normalize(weighted_average(phrase_embeddings, format_weights)),
        "context_vector": l2_normalize(weighted_average(phrase_embeddings, context_weights)),
        "style_vector": l2_normalize(weighted_average(phrase_embeddings, style_weights)),
    }


def print_ranked(label: str, items: list[dict]) -> None:
    print(label)
    for index, item in enumerate(items, start=1):
        row = item["row"]
        score = item["score"]
        print(f"{index}. {row['title']}")
        print(
            f"   total={score['total']:.4f} topic={score['topic']:.4f} "
            f"format={score['format']:.4f} context={score['context']:.4f} style={score['style']:.4f}"
        )
        print(f"   views={row['views']} channel={row['channel_title']}")
        print(f"   tags={', '.join(row['tags'])}")
        print()


def main() -> None:
    query = {
        "title": "I Tried Surviving 24 Hours in Tokyo's Weirdest Capsule Hotel",
        "tags": ["tokyo travel", "capsule hotel", "japan vlog", "24 hour challenge"],
    }

    subset = load_subset(DATASET_PATH, SUBSET_SIZE)
    subset.extend(build_synthetic_rows())
    query_phrases = build_phrase_units(query["title"], query["tags"])

    all_phrases = FORMAT_SEEDS + CONTEXT_SEEDS + STYLE_SEEDS + query_phrases
    for row in subset:
        row["phrase_units"] = build_phrase_units(row["title"], row["tags"])
        all_phrases.extend(row["phrase_units"])

    embedding_map = build_phrase_embedding_map(all_phrases)
    prototypes = build_prototypes(embedding_map)

    query_signature = encode_signature(query_phrases, embedding_map, prototypes)

    items_with_signatures = []
    for row in subset:
        row_signature = encode_signature(row["phrase_units"], embedding_map, prototypes)
        items_with_signatures.append((row, row_signature))

    coarse_candidates = build_coarse_candidates(
        query_signature=query_signature,
        items_with_signatures=items_with_signatures,
        topic_percent=0.10,
        topic_hard_cap=1000,
        topic_min_keep=20,
    )
    reranked_candidates = rerank_candidates(query_signature, coarse_candidates)
    relevant_candidates, selected_by_views = select_reference_videos(
        reranked_candidates,
        top_k=5,
        absolute_threshold=0.60,
        relative_ratio=0.85,
    )

    top_by_similarity = [
        {
            "row": candidate.item,
            "score": candidate.final_score,
        }
        for candidate in reranked_candidates[:5]
        if candidate.final_score
    ]
    top_by_views = [
        {
            "row": candidate.item,
            "score": candidate.final_score,
        }
        for candidate in selected_by_views
        if candidate.final_score
    ]

    print("=== Real Subset Similarity Demo ===\n")
    print("Dataset:", DATASET_PATH.name)
    print("Subset size:", len(subset))
    print("Synthetic rows added:", len(build_synthetic_rows()))
    print("Stage 1 coarse pool size:", len(coarse_candidates))
    print("Stage 2 relevant pool size:", len(relevant_candidates))
    print("Query title:", query["title"])
    print("Query tags:", ", ".join(query["tags"]))
    print()

    print_ranked("=== Top 5 By Final Similarity ===", top_by_similarity)
    print_ranked("=== Top 5 Selected References (Views Within Relevance Gate) ===", top_by_views)


if __name__ == "__main__":
    main()
