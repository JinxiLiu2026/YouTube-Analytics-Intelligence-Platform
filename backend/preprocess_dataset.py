from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import OPENAI_CONFIG
from backend.app.dimension_vectors import (
    CONTEXT_SEEDS,
    FORMAT_SEEDS,
    STYLE_SEEDS,
    build_phrase_units,
)
from backend.app.openai_client import get_openai_client


DEFAULT_INPUT = ROOT_DIR / "archive" / "US_Trending.csv"
DEFAULT_OUTPUT = ROOT_DIR / "processed" / "US_Trending.processed.jsonl"
DEFAULT_EMBEDDING_BATCH_SIZE = 200


def chunked(seq: list[str], size: int) -> list[list[str]]:
    return [seq[index:index + size] for index in range(0, len(seq), size)]


def l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def l2_normalize_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / max(norm, eps)


def softmax_np(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def infer_country(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0].upper()
    return stem[:2].upper()


def normalize_tags(raw_tags: str) -> list[str]:
    if not raw_tags or raw_tags == "[none]":
        return []

    tags = []
    for tag in raw_tags.split("|"):
        cleaned = " ".join(tag.strip().split())
        if cleaned:
            tags.append(cleaned)
    return tags


def load_best_rows_by_video_id(path: Path, country: str) -> list[dict]:
    best_by_video_id: dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_id = (row.get("video_id") or "").strip()
            if not video_id:
                continue

            views = int(row.get("views", "0") or 0)
            normalized_row = {
                "video_id": video_id,
                "country": country,
                "title": " ".join((row.get("title") or "").split()),
                "tags": normalize_tags(row.get("tags", "")),
                "description": (row.get("description") or "").strip(),
                "views": views,
                "channel_title": " ".join((row.get("channel_title") or "").split()),
            }

            if not normalized_row["title"] and not normalized_row["tags"]:
                continue

            existing = best_by_video_id.get(video_id)
            if existing is None or views > existing["views"]:
                best_by_video_id[video_id] = normalized_row

    return list(best_by_video_id.values())


def build_phrase_embedding_map(
    phrases: list[str],
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> dict[str, np.ndarray]:
    client = get_openai_client()
    mapping: dict[str, np.ndarray] = {}
    unique_phrases = []
    seen = set()

    for phrase in phrases:
        key = phrase.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_phrases.append(key)

    batches = chunked(unique_phrases, batch_size)
    total_batches = len(batches)
    total_phrases = len(unique_phrases)
    completed_phrases = 0

    for batch_index, batch in enumerate(batches, start=1):
        response = client.embeddings.create(model=OPENAI_CONFIG.embedding_model, input=batch)
        batch_matrix = np.array([item.embedding for item in response.data], dtype=np.float32)
        batch_matrix = l2_normalize_rows(batch_matrix)
        for phrase, vector in zip(batch, batch_matrix):
            mapping[phrase] = vector
        completed_phrases += len(batch)
        percent = (completed_phrases / max(total_phrases, 1)) * 100
        print(
            f"Embedding batches: {batch_index}/{total_batches} "
            f"({completed_phrases}/{total_phrases} phrases, {percent:.1f}%)"
        )

    return mapping


def build_prototypes(embedding_map: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "format": l2_normalize_vector(np.mean(np.stack([embedding_map[seed] for seed in FORMAT_SEEDS]), axis=0)),
        "context": l2_normalize_vector(np.mean(np.stack([embedding_map[seed] for seed in CONTEXT_SEEDS]), axis=0)),
        "style": l2_normalize_vector(np.mean(np.stack([embedding_map[seed] for seed in STYLE_SEEDS]), axis=0)),
    }


def encode_signature(
    title: str,
    tags: list[str],
    embedding_map: dict[str, np.ndarray],
    prototypes: dict[str, np.ndarray],
) -> dict[str, list[float]]:
    phrase_units = build_phrase_units(title, tags)
    if not phrase_units:
        return {
            "topic_vector": [],
            "format_vector": [],
            "context_vector": [],
            "style_vector": [],
        }

    phrase_matrix = np.stack([embedding_map[phrase] for phrase in phrase_units], axis=0)
    prototype_matrix = np.stack(
        [
            prototypes["format"],
            prototypes["context"],
            prototypes["style"],
        ],
        axis=0,
    )

    scores = phrase_matrix @ prototype_matrix.T
    scores = np.maximum(scores, 0.0)
    all_zero_mask = np.all(scores == 0.0, axis=1)
    weights = softmax_np(scores * 6.0, axis=1)
    if np.any(all_zero_mask):
        weights[all_zero_mask] = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)

    source_weights = np.ones((len(phrase_units),), dtype=np.float32)
    source_weights[0] = 2.0
    weighted_scores = weights * source_weights[:, None]

    weighted_sum = weighted_scores.T @ phrase_matrix
    weight_totals = np.sum(weighted_scores, axis=0)
    representation_matrix = weighted_sum / np.maximum(weight_totals[:, None], 1e-12)
    representation_matrix = l2_normalize_rows(representation_matrix)

    topic_vector = l2_normalize_vector(np.mean(phrase_matrix, axis=0))

    return {
        "topic_vector": topic_vector.tolist(),
        "format_vector": representation_matrix[0].tolist(),
        "context_vector": representation_matrix[1].tolist(),
        "style_vector": representation_matrix[2].tolist(),
    }


def process_rows(
    rows: list[dict],
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> list[dict]:
    all_phrases = FORMAT_SEEDS + CONTEXT_SEEDS + STYLE_SEEDS
    for row in rows:
        row["phrase_units"] = build_phrase_units(row["title"], row["tags"])
        all_phrases.extend(row["phrase_units"])

    print("Building phrase embedding map...")
    embedding_map = build_phrase_embedding_map(
        all_phrases,
        batch_size=embedding_batch_size,
    )
    print(f"Unique phrases embedded: {len(embedding_map)}")

    prototypes = build_prototypes(embedding_map)
    processed_rows = []
    total_rows = len(rows)
    for index, row in enumerate(rows, start=1):
        signature = encode_signature(
            title=row["title"],
            tags=row["tags"],
            embedding_map=embedding_map,
            prototypes=prototypes,
        )
        processed_rows.append(
            {
                "video_id": row["video_id"],
                "country": row["country"],
                "title": row["title"],
                "tags": row["tags"],
                "description": row["description"],
                "views": row["views"],
                "channel_title": row["channel_title"],
                "topic_vector": signature["topic_vector"],
                "format_vector": signature["format_vector"],
                "context_vector": signature["context_vector"],
                "style_vector": signature["style_vector"],
            }
        )
        if index % 100 == 0:
            percent = (index / max(total_rows, 1)) * 100
            print(f"Processed videos: {index}/{total_rows} ({percent:.1f}%)")
    return processed_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a processed JSONL retrieval dataset from a Kaggle trending CSV file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to a country CSV file such as archive/US_Trending.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the processed JSONL output file",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Number of unique phrases to send per embeddings API request",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    country = infer_country(args.input)

    print(f"Loading source file: {args.input}")
    rows = load_best_rows_by_video_id(args.input, country)
    print(f"Deduplicated videos: {len(rows)}")

    processed_rows = process_rows(
        rows,
        embedding_batch_size=args.embedding_batch_size,
    )
    write_jsonl(args.output, processed_rows)

    print(f"Wrote processed dataset: {args.output}")
    print(f"Country: {country}")
    print(f"Final rows: {len(processed_rows)}")


if __name__ == "__main__":
    main()
