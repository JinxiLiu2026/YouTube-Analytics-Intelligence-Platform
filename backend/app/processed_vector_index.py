from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .dimension_vectors import VideoDimensionSignature


BASE_DIMENSION_WEIGHTS = {
    "topic": 0.40,
    "format": 0.20,
    "context": 0.25,
    "style": 0.15,
}


@dataclass
class ProcessedVectorSearchResult:
    row_index: int
    score: dict[str, float]


class ProcessedVectorIndex:
    def __init__(
        self,
        path: Path,
        country: str,
        topic_matrix: np.ndarray,
        format_matrix: np.ndarray,
        context_matrix: np.ndarray,
        style_matrix: np.ndarray,
        views_array: np.ndarray,
        offsets: list[int],
    ) -> None:
        self.path = path
        self.country = country
        self.topic_matrix = topic_matrix
        self.format_matrix = format_matrix
        self.context_matrix = context_matrix
        self.style_matrix = style_matrix
        self.views_array = views_array
        self.offsets = offsets
        self.size = topic_matrix.shape[0]

    @classmethod
    def load(cls, path: Path) -> "ProcessedVectorIndex":
        country = path.name.split("_", 1)[0].upper()

        topic_rows: list[np.ndarray] = []
        format_rows: list[np.ndarray] = []
        context_rows: list[np.ndarray] = []
        style_rows: list[np.ndarray] = []
        views: list[int] = []
        offsets: list[int] = []

        with path.open("r", encoding="utf-8") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                row = json.loads(line)
                offsets.append(offset)
                topic_rows.append(np.array(row["topic_vector"], dtype=np.float32))
                format_rows.append(np.array(row["format_vector"], dtype=np.float32))
                context_rows.append(np.array(row["context_vector"], dtype=np.float32))
                style_rows.append(np.array(row["style_vector"], dtype=np.float32))
                views.append(int(row["views"]))

        return cls(
            path=path,
            country=country,
            topic_matrix=np.stack(topic_rows, axis=0),
            format_matrix=np.stack(format_rows, axis=0),
            context_matrix=np.stack(context_rows, axis=0),
            style_matrix=np.stack(style_rows, axis=0),
            views_array=np.array(views, dtype=np.int64),
            offsets=offsets,
        )

    def fetch_metadata(self, row_index: int) -> dict:
        offset = self.offsets[row_index]
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            return json.loads(handle.readline())

    def fetch_many_metadata(self, row_indices: list[int]) -> list[dict]:
        return [self.fetch_metadata(row_index) for row_index in row_indices]

    def _score_components(
        self,
        query_signature: VideoDimensionSignature,
        indices: Optional[np.ndarray] = None,
        enabled_features: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        topic_query = np.array(query_signature.topic_vector, dtype=np.float32)
        format_query = np.array(query_signature.format_vector, dtype=np.float32)
        context_query = np.array(query_signature.context_vector, dtype=np.float32)
        style_query = np.array(query_signature.style_vector, dtype=np.float32)

        topic_matrix = self.topic_matrix if indices is None else self.topic_matrix[indices]
        format_matrix = self.format_matrix if indices is None else self.format_matrix[indices]
        context_matrix = self.context_matrix if indices is None else self.context_matrix[indices]
        style_matrix = self.style_matrix if indices is None else self.style_matrix[indices]

        topic_scores = topic_matrix @ topic_query
        format_scores = format_matrix @ format_query
        context_scores = context_matrix @ context_query
        style_scores = style_matrix @ style_query

        normalized_weights = _resolve_dimension_weights(enabled_features)
        total_scores = (
            normalized_weights["topic"] * topic_scores
            + normalized_weights["format"] * format_scores
            + normalized_weights["context"] * context_scores
            + normalized_weights["style"] * style_scores
        )
        return topic_scores, format_scores, context_scores, style_scores, total_scores

    def search(
        self,
        query_signature: VideoDimensionSignature,
        enabled_features: Optional[list[str]] = None,
        top_k: int = 5,
        topic_percent: float = 0.10,
        topic_hard_cap: int = 1000,
        topic_min_keep: int = 50,
        absolute_threshold: float = 0.60,
        relative_ratio: float = 0.85,
    ) -> dict:
        total_candidates = self.size
        coarse_limit = min(
            total_candidates,
            max(topic_min_keep, min(topic_hard_cap, int(np.ceil(total_candidates * topic_percent)))),
        )

        _, _, _, _, coarse_scores = self._score_components(
            query_signature,
            enabled_features=enabled_features,
        )
        coarse_idx_unsorted = np.argpartition(coarse_scores, -coarse_limit)[-coarse_limit:]
        coarse_idx = coarse_idx_unsorted[np.argsort(coarse_scores[coarse_idx_unsorted])[::-1]]

        topic_sub, format_sub, context_sub, style_sub, total_sub = self._score_components(
            query_signature,
            indices=coarse_idx,
            enabled_features=enabled_features,
        )
        rerank_order = np.argsort(total_sub)[::-1]
        reranked_idx = coarse_idx[rerank_order]
        topic_sub = topic_sub[rerank_order]
        format_sub = format_sub[rerank_order]
        context_sub = context_sub[rerank_order]
        style_sub = style_sub[rerank_order]
        total_sub = total_sub[rerank_order]

        top_results = [
            ProcessedVectorSearchResult(
                row_index=int(row_index),
                score={
                    "topic": float(topic_score),
                    "format": float(format_score),
                    "context": float(context_score),
                    "style": float(style_score),
                    "total": float(total_score),
                },
            )
            for row_index, topic_score, format_score, context_score, style_score, total_score in zip(
                reranked_idx[:top_k],
                topic_sub[:top_k],
                format_sub[:top_k],
                context_sub[:top_k],
                style_sub[:top_k],
                total_sub[:top_k],
            )
        ]

        best_total = float(total_sub[0]) if len(total_sub) else 0.0
        relevant_mask = (total_sub >= absolute_threshold) & (total_sub >= best_total * relative_ratio)
        relevant_idx = reranked_idx[relevant_mask]
        relevant_total = total_sub[relevant_mask]
        relevant_topic = topic_sub[relevant_mask]
        relevant_format = format_sub[relevant_mask]
        relevant_context = context_sub[relevant_mask]
        relevant_style = style_sub[relevant_mask]

        if len(relevant_idx):
            ranked_rows = sorted(
                zip(
                    relevant_idx,
                    relevant_total,
                    relevant_topic,
                    relevant_format,
                    relevant_context,
                    relevant_style,
                ),
                key=lambda item: (
                    int(self.views_array[int(item[0])]),
                    float(item[1]),
                ),
                reverse=True,
            )[:top_k]
            selected_idx = np.array([int(item[0]) for item in ranked_rows], dtype=np.int64)
            selected_total = np.array([float(item[1]) for item in ranked_rows], dtype=np.float32)
            selected_topic = np.array([float(item[2]) for item in ranked_rows], dtype=np.float32)
            selected_format = np.array([float(item[3]) for item in ranked_rows], dtype=np.float32)
            selected_context = np.array([float(item[4]) for item in ranked_rows], dtype=np.float32)
            selected_style = np.array([float(item[5]) for item in ranked_rows], dtype=np.float32)
        else:
            selected_idx = np.array([], dtype=np.int64)
            selected_total = np.array([], dtype=np.float32)
            selected_topic = np.array([], dtype=np.float32)
            selected_format = np.array([], dtype=np.float32)
            selected_context = np.array([], dtype=np.float32)
            selected_style = np.array([], dtype=np.float32)

        selected_results = [
            ProcessedVectorSearchResult(
                row_index=int(row_index),
                score={
                    "topic": float(topic_score),
                    "format": float(format_score),
                    "context": float(context_score),
                    "style": float(style_score),
                    "total": float(total_score),
                },
            )
            for row_index, topic_score, format_score, context_score, style_score, total_score in zip(
                selected_idx,
                selected_topic,
                selected_format,
                selected_context,
                selected_style,
                selected_total,
            )
        ]

        return {
            "coarse_pool_size": int(len(coarse_idx)),
            "relevant_pool_size": int(len(relevant_idx)),
            "top_results": top_results,
            "selected_results": selected_results,
        }


def _resolve_dimension_weights(enabled_features: Optional[list[str]]) -> dict[str, float]:
    enabled = {
        str(feature).strip().lower()
        for feature in (enabled_features or [])
        if str(feature).strip().lower() in BASE_DIMENSION_WEIGHTS
    }
    if not enabled:
        enabled = set(BASE_DIMENSION_WEIGHTS.keys())

    total = sum(BASE_DIMENSION_WEIGHTS[name] for name in enabled)
    if total <= 0:
        total = 1.0

    return {
        name: (BASE_DIMENSION_WEIGHTS[name] / total if name in enabled else 0.0)
        for name in BASE_DIMENSION_WEIGHTS
    }
