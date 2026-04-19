from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .config import OPENAI_CONFIG
from .openai_client import get_openai_client


FORMAT_SEEDS = [
    "vlog",
    "reaction",
    "review",
    "interview",
    "breakdown",
    "tutorial",
    "livestream",
    "highlights",
    "commentary",
    "analysis",
]

CONTEXT_SEEDS = [
    "city",
    "country",
    "place",
    "location",
    "setting",
    "environment",
    "venue",
    "destination",
    "region",
]

STYLE_SEEDS = [
    "dramatic",
    "funny",
    "weird",
    "shocking",
    "intense",
    "emotional",
    "exclusive",
    "chaotic",
    "surprising",
    "curious",
]

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vec))
    if norm == 0:
        return vec
    return [value / norm for value in vec]


def mean_vector(vectors: Iterable[list[float]]) -> list[float]:
    vectors = list(vectors)
    if not vectors:
        return []
    size = len(vectors[0])
    totals = [0.0] * size
    for vector in vectors:
        for idx, value in enumerate(vector):
            totals[idx] += value
    return [value / len(vectors) for value in totals]


def weighted_average(vectors: list[list[float]], weights: list[float]) -> list[float]:
    if not vectors or not weights:
        return []
    weight_sum = sum(weights)
    if weight_sum == 0:
        return [0.0] * len(vectors[0])
    totals = [0.0] * len(vectors[0])
    for vector, weight in zip(vectors, weights):
        for idx, value in enumerate(vector):
            totals[idx] += weight * value
    return [value / weight_sum for value in totals]


def softmax(values: list[float], temperature: float = 6.0) -> list[float]:
    if not values:
        return []
    scaled = [value * temperature for value in values]
    max_value = max(scaled)
    exp_values = [math.exp(value - max_value) for value in scaled]
    total = sum(exp_values)
    if total == 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exp_values]


def split_title_phrases(title: str) -> list[str]:
    cleaned = " ".join(title.strip().split())
    return [cleaned] if cleaned else []


def split_tag_phrases(tags: list[str]) -> list[str]:
    phrases: list[str] = []
    for tag in tags:
        cleaned = " ".join(tag.strip().split())
        if cleaned:
            phrases.append(cleaned)
    return phrases


def build_phrase_units(title: str, tags: list[str]) -> list[str]:
    title_units = split_title_phrases(title)
    seen = {phrase.lower() for phrase in title_units}
    tag_units = []
    for phrase in split_tag_phrases(tags):
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        tag_units.append(phrase)
    return title_units + tag_units


@dataclass
class PhraseProjection:
    phrase: str
    source: str
    format_weight: float
    context_weight: float
    style_weight: float


@dataclass
class VideoDimensionSignature:
    phrase_units: list[str]
    projections: list[PhraseProjection]
    topic_vector: list[float]
    format_vector: list[float]
    context_vector: list[float]
    style_vector: list[float]


@dataclass
class RetrievalCandidate:
    item: dict
    signature: VideoDimensionSignature
    topic_score: float
    final_score: dict[str, float] | None = None


class DimensionVectorEncoder:
    def __init__(self) -> None:
        self.client = get_openai_client()
        self.model = OPENAI_CONFIG.embedding_model
        self._prototype_cache: dict[str, list[float]] = {}

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def prototype_vector(self, name: str, seeds: list[str]) -> list[float]:
        if name not in self._prototype_cache:
            seed_embeddings = self.embed_texts(seeds)
            self._prototype_cache[name] = l2_normalize(mean_vector(seed_embeddings))
        return self._prototype_cache[name]

    def prototypes(self) -> dict[str, list[float]]:
        return {
            "format": self.prototype_vector("format", FORMAT_SEEDS),
            "context": self.prototype_vector("context", CONTEXT_SEEDS),
            "style": self.prototype_vector("style", STYLE_SEEDS),
        }

    def encode_video(self, title: str, tags: list[str]) -> VideoDimensionSignature:
        phrase_units = build_phrase_units(title, tags)
        if not phrase_units:
            return VideoDimensionSignature([], [], [], [], [], [])

        phrase_embeddings = [l2_normalize(vec) for vec in self.embed_texts(phrase_units)]
        prototypes = self.prototypes()

        projections: list[PhraseProjection] = []
        format_weights: list[float] = []
        context_weights: list[float] = []
        style_weights: list[float] = []

        title_phrase_count = len(split_title_phrases(title))

        for idx, (phrase, embedding) in enumerate(zip(phrase_units, phrase_embeddings)):
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

            source = "title" if idx < title_phrase_count else "tag"
            source_weight = 2.0 if source == "title" else 1.0

            projections.append(
                PhraseProjection(
                    phrase=phrase,
                    source=source,
                    format_weight=format_weight,
                    context_weight=context_weight,
                    style_weight=style_weight,
                )
            )
            format_weights.append(format_weight * source_weight)
            context_weights.append(context_weight * source_weight)
            style_weights.append(style_weight * source_weight)

        topic_vector = l2_normalize(mean_vector(phrase_embeddings))
        format_vector = l2_normalize(weighted_average(phrase_embeddings, format_weights))
        context_vector = l2_normalize(weighted_average(phrase_embeddings, context_weights))
        style_vector = l2_normalize(weighted_average(phrase_embeddings, style_weights))

        return VideoDimensionSignature(
            phrase_units=phrase_units,
            projections=projections,
            topic_vector=topic_vector,
            format_vector=format_vector,
            context_vector=context_vector,
            style_vector=style_vector,
        )


def signature_similarity(
    left: VideoDimensionSignature,
    right: VideoDimensionSignature,
    topic_weight: float = 0.40,
    format_weight: float = 0.20,
    context_weight: float = 0.25,
    style_weight: float = 0.15,
) -> dict[str, float]:
    sim_topic = cosine_similarity(left.topic_vector, right.topic_vector) if left.topic_vector and right.topic_vector else 0.0
    sim_format = cosine_similarity(left.format_vector, right.format_vector) if left.format_vector and right.format_vector else 0.0
    sim_context = cosine_similarity(left.context_vector, right.context_vector) if left.context_vector and right.context_vector else 0.0
    sim_style = cosine_similarity(left.style_vector, right.style_vector) if left.style_vector and right.style_vector else 0.0
    sim_total = (
        topic_weight * sim_topic
        + format_weight * sim_format
        + context_weight * sim_context
        + style_weight * sim_style
    )
    return {
        "topic": sim_topic,
        "format": sim_format,
        "context": sim_context,
        "style": sim_style,
        "total": sim_total,
    }


def topic_similarity(
    left: VideoDimensionSignature,
    right: VideoDimensionSignature,
) -> float:
    if not left.topic_vector or not right.topic_vector:
        return 0.0
    return cosine_similarity(left.topic_vector, right.topic_vector)


def coarse_pool_limit(
    total_candidates: int,
    percent: float = 0.10,
    hard_cap: int = 1000,
    min_keep: int = 50,
) -> int:
    if total_candidates <= 0:
        return 0
    percent_count = math.ceil(total_candidates * percent)
    return min(total_candidates, max(min_keep, min(hard_cap, percent_count)))


def passes_relevance_gate(
    score: float,
    best_score: float,
    absolute_threshold: float = 0.60,
    relative_ratio: float = 0.85,
) -> bool:
    return score >= absolute_threshold and score >= (best_score * relative_ratio)


def build_coarse_candidates(
    query_signature: VideoDimensionSignature,
    items_with_signatures: list[tuple[dict, VideoDimensionSignature]],
    topic_percent: float = 0.10,
    topic_hard_cap: int = 1000,
    topic_min_keep: int = 50,
) -> list[RetrievalCandidate]:
    candidates = [
        RetrievalCandidate(
            item=item,
            signature=signature,
            topic_score=topic_similarity(query_signature, signature),
        )
        for item, signature in items_with_signatures
    ]
    candidates.sort(key=lambda candidate: candidate.topic_score, reverse=True)
    limit = coarse_pool_limit(
        total_candidates=len(candidates),
        percent=topic_percent,
        hard_cap=topic_hard_cap,
        min_keep=topic_min_keep,
    )
    return candidates[:limit]


def rerank_candidates(
    query_signature: VideoDimensionSignature,
    coarse_candidates: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    reranked: list[RetrievalCandidate] = []
    for candidate in coarse_candidates:
        candidate.final_score = signature_similarity(query_signature, candidate.signature)
        reranked.append(candidate)
    reranked.sort(
        key=lambda candidate: (
            candidate.final_score["total"] if candidate.final_score else 0.0
        ),
        reverse=True,
    )
    return reranked


def select_reference_videos(
    reranked_candidates: list[RetrievalCandidate],
    top_k: int = 5,
    absolute_threshold: float = 0.60,
    relative_ratio: float = 0.85,
) -> tuple[list[RetrievalCandidate], list[RetrievalCandidate]]:
    if not reranked_candidates:
        return [], []

    best_score = reranked_candidates[0].final_score["total"] if reranked_candidates[0].final_score else 0.0
    relevant_candidates = [
        candidate
        for candidate in reranked_candidates
        if candidate.final_score
        and passes_relevance_gate(
            score=candidate.final_score["total"],
            best_score=best_score,
            absolute_threshold=absolute_threshold,
            relative_ratio=relative_ratio,
        )
    ]

    selected = sorted(
        relevant_candidates,
        key=lambda candidate: (
            int(candidate.item.get("views", 0) or 0),
            candidate.final_score["total"] if candidate.final_score else 0.0,
        ),
        reverse=True,
    )[:top_k]

    return relevant_candidates, selected
