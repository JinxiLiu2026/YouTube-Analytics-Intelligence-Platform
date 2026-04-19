from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .config import ARCHIVE_DIR, DEFAULT_TOP_K, MAX_TOP_K


TOKEN_RE = re.compile(r"[A-Za-z0-9\u00C0-\u024F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF#@']+")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "with",
    "from",
    "by",
    "is",
    "it",
    "this",
    "that",
    "my",
    "our",
    "your",
    "video",
    "official",
    "new",
    "2026",
}


@dataclass
class VideoRecord:
    video_id: str
    country: str
    title: str
    channel_title: str
    category_id: str
    publish_time: str
    trending_date: str
    tags: list[str]
    description: str
    views: int
    likes: int
    comments: int
    normalized_tags: list[str]
    text: str
    token_counts: Counter[str]
    token_norm: float

    @property
    def engagement_rate(self) -> float:
        if self.views <= 0:
            return 0.0
        return round((self.likes + self.comments) / self.views, 4)


def normalize_tag(tag: str) -> str:
    return " ".join(tag.lower().strip().split())


def normalize_tag_list(raw_tags: str) -> list[str]:
    if not raw_tags or raw_tags == "[none]":
        return []
    normalized = []
    for part in raw_tags.split("|"):
        cleaned = normalize_tag(part)
        if cleaned:
            normalized.append(cleaned)
    return normalized


def tokenize(text: str) -> list[str]:
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def counts_to_norm(token_counts: Counter[str]) -> float:
    return math.sqrt(sum(value * value for value in token_counts.values())) or 1.0


def build_query_text(title: str, description: str) -> str:
    parts = [title.strip(), description.strip()]
    return "\n".join(part for part in parts if part)


def safe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def candidate_pool_size(total_candidates: int, top_k: int) -> int:
    if total_candidates <= 0:
        return top_k
    top_five_percent = math.ceil(total_candidates * 0.05)
    return max(top_k, min(50, top_five_percent))


class VideoSearchIndex:
    def __init__(self, archive_dir: Path) -> None:
        self.archive_dir = archive_dir
        self.videos: list[VideoRecord] = []
        self.countries: set[str] = set()
        self.inverted_index: dict[str, list[int]] = defaultdict(list)
        self.tag_index: dict[str, list[int]] = defaultdict(list)
        self.tag_token_index: dict[str, list[int]] = defaultdict(list)
        self.document_frequency: Counter[str] = Counter()
        self.total_documents: int = 0

    def load(self) -> None:
        csv_files = sorted(self.archive_dir.glob("*_Trending.csv"))
        for csv_path in csv_files:
            country = csv_path.stem.split("_", 1)[0]
            self.countries.add(country)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    tags = normalize_tag_list(row.get("tags", ""))
                    description = (row.get("description") or "").strip()
                    text = build_query_text(row.get("title", ""), description)
                    token_counts = Counter(tokenize(text))
                    record = VideoRecord(
                        video_id=row.get("video_id", ""),
                        country=country,
                        title=row.get("title", ""),
                        channel_title=row.get("channel_title", ""),
                        category_id=row.get("category_id", ""),
                        publish_time=row.get("publish_time", ""),
                        trending_date=row.get("trending_date", ""),
                        tags=tags,
                        description=description,
                        views=safe_int(row.get("views")),
                        likes=safe_int(row.get("likes")),
                        comments=safe_int(row.get("comments")),
                        normalized_tags=tags,
                        text=text,
                        token_counts=token_counts,
                        token_norm=counts_to_norm(token_counts),
                    )
                    index = len(self.videos)
                    self.videos.append(record)
                    for token in token_counts:
                        self.inverted_index[token].append(index)
                        self.document_frequency[token] += 1
                    for tag in tags:
                        self.tag_index[tag].append(index)
                        for token in tokenize(tag):
                            self.tag_token_index[token].append(index)
                    self.total_documents += 1

    def idf(self, token: str) -> float:
        df = self.document_frequency.get(token, 0)
        return math.log((self.total_documents + 1) / (df + 1)) + 1.0

    def tag_score(self, query_tags: list[str], candidate_tags: list[str]) -> tuple[float, list[str]]:
        if not query_tags or not candidate_tags:
            return 0.0, []

        candidate_set = set(candidate_tags)
        matched_tags = []
        total_score = 0.0

        for query_tag in query_tags:
            best_score = 0.0
            best_match = ""
            query_tokens = set(tokenize(query_tag))
            for candidate_tag in candidate_set:
                if query_tag == candidate_tag:
                    best_score = 1.0
                    best_match = candidate_tag
                    break
                candidate_tokens = set(tokenize(candidate_tag))
                if not query_tokens or not candidate_tokens:
                    continue
                overlap = len(query_tokens & candidate_tokens)
                if overlap == 0:
                    continue
                partial_score = overlap / max(len(query_tokens), len(candidate_tokens))
                if partial_score > best_score:
                    best_score = partial_score
                    best_match = candidate_tag
            total_score += best_score
            if best_match and best_match not in matched_tags:
                matched_tags.append(best_match)

        return round(total_score / len(query_tags), 4), matched_tags[:8]

    def text_score(self, query_counts: Counter[str], query_weight_norm: float, record: VideoRecord) -> tuple[float, list[str]]:
        dot = 0.0
        matched_terms = []
        record_weight_norm = math.sqrt(
            sum((count * self.idf(token)) ** 2 for token, count in record.token_counts.items())
        ) or 1.0
        for token, weight in query_counts.items():
            if token in record.token_counts:
                token_weight = self.idf(token)
                dot += (weight * token_weight) * (record.token_counts[token] * token_weight)
                matched_terms.append(token)
        similarity = dot / (query_weight_norm * record_weight_norm)
        return round(similarity, 4), matched_terms[:8]

    def _score_candidates(
        self,
        title: str,
        description: str,
        tags: list[str],
        country: Optional[str],
    ) -> tuple[str, list[str], list[dict]]:
        query_text = build_query_text(title, description)
        query_counts = Counter(tokenize(query_text))
        query_weight_norm = math.sqrt(
            sum((count * self.idf(token)) ** 2 for token, count in query_counts.items())
        ) or 1.0
        keywords = [token for token, _ in query_counts.most_common(12)]
        query_tags = [normalize_tag(tag) for tag in tags if normalize_tag(tag)]

        candidate_counter: Counter[int] = Counter()
        tag_matched_candidates = set()

        if query_tags:
            for query_tag in query_tags:
                for video_idx in self.tag_index.get(query_tag, []):
                    if country and self.videos[video_idx].country != country.upper():
                        continue
                    candidate_counter[video_idx] += 3
                    tag_matched_candidates.add(video_idx)
                for token in tokenize(query_tag):
                    for video_idx in self.tag_token_index.get(token, []):
                        if country and self.videos[video_idx].country != country.upper():
                            continue
                        candidate_counter[video_idx] += 1
                        tag_matched_candidates.add(video_idx)

        if not tag_matched_candidates:
            for token in query_counts:
                for video_idx in self.inverted_index.get(token, []):
                    if country and self.videos[video_idx].country != country.upper():
                        continue
                    candidate_counter[video_idx] += 1
        else:
            for token in query_counts:
                for video_idx in self.inverted_index.get(token, []):
                    if video_idx not in tag_matched_candidates:
                        continue
                    if country and self.videos[video_idx].country != country.upper():
                        continue
                    candidate_counter[video_idx] += 1

        if not candidate_counter:
            for token in query_counts:
                for video_idx in self.inverted_index.get(token, []):
                    if country and self.videos[video_idx].country != country.upper():
                        continue
                    candidate_counter[video_idx] += 1

        if not candidate_counter:
            for idx, record in enumerate(self.videos):
                if country and record.country != country.upper():
                    continue
                candidate_counter[idx] = 1
                if len(candidate_counter) >= 2000:
                    break

        scored = []
        for idx in candidate_counter:
            record = self.videos[idx]
            tag_score, matched_tags = self.tag_score(query_tags, record.normalized_tags)
            text_score, matched_terms = self.text_score(query_counts, query_weight_norm, record)

            if query_tags:
                final_score = (0.65 * tag_score) + (0.35 * text_score)
            else:
                final_score = text_score

            scored.append(
                {
                    "record": record,
                    "score": round(final_score, 4),
                    "tag_score": tag_score,
                    "text_score": text_score,
                    "matched_terms": matched_terms[:8],
                    "matched_tags": matched_tags,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return query_text, keywords, scored

    def search_debug(
        self,
        title: str,
        description: str,
        tags: list[str],
        country: Optional[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> tuple[str, list[str], list[dict], list[dict], list[dict]]:
        query_text, keywords, scored = self._score_candidates(title, description, tags, country)
        pool_size = candidate_pool_size(len(scored), min(top_k, MAX_TOP_K))
        candidate_pool = scored[:pool_size]
        final_ranked = sorted(
            candidate_pool,
            key=lambda item: (
                item["record"].views,
                item["score"],
                item["text_score"],
            ),
            reverse=True,
        )
        return (
            query_text,
            keywords,
            scored[: min(top_k, MAX_TOP_K)],
            candidate_pool,
            final_ranked[: min(top_k, MAX_TOP_K)],
        )

    def search(self, title: str, description: str, content: str, tags: list[str], country: Optional[str], top_k: int = DEFAULT_TOP_K) -> tuple[str, list[str], list[dict]]:
        _query_text, _keywords, _top_by_score, _candidate_pool, final_ranked = self.search_debug(
            title=title,
            description=description,
            tags=tags,
            country=country,
            top_k=top_k,
        )
        return _query_text, _keywords, final_ranked


@lru_cache(maxsize=1)
def get_index() -> VideoSearchIndex:
    index = VideoSearchIndex(ARCHIVE_DIR)
    index.load()
    return index
