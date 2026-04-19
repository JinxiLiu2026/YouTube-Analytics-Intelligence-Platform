from __future__ import annotations

from collections import defaultdict
from typing import Iterable
from typing import Optional

import requests
from youtube_transcript_api import YouTubeTranscriptApi

from youtube_insights.schemas import (
    CommentSnippet,
    TranscriptChunk,
    TranscriptSegment,
    VideoMetadata,
    VideoStats,
)


class YouTubePublicDataClient:
    def __init__(self, api_key: Optional[str], timeout_seconds: int = 20) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def fetch_video_metadata(self, video_ids: Iterable[str]) -> dict[str, VideoMetadata]:
        ids = [video_id.strip() for video_id in video_ids if video_id.strip()]
        if not ids:
            return {}

        if not self.api_key:
            return {video_id: VideoMetadata(video_id=video_id) for video_id in ids}

        response = self.session.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(ids),
                "key": self.api_key,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        result: dict[str, VideoMetadata] = {video_id: VideoMetadata(video_id=video_id) for video_id in ids}
        for item in payload.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            result[item["id"]] = VideoMetadata(
                video_id=item["id"],
                title=snippet.get("title"),
                channel_title=snippet.get("channelTitle"),
                published_at=snippet.get("publishedAt"),
                description=snippet.get("description"),
                duration=item.get("contentDetails", {}).get("duration"),
                stats=VideoStats(
                    view_count=_safe_int(stats.get("viewCount")),
                    like_count=_safe_int(stats.get("likeCount")),
                    comment_count=_safe_int(stats.get("commentCount")),
                ),
            )
        return result

    def fetch_comments(self, video_id: str, max_results: int = 40) -> list[CommentSnippet]:
        if not self.api_key or max_results <= 0:
            return []

        comments: list[CommentSnippet] = []
        next_page_token: Optional[str] = None

        while len(comments) < max_results:
            remaining = max_results - len(comments)
            response = self.session.get(
                "https://www.googleapis.com/youtube/v3/commentThreads",
                params={
                    "part": "snippet",
                    "videoId": video_id,
                    "maxResults": min(remaining, 100),
                    "order": "relevance",
                    "pageToken": next_page_token,
                    "textFormat": "plainText",
                    "key": self.api_key,
                },
                timeout=self.timeout_seconds,
            )
            if response.status_code in {403, 404}:
                return comments

            response.raise_for_status()
            payload = response.json()
            for item in payload.get("items", []):
                top_level = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                text = (top_level.get("textDisplay") or "").strip()
                if not text:
                    continue
                comments.append(
                    CommentSnippet(
                        author=top_level.get("authorDisplayName", "unknown"),
                        text=text,
                        like_count=_safe_int(top_level.get("likeCount")),
                        published_at=top_level.get("publishedAt"),
                    )
                )
            next_page_token = payload.get("nextPageToken")
            if not next_page_token:
                break

        return comments


class TranscriptProvider:
    def __init__(self, languages: Iterable[str]) -> None:
        self.languages = tuple(languages)
        self.api = YouTubeTranscriptApi()

    def fetch_transcript(self, video_id: str) -> list[TranscriptSegment]:
        transcript = self.api.fetch(video_id, languages=list(self.languages))
        segments: list[TranscriptSegment] = []
        for item in transcript:
            start = float(getattr(item, "start", 0.0))
            duration = float(getattr(item, "duration", 0.0))
            text = " ".join(str(getattr(item, "text", "")).split())
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start=start,
                    duration=duration,
                    end=start + duration,
                    text=text,
                )
            )
        return segments


def chunk_transcript(segments: list[TranscriptSegment], chunk_seconds: int) -> list[TranscriptChunk]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive.")

    grouped: dict[int, list[TranscriptSegment]] = defaultdict(list)
    for segment in segments:
        bucket = int(segment.start // chunk_seconds)
        grouped[bucket].append(segment)

    chunks: list[TranscriptChunk] = []
    for bucket in sorted(grouped):
        bucket_segments = grouped[bucket]
        chunks.append(
            TranscriptChunk(
                start=float(bucket * chunk_seconds),
                end=float((bucket + 1) * chunk_seconds),
                text=" ".join(segment.text for segment in bucket_segments).strip(),
                segment_count=len(bucket_segments),
            )
        )
    return chunks


def sample_transcript_segments(
    segments: list[TranscriptSegment],
    window_seconds: int,
) -> list[TranscriptSegment]:
    if not segments or window_seconds <= 0:
        return segments

    video_end = max(segment.end for segment in segments)
    if video_end <= window_seconds:
        return segments

    windows = [(0.0, float(window_seconds))]

    middle_start = max(0.0, (video_end - float(window_seconds)) / 2.0)
    middle_end = min(video_end, middle_start + float(window_seconds))
    windows.append((middle_start, middle_end))

    tail_start = max(0.0, video_end - float(window_seconds))
    windows.append((tail_start, video_end))

    selected: list[TranscriptSegment] = []
    seen: set[tuple[float, float, str]] = set()
    for segment in segments:
        overlaps_window = any(
            (segment.start < window_end) and (segment.end > window_start)
            for window_start, window_end in windows
        )
        if not overlaps_window:
            continue
        key = (segment.start, segment.end, segment.text)
        if key in seen:
            continue
        seen.add(key)
        selected.append(segment)

    selected.sort(key=lambda item: item.start)
    return selected


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
