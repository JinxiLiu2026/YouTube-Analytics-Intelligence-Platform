from __future__ import annotations

import json
import math
import re
from textwrap import dedent
from typing import Iterable

from openai import OpenAI

from youtube_insights.config import Settings
from youtube_insights.schemas import (
    CrossVideoSummary,
    HighlightMoment,
    VideoContext,
    VideoFinalAnalysis,
    WindowCandidateMoments,
)


class LLMInsightClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    def analyze_video(
        self,
        user_description: str,
        context: VideoContext,
        mode: str = "fast",
    ) -> VideoFinalAnalysis:
        if not context.transcript_chunks:
            return VideoFinalAnalysis(
                creator_takeaways=[
                    "Transcript could not be retrieved, so no moment-level LLM analysis was possible."
                ],
                analyzed_chunk_count=0,
            )

        working_chunks = context.transcript_chunks
        if mode == "fast":
            working_chunks = self._select_fast_chunks(
                user_description=user_description,
                context=context,
            )

        if not working_chunks:
            return VideoFinalAnalysis(
                creator_takeaways=[
                    "No transcript chunks were selected for analysis after fast-mode filtering."
                ],
                analyzed_chunk_count=0,
            )

        candidates: list[HighlightMoment] = []

        if mode == "fast":
            window_candidates = self._extract_window_candidates(
                user_description=user_description,
                context=context,
                window=working_chunks,
            )
            candidates.extend(window_candidates.moments)
        else:
            window_size = self.settings.analysis_window_chunks
            for start_idx in range(0, len(working_chunks), window_size):
                window = working_chunks[start_idx : start_idx + window_size]
                window_candidates = self._extract_window_candidates(
                    user_description=user_description,
                    context=context,
                    window=window,
                )
                candidates.extend(window_candidates.moments)

        deduped_candidates = _dedupe_moments(candidates)
        return self._finalize_video_analysis(
            user_description=user_description,
            context=context,
            candidates=deduped_candidates,
            transcript_outline=working_chunks,
        )

    def summarize_across_videos(
        self,
        user_description: str,
        analyses: list[VideoFinalAnalysis],
        contexts: list[VideoContext],
    ) -> CrossVideoSummary:
        compact_videos = []
        for analysis, context in zip(analyses, contexts):
            compact_videos.append(
                {
                    "title": context.metadata.title,
                    "channel_title": context.metadata.channel_title,
                    "stats": context.metadata.stats.model_dump(),
                    "highlight_moments": [moment.model_dump() for moment in analysis.highlight_moments],
                    "creator_takeaways": analysis.creator_takeaways,
                }
            )

        prompt = dedent(
            """
            You are a YouTube strategy analyst.

            The user gave a target content description. We already analyzed up to five public YouTube videos and predicted
            their strongest moments from transcript and comments.

            Your job:
            - find recurring creative patterns across the videos
            - produce practical suggestions the user can apply to make a stronger video
            - keep all reasoning grounded in the provided analysis rather than generic YouTube advice
            - adapt every tip to the user's target video idea, not to the exact subject matter of one matched video
            - extract transferable patterns, not topic-specific details
            - prefer patterns that appear across multiple matched videos, not just one isolated example
            - do not let one single video dominate all grounded suggestions unless the evidence is truly overwhelming
            - do not mention that retention data is unavailable
            - do not invent details beyond the provided evidence
            - do not tell the user to mention specific characters, plot points, locations, products, or named entities unless they are already part of the user's target idea
            - if a matched video contains a specific example, generalize it into a reusable storytelling principle
            - return grounded_suggestions as tip/pattern/source_title entries
            - each grounded suggestion must include the title of one matched video that most directly supports the tip
            - source_title must be the exact video title string, not a video id, not a channel name, and not null
            - never output video ids in source_title or supporting_pattern
            - supporting_pattern should clearly explain why that specific video supports the tip, while keeping the tip itself broadly useful across the matched set
            - each supporting_pattern must be a complete sentence, not a fragment
            - keep tips short, actionable, and specific
            - produce at most 5 grounded_suggestions
            """
        ).strip()

        return self._parse_response(
            system_prompt=prompt,
            user_payload={
                "user_description": user_description,
                "video_analyses": compact_videos,
            },
            output_model=CrossVideoSummary,
        )

    def _extract_window_candidates(
        self,
        user_description: str,
        context: VideoContext,
        window: list,
    ) -> WindowCandidateMoments:
        prompt = dedent(
            f"""
            You are analyzing one slice of a YouTube video's transcript.

            Find up to {self.settings.candidate_moments_per_window} moments that are most likely to be exciting,
            memorable, heavily rewatched, or strategically useful for the user's target video idea.

            Rules:
            - Use only the transcript chunk text, comments, and public metadata that were provided.
            - Prefer moments with novelty, payoff, tension, surprise, clarity, strong claims, concrete examples,
              emotional peaks, useful phrasing, or obvious audience payoff.
            - If comments mention timestamps or quote moments, use that as evidence.
            - Do not use hard-coded social media rules; infer from the content itself.
            - Return an empty list if the window contains nothing notable.
            """
        ).strip()

        return self._parse_response(
            system_prompt=prompt,
            user_payload={
                "user_description": user_description,
                "video": {
                    "video_id": context.metadata.video_id,
                    "title": context.metadata.title,
                    "channel_title": context.metadata.channel_title,
                    "stats": context.metadata.stats.model_dump(),
                },
                "top_comments": [comment.model_dump() for comment in context.comments[:20]],
                "transcript_window": [chunk.model_dump() for chunk in window],
            },
            output_model=WindowCandidateMoments,
        )

    def _finalize_video_analysis(
        self,
        user_description: str,
        context: VideoContext,
        candidates: list[HighlightMoment],
        transcript_outline: list,
    ) -> VideoFinalAnalysis:
        prompt = dedent(
            f"""
            You are selecting the strongest moments from a public YouTube video.

            Choose up to {self.settings.top_moments_per_video} strong moments overall and give actionable takeaways
            for the user's target video description, based on transcript, comments, public stats, and the already
            extracted candidate moments.

            Rules:
            - Anchor every claim to the evidence provided.
            - Keep takeaways specific to this video's content and wording.
            - Prefer concrete language over vague marketing advice.
            """
        ).strip()

        candidate_payload = [candidate.model_dump() for candidate in candidates]
        result = self._parse_response(
            system_prompt=prompt,
            user_payload={
                "user_description": user_description,
                "video": {
                    "video_id": context.metadata.video_id,
                    "title": context.metadata.title,
                    "channel_title": context.metadata.channel_title,
                    "description": context.metadata.description,
                    "stats": context.metadata.stats.model_dump(),
                },
                "top_comments": [comment.model_dump() for comment in context.comments[:30]],
                "candidate_moments": candidate_payload,
                "transcript_outline": [chunk.model_dump() for chunk in transcript_outline[:80]],
            },
            output_model=VideoFinalAnalysis,
        )
        result.analyzed_chunk_count = len(transcript_outline)
        return result

    def _parse_response(self, system_prompt: str, user_payload: dict, output_model):
        request_kwargs = {
            "model": self.settings.openai_model,
            "input": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(user_payload, ensure_ascii=False),
                        }
                    ],
                },
            ],
            "text_format": output_model,
        }
        if _supports_reasoning_effort(self.settings.openai_model):
            request_kwargs["reasoning"] = {"effort": self.settings.openai_reasoning_effort}

        response = self.client.responses.parse(**request_kwargs)
        return response.output_parsed

    def _select_fast_chunks(self, user_description: str, context: VideoContext) -> list:
        chunks = context.transcript_chunks
        if not chunks:
            return []

        intro_chunks = [
            chunk for chunk in chunks if chunk.start < float(self.settings.fast_intro_seconds)
        ]
        numeric_chunks = [
            chunk for chunk in chunks if re.search(r"\d", chunk.text)
        ][: self.settings.fast_numeric_chunks]
        relevant_chunks = self._rank_chunks_by_relevance(
            user_description=user_description,
            chunks=chunks,
            limit=self.settings.fast_relevant_chunks,
        )

        selected_by_key = {}
        for chunk in intro_chunks + relevant_chunks + numeric_chunks:
            selected_by_key[(chunk.start, chunk.end)] = chunk

        selected = list(selected_by_key.values())
        selected.sort(key=lambda chunk: chunk.start)
        return selected

    def _rank_chunks_by_relevance(self, user_description: str, chunks: list, limit: int) -> list:
        if limit <= 0 or not chunks:
            return []

        embeddings = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=[user_description] + [chunk.text for chunk in chunks],
        )
        vectors = [item.embedding for item in embeddings.data]
        query_vector = vectors[0]
        chunk_vectors = vectors[1:]

        scored = []
        for chunk, vector in zip(chunks, chunk_vectors):
            scored.append((_cosine_similarity(query_vector, vector), chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:limit]]


def _dedupe_moments(moments: list[HighlightMoment]) -> list[HighlightMoment]:
    deduped: list[HighlightMoment] = []
    seen_keys: set[tuple[int, int, str]] = set()
    for moment in moments:
        key = (round(moment.start), round(moment.end), moment.headline.strip().lower())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(moment)
    deduped.sort(key=lambda item: (-item.confidence, item.start))
    return deduped


def _cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_list = list(left)
    right_list = list(right)
    numerator = sum(a * b for a, b in zip(left_list, right_list))
    left_norm = math.sqrt(sum(a * a for a in left_list))
    right_norm = math.sqrt(sum(b * b for b in right_list))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _supports_reasoning_effort(model_name: str) -> bool:
    normalized = (model_name or "").strip().lower()
    return normalized.startswith("gpt-5")
