from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from youtube_insights.config import Settings
from youtube_insights.llm_client import LLMInsightClient
from youtube_insights.schemas import (
    AnalyzeVideosRequest,
    AnalyzeVideosResponse,
    VideoAnalysis,
    VideoContext,
    VideoMetadata,
)
from youtube_insights.youtube_client import (
    TranscriptProvider,
    YouTubePublicDataClient,
    chunk_transcript,
    sample_transcript_segments,
)


class YouTubeInsightService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.youtube_client = YouTubePublicDataClient(api_key=settings.youtube_api_key)
        self.transcript_provider = TranscriptProvider(settings.transcript_languages)
        self.llm_client = LLMInsightClient(settings)

    def analyze(self, request: AnalyzeVideosRequest) -> AnalyzeVideosResponse:
        mode = request.mode or self.settings.analysis_mode
        chunk_seconds = self._resolve_chunk_seconds(mode)
        metadata_map = self.youtube_client.fetch_video_metadata(request.video_ids)
        contexts: list[VideoContext] = []

        for video_id in request.video_ids:
            metadata = metadata_map.get(video_id) or VideoMetadata(video_id=video_id)

            warnings: list[str] = []
            try:
                transcript_segments = self.transcript_provider.fetch_transcript(video_id)
            except Exception as exc:  # pragma: no cover - depends on live external data
                transcript_segments = []
                warnings.append(f"Transcript retrieval failed: {exc}")

            raw_transcript_segment_count = len(transcript_segments)
            transcript_segments = sample_transcript_segments(
                transcript_segments,
                window_seconds=self.settings.transcript_sample_window_seconds,
            )
            if transcript_segments and len(transcript_segments) < raw_transcript_segment_count:
                warnings.append(
                    "Transcript was sampled to the first 5 minutes, middle 5 minutes, and last 5 minutes for faster analysis."
                )

            transcript_chunks = (
                chunk_transcript(transcript_segments, chunk_seconds)
                if transcript_segments
                else []
            )

            max_comments = (
                request.max_comments
                if request.max_comments is not None
                else self.settings.max_comments_default
            )
            try:
                comments = self.youtube_client.fetch_comments(video_id, max_results=max_comments)
            except Exception as exc:  # pragma: no cover - depends on live external data
                comments = []
                warnings.append(f"Comment retrieval failed: {exc}")

            if not self.settings.youtube_api_key:
                warnings.append("Running in transcript-only mode because YOUTUBE_API_KEY is not set.")

            contexts.append(
                VideoContext(
                    metadata=metadata,
                    transcript_segments=transcript_segments,
                    transcript_chunks=transcript_chunks,
                    comments=comments,
                    warnings=warnings,
                )
            )

        max_workers = min(max(self.settings.max_parallel_video_analyses, 1), max(len(contexts), 1))
        if max_workers == 1 or len(contexts) <= 1:
            video_final_analyses = [
                LLMInsightClient(self.settings).analyze_video(
                    user_description=request.description,
                    context=context,
                    mode=mode,
                )
                for context in contexts
            ]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        LLMInsightClient(self.settings).analyze_video,
                        request.description,
                        context,
                        mode,
                    )
                    for context in contexts
                ]
                video_final_analyses = [future.result() for future in futures]

        videos = [
            VideoAnalysis(
                video_id=context.metadata.video_id,
                title=context.metadata.title,
                channel_title=context.metadata.channel_title,
                stats=context.metadata.stats,
                transcript_found=bool(context.transcript_segments),
                warnings=context.warnings,
                highlight_moments=analysis.highlight_moments,
                creator_takeaways=analysis.creator_takeaways,
                comments_considered=context.comments[:10],
                transcript_chunks_analyzed=analysis.analyzed_chunk_count,
            )
            for context, analysis in zip(contexts, video_final_analyses)
        ]

        cross_video_summary = self.llm_client.summarize_across_videos(
            user_description=request.description,
            analyses=video_final_analyses,
            contexts=contexts,
        )

        return AnalyzeVideosResponse(
            description=request.description,
            videos=videos,
            cross_video_summary=cross_video_summary,
            debug={
                "analysis_mode": mode,
                "chunk_seconds": chunk_seconds,
                "analysis_window_chunks": self.settings.analysis_window_chunks,
                "model": self.settings.openai_model,
                "embedding_model": self.settings.openai_embedding_model,
                "reasoning_effort": self.settings.openai_reasoning_effort,
                "transcript_sample_window_seconds": self.settings.transcript_sample_window_seconds,
                "max_parallel_video_analyses": max_workers,
            },
        )

    def _resolve_chunk_seconds(self, mode: str) -> int:
        if mode == "fast":
            return self.settings.fast_chunk_seconds
        return self.settings.chunk_seconds
