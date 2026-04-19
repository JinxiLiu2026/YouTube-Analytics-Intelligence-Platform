from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


load_dotenv()
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
ROOT_DIR = PROJECT_DIR.parent
DEFAULT_ROOT_CONFIG = ROOT_DIR / "config.json"
DEFAULT_LOCAL_CONFIG = PROJECT_DIR / "config.json"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: Optional[str] = "https://api.openai.com/v1"
    openai_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_reasoning_effort: str = "low"
    youtube_api_key: Optional[str] = None
    analysis_mode: str = "fast"
    transcript_languages: tuple[str, ...] = ("en", "en-US", "en-GB")
    chunk_seconds: int = 10
    analysis_window_chunks: int = 30
    candidate_moments_per_window: int = 3
    top_moments_per_video: int = 3
    max_comments_default: int = 40
    fast_chunk_seconds: int = 20
    fast_intro_seconds: int = 90
    fast_relevant_chunks: int = 18
    fast_numeric_chunks: int = 8
    transcript_sample_window_seconds: int = 300
    max_parallel_video_analyses: int = 3
    config_path: Optional[str] = None

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Settings":
        resolved_path = _resolve_config_path(config_path)
        file_config = _load_json_config(resolved_path)
        env_config = _load_env_config()
        merged = {**env_config, **file_config}

        openai_api_key = str(merged.get("openai_api_key", "")).strip()
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required. Put it in config.json or the environment.")

        transcript_languages = merged.get("transcript_languages", ("en", "en-US", "en-GB"))
        if isinstance(transcript_languages, str):
            transcript_languages = [item.strip() for item in transcript_languages.split(",") if item.strip()]

        return cls(
            openai_api_key=openai_api_key,
            openai_base_url=_optional_str(merged.get("openai_base_url")) or "https://api.openai.com/v1",
            openai_model=str(
                merged.get("youtube_insights_openai_model")
                or merged.get("openai_model", "gpt-5-mini")
            ).strip() or "gpt-5-mini",
            openai_embedding_model=str(
                merged.get("youtube_insights_openai_embedding_model")
                or merged.get("openai_embedding_model", "text-embedding-3-small")
            ).strip() or "text-embedding-3-small",
            openai_reasoning_effort=str(
                merged.get("youtube_insights_openai_reasoning_effort")
                or merged.get("openai_reasoning_effort", "low")
            ).strip() or "low",
            youtube_api_key=_optional_str(merged.get("youtube_api_key")),
            analysis_mode=str(merged.get("analysis_mode", "fast")).strip() or "fast",
            transcript_languages=tuple(transcript_languages) or ("en", "en-US", "en-GB"),
            chunk_seconds=_int_or_default(merged.get("chunk_seconds"), 10),
            analysis_window_chunks=_int_or_default(merged.get("analysis_window_chunks"), 30),
            candidate_moments_per_window=_int_or_default(merged.get("candidate_moments_per_window"), 3),
            top_moments_per_video=_int_or_default(merged.get("top_moments_per_video"), 3),
            max_comments_default=_int_or_default(merged.get("max_comments_default"), 40),
            fast_chunk_seconds=_int_or_default(merged.get("fast_chunk_seconds"), 20),
            fast_intro_seconds=_int_or_default(merged.get("fast_intro_seconds"), 90),
            fast_relevant_chunks=_int_or_default(merged.get("fast_relevant_chunks"), 18),
            fast_numeric_chunks=_int_or_default(merged.get("fast_numeric_chunks"), 8),
            transcript_sample_window_seconds=_int_or_default(
                merged.get("transcript_sample_window_seconds"),
                300,
            ),
            max_parallel_video_analyses=_int_or_default(merged.get("max_parallel_video_analyses"), 3),
            config_path=str(resolved_path) if resolved_path.exists() else None,
        )

    @classmethod
    def from_env(cls) -> "Settings":
        return cls.load(config_path=None)


def _load_json_config(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} must contain a JSON object.")
    return payload


def _resolve_config_path(config_path: Optional[str]) -> Path:
    if config_path:
        return Path(config_path)
    if os.getenv("YOUTUBE_INSIGHTS_CONFIG"):
        return Path(os.getenv("YOUTUBE_INSIGHTS_CONFIG", ""))
    if DEFAULT_ROOT_CONFIG.exists():
        return DEFAULT_ROOT_CONFIG
    return DEFAULT_LOCAL_CONFIG


def _load_env_config() -> dict:
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        "youtube_insights_openai_model": os.getenv("YOUTUBE_INSIGHTS_OPENAI_MODEL"),
        "youtube_insights_openai_embedding_model": os.getenv("YOUTUBE_INSIGHTS_OPENAI_EMBEDDING_MODEL"),
        "youtube_insights_openai_reasoning_effort": os.getenv("YOUTUBE_INSIGHTS_OPENAI_REASONING_EFFORT"),
        "openai_model": os.getenv("OPENAI_MODEL"),
        "openai_embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL"),
        "openai_reasoning_effort": os.getenv("OPENAI_REASONING_EFFORT"),
        "youtube_api_key": os.getenv("YOUTUBE_API_KEY"),
        "analysis_mode": os.getenv("ANALYSIS_MODE"),
        "transcript_languages": os.getenv("TRANSCRIPT_LANGUAGES"),
        "chunk_seconds": os.getenv("TRANSCRIPT_CHUNK_SECONDS"),
        "analysis_window_chunks": os.getenv("ANALYSIS_WINDOW_CHUNKS"),
        "candidate_moments_per_window": os.getenv("CANDIDATE_MOMENTS_PER_WINDOW"),
        "top_moments_per_video": os.getenv("TOP_MOMENTS_PER_VIDEO"),
        "max_comments_default": os.getenv("MAX_COMMENTS_DEFAULT"),
        "fast_chunk_seconds": os.getenv("FAST_CHUNK_SECONDS"),
        "fast_intro_seconds": os.getenv("FAST_INTRO_SECONDS"),
        "fast_relevant_chunks": os.getenv("FAST_RELEVANT_CHUNKS"),
        "fast_numeric_chunks": os.getenv("FAST_NUMERIC_CHUNKS"),
        "transcript_sample_window_seconds": os.getenv("TRANSCRIPT_SAMPLE_WINDOW_SECONDS"),
        "max_parallel_video_analyses": os.getenv("MAX_PARALLEL_VIDEO_ANALYSES"),
    }


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_default(value: object, default: int) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return int(text)
