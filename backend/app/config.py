from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT_DIR / "archive"
DEFAULT_TOP_K = 5
MAX_TOP_K = 10

load_dotenv(ROOT_DIR / ".env")
ROOT_CONFIG_PATH = ROOT_DIR / "config.json"


def _load_root_config() -> dict:
    if not ROOT_CONFIG_PATH.exists():
        return {}
    payload = json.loads(ROOT_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {ROOT_CONFIG_PATH} must contain a JSON object.")
    return payload


ROOT_CONFIG = _load_root_config()


class OpenAIConfig:
    api_key: str = str(ROOT_CONFIG.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")).strip()
    base_url: Optional[str] = str(
        ROOT_CONFIG.get("openai_base_url")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).strip()

    # Primary model settings for the hackathon MVP.
    suggestion_model: str = str(
        ROOT_CONFIG.get("openai_suggestion_model")
        or os.getenv("OPENAI_SUGGESTION_MODEL", "gpt-4.1-mini")
    ).strip() or "gpt-4.1-mini"
    rewrite_model: str = str(
        ROOT_CONFIG.get("openai_rewrite_model")
        or os.getenv("OPENAI_REWRITE_MODEL", "gpt-4.1-mini")
    ).strip() or "gpt-4.1-mini"
    embedding_model: str = str(
        ROOT_CONFIG.get("openai_embedding_model")
        or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    ).strip() or "text-embedding-3-small"


OPENAI_CONFIG = OpenAIConfig()
