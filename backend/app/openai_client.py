from __future__ import annotations

from openai import OpenAI

from .config import OPENAI_CONFIG


def get_openai_client() -> OpenAI:
    if not OPENAI_CONFIG.api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your API key before calling OpenAI."
        )

    kwargs = {}
    kwargs["api_key"] = OPENAI_CONFIG.api_key
    if OPENAI_CONFIG.base_url:
        kwargs["base_url"] = OPENAI_CONFIG.base_url
    return OpenAI(**kwargs)
