"""Helpers for the Streamlit/OpenRouter chatbot demo."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
DEFAULT_SYSTEM_PROMPT = (
    "You are DeepRetro Chat, a concise assistant for retrosynthesis and "
    "chemistry software questions. Be direct, careful, and practical."
)
SUPPORTED_CHAT_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class ChatbotSettings:
    """Runtime settings for the Streamlit chatbot."""

    api_key: str | None
    model: str
    base_url: str
    app_name: str
    site_url: str | None


def _read_setting(
    name: str,
    environ: Mapping[str, str],
    secrets: Mapping[str, Any] | None,
) -> str | None:
    """Read a setting from environment variables, then Streamlit secrets."""
    env_value = environ.get(name)
    if env_value and env_value.strip():
        return env_value.strip()
    if secrets is None:
        return None
    try:
        secret_value = secrets.get(name)
    except (AttributeError, FileNotFoundError):
        secret_value = None
    if secret_value is None:
        return None
    normalized_value = str(secret_value).strip()
    return normalized_value or None


def resolve_chatbot_settings(
    environ: Mapping[str, str] | None = None,
    secrets: Mapping[str, Any] | None = None,
) -> ChatbotSettings:
    """Resolve chatbot settings without hard-coding secrets."""
    source_environ = os.environ if environ is None else environ
    return ChatbotSettings(
        api_key=_read_setting("OPENROUTER_API_KEY", source_environ, secrets),
        model=(
            _read_setting("OPENROUTER_MODEL", source_environ, secrets)
            or DEFAULT_OPENROUTER_MODEL
        ),
        base_url=(
            _read_setting("OPENROUTER_BASE_URL", source_environ, secrets)
            or DEFAULT_OPENROUTER_BASE_URL
        ),
        app_name=(
            _read_setting("OPENROUTER_APP_NAME", source_environ, secrets)
            or "DeepRetro Streamlit Chatbot"
        ),
        site_url=_read_setting("OPENROUTER_SITE_URL", source_environ, secrets),
    )


def build_openrouter_messages(
    chat_history: list[dict[str, str]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build OpenAI-compatible chat messages for OpenRouter."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(
        {"role": message["role"], "content": message["content"]}
        for message in chat_history
        if message.get("role") in SUPPORTED_CHAT_ROLES
        and isinstance(message.get("content"), str)
    )
    return messages


def build_openrouter_headers(settings: ChatbotSettings) -> dict[str, str]:
    """Build optional OpenRouter attribution headers."""
    headers = {"X-Title": settings.app_name}
    if settings.site_url:
        headers["HTTP-Referer"] = settings.site_url
    return headers
