from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from settings import get_api_key


class TranscriptionProvider(Protocol):
    id: str
    name: str
    requires_api_key: bool
    models: list[dict[str, Any]]

    def is_configured(self) -> bool: ...

    def transcribe(
        self,
        path: Path,
        *,
        model: str,
        task: str,
        language: str,
        device: str,
    ) -> dict[str, Any]: ...


PROVIDER_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": "local",
        "name": "Local Whisper",
        "requiresApiKey": False,
        "description": "Run OpenAI Whisper on this device (GPU or RAM).",
        "models": [
            {"id": "tiny", "label": "Tiny"},
            {"id": "base", "label": "Base"},
            {"id": "small", "label": "Small", "recommended": True},
            {"id": "medium", "label": "Medium"},
            {"id": "large", "label": "Large"},
            {"id": "turbo", "label": "Turbo"},
        ],
    },
    {
        "id": "openai",
        "name": "OpenAI API",
        "requiresApiKey": True,
        "description": "Cloud transcription via OpenAI Whisper API.",
        "models": [
            {"id": "whisper-1", "label": "Whisper v1"},
        ],
    },
    {
        "id": "deepgram",
        "name": "Deepgram",
        "requiresApiKey": True,
        "description": "Cloud transcription via Deepgram Nova models.",
        "models": [
            {"id": "nova-2", "label": "Nova-2"},
            {"id": "nova-2-general", "label": "Nova-2 General"},
            {"id": "nova-3", "label": "Nova-3"},
        ],
    },
    {
        "id": "assemblyai",
        "name": "AssemblyAI",
        "requiresApiKey": True,
        "description": "Cloud transcription via AssemblyAI.",
        "models": [
            {"id": "best", "label": "Best quality"},
            {"id": "nano", "label": "Nano (fast)"},
        ],
    },
]


def list_providers(configured_keys: dict[str, bool]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for provider in PROVIDER_DEFINITIONS:
        item = dict(provider)
        item["configured"] = configured_keys.get(provider["id"], False) if provider["requiresApiKey"] else True
        items.append(item)
    return items


def get_provider(provider_id: str):
    if provider_id == "local":
        from providers.local import LocalWhisperProvider

        return LocalWhisperProvider()
    if provider_id == "openai":
        from providers.openai_api import OpenAIProvider

        return OpenAIProvider()
    if provider_id == "deepgram":
        from providers.deepgram import DeepgramProvider

        return DeepgramProvider()
    if provider_id == "assemblyai":
        from providers.assemblyai import AssemblyAIProvider

        return AssemblyAIProvider()
    raise ValueError(f"Unknown provider: {provider_id}")


def provider_ready(provider_id: str) -> bool:
    definition = next((item for item in PROVIDER_DEFINITIONS if item["id"] == provider_id), None)
    if not definition:
        return False
    if not definition["requiresApiKey"]:
        return True
    return bool(get_api_key(provider_id))
