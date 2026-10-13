from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from providers.openai_api import LANGUAGE_NAMES
from settings import get_api_key

BASE_URL = "https://api.avalai.ir/v1"

WHISPER_MODELS = {
    "whisper-1",
    "groq.whisper-large-v3",
    "groq.whisper-large-v3-turbo",
}
GPT_TRANSCRIBE_MODELS = {
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
}
ALLOWED_MODELS = WHISPER_MODELS | GPT_TRANSCRIBE_MODELS


def translate_text(text: str, target_language: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    api_key = get_api_key("avalai")
    if not api_key:
        raise RuntimeError("AvalAI API key is not configured. Add it in Settings.")

    code = target_language.strip().lower()
    target_name = LANGUAGE_NAMES.get(code, code or "English")
    body = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Translate the user's speech transcript into {target_name}. "
                    "Return only the translation, with no quotes or explanation. "
                    f"If the text is already in {target_name}, return it unchanged."
                ),
            },
            {"role": "user", "content": cleaned},
        ],
    }
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail_body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail_body).get("error", {}).get("message", detail_body)
        except json.JSONDecodeError:
            detail = detail_body or exc.reason
        raise RuntimeError(f"AvalAI API error: {detail}") from exc

    message = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(message, list):
        message = "".join(str(part.get("text", "")) for part in message if isinstance(part, dict))
    return str(message).strip()


class AvalAIProvider:
    id = "avalai"
    name = "AvalAI API"
    requires_api_key = True
    models = [
        {"id": "whisper-1", "label": "Whisper v1", "recommended": True},
        {"id": "gpt-4o-mini-transcribe", "label": "GPT-4o Mini Transcribe"},
        {"id": "gpt-4o-transcribe", "label": "GPT-4o Transcribe"},
        {"id": "groq.whisper-large-v3-turbo", "label": "Groq Whisper Large v3 Turbo"},
        {"id": "groq.whisper-large-v3", "label": "Groq Whisper Large v3"},
    ]

    def is_configured(self) -> bool:
        return bool(get_api_key(self.id))

    def transcribe(
        self,
        path: Path,
        *,
        model: str,
        task: str,
        language: str,
        device: str,
    ) -> dict[str, Any]:
        api_key = get_api_key(self.id)
        if not api_key:
            raise RuntimeError("AvalAI API key is not configured. Add it in Settings.")

        if model not in ALLOWED_MODELS:
            raise ValueError(f"Unknown AvalAI model: {model}")

        use_whisper = model in WHISPER_MODELS
        endpoint_model = model
        if task == "translate":
            if not use_whisper:
                endpoint_model = "whisper-1"
                use_whisper = True
            url = f"{BASE_URL}/audio/translations"
        else:
            url = f"{BASE_URL}/audio/transcriptions"

        boundary = "----WhisperUIBoundary"
        parts: list[bytes] = []

        def add_field(name: str, value: str) -> None:
            parts.append(f"--{boundary}\r\n".encode())
            parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            parts.append(f"{value}\r\n".encode())

        add_field("model", endpoint_model)
        if language:
            add_field("language", language)
        if use_whisper:
            add_field("response_format", "verbose_json")
            add_field("timestamp_granularities[]", "segment")
        else:
            add_field("response_format", "json")

        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode()
        )
        parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
        parts.append(path.read_bytes())
        parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())

        request = urllib.request.Request(
            url,
            data=b"".join(parts),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("error", {}).get("message", body)
            except json.JSONDecodeError:
                detail = body or exc.reason
            raise RuntimeError(f"AvalAI API error: {detail}") from exc

        segments = [
            {
                "id": index,
                "start": round(float(segment.get("start", 0)), 2),
                "end": round(float(segment.get("end", 0)), 2),
                "text": str(segment.get("text", "")).strip(),
            }
            for index, segment in enumerate(payload.get("segments", []))
        ]
        return {
            "text": str(payload.get("text", "")).strip(),
            "language": payload.get("language"),
            "model": endpoint_model,
            "task": task,
            "provider": self.id,
            "device": "cloud",
            "segments": segments,
        }
