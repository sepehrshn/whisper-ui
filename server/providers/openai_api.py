from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from settings import get_api_key


class OpenAIProvider:
    id = "openai"
    name = "OpenAI API"
    requires_api_key = True
    models = [{"id": "whisper-1", "label": "Whisper v1"}]

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
            raise RuntimeError("OpenAI API key is not configured. Add it in Settings.")

        if model not in {"whisper-1"}:
            raise ValueError(f"Unknown OpenAI model: {model}")

        boundary = "----WhisperUIBoundary"
        parts: list[bytes] = []

        def add_field(name: str, value: str) -> None:
            parts.append(f"--{boundary}\r\n".encode())
            parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            parts.append(f"{value}\r\n".encode())

        add_field("model", model)
        if task == "translate":
            add_field("task", "translate")
        if language:
            add_field("language", language)
        add_field("response_format", "verbose_json")
        add_field("timestamp_granularities[]", "segment")

        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode()
        )
        parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
        parts.append(path.read_bytes())
        parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())

        request = urllib.request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
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
            raise RuntimeError(f"OpenAI API error: {detail}") from exc

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
            "model": model,
            "task": task,
            "provider": self.id,
            "device": "cloud",
            "segments": segments,
        }
