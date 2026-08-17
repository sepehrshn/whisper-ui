from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from settings import get_api_key


class DeepgramProvider:
    id = "deepgram"
    name = "Deepgram"
    requires_api_key = True
    models = [
        {"id": "nova-2", "label": "Nova-2"},
        {"id": "nova-2-general", "label": "Nova-2 General"},
        {"id": "nova-3", "label": "Nova-3"},
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
            raise RuntimeError("Deepgram API key is not configured. Add it in Settings.")

        params = ["model=" + model, "punctuate=true", "utterances=true"]
        if language:
            params.append("language=" + language)
        if task == "translate":
            params.append("translate=true")

        url = "https://api.deepgram.com/v1/listen?" + "&".join(params)
        request = urllib.request.Request(
            url,
            data=path.read_bytes(),
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/octet-stream",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Deepgram API error: {body or exc.reason}") from exc

        channel = payload.get("results", {}).get("channels", [{}])[0]
        alternative = channel.get("alternatives", [{}])[0]
        text = str(alternative.get("transcript", "")).strip()
        segments: list[dict[str, Any]] = []
        for index, utterance in enumerate(payload.get("results", {}).get("utterances", [])):
            segments.append(
                {
                    "id": index,
                    "start": round(float(utterance.get("start", 0)), 2),
                    "end": round(float(utterance.get("end", 0)), 2),
                    "text": str(utterance.get("transcript", "")).strip(),
                }
            )
        return {
            "text": text,
            "language": payload.get("metadata", {}).get("language"),
            "model": model,
            "task": task,
            "provider": self.id,
            "device": "cloud",
            "segments": segments,
        }
