from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from settings import get_api_key


class AssemblyAIProvider:
    id = "assemblyai"
    name = "AssemblyAI"
    requires_api_key = True
    models = [
        {"id": "best", "label": "Best quality"},
        {"id": "nano", "label": "Nano (fast)"},
    ]

    def is_configured(self) -> bool:
        return bool(get_api_key(self.id))

    def _request(self, method: str, url: str, *, data: dict[str, Any] | None = None) -> dict[str, Any]:
        api_key = get_api_key(self.id)
        if not api_key:
            raise RuntimeError("AssemblyAI API key is not configured. Add it in Settings.")
        body = None if data is None else json.dumps(data).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": api_key,
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"AssemblyAI API error: {detail or exc.reason}") from exc

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
        upload_request = urllib.request.Request(
            "https://api.assemblyai.com/v2/upload",
            data=path.read_bytes(),
            headers={
                "Authorization": api_key or "",
                "Content-Type": "application/octet-stream",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(upload_request, timeout=600) as response:
                upload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"AssemblyAI upload error: {detail or exc.reason}") from exc

        upload_url = upload.get("upload_url")
        if not upload_url:
            raise RuntimeError("AssemblyAI upload failed")

        payload: dict[str, Any] = {"audio_url": upload_url}
        if language:
            payload["language_code"] = language
        if model == "nano":
            payload["speech_model"] = "nano"
        else:
            payload["speech_model"] = "best"

        job = self._request("POST", "https://api.assemblyai.com/v2/transcript", data=payload)
        transcript_id = job.get("id")
        if not transcript_id:
            raise RuntimeError("AssemblyAI did not return a transcript id")

        for _ in range(120):
            status = self._request("GET", f"https://api.assemblyai.com/v2/transcript/{transcript_id}")
            state = status.get("status")
            if state == "completed":
                segments = [
                    {
                        "id": index,
                        "start": round(float(item.get("start", 0)) / 1000, 2),
                        "end": round(float(item.get("end", 0)) / 1000, 2),
                        "text": str(item.get("text", "")).strip(),
                    }
                    for index, item in enumerate(status.get("utterances") or [])
                ]
                text = str(status.get("text", "")).strip()
                if task == "translate" and text:
                    translated = self._request(
                        "POST",
                        "https://api.assemblyai.com/v2/transcript",
                        data={"audio_url": upload_url, "speech_model": payload["speech_model"], "language_detection": True},
                    )
                    # AssemblyAI translate is a separate feature; keep original text if unavailable.
                    text = str(translated.get("text", text)).strip()
                return {
                    "text": text,
                    "language": status.get("language_code"),
                    "model": model,
                    "task": task,
                    "provider": self.id,
                    "device": "cloud",
                    "segments": segments,
                }
            if state == "error":
                raise RuntimeError(status.get("error") or "AssemblyAI transcription failed")
            time.sleep(2)

        raise RuntimeError("AssemblyAI transcription timed out")
