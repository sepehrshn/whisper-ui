from __future__ import annotations

from pathlib import Path
from typing import Any

from audio import decode_audio
from transcribe_local import ALLOWED_MODELS, MODELS, run_transcribe


class LocalWhisperProvider:
    id = "local"
    name = "Local Whisper"
    requires_api_key = False
    models = MODELS

    def is_configured(self) -> bool:
        return True

    def transcribe(
        self,
        path: Path,
        *,
        model: str,
        task: str,
        language: str,
        device: str,
    ) -> dict[str, Any]:
        if model not in ALLOWED_MODELS:
            raise ValueError(f"Unknown local model: {model}")
        audio = decode_audio(path)
        return run_transcribe(audio, model, task, language, device)
