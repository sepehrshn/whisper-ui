from __future__ import annotations

import logging
import tempfile
import wave
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from providers.avalai import translate_text as translate_text_avalai
from providers.openai_api import LANGUAGE_NAMES, translate_text as translate_text_openai
from providers.registry import get_provider, provider_ready
from settings import get_api_key
from transcribe_local import ALLOWED_MODELS, run_transcribe, unload_model

logger = logging.getLogger("whisper-ui.stream")

SAMPLE_RATE = 16000
WINDOW_SECONDS = 5.0
OVERLAP_SECONDS = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
HOP_SAMPLES = int(SAMPLE_RATE * (WINDOW_SECONDS - OVERLAP_SECONDS))
MIN_FINAL_SAMPLES = int(SAMPLE_RATE * 0.75)
MAX_BUFFER_SAMPLES = SAMPLE_RATE * 60
SILENCE_RMS = 0.01
ALLOWED_DEVICES = {"auto", "cuda", "gpu", "cpu", "ram", "memory"}


def _words(text: str) -> list[str]:
    return text.split()


def _normalize(words: list[str]) -> list[str]:
    return [word.lower().strip(".,!?;:\"'`()[]{}") for word in words]


def _overlap_count(previous: str, current: str) -> int:
    previous_words = _words(previous)
    current_words = _words(current)
    if not previous_words or not current_words:
        return 0
    previous_norm = _normalize(previous_words)
    current_norm = _normalize(current_words)
    limit = min(len(previous_words), len(current_words))
    for count in range(limit, 0, -1):
        if previous_norm[-count:] == current_norm[:count]:
            return count
    return 0


def committed_delta(previous: str, current: str) -> str:
    """Return the part of `previous` that is not repeated at the start of `current`."""
    previous_words = _words(previous)
    if not previous_words:
        return ""
    overlap = _overlap_count(previous, current)
    if overlap:
        previous_words = previous_words[:-overlap]
    return " ".join(previous_words).strip()


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def _join(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if left and right:
        return f"{left} {right}"
    return left or right


def _write_wav(audio: np.ndarray) -> Path:
    pcm = np.clip(np.asarray(audio, dtype=np.float32) * 32768.0, -32768, 32767).astype("<i2")
    handle = tempfile.NamedTemporaryFile(prefix="whisper-live-", suffix=".wav", delete=False)
    path = Path(handle.name)
    handle.close()
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())
    return path


class StreamSession:
    """Chunk microphone PCM through the selected engine, then translate committed text."""

    def __init__(
        self,
        model: str,
        language: str,
        target_language: str,
        device: str,
        provider: str = "local",
    ):
        self.model = model.strip()
        self.language = language.strip()
        self.device = (device or "auto").strip().lower()
        self.provider_id = (provider or "local").strip().lower()
        self.target = target_language.strip().lower()
        if self.target in {"none", "off", "transcript"}:
            self.target = ""

        try:
            self._engine = get_provider(self.provider_id)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        self.engine_name = self._engine.name

        if self.provider_id == "local":
            if self.model not in ALLOWED_MODELS:
                raise ValueError(f"Unknown model: {self.model}")
            if self.device not in ALLOWED_DEVICES:
                raise ValueError(f"Unknown device: {self.device}")
        else:
            if self._engine.requires_api_key and not provider_ready(self.provider_id):
                raise ValueError(f"{self.engine_name} API key is not configured. Add it in Settings.")
            allowed = {item["id"] for item in self._engine.models}
            if self.model not in allowed:
                raise ValueError(f"Unknown {self.engine_name} model: {self.model}")
            unload_model()

        if self.target and self.target not in LANGUAGE_NAMES:
            raise ValueError(f"Unknown target language: {self.target}")

        self.buffer = np.zeros(0, dtype=np.float32)
        self._buffer_lock = Lock()
        self._received_samples = 0
        self._pcm_remainder = b""
        self.pending_text = ""
        self.pending_english = ""
        self._translation_failed = False
        self._fell_behind = False

        self._cloud_translate = None
        if self.target:
            if self.provider_id == "avalai" and get_api_key("avalai"):
                self._cloud_translate = translate_text_avalai
            elif self.provider_id == "openai" and get_api_key("openai"):
                self._cloud_translate = translate_text_openai
            elif get_api_key("openai"):
                self._cloud_translate = translate_text_openai
            elif get_api_key("avalai"):
                self._cloud_translate = translate_text_avalai
        self._use_whisper_translate = (
            self.provider_id == "local" and self.target == "en" and not self._cloud_translate
        )
        self._translation_blocked = bool(self.target) and not self._cloud_translate and not self._use_whisper_translate
        self._blocked_message = ""
        if self._use_whisper_translate and self.model.endswith(".en"):
            self._use_whisper_translate = False
            self._translation_blocked = True
            self._blocked_message = (
                "This English-only model cannot translate. "
                "Choose a multilingual model, or add an OpenAI or AvalAI key in Settings."
            )
        elif self._translation_blocked:
            name = LANGUAGE_NAMES.get(self.target, self.target)
            self._blocked_message = f"Add an OpenAI or AvalAI key in Settings to translate into {name}."

    def start_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = [{"type": "ready"}]
        if self._blocked_message:
            events.append({"type": "error", "scope": "translation", "text": self._blocked_message})
        return events

    def push_pcm(self, data: bytes) -> float:
        """Store microphone audio and return how many seconds have arrived."""
        if self._pcm_remainder:
            data = self._pcm_remainder + data
            self._pcm_remainder = b""
        if len(data) % 2:
            self._pcm_remainder = data[-1:]
            data = data[:-1]
        if not data:
            return self.heard_seconds()

        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        with self._buffer_lock:
            self.buffer = samples if self.buffer.size == 0 else np.concatenate([self.buffer, samples])
            self._received_samples += int(samples.size)
            if self.buffer.size > MAX_BUFFER_SAMPLES:
                self.buffer = self.buffer[-WINDOW_SAMPLES:]
                self._fell_behind = True
        return self.heard_seconds()

    def heard_seconds(self) -> float:
        return self._received_samples / SAMPLE_RATE

    def ready(self) -> bool:
        with self._buffer_lock:
            return self.buffer.size >= WINDOW_SAMPLES

    def fell_behind(self) -> bool:
        with self._buffer_lock:
            behind = self._fell_behind
            self._fell_behind = False
            return behind

    def finish(self) -> list[dict[str, Any]]:
        return self._consume(final=True)

    def _consume(self, *, final: bool) -> list[dict[str, Any]]:
        with self._buffer_lock:
            if not final and self.buffer.size < WINDOW_SAMPLES:
                return []
            if final and self.buffer.size < MIN_FINAL_SAMPLES:
                window = None
            elif final:
                window = self.buffer.copy()
                self.buffer = np.zeros(0, dtype=np.float32)
            else:
                window = self.buffer[:WINDOW_SAMPLES].copy()
                self.buffer = self.buffer[HOP_SAMPLES:]
        if window is None:
            return self._commit_pending_all()

        if _rms(window) < SILENCE_RMS:
            return self._commit_pending_all()

        text, english = self._transcribe_window(window)
        if not text:
            return self._commit_pending_all()

        if final:
            return self._commit_final(text, english)

        events: list[dict[str, Any]] = []
        if self.pending_text or self.pending_english:
            delta = committed_delta(self.pending_text, text)
            delta_english = committed_delta(self.pending_english, english)
            if delta:
                events.append({"type": "transcript", "text": delta})
                events.extend(self._translation_events(delta, delta_english))
        events.append({"type": "partial", "text": text})
        self.pending_text = text
        self.pending_english = english
        return events

    def _commit_final(self, text: str, english: str) -> list[dict[str, Any]]:
        spoken = text
        spoken_english = english
        if self.pending_text or self.pending_english:
            spoken = _join(committed_delta(self.pending_text, text), text)
            spoken_english = _join(committed_delta(self.pending_english, english), english)
        events: list[dict[str, Any]] = []
        if spoken:
            events.append({"type": "transcript", "text": spoken})
            events.extend(self._translation_events(spoken, spoken_english))
        events.append({"type": "partial", "text": ""})
        self.pending_text = ""
        self.pending_english = ""
        return events

    def _commit_pending_all(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self.pending_text:
            events.append({"type": "transcript", "text": self.pending_text})
        events.extend(self._translation_events(self.pending_text, self.pending_english))
        if self.pending_text or self.pending_english:
            events.append({"type": "partial", "text": ""})
        self.pending_text = ""
        self.pending_english = ""
        return events

    def _transcribe_window(self, window: np.ndarray) -> tuple[str, str]:
        if self.provider_id == "local":
            result = run_transcribe(window, self.model, "transcribe", self.language, self.device)
            text = str(result.get("text", "")).strip()
            english = ""
            if self._use_whisper_translate:
                translated = run_transcribe(window, self.model, "translate", self.language, self.device)
                english = str(translated.get("text", "")).strip()
            return text, english

        path = _write_wav(window)
        try:
            result = self._engine.transcribe(
                path,
                model=self.model,
                task="transcribe",
                language=self.language,
                device="cloud",
            )
        finally:
            path.unlink(missing_ok=True)
        return str(result.get("text", "")).strip(), ""

    def _translation_events(self, source_text: str, english_text: str) -> list[dict[str, Any]]:
        if not self.target or self._translation_blocked or self._translation_failed:
            return []
        if self._cloud_translate:
            source_text = source_text.strip()
            if not source_text:
                return []
            try:
                translated = self._cloud_translate(source_text, self.target)
            except Exception as exc:
                logger.warning("Live translation failed: %s", exc)
                self._translation_failed = True
                return [{"type": "error", "scope": "translation", "text": str(exc)}]
            if not translated:
                return []
            return [{"type": "translation", "text": translated}]
        if self._use_whisper_translate:
            english_text = english_text.strip()
            if not english_text:
                return []
            return [{"type": "translation", "text": english_text}]
        return []
