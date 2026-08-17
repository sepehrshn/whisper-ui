from __future__ import annotations

import os
import re
import subprocess
import uuid
from pathlib import Path

import numpy as np

ALLOWED_EXTS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".ogg",
    ".webm",
    ".aac",
    ".mp4",
    ".wma",
    ".opus",
    ".mkv",
    ".mpeg",
    ".mpga",
    ".oga",
}


def safe_suffix(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    ext = re.sub(r"[^a-z0-9.]", "", ext)
    return ext if ext in ALLOWED_EXTS else ".wav"


def save_upload(data: bytes, filename: str, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid.uuid4().hex}{safe_suffix(filename)}"
    path.write_bytes(data)
    return path


def decode_audio(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        "-i",
        str(path.resolve()),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "pipe:1",
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            creationflags=flags,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or "ffmpeg could not read this audio file") from exc

    if not completed.stdout:
        raise RuntimeError("ffmpeg returned empty audio")
    return np.frombuffer(completed.stdout, np.int16).astype(np.float32) / 32768.0
