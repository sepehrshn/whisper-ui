from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import urllib.request
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
import whisper
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-ui")

ALLOWED_MODELS = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "turbo",
}

ALLOWED_TASKS = {"transcribe", "translate"}
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

MODELS = [
    {
        "id": "tiny",
        "label": "Tiny",
        "params": "39 M",
        "vram": "~1 GB",
        "vramGb": 1,
        "speed": "~10x",
        "englishOnly": False,
    },
    {
        "id": "tiny.en",
        "label": "Tiny EN",
        "params": "39 M",
        "vram": "~1 GB",
        "vramGb": 1,
        "speed": "~10x",
        "englishOnly": True,
    },
    {
        "id": "base",
        "label": "Base",
        "params": "74 M",
        "vram": "~1 GB",
        "vramGb": 1,
        "speed": "~7x",
        "englishOnly": False,
    },
    {
        "id": "base.en",
        "label": "Base EN",
        "params": "74 M",
        "vram": "~1 GB",
        "vramGb": 1,
        "speed": "~7x",
        "englishOnly": True,
    },
    {
        "id": "small",
        "label": "Small",
        "params": "244 M",
        "vram": "~2 GB",
        "vramGb": 2,
        "speed": "~4x",
        "englishOnly": False,
        "recommended": True,
    },
    {
        "id": "small.en",
        "label": "Small EN",
        "params": "244 M",
        "vram": "~2 GB",
        "vramGb": 2,
        "speed": "~4x",
        "englishOnly": True,
    },
    {
        "id": "medium",
        "label": "Medium",
        "params": "769 M",
        "vram": "~5 GB RAM",
        "vramGb": 5,
        "speed": "~2x",
        "englishOnly": False,
    },
    {
        "id": "medium.en",
        "label": "Medium EN",
        "params": "769 M",
        "vram": "~5 GB RAM",
        "vramGb": 5,
        "speed": "~2x",
        "englishOnly": True,
    },
    {
        "id": "large",
        "label": "Large",
        "params": "1550 M",
        "vram": "~10 GB RAM",
        "vramGb": 10,
        "speed": "1x",
        "englishOnly": False,
    },
    {
        "id": "turbo",
        "label": "Turbo",
        "params": "809 M",
        "vram": "~6 GB RAM",
        "vramGb": 6,
        "speed": "~8x",
        "englishOnly": False,
    },
]

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
CACHE_DIR = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"


def resolve_device() -> torch.device:
    requested = os.environ.get("WHISPER_DEVICE", "auto").strip().lower()
    cuda_ok = torch.cuda.is_available()
    if requested in {"cuda", "gpu"}:
        if not cuda_ok:
            raise RuntimeError(
                "WHISPER_DEVICE is set to cuda, but PyTorch cannot see an NVIDIA GPU. "
                "Install a CUDA build of torch: pip install torch --index-url https://download.pytorch.org/whl/cu128"
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if cuda_ok else "cpu")


DEVICE = resolve_device()


def use_fp16() -> bool:
    env = os.environ.get("WHISPER_FP16", "").strip().lower()
    if env in {"0", "false", "no"}:
        return False
    if env in {"1", "true", "yes"}:
        return DEVICE.type == "cuda"
    if DEVICE.type != "cuda":
        return False
    name = torch.cuda.get_device_name(0).upper()
    # GTX 16-series supports FP16 poorly and Whisper often emits NaN logits.
    if "GTX 16" in name:
        return False
    return True


USE_FP16 = use_fp16()
if DEVICE.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    logger.info(
        "Using GPU %s (%.1f GB VRAM), precision=%s",
        torch.cuda.get_device_name(0),
        props.total_memory / 1024**3,
        "fp16" if USE_FP16 else "fp32",
    )
else:
    logger.warning(
        "CUDA not available; using CPU. Install CUDA torch to use the GPU: "
        "pip install torch --index-url https://download.pytorch.org/whl/cu128"
    )

app = FastAPI(title="Whisper UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_lock = Lock()
_current_name: str | None = None
_current_device: str | None = None
_current_model: Any = None


def safe_suffix(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    ext = re.sub(r"[^a-z0-9.]", "", ext)
    return ext if ext in ALLOWED_EXTS else ".wav"


def save_upload(data: bytes, filename: str) -> Path:
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{safe_suffix(filename)}"
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as source, dest.open("wb") as output:
        expected = int(source.headers.get("Content-Length") or 0)
        copied = 0
        last_pct = -10
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            copied += len(chunk)
            if expected:
                pct = copied * 100 / expected
                if pct - last_pct >= 10:
                    last_pct = pct
                    logger.info("Downloading %s: %.0f%%", dest.name.replace(".part", ""), pct)
        output.flush()
        os.fsync(output.fileno())
    if expected and dest.stat().st_size != expected:
        raise RuntimeError(
            f"Incomplete download of {dest.name}: got {dest.stat().st_size} bytes, expected {expected}"
        )


def ensure_model_file(name: str) -> Path:
    url = whisper._MODELS[name]
    expected_sha = url.split("/")[-2]
    dest = CACHE_DIR / Path(url).name
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and dest.stat().st_size > 0 and sha256_file(dest) == expected_sha:
        return dest

    if dest.exists():
        logger.warning("Removing corrupt model cache %s (%s bytes)", dest, dest.stat().st_size)
        dest.unlink(missing_ok=True)

    part = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            logger.info("Downloading Whisper model %s (attempt %s/3)", name, attempt)
            part.unlink(missing_ok=True)
            download_file(url, part)
            digest = sha256_file(part)
            if digest != expected_sha:
                raise RuntimeError(f"SHA256 mismatch for {name}: {digest}")
            part.replace(dest)
            return dest
        except Exception as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            dest.unlink(missing_ok=True)
            logger.warning("Download attempt %s for %s failed: %s", attempt, name, exc)

    raise RuntimeError(
        f"Could not download a valid '{name}' model. {last_error}. "
        "Pick tiny or try again on a more stable connection."
    )


def resolve_run_device(requested: str, model_name: str) -> torch.device:
    choice = requested.strip().lower()
    if choice in {"cpu", "ram", "memory"}:
        return torch.device("cpu")
    if choice in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA is unavailable; using RAM")
            return torch.device("cpu")
        return torch.device("cuda")

    if DEVICE.type != "cuda" or not torch.cuda.is_available():
        return torch.device("cpu")
    info = next((item for item in MODELS if item["id"] == model_name), None)
    needed = float(info["vramGb"]) if info else 0
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if needed > gpu_vram:
        logger.info(
            "Model %s needs ~%.0f GB VRAM; GPU has %.1f GB, using RAM",
            model_name,
            needed,
            gpu_vram,
        )
        return torch.device("cpu")
    return torch.device("cuda")


def get_model(name: str, device: torch.device):
    global _current_name, _current_device, _current_model
    key = str(device)
    with _model_lock:
        if _current_name != name or _current_device != key:
            if _current_model is not None:
                del _current_model
                _current_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logger.info("Loading Whisper model %s on %s", name, device)
            ensure_model_file(name)
            _current_model = whisper.load_model(name, device=device, in_memory=True)
            if device.type == "cpu" or not USE_FP16:
                _current_model = _current_model.float()
            _current_name = name
            _current_device = key
        return _current_model


def run_transcribe(
    path: Path,
    model_name: str,
    task: str,
    language: str,
    device_choice: str,
) -> dict[str, Any]:
    audio = decode_audio(path)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    device = resolve_run_device(device_choice, model_name)
    model = get_model(model_name, device)
    use_half = device.type == "cuda" and USE_FP16
    options: dict[str, Any] = {
        "task": task,
        "fp16": use_half,
    }
    if language:
        options["language"] = language
    try:
        result = model.transcribe(audio, **options)
    except ValueError as exc:
        message = str(exc).lower()
        if "nan" not in message and "invalid values" not in message:
            raise
        logger.warning("Decoder produced NaN logits; retrying in FP32")
        model.float()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        options["fp16"] = False
        result = model.transcribe(audio, **options)
    segments = [
        {
            "id": segment.get("id"),
            "start": round(float(segment.get("start", 0)), 2),
            "end": round(float(segment.get("end", 0)), 2),
            "text": str(segment.get("text", "")).strip(),
        }
        for segment in result.get("segments", [])
    ]
    return {
        "text": str(result.get("text", "")).strip(),
        "language": result.get("language"),
        "model": model_name,
        "task": task,
        "device": "ram" if device.type == "cpu" else "gpu",
        "segments": segments,
    }


def gpu_info() -> dict[str, Any]:
    if DEVICE.type != "cuda":
        return {
            "cuda": False,
            "device": str(DEVICE),
            "gpu": None,
            "vramGb": None,
        }
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "vramGb": round(props.total_memory / 1024**3, 1),
        "fp16": USE_FP16,
    }


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "loadedModel": _current_name,
        "loadedDevice": _current_device,
        **gpu_info(),
    }


@app.get("/api/models")
def list_models():
    return {"models": MODELS}


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("base"),
    task: str = Form("transcribe"),
    language: str = Form(""),
    device: str = Form("auto"),
):
    if model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
    if task not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")
    if device.strip().lower() not in {"auto", "cuda", "gpu", "cpu", "ram", "memory"}:
        raise HTTPException(status_code=400, detail=f"Unknown device: {device}")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Audio file is required")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    tmp_path: Path | None = None
    try:
        tmp_path = save_upload(data, file.filename)
        return run_transcribe(tmp_path, model, task, language.strip(), device)
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            status_code=500,
            detail="GPU ran out of memory. Switch to RAM and try again, or use tiny/base/small on the GPU.",
        ) from exc
    except Exception as exc:
        logger.exception("Transcription failed")
        message = str(exc)
        if "SHA256" in message or "checksum" in message.lower():
            message = (
                "The model file was incomplete or corrupted. "
                "It was removed; click Transcribe again to re-download it. "
                "On this GPU prefer tiny, base, or small."
            )
        raise HTTPException(status_code=500, detail=message) from exc
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not delete temp file %s", tmp_path)
