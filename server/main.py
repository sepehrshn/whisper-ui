from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from audio import save_upload
from dependencies import dependency_report, install_dependencies
from providers.registry import get_provider, list_providers, provider_ready
from settings import delete_api_key, public_settings, set_api_key, set_default_provider
from transcribe_local import ALLOWED_MODELS, ALLOWED_TASKS, gpu_info, loaded_model_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-ui")

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

STATIC_DIR = ROOT.parent / "web" / "dist"

app = FastAPI(title="Whisper UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ApiKeyUpdate(BaseModel):
    provider: str
    apiKey: str = ""


class DefaultProviderUpdate(BaseModel):
    provider: str


class InstallRequest(BaseModel):
    python: bool = True
    ffmpeg: bool = True


@app.get("/api/health")
def health():
    return {
        "ok": True,
        **gpu_info(),
        **loaded_model_info(),
    }


@app.get("/api/dependencies")
def dependencies():
    return dependency_report()


@app.post("/api/setup/install")
def setup_install(body: InstallRequest):
    try:
        return install_dependencies(python=body.python, ffmpeg=body.ffmpeg)
    except Exception as exc:
        logger.exception("Dependency install failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/providers")
def providers():
    settings = public_settings()
    return {"providers": list_providers(settings["configuredKeys"])}


@app.get("/api/settings")
def settings():
    return public_settings()


@app.put("/api/settings/keys")
def update_api_key(body: ApiKeyUpdate):
    allowed = {"openai", "deepgram", "assemblyai"}
    if body.provider not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {body.provider}")
    if body.apiKey.strip():
        set_api_key(body.provider, body.apiKey)
    else:
        delete_api_key(body.provider)
    settings = public_settings()
    return {"ok": True, **settings}


@app.put("/api/settings/default-provider")
def update_default_provider(body: DefaultProviderUpdate):
    set_default_provider(body.provider)
    return {"ok": True, "defaultProvider": body.provider}


@app.get("/api/models")
def list_models(provider: str = "local"):
    try:
        return {"models": get_provider(provider).models}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    provider: str = Form("local"),
    model: str = Form("base"),
    task: str = Form("transcribe"),
    language: str = Form(""),
    device: str = Form("auto"),
):
    if task not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

    try:
        engine = get_provider(provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if engine.requires_api_key and not provider_ready(provider):
        raise HTTPException(
            status_code=400,
            detail=f"{engine.name} API key is not configured. Open Settings and add your key.",
        )

    if provider == "local":
        if model not in ALLOWED_MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
        if device.strip().lower() not in {"auto", "cuda", "gpu", "cpu", "ram", "memory"}:
            raise HTTPException(status_code=400, detail=f"Unknown device: {device}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Audio file is required")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    tmp_path: Path | None = None
    try:
        tmp_path = save_upload(data, file.filename, UPLOAD_DIR)
        return engine.transcribe(
            tmp_path,
            model=model,
            task=task,
            language=language.strip(),
            device=device,
        )
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
                "It was removed; click Transcribe again to re-download it."
            )
        raise HTTPException(status_code=500, detail=message) from exc
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not delete temp file %s", tmp_path)


if STATIC_DIR.is_dir():
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    def index_page():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.svg")
    def favicon():
        icon = STATIC_DIR / "favicon.svg"
        if icon.is_file():
            return FileResponse(icon)
        raise HTTPException(status_code=404)
