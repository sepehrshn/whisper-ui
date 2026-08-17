from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("whisper-ui.deps")

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
VENV_PYTHON = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")


def python_executable() -> str:
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable


def _run_command(cmd: list[str], *, timeout: int = 900) -> tuple[bool, str]:
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=flags,
        )
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return False, f"Command timed out: {' '.join(cmd)}"
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, output.strip()


def check_ffmpeg() -> dict[str, Any]:
    path = shutil.which("ffmpeg")
    if not path:
        return {"ok": False, "path": None, "message": "ffmpeg is not on PATH"}
    ok, output = _run_command(["ffmpeg", "-version"], timeout=30)
    version = output.splitlines()[0] if output else "unknown"
    return {"ok": ok, "path": path, "version": version}


def check_python() -> dict[str, Any]:
    exe = python_executable()
    ok, output = _run_command([exe, "--version"], timeout=30)
    return {"ok": ok, "executable": exe, "version": output}


def check_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_python_packages() -> dict[str, Any]:
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "whisper": "openai-whisper",
        "torch": "torch",
        "numpy": "numpy",
    }
    missing = [pip_name for module, pip_name in required.items() if not check_package(module)]
    return {
        "ok": not missing,
        "missing": missing,
        "venv": VENV_PYTHON.is_file(),
    }


def check_cuda() -> dict[str, Any]:
    if not check_package("torch"):
        return {"ok": False, "available": False, "gpu": None, "message": "torch is not installed"}
    try:
        import torch

        available = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if available else None
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if available else None
        return {"ok": True, "available": available, "gpu": gpu, "vramGb": vram_gb}
    except Exception as exc:
        return {"ok": False, "available": False, "gpu": None, "message": str(exc)}


def check_disk_space() -> dict[str, Any]:
    cache_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"
    target = cache_dir if cache_dir.exists() else Path.home()
    usage = shutil.disk_usage(target)
    free_gb = round(usage.free / 1024**3, 1)
    return {"ok": free_gb >= 2, "freeGb": free_gb, "cacheDir": str(cache_dir)}


def check_node_build() -> dict[str, Any]:
    dist = ROOT.parent / "web" / "dist"
    return {"ok": dist.is_dir() and (dist / "index.html").is_file(), "path": str(dist)}


def dependency_report() -> dict[str, Any]:
    ffmpeg = check_ffmpeg()
    packages = check_python_packages()
    cuda = check_cuda()
    disk = check_disk_space()
    node_build = check_node_build()
    issues: list[str] = []
    if not ffmpeg["ok"]:
        issues.append("ffmpeg")
    if not packages["ok"]:
        issues.append("python-packages")
    if not disk["ok"]:
        issues.append("disk-space")
    return {
        "ready": not issues,
        "issues": issues,
        "ffmpeg": ffmpeg,
        "python": check_python(),
        "packages": packages,
        "cuda": cuda,
        "disk": disk,
        "frontend": node_build,
    }


def _has_nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi"):
        ok, _ = _run_command(["nvidia-smi"], timeout=20)
        return ok
    return False


def install_python_packages(*, use_cuda: bool | None = None) -> tuple[bool, str]:
    exe = python_executable()
    steps: list[list[str]] = [
        [exe, "-m", "pip", "install", "--upgrade", "pip"],
        [exe, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
    ]
    want_cuda = _has_nvidia_gpu() if use_cuda is None else use_cuda
    if want_cuda:
        steps.append(
            [
                exe,
                "-m",
                "pip",
                "install",
                "torch",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ]
        )
    else:
        steps.append([exe, "-m", "pip", "install", "torch"])

    logs: list[str] = []
    for cmd in steps:
        logger.info("Running %s", " ".join(cmd))
        ok, output = _run_command(cmd, timeout=1800)
        logs.append(output)
        if not ok:
            return False, "\n".join(logs)
    return True, "\n".join(logs)


def install_ffmpeg() -> tuple[bool, str]:
    if shutil.which("ffmpeg"):
        return True, "ffmpeg already installed"

    if shutil.which("winget"):
        ok, output = _run_command(
            ["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--accept-package-agreements", "--accept-source-agreements"],
            timeout=900,
        )
        if ok and shutil.which("ffmpeg"):
            return True, output
        logs = [f"winget: {output}"]

        ok, output = _run_command(
            ["winget", "install", "--id", "FFmpeg.FFmpeg", "-e", "--accept-package-agreements", "--accept-source-agreements"],
            timeout=900,
        )
        if ok and shutil.which("ffmpeg"):
            return True, output
        logs.append(f"winget fallback: {output}")
    else:
        logs = ["winget not found"]

    if shutil.which("choco"):
        ok, output = _run_command(["choco", "install", "ffmpeg", "-y"], timeout=900)
        if ok and shutil.which("ffmpeg"):
            return True, output
        logs.append(f"choco: {output}")
    else:
        logs.append("choco not found")

    return False, "\n".join(logs) + "\nInstall ffmpeg manually: https://ffmpeg.org/download.html"


def install_dependencies(*, python: bool = True, ffmpeg: bool = True) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if python:
        ok, log = install_python_packages()
        results["python"] = {"ok": ok, "log": log}
    if ffmpeg and not check_ffmpeg()["ok"]:
        ok, log = install_ffmpeg()
        results["ffmpeg"] = {"ok": ok, "log": log}
    results["report"] = dependency_report()
    return results


def ensure_venv() -> tuple[bool, str]:
    if VENV_PYTHON.is_file():
        return True, str(VENV_PYTHON)
    if not shutil.which("python"):
        return False, "Python is not installed or not on PATH"
    ok, output = _run_command([sys.executable, "-m", "venv", str(ROOT / ".venv")], timeout=120)
    if not ok or not VENV_PYTHON.is_file():
        return False, output or "Could not create virtual environment"
    return True, str(VENV_PYTHON)
