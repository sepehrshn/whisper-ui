from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dependencies import dependency_report, ensure_venv, install_dependencies  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-ui.launcher")

HOST = os.getenv("WHISPER_UI_HOST", "127.0.0.1")
PORT = int(os.getenv("WHISPER_UI_PORT", "8000"))
AUTO_INSTALL = os.getenv("WHISPER_UI_AUTO_INSTALL", "1") != "0"
APP_MODE = os.getenv("WHISPER_UI_APP_MODE", "1") != "0"


def maybe_install_dependencies() -> None:
    report = dependency_report()
    if report["ready"]:
        logger.info("All dependencies are ready")
        return
    if not AUTO_INSTALL:
        logger.warning("Missing dependencies: %s", ", ".join(report["issues"]))
        return

    logger.info("Missing dependencies detected: %s", ", ".join(report["issues"]))
    if not report["packages"]["venv"]:
        ok, message = ensure_venv()
        if not ok:
            logger.error("Could not create virtual environment: %s", message)
            return
        logger.info("Created virtual environment at %s", message)

    install_python = "python-packages" in report["issues"]
    install_ffmpeg_flag = "ffmpeg" in report["issues"]
    if install_python or install_ffmpeg_flag:
        result = install_dependencies(python=install_python, ffmpeg=install_ffmpeg_flag)
        logger.info("Dependency install finished: %s", result)


def _find_browser() -> list[list[str]]:
    url = f"http://{HOST}:{PORT}"
    candidates: list[list[str]] = []

    if APP_MODE:
        edge_paths = [
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft/Edge/Application/msedge.exe",
            Path(os.environ.get("ProgramFiles", "")) / "Microsoft/Edge/Application/msedge.exe",
            Path(os.environ.get("LocalAppData", "")) / "Microsoft/Edge/Application/msedge.exe",
        ]
        chrome_paths = [
            Path(os.environ.get("ProgramFiles", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("LocalAppData", "")) / "Google/Chrome/Application/chrome.exe",
        ]
        for path in edge_paths + chrome_paths:
            if path.is_file():
                candidates.append([str(path), f"--app={url}", "--window-size=1280,800"])
                break

    return candidates


def open_app_window() -> None:
    time.sleep(1.5)
    url = f"http://{HOST}:{PORT}"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    for cmd in _find_browser():
        try:
            logger.info("Opening app window: %s", cmd[0])
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
            return
        except OSError as exc:
            logger.warning("Could not launch %s: %s", cmd[0], exc)

    import webbrowser

    logger.info("Opening in default browser: %s", url)
    webbrowser.open(url)


def main() -> None:
    maybe_install_dependencies()
    thread = threading.Thread(target=open_app_window, daemon=True)
    thread.start()
    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
