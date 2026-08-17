from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(os.getenv("WHISPER_UI_CONFIG", Path.home() / "AppData" / "Roaming" / "WhisperUI"))
CONFIG_FILE = CONFIG_DIR / "settings.json"


def _dpapi_available() -> bool:
    return sys.platform == "win32"


def _encrypt_secret(value: str) -> str:
    if not value:
        return ""
    if _dpapi_available():
        import ctypes
        import ctypes.wintypes

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [
                ("cbData", ctypes.wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_byte)),
            ]

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32

        def blob_from_bytes(data: bytes) -> DATA_BLOB:
            buffer = ctypes.create_string_buffer(data, len(data))
            blob = DATA_BLOB(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
            return blob

        in_blob = blob_from_bytes(value.encode("utf-8"))
        out_blob = DATA_BLOB()
        if not crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            None,
            None,
            None,
            None,
            0,
            ctypes.byref(out_blob),
        ):
            raise OSError("Could not encrypt API key with Windows DPAPI")
        encrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        kernel32.LocalFree(out_blob.pbData)
        return base64.b64encode(encrypted).decode("ascii")

    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _decrypt_secret(value: str) -> str:
    if not value:
        return ""
    raw = base64.b64decode(value.encode("ascii"))
    if _dpapi_available():
        import ctypes
        import ctypes.wintypes

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [
                ("cbData", ctypes.wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_byte)),
            ]

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32

        in_blob = DATA_BLOB(len(raw), ctypes.cast(ctypes.create_string_buffer(raw, len(raw)), ctypes.POINTER(ctypes.c_byte)))
        out_blob = DATA_BLOB()
        if not crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            None,
            None,
            None,
            None,
            0,
            ctypes.byref(out_blob),
        ):
            raise OSError("Could not decrypt stored API key")
        decrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        kernel32.LocalFree(out_blob.pbData)
        return decrypted.decode("utf-8")

    return raw.decode("utf-8")


def _default_settings() -> dict[str, Any]:
    return {
        "defaultProvider": "local",
        "keys": {},
    }


def load_settings() -> dict[str, Any]:
    if not CONFIG_FILE.is_file():
        return _default_settings()
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_settings()
    if not isinstance(data, dict):
        return _default_settings()
    data.setdefault("defaultProvider", "local")
    data.setdefault("keys", {})
    return data


def save_settings(data: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_api_key(provider: str) -> str | None:
    data = load_settings()
    encrypted = data.get("keys", {}).get(provider)
    if not encrypted:
        return None
    try:
        return _decrypt_secret(str(encrypted))
    except OSError:
        return None


def set_api_key(provider: str, api_key: str) -> None:
    data = load_settings()
    keys = data.setdefault("keys", {})
    if api_key.strip():
        keys[provider] = _encrypt_secret(api_key.strip())
    elif provider in keys:
        del keys[provider]
    save_settings(data)


def delete_api_key(provider: str) -> None:
    data = load_settings()
    keys = data.setdefault("keys", {})
    keys.pop(provider, None)
    save_settings(data)


def set_default_provider(provider: str) -> None:
    data = load_settings()
    data["defaultProvider"] = provider
    save_settings(data)


def public_settings() -> dict[str, Any]:
    data = load_settings()
    keys = data.get("keys", {})
    configured = {provider: bool(value) for provider, value in keys.items()}
    return {
        "defaultProvider": data.get("defaultProvider", "local"),
        "configuredKeys": configured,
        "configPath": str(CONFIG_FILE),
        "secureStorage": _dpapi_available(),
    }
