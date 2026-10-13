# Whisper UI

Desktop speech-to-text app for Windows — **local Whisper** (GPU/RAM) or **cloud APIs** (OpenAI, AvalAI, Deepgram, AssemblyAI).

## Install on Windows

### Option A — Setup installer (recommended)

```powershell
.\build-installer.ps1
```

Then distribute or run:
- **`release\WhisperUI-Setup.exe`** — full Windows installer (requires [Inno Setup 6](https://jrsoftware.org/isdl.php))
- **`release\Setup-WhisperUI.ps1`** — GUI installer (no extra tools needed)

The installer:
- Copies the app to `%LOCALAPPDATA%\Programs\WhisperUI`
- Creates **Start Menu** and **Desktop** shortcuts
- Runs first-time Python/ffmpeg setup automatically

### Option B — Run from source

```powershell
.\install-whisper-ui.ps1    # first time
.\start-whisper-ui.bat      # launch app
```

Opens in a dedicated **app window** (Edge/Chrome app mode) at `http://127.0.0.1:8000`.

## App features

- **Desktop-style UI** — sidebar navigation (Transcribe, Settings, Setup)
- **Local Whisper** — GPU or RAM, all model sizes
- **Cloud APIs** — add your API keys in Settings (encrypted with Windows DPAPI)
- **Auto setup** — dependency checker installs ffmpeg and Python packages

## API keys

Open **Settings → API keys** and paste keys for OpenAI, AvalAI, Deepgram, or AssemblyAI. Keys never leave your PC. AvalAI uses the OpenAI-compatible endpoint `https://api.avalai.ir/v1`.

## Build standalone exe (advanced)

```powershell
.\build.ps1
```

Creates `dist\WhisperUI.exe` (very large — bundles PyTorch).

## Requirements

- Windows 10/11
- Python 3.9–3.11
- Node.js 18+ (build only)
- ffmpeg (auto-installed when possible)
- NVIDIA GPU (optional)

## Environment variables

| Variable | Description |
| --- | --- |
| `WHISPER_UI_APP_MODE` | `0` to open in normal browser tab |
| `WHISPER_UI_PORT` | Server port (default `8000`) |
| `WHISPER_DEVICE` | `auto`, `cuda`, or `cpu` |
