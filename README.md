# Whisper UI

TypeScript web app for [OpenAI Whisper](https://github.com/openai/whisper): upload an audio file, pick a model, and read the transcript.

## What you get

- Simple upload UI (drag and drop or file picker)
- Model picker: `tiny`, `base`, `small`, `medium`, `large`, `turbo`, plus English-only variants
- Task: transcribe or translate to English
- Language: auto-detect or choose one
- Result: full text plus timestamped segments

## Requirements

- Node.js 18+
- Python 3.9–3.11
- [ffmpeg](https://ffmpeg.org/) on your PATH
- NVIDIA GPU + current Game Ready / Studio driver (this app is set up for a **GTX 1630 4 GB**)

Windows (Chocolatey):

```powershell
choco install ffmpeg
```

## Setup

### 1. Python server

```powershell
cd server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
uvicorn main:app --reload --port 8000
```

Or run `.\install-gpu.ps1` from `server` to create the venv and install CUDA PyTorch.

The first transcription downloads the selected Whisper model. That can take a few minutes.

On startup the server log should show `Using GPU NVIDIA GeForce GTX 1630`. The UI should say **GPU ready**. If it says CPU mode, the CPU torch wheel is still installed.

Force GPU or CPU with `WHISPER_DEVICE=cuda` or `WHISPER_DEVICE=cpu`.

### 2. TypeScript UI

In another terminal:

```powershell
cd web
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The Vite app proxies `/api` to the Python server.

## Models

| Size | Parameters | English-only | Multilingual | VRAM | Relative speed |
| --- | --- | --- | --- | --- | --- |
| tiny | 39 M | `tiny.en` | `tiny` | ~1 GB | ~10x |
| base | 74 M | `base.en` | `base` | ~1 GB | ~7x |
| small | 244 M | `small.en` | `small` | ~2 GB | ~4x |
| medium | 769 M | `medium.en` | `medium` | ~5 GB | ~2x |
| large | 1550 M | — | `large` | ~10 GB | 1x |
| turbo | 809 M | — | `turbo` | ~6 GB | ~8x |

Start with `small` on a 4 GB GPU (GTX 1630). Use `tiny` or `base` if you run out of VRAM. Do not use `medium`, `turbo`, or `large` on this card.

`turbo` is not trained for translation. For translate-to-English, use a multilingual model such as `small`.
