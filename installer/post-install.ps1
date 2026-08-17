# Whisper UI — post-install script (runs after files are copied)
$ErrorActionPreference = "Stop"
$InstallDir = $PSScriptRoot
$server = Join-Path $InstallDir "server"
$python = Join-Path $server ".venv\Scripts\python.exe"

Write-Host ""
Write-Host "  Whisper UI — First-time setup" -ForegroundColor Cyan
Write-Host "  Installing Python environment (may take several minutes)..." -ForegroundColor Gray
Write-Host ""

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "  ERROR: Python 3.9–3.11 is required." -ForegroundColor Red
    Write-Host "  Download: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path $python)) {
    python -m venv (Join-Path $server ".venv")
}

& $python -m pip install --upgrade pip --quiet
& $python -m pip install -r (Join-Path $server "requirements.txt") --quiet

$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    & $python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --quiet
} else {
    & $python -m pip install torch --quiet
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "  Installing ffmpeg..." -ForegroundColor Gray
        winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements 2>$null
    }
}

Write-Host ""
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
