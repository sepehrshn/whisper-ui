$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$server = Join-Path $root "server"
$web = Join-Path $root "web"
$venv = Join-Path $server ".venv"
$python = Join-Path $venv "Scripts\python.exe"

Write-Host "=== Whisper UI setup ===" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not installed. Install Python 3.9-3.11 and try again."
}

if (-not (Test-Path $python)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venv
}

Write-Host "Installing Python packages..."
& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $server "requirements.txt")

$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    Write-Host "NVIDIA GPU detected. Installing CUDA PyTorch..."
    & $python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "No NVIDIA GPU detected. Installing CPU PyTorch..."
    & $python -m pip install torch
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ffmpeg not found. Attempting install via winget..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        choco install ffmpeg -y
    } else {
        Write-Warning "Install ffmpeg manually: https://ffmpeg.org/download.html"
    }
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js 18+ is required to build the UI. Install Node and run this script again."
}

Write-Host "Building web UI..."
Push-Location $web
npm install
npm run build
Pop-Location

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run: .\start-whisper-ui.ps1"
