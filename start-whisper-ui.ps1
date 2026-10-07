$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$server = Join-Path $root "server"
$venvPython = Join-Path $server ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "First launch - setting up Whisper UI..." -ForegroundColor Cyan
    & (Join-Path $root "install-whisper-ui.ps1")
}

$dist = Join-Path $root "web\dist\index.html"
if (-not (Test-Path $dist)) {
    Write-Host "Building application UI..." -ForegroundColor Cyan
    Push-Location (Join-Path $root "web")
    npm install --silent
    npm run build
    Pop-Location
}

Write-Host "Launching Whisper UI..." -ForegroundColor Green
Push-Location $server
$env:WHISPER_UI_APP_MODE = "1"
& $venvPython launcher.py
Pop-Location
