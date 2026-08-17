$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Running setup..."
& (Join-Path $root "install-whisper-ui.ps1")

$server = Join-Path $root "server"
$python = Join-Path $server ".venv\Scripts\python.exe"
& $python -m pip install pyinstaller

Push-Location $root
& $python -m PyInstaller WhisperUI.spec --noconfirm
Pop-Location

Write-Host ""
Write-Host "Build finished: dist\WhisperUI.exe" -ForegroundColor Green
Write-Host "Note: the exe is large because it bundles PyTorch and Whisper."
