$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$release = Join-Path $root "release\WhisperUI"
$iscc = Get-Command iscc -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "  Whisper UI — Build Installer" -ForegroundColor Cyan
Write-Host "  ============================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Full setup (venv, deps, UI build)
Write-Host "[1/4] Running setup..." -ForegroundColor Yellow
& (Join-Path $root "install-whisper-ui.ps1")

# Step 2: Create release folder
Write-Host "[2/4] Packaging application..." -ForegroundColor Yellow
if (Test-Path $release) { Remove-Item $release -Recurse -Force }
New-Item -ItemType Directory -Path $release -Force | Out-Null

$items = @(
    @{ Src = "server"; Dest = "server" },
    @{ Src = "web\dist"; Dest = "web\dist" },
    @{ Src = "installer"; Dest = "installer" },
    @{ Src = "start-whisper-ui.ps1"; Dest = "start-whisper-ui.ps1" },
    @{ Src = "start-whisper-ui.bat"; Dest = "start-whisper-ui.bat" },
    @{ Src = "install-whisper-ui.ps1"; Dest = "install-whisper-ui.ps1" },
    @{ Src = "README.md"; Dest = "README.md" }
)
foreach ($item in $items) {
    $src = Join-Path $root $item.Src
    $dest = Join-Path $release $item.Dest
    if (Test-Path $src) {
        $destParent = Split-Path $dest -Parent
        if ($destParent -and -not (Test-Path $destParent)) {
            New-Item -ItemType Directory -Path $destParent -Force | Out-Null
        }
        if (Test-Path $src -PathType Container) {
            Copy-Item $src $dest -Recurse -Force
        } else {
            Copy-Item $src $dest -Force
        }
    }
}

# Root launcher for installed app
@'
@echo off
title Whisper UI
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0start-whisper-ui.ps1"
'@ | Set-Content (Join-Path $release "WhisperUI.bat") -Encoding ASCII

# Step 3: Build Setup.exe via Inno Setup (if available)
Write-Host "[3/4] Building Windows installer..." -ForegroundColor Yellow
$setupExe = Join-Path $root "release\WhisperUI-Setup.exe"
if ($iscc) {
    & iscc (Join-Path $root "installer\WhisperUI.iss")
    if (Test-Path $setupExe) {
        Write-Host "  Created: release\WhisperUI-Setup.exe" -ForegroundColor Green
    }
} else {
    Write-Host "  Inno Setup not found — creating PowerShell installer instead." -ForegroundColor Gray
    Copy-Item (Join-Path $root "installer\Setup-WhisperUI.ps1") (Join-Path $root "release\Setup-WhisperUI.ps1") -Force
    Write-Host "  Created: release\Setup-WhisperUI.ps1" -ForegroundColor Green
    Write-Host "  Install Inno Setup 6 for a proper Setup.exe: https://jrsoftware.org/isdl.php" -ForegroundColor Gray
}

# Step 4: Summary
Write-Host "[4/4] Done!" -ForegroundColor Green
Write-Host ""
Write-Host "  Portable app:  release\WhisperUI\" -ForegroundColor White
Write-Host "  Run locally:   release\WhisperUI\WhisperUI.bat" -ForegroundColor White
if (Test-Path $setupExe) {
    Write-Host "  Installer:     release\WhisperUI-Setup.exe" -ForegroundColor White
} else {
    Write-Host "  Installer:     release\Setup-WhisperUI.ps1" -ForegroundColor White
}
Write-Host ""
