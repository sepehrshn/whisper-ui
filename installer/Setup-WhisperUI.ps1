# Whisper UI — Windows Installer (no Inno Setup required)
# Double-click or run: powershell -ExecutionPolicy Bypass -File Setup-WhisperUI.ps1

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms

$AppName = "Whisper UI"
$SourceDir = $PSScriptRoot
if (Test-Path (Join-Path $SourceDir "WhisperUI")) {
    $SourceDir = Join-Path $SourceDir "WhisperUI"
}

$InstallDir = Join-Path $env:LOCALAPPDATA "Programs\WhisperUI"
$Desktop = [Environment]::GetFolderPath("Desktop")
$StartMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"

$form = New-Object System.Windows.Forms.Form
$form.Text = "$AppName Setup"
$form.Size = New-Object System.Drawing.Size(480, 220)
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false

$label = New-Object System.Windows.Forms.Label
$label.Location = New-Object System.Drawing.Point(20, 20)
$label.Size = New-Object System.Drawing.Size(430, 60)
$label.Text = "Install $AppName on this computer.`n`nLocation: $InstallDir"
$form.Controls.Add($label)

$progress = New-Object System.Windows.Forms.Label
$progress.Location = New-Object System.Drawing.Point(20, 90)
$progress.Size = New-Object System.Drawing.Size(430, 30)
$progress.Text = "Ready to install."
$form.Controls.Add($progress)

$installBtn = New-Object System.Windows.Forms.Button
$installBtn.Location = New-Object System.Drawing.Point(280, 140)
$installBtn.Size = New-Object System.Drawing.Size(90, 32)
$installBtn.Text = "Install"
$form.Controls.Add($installBtn)

$cancelBtn = New-Object System.Windows.Forms.Button
$cancelBtn.Location = New-Object System.Drawing.Point(380, 140)
$cancelBtn.Size = New-Object System.Drawing.Size(75, 32)
$cancelBtn.Text = "Cancel"
$form.Controls.Add($cancelBtn)

$installBtn.Add_Click({
    $installBtn.Enabled = $false
    $cancelBtn.Enabled = $false
    try {
        $progress.Text = "Copying files..."
        [System.Windows.Forms.Application]::DoEvents()

        if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force }
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
        Copy-Item "$SourceDir\*" $InstallDir -Recurse -Force

        $progress.Text = "Setting up Python environment..."
        [System.Windows.Forms.Application]::DoEvents()

        $postInstall = Join-Path $InstallDir "installer\post-install.ps1"
        if (Test-Path $postInstall) {
            & powershell -NoProfile -ExecutionPolicy Bypass -File $postInstall
        }

        $progress.Text = "Creating shortcuts..."
        [System.Windows.Forms.Application]::DoEvents()

        $launcher = Join-Path $InstallDir "WhisperUI.bat"
        if (-not (Test-Path $launcher)) {
            $launcher = Join-Path $InstallDir "start-whisper-ui.bat"
        }

        $WshShell = New-Object -ComObject WScript.Shell
        $shortcut = $WshShell.CreateShortcut((Join-Path $StartMenu "$AppName.lnk"))
        $shortcut.TargetPath = $launcher
        $shortcut.WorkingDirectory = $InstallDir
        $shortcut.Description = "Whisper UI — Speech to Text"
        $shortcut.Save()

        $desktopShortcut = $WshShell.CreateShortcut((Join-Path $Desktop "$AppName.lnk"))
        $desktopShortcut.TargetPath = $launcher
        $desktopShortcut.WorkingDirectory = $InstallDir
        $desktopShortcut.Save()

        $progress.Text = "Installation complete!"
        [System.Windows.Forms.MessageBox]::Show(
            "$AppName has been installed.`n`nOpen it from the Start Menu or Desktop shortcut.",
            "Setup Complete",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        )

        $launch = [System.Windows.Forms.MessageBox]::Show("Launch $AppName now?", "Launch", "YesNo", "Question")
        if ($launch -eq "Yes") { Start-Process $launcher }
        $form.Close()
    } catch {
        [System.Windows.Forms.MessageBox]::Show("Installation failed: $_", "Error", "OK", "Error")
        $installBtn.Enabled = $true
        $cancelBtn.Enabled = $true
    }
})

$cancelBtn.Add_Click({ $form.Close() })
[void]$form.ShowDialog()
