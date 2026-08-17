; Whisper UI — Windows Installer (Inno Setup)
; Build: iscc installer\WhisperUI.iss
; Requires Inno Setup 6: https://jrsoftware.org/isdl.php

#define AppName "Whisper UI"
#define AppVersion "1.0.0"
#define AppPublisher "Whisper UI"
#define AppURL "https://github.com/openai/whisper"
#define AppExeName "WhisperUI.bat"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=no
OutputDir=..\release
OutputBaseFilename=WhisperUI-Setup
SetupIconFile=..\web\public\favicon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\WhisperUI.bat

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "..\release\WhisperUI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon; WorkingDir: "{app}"

[Run]
Filename: "{app}\installer\post-install.ps1"; Flags: runhidden waituntilterminated; StatusMsg: "Setting up Python environment..."
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\server\.venv"
Type: filesandordirs; Name: "{app}\server\uploads"

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  Result := '';
  if not Exec('powershell.exe', '-NoProfile -ExecutionPolicy Bypass -Command "if (-not (Get-Command python -EA SilentlyContinue)) { exit 1 } else { exit 0 }"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    Result := 'Python 3.9–3.11 is required. Please install Python from python.org and run setup again.';
    Exit;
  end;
  if ResultCode <> 0 then
  begin
    Result := 'Python 3.9–3.11 is required. Please install Python from python.org and run setup again.';
  end;
end;
