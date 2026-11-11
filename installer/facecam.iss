; Facecam-Setup.exe. Built by CI with Inno Setup: iscc installer\facecam.iss
; Installs per user (no admin prompt), then runs install.bat to fetch Python,
; the engine, packages and models. Re-running it skips whatever is already there.

#define AppVersion "1.1"

[Setup]
AppId={{6C0B7E52-3A1F-4D8E-9B57-FACECA4A0001}
AppName=Facecam
AppVersion={#AppVersion}
AppPublisher=Austin Radliff
AppPublisherURL=https://github.com/AUstnnnnnn/facecam-windows
DefaultDirName={localappdata}\Programs\Facecam
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=Facecam-Setup
SetupIconFile=..\assets\facecam.ico
UninstallDisplayIcon={app}\assets\facecam.ico
UninstallDisplayName=Facecam
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
CloseApplications=yes

[Tasks]
Name: desktopicon; Description: "Add a desktop shortcut"

[Files]
Source: "..\facecam.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\install.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\tools\camera_check.py"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "..\assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Facecam"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\facecam.py"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\facecam.ico"
Name: "{autodesktop}\Facecam"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\facecam.py"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\facecam.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\facecam.py"""; WorkingDir: "{app}"; Description: "Open Facecam"; Flags: postinstall nowait skipifsilent; Check: SetupWorked

[UninstallDelete]
; Everything install.bat and the app made: engine, Python, models, saved photos and presets.
Type: filesandordirs; Name: "{app}"

[Code]
var
  Worked: Boolean;

function SetupWorked: Boolean;
begin
  Result := Worked;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Code: Integer;
  App: String;
begin
  if CurStep <> ssPostInstall then
    Exit;
  App := ExpandConstant('{app}');
  WizardForm.StatusLabel.Caption := 'Setting up the face swap engine. The first install downloads about 1.5 GB, so this can take a while. Updates are quick.';
  WizardForm.FilenameLabel.Caption := '';
  WizardForm.ProgressGauge.Style := npbstMarquee;
  Worked := Exec(ExpandConstant('{cmd}'), '/c ""' + App + '\install.bat" /quiet > "' + App + '\install.log" 2>&1"',
                 App, SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0);
  WizardForm.ProgressGauge.Style := npbstNormal;
  if not Worked then
    SuppressibleMsgBox('Facecam couldn''t finish setting up. Check your internet connection and run the installer again; it picks up where it stopped.' + #13#10#13#10 +
                       'Details are in ' + App + '\install.log', mbError, MB_OK, IDOK);
end;

function InitializeSetup: Boolean;
begin
  Worked := False;
  Result := True;
end;
