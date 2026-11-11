@echo off
setlocal
cd /d "%~dp0"
title Facecam setup
rem Run by Facecam-Setup.exe with /quiet. Safe to run again: every step skips
rem what's already installed.
if /i "%~1"=="/quiet" set NOPAUSE=1
if defined FACECAM_CI set NOPAUSE=1

rem Deep-Live-Cam version this was tested against.
set DLC_REF=08896a51c9a6198f50dc9f7508acf39fc460c8f1
if not defined FACECAM_WHEEL set FACECAM_WHEEL=https://github.com/AUstnnnnnn/facecam-windows/releases/download/wheels/insightface-0.7.3-cp312-cp312-win_amd64.whl

echo [1/6] Python manager (uv)...
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>&1 && (echo   Already installed.) || (powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" || goto :fail)

echo [2/6] Face swap engine (Deep-Live-Cam)...
if exist engine\run.py (echo   Already installed.) else (
  curl -fL -o dlc.zip https://github.com/hacksider/Deep-Live-Cam/archive/%DLC_REF%.zip || goto :fail
  tar -xf dlc.zip || goto :fail
  move Deep-Live-Cam-%DLC_REF% engine >nul || goto :fail
  del dlc.zip
)

echo [3/6] Python 3.12 and packages...
if not exist .venv\Scripts\python.exe (uv venv -p 3.12 .venv || goto :fail)
set PY=.venv\Scripts\python.exe
rem Skip the NVIDIA-only runtime and the unused NSFW filter; insightface comes prebuilt.
findstr /v /i "onnxruntime opennsfw2 keras insightface" engine\requirements.txt > reqs.txt
(type reqs.txt & echo %FACECAM_WHEEL% onnxruntime-directml pyvirtualcam) > .venv\deps.want
fc /b .venv\deps.want .venv\deps.done >nul 2>&1
if errorlevel 1 (
  uv pip install --python %PY% "%FACECAM_WHEEL%" || goto :fail
  uv pip install --python %PY% -r reqs.txt onnxruntime-directml pyvirtualcam || goto :fail
  copy /y .venv\deps.want .venv\deps.done >nul
) else echo   Already installed.

echo [4/6] Face models (about 800 MB)...
%PY% facecam.py --prepare || goto :fail

echo [5/6] OBS Studio (provides the virtual camera Discord sees)...
if exist "%ProgramFiles%\obs-studio\bin\64bit\obs64.exe" (echo   Already installed.) else (
  winget install -e --id OBSProject.OBSStudio --accept-package-agreements --accept-source-agreements || echo   Couldn't install OBS automatically. Get it from https://obsproject.com
)
echo [6/6] Visual C++ runtime...
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1 && (echo   Already installed.) || (
  winget install -e --id Microsoft.VCRedist.2015+.x64 --accept-package-agreements --accept-source-agreements >nul 2>&1
)

echo.
echo Done. Open Facecam from the Start menu.
if not defined NOPAUSE pause
exit /b 0

:fail
echo.
echo Setup failed at the step above.
if not defined NOPAUSE pause
exit /b 1
