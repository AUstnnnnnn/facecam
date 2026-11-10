@echo off
setlocal
cd /d "%~dp0"
title Facecam setup

rem Deep-Live-Cam version this was tested against.
set DLC_REF=08896a51c9a6198f50dc9f7508acf39fc460c8f1
if not defined FACECAM_WHEEL set FACECAM_WHEEL=https://github.com/AUstnnnnnn/facecam-windows/releases/download/wheels/insightface-0.7.3-cp312-cp312-win_amd64.whl

echo [1/5] Python manager (uv)...
where uv >nul 2>&1 || powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" || goto :fail
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

echo [2/5] Face swap engine (Deep-Live-Cam)...
if not exist engine\run.py (
  curl -fL -o dlc.zip https://github.com/hacksider/Deep-Live-Cam/archive/%DLC_REF%.zip || goto :fail
  tar -xf dlc.zip || goto :fail
  move Deep-Live-Cam-%DLC_REF% engine >nul || goto :fail
  del dlc.zip
)

echo [3/5] Python 3.12 and packages (a few minutes)...
if not exist .venv\Scripts\python.exe (uv venv -p 3.12 .venv || goto :fail)
set PY=.venv\Scripts\python.exe
rem Skip the NVIDIA-only runtime and the unused NSFW filter; insightface comes prebuilt.
findstr /v /i "onnxruntime opennsfw2 keras insightface" engine\requirements.txt > reqs.txt
uv pip install --python %PY% "%FACECAM_WHEEL%" || goto :fail
uv pip install --python %PY% -r reqs.txt onnxruntime-directml pyvirtualcam || goto :fail

echo [4/5] OBS Studio (provides the virtual camera Discord sees)...
if not exist "%ProgramFiles%\obs-studio\bin\64bit\obs64.exe" (
  winget install -e --id OBSProject.OBSStudio --accept-package-agreements --accept-source-agreements || echo   Couldn't install OBS automatically. Get it from https://obsproject.com
)
echo [5/5] Visual C++ runtime...
winget install -e --id Microsoft.VCRedist.2015+.x64 --accept-package-agreements --accept-source-agreements >nul 2>&1

echo.
echo Done. Double-click Facecam.bat to start.
if not defined FACECAM_CI pause
exit /b 0

:fail
echo.
echo Setup failed at the step above. Screenshot this window and send it over.
if not defined FACECAM_CI pause
exit /b 1
