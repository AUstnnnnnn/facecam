#!/bin/sh
# Facecam's first-run setup on macOS. The app runs this and shows each "[n/5]"
# line as progress. Safe to run again: every step skips what's already there.
#
#   FACECAM_SUPPORT  install folder (default ~/Library/Application Support/Facecam)
#   FACECAM_WHEEL    insightface wheel path or URL (CI passes the one it just built)
#   FACECAM_SKIP_OBS set to skip installing OBS (CI)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SUPPORT=${FACECAM_SUPPORT:-"$HOME/Library/Application Support/Facecam"}
DLC_REF=08896a51c9a6198f50dc9f7508acf39fc460c8f1  # Deep-Live-Cam version this was tested against
OBS_VER=33.0.1
WHEEL=${FACECAM_WHEEL:-https://github.com/AUstnnnnnn/facecam/releases/download/wheels/insightface-0.7.3-cp312-cp312-macosx_11_0_arm64.whl}
mkdir -p "$SUPPORT"
cd "$SUPPORT"
export PATH="$SUPPORT/bin:$PATH"

echo "[1/5] Python manager (uv)"
if command -v uv >/dev/null 2>&1; then echo "  Already installed."
else curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SUPPORT/bin" UV_NO_MODIFY_PATH=1 sh
fi

echo "[2/5] Face swap engine"
if [ -f engine/run.py ]; then echo "  Already installed."
else
  rm -rf engine "Deep-Live-Cam-$DLC_REF"
  curl -fsSL "https://github.com/hacksider/Deep-Live-Cam/archive/$DLC_REF.tar.gz" | tar -xz
  mv "Deep-Live-Cam-$DLC_REF" engine
fi

echo "[3/5] Python and packages"
[ -x venv/bin/python ] || uv venv -q -p 3.12 venv
# Skip the unused NSFW filter; insightface comes prebuilt
# because building it needs Xcode.
grep -viE '^(opennsfw2|keras|insightface)' engine/requirements.txt > venv/reqs.txt
echo pyvirtualcam==0.15.0 >> venv/reqs.txt
{ cat venv/reqs.txt; echo "$WHEEL"; } > venv/deps.want
if cmp -s venv/deps.want venv/deps.done; then echo "  Already installed."
else
  uv pip install -q --python venv/bin/python "$WHEEL"
  uv pip install -q --python venv/bin/python -r venv/reqs.txt
  cp venv/deps.want venv/deps.done
fi

echo "[4/5] Face models (about 800 MB)"
FACECAM_SUPPORT="$SUPPORT" venv/bin/python "$HERE/bridge.py" prepare

echo "[5/5] OBS (makes the virtual camera Discord sees)"
if [ -d /Applications/OBS.app ] || [ -n "${FACECAM_SKIP_OBS:-}" ]; then echo "  Already installed."
else
  curl -fsSL -o obs.dmg "https://github.com/obsproject/obs-studio/releases/download/$OBS_VER/OBS-Studio-$OBS_VER-macOS-Apple.dmg"
  MNT=$(mktemp -d)
  hdiutil attach -nobrowse -quiet -mountpoint "$MNT" obs.dmg
  # The camera extension only activates from /Applications.
  ditto "$MNT/OBS.app" /Applications/OBS.app || { hdiutil detach -quiet "$MNT"; echo "  Couldn't copy OBS into Applications."; exit 1; }
  hdiutil detach -quiet "$MNT"
  rm obs.dmg
fi

echo "1" > "$SUPPORT/ready"  # the app skips this step once it sees this
