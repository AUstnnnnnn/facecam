#!/bin/sh
# One-line install for Facecam on Mac (skips the "unverified developer" warning,
# since files curl downloads aren't flagged the way browser downloads are):
#   curl -fsSL https://raw.githubusercontent.com/AUstnnnnnn/facecam/main/mac/install.sh | sh
set -e
[ "$(uname -m)" = arm64 ] || { echo "Facecam needs a Mac with Apple silicon (M1 or newer)."; exit 1; }
TMP=$(mktemp -d)
echo "Downloading Facecam..."
curl -fsSL -o "$TMP/Facecam.dmg" https://github.com/AUstnnnnnn/facecam/releases/download/mac/Facecam.dmg
hdiutil attach -nobrowse -quiet -mountpoint "$TMP/mnt" "$TMP/Facecam.dmg"
rm -rf /Applications/Facecam.app
ditto "$TMP/mnt/Facecam.app" /Applications/Facecam.app
hdiutil detach -quiet "$TMP/mnt"
rm -rf "$TMP"
echo "Installed. Opening Facecam..."
open /Applications/Facecam.app
