#!/bin/sh
# Builds dist/Facecam.app and dist/Facecam.dmg (Apple Silicon).
#   ./build.sh            build
#   ./build.sh --install  also copy the app to ~/Applications
set -e
cd "$(dirname "$0")"
VERSION=1.0
APP=dist/Facecam.app
rm -rf dist && mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# This Mac's xcrun is blocked by the unaccepted Xcode licence, so use the
# CommandLineTools compiler plus Xcode's macro plugins when they're there.
CLT=/Library/Developer/CommandLineTools
PLUGINS=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/usr/lib/swift/host/plugins
if [ -z "$CI" ] && [ -x $CLT/usr/bin/swiftc ] && [ -d "$PLUGINS" ]; then
  DEVELOPER_DIR=$CLT $CLT/usr/bin/swiftc -O -target arm64-apple-macos14 -sdk $CLT/SDKs/MacOSX.sdk \
    -plugin-path "$PLUGINS" main.swift -o "$APP/Contents/MacOS/Facecam"
else
  swiftc -O -target arm64-apple-macos14 main.swift -o "$APP/Contents/MacOS/Facecam"
fi
cp bridge.py setup.sh "$APP/Contents/Resources/"

ICONSET=dist/Facecam.iconset
mkdir -p $ICONSET
for s in 16 32 128 256 512; do
  sips -z $s $s icon.png --out $ICONSET/icon_${s}x${s}.png >/dev/null
  sips -z $((s*2)) $((s*2)) icon.png --out $ICONSET/icon_${s}x${s}@2x.png >/dev/null
done
iconutil -c icns $ICONSET -o "$APP/Contents/Resources/Facecam.icns"
rm -rf $ICONSET

cat > "$APP/Contents/Info.plist" <<P
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>com.austn.facecam</string>
<key>CFBundleName</key><string>Facecam</string>
<key>CFBundleDisplayName</key><string>Facecam</string>
<key>CFBundleExecutable</key><string>Facecam</string>
<key>CFBundleIconFile</key><string>Facecam</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>$VERSION</string>
<key>CFBundleVersion</key><string>$VERSION</string>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>NSCameraUsageDescription</key><string>Facecam reads your camera to swap your face live.</string>
</dict></plist>
P
codesign -fs - "$APP"

# Disk image: a drag-to-Applications window with instructions in the background (see dmg.py).
uvx --from dmgbuild==1.6.7 dmgbuild -s dmg.py Facecam dist/Facecam.dmg >/dev/null
echo "built $APP and dist/Facecam.dmg"

if [ "$1" = "--install" ]; then
  rm -rf ~/Applications/Facecam.app && cp -R "$APP" ~/Applications/ && echo "installed ~/Applications/Facecam.app"
fi
