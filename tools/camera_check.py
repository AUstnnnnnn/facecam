"""Diagnose "Camera didn't open. Close other apps using it and try again."

1. Asks Windows which apps are holding the webcam right now (the same
   registry data behind the camera-in-use icon in Settings > Privacy).
2. Tries every way engine/modules/video_capture.py opens a camera
   (DSHOW / MSMF, with and without the MJPG 1280x720@30 open params).

    .venv\\Scripts\\python.exe tools\\camera_check.py

Run it while Facecam is stopped, or it will report Facecam itself as the holder.
"""
import sys
import winreg
from datetime import datetime

WEBCAM = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"


def _filetime(ft):
    """Windows FILETIME (100 ns ticks since 1601, UTC) to local time."""
    return datetime.fromtimestamp((ft - 116444736000000000) / 1e7)


def who_has_webcam():
    """[(started, app)] for apps Windows says have the webcam open, newest first.

    LastUsedTimeStop == 0 means "still open". Apps that crashed or updated while
    using the camera leave stale entries (Discord leaves one per app version), so
    only the newest entry per app is kept and the first result is the likely culprit.
    """
    held = []
    for sub in ("NonPackaged", ""):  # desktop apps, then Store apps
        try:
            root = winreg.OpenKey(winreg.HKEY_CURRENT_USER, WEBCAM + ("\\" + sub if sub else ""))
        except OSError:
            continue
        with root:
            i = 0
            while True:
                try:
                    name = winreg.EnumKey(root, i)
                except OSError:
                    break
                i += 1
                if name == "NonPackaged":
                    continue
                try:
                    with winreg.OpenKey(root, name) as k:
                        start = winreg.QueryValueEx(k, "LastUsedTimeStart")[0]
                        stop = winreg.QueryValueEx(k, "LastUsedTimeStop")[0]
                except OSError:
                    continue
                if start and stop == 0:
                    held.append((_filetime(start), name.replace("#", "\\").rsplit("\\", 1)[-1]))
    newest = {}
    for started, app in held:
        newest[app] = max(started, newest.get(app, started))
    return sorted(((t, app) for app, t in newest.items()), reverse=True)


def try_open():
    import cv2
    from pygrabber.dshow_graph import FilterGraph
    names = FilterGraph().get_input_devices()
    print("DirectShow devices:", list(enumerate(names)))
    params = [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"), cv2.CAP_PROP_FRAME_WIDTH, 1280,
              cv2.CAP_PROP_FRAME_HEIGHT, 720, cv2.CAP_PROP_FPS, 30]
    for i, name in enumerate(names):
        if "OBS" in name:
            continue
        for label, backend, p in [("DSHOW + MJPG 720p params", cv2.CAP_DSHOW, params),
                                  ("DSHOW plain", cv2.CAP_DSHOW, None),
                                  ("MSMF + params", cv2.CAP_MSMF, params),
                                  ("MSMF plain", cv2.CAP_MSMF, None)]:
            cap = cv2.VideoCapture(i, backend, p) if p else cv2.VideoCapture(i, backend)
            opened = cap.isOpened()
            ok, f = cap.read() if opened else (False, None)
            print(f"  [{i}] {name!r:32} {label:26} opened={opened!s:5} frame={ok!s:5} "
                  f"{f.shape if ok else ''}")
            cap.release()
    print("A busy camera fails every row, or 'opens' on MSMF but returns no frame.")


if __name__ == "__main__":
    held = who_has_webcam()
    print("Apps holding the webcam:", ", ".join(f"{app} (since {t:%Y-%m-%d %H:%M})"
                                                 for t, app in held) or "none")
    if "--no-open" not in sys.argv:
        try_open()
