"""End-to-end check on a Windows runner: real window, real drops,
real models, real OBS Virtual Camera. Only the webcam is faked (CI has none).
The camera opens with the window and sends you unchanged; Start scans the face
in, Stop scans it back out.

    set FACECAM_CPU=1
    .venv\\Scripts\\python.exe .github\\e2e.py
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import facecam as fc  # noqa: E402  (chdirs into engine/)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = os.path.abspath(os.environ.get("E2E_OUT", os.path.join(tempfile.gettempdir(), "facecam-e2e")))
os.makedirs(OUT, exist_ok=True)
fc.FACES = tempfile.mkdtemp()  # keep test photos out of the real faces/ folder
app = QApplication(sys.argv)


def gif(name, at):
    cap = cv2.VideoCapture(os.path.join(fc.ENGINE, "media", name))
    cap.set(cv2.CAP_PROP_POS_FRAMES, at)
    ok, f = cap.read()
    assert ok, name
    return f


def pump(cond, secs, what):
    end = time.time() + secs
    while time.time() < end:
        app.processEvents()
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}; status: {w.status.text()!r}")


def drop(w, urls=(), html=None):
    md = QMimeData()
    if urls:
        md.setUrls([QUrl(u) for u in urls])
    if html:
        md.setHtml(html)
    w.dropEvent(QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, md,
                           Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))


def snap(w, name):
    w.grab().save(os.path.join(OUT, name))


# ── 1. detection under dim light (the flicker fix) ──
get_one_face, fs = fc.load()
movie = gif("movie.gif", 10)
dark = cv2.convertScaleAbs(movie, alpha=0.06)
assert get_one_face(fc.lift(dark)) is not None, "lift() didn't rescue a dim frame"
print("dim:", "plain detect", get_one_face(dark) is not None, "| lifted detect True")

# ── 2. real camera enumeration on a machine without one ──
w = fc.Window()
assert "No camera" in w.status.text() and not w.start.isEnabled(), w.status.text()
snap(w, "0-no-camera.png")
w.close()
# Free this window here, on the UI thread. Left to the garbage collector it can be
# collected on the camera thread later, and PySide crashes deleting its buttons.
import gc  # noqa: E402
del w
gc.collect()

# ── fake webcam: plain movie frames, or a scripted FEED once one is queued ──
queue, served = [], []  # served: FEED index (or None) for every frame read, in order


class FakeCam:
    def __init__(self, i): pass
    def start(self, *a): return True
    def release(self): pass

    def read(self):
        time.sleep(0.03)
        k = queue.pop(0) if queue else None
        served.append(k)
        return True, (FEED[k] if k is not None else movie).copy()


import modules.video_capture as vc  # noqa: E402
import pyvirtualcam  # noqa: E402
vc.VideoCapturer = FakeCam
sent = []
real_send = pyvirtualcam.Camera.send
pyvirtualcam.Camera.send = lambda self, f: (sent.append(f.copy()), real_send(self, f))[1]
FEED = [movie] * 6 + [dark] * 4 + [np.zeros_like(movie)] * 12  # face, dim face, no face

# ── 3. drops: Explorer files, browser URL, data URL, blocked network paths ──
fc.list_cameras = lambda: [(0, "Fake Webcam")]
w = fc.Window()
w.show()
assert w.status.text().startswith("Pick a face") and not w.start.isEnabled()
snap(w, "1-empty.png")

photos = []
for i, at in enumerate((0, 3, 6)):
    photos.append(os.path.join(OUT, f"ludwig{i}.png"))
    cv2.imwrite(photos[-1], gif("ludwig.gif", at))

drop(w, [QUrl.fromLocalFile(photos[0]).toString()])
assert len(fc.face_files()) == 1 and w.start.isEnabled() and w.status.text() == w.ready_text(), w.status.text()
drop(w, [QUrl.fromLocalFile(p).toString() for p in photos[1:]])  # multi-select from Explorer
assert len(fc.face_files()) == 3 and w.status.text() == w.ready_text() and "3 photos" in w.status.text(), w.status.text()
snap(w, "2-three-photos.png")

for bad in ("file://evil-host/share/x.png", "file:////evil-host/share/x.png", "ftp://x/y.png"):
    drop(w, [bad])
    assert "wasn't an image" in w.status.text() and len(fc.face_files()) == 3, bad

ok, buf = cv2.imencode(".png", gif("ludwig.gif", 9))
import base64  # noqa: E402
drop(w, html=f'<img src="data:image/png;base64,{base64.b64encode(buf).decode()}">')
assert len(fc.face_files()) == 4, "data: URL in HTML drop not used"
drop(w, ["https://raw.githubusercontent.com/hacksider/Deep-Live-Cam/main/media/ludwig.gif"])
assert len(fc.face_files()) == 5, f"https drop failed: {w.status.text()}"
drop(w, [QUrl.fromLocalFile(photos[0]).toString()])
assert len(fc.face_files()) == fc.MAX_FACES and len(w.thumbs) == fc.MAX_FACES, "MAX_FACES not enforced"

w.clear.click()
assert not fc.face_files() and not w.start.isEnabled() and not w.clear.isVisible()
for p in photos:
    drop(w, [QUrl.fromLocalFile(p).toString()])
print("drops ok")

# ── 3b. presets: save, sanitized names, load, menu, delete ──
fc.PRESETS = tempfile.mkdtemp()
w.save_preset("Ludwig")
assert fc.preset_names() == ["Ludwig"] and w.presets.text() == "Ludwig", w.status.text()
w.save_preset("..\\..\\evil:/*")  # path tricks collapse to a plain name inside PRESETS
assert fc.preset_names() == ["Ludwig", "evil"], fc.preset_names()
assert fc.preset_dir(" ... ") is None
w.clear.click()
assert w.presets.text() == "Presets" and not w.start.isEnabled()
w.load_preset("Ludwig")
assert len(fc.face_files()) == 3 and w.start.isEnabled() and w.presets.text() == "Ludwig"
assert w.status.text() == w.ready_text() and "3 photos" in w.status.text(), w.status.text()
w.fill_presets_menu()
items = [a.text() for a in w.presets_menu.actions() if a.text()]
assert items[:2] == ["Ludwig", "evil"] and items[2].startswith("Save"), items
w.delete_preset("evil")
assert fc.preset_names() == ["Ludwig"]
snap(w, "2b-preset.png")
print("presets ok")

# ── 4. live run: you → scan in → swap → OBS Virtual Camera → read back → scan out ──
pump(lambda: w.live and len(sent) > 5, 120, "camera on before Start")
assert all((f == movie).all() for f in sent[-3:]), "before Start the camera should send you unchanged"
assert w.status.text().startswith("Showing you"), w.status.text()
snap(w, "3-you.png")


def orange(f):
    """Pixels the scan's orange glow warmed up: red rose well past blue vs the camera frame."""
    d = f.astype(int) - movie.astype(int)
    return int(((d[..., 2] - d[..., 0]) > 40).sum())


def scan_seen(start):
    """True once any frame sent since `start` shows the glow. Checks each frame once:
    rescanning the whole list every pump starves the Qt event loop."""
    while seen[0] < len(sent):
        if seen[0] >= start and orange(sent[seen[0]]) > 1000:
            return True
        seen[0] += 1
    return False


seen = [0]


w.start.click()
assert w.start.text() == "Stop" and w.cameras.isEnabled()
before = len(sent)
pump(lambda: w.status.text().startswith("Live"), 300, "Live status")
seen[0] = before
pump(lambda: scan_seen(before), 120, "the scan line on screen")
# Let the scan finish before the scripted feed (slow CPU runners take a few seconds).
pump(lambda: len(sent) > before + 5 and all(orange(f) < 200 for f in sent[-5:]), 300, "the scan to finish")
start = len(served)
queue.extend(range(len(FEED)))
pump(lambda: not queue and len(sent) >= len(served) - 1 and len(served) > start + len(FEED) + 3, 600,
     "whole feed into OBS camera")
from pygrabber.dshow_graph import FilterGraph  # noqa: E402
names = FilterGraph().get_input_devices()
print("DirectShow devices:", names)
idx = next(i for i, n in enumerate(names) if "OBS" in n)
rb = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
got = None
for _ in range(60):
    app.processEvents()
    ok, got = rb.read()
    if ok and got is not None and got.mean() > 5:
        break
rb.release()
assert got is not None, "couldn't read OBS Virtual Camera back"
cv2.imwrite(os.path.join(OUT, "3-obs-readback.png"), got)
snap(w, "4-live.png")

# The size probe is read but never sent, so frame served[i + 1] went out as sent[i].
swapped = [served[i + 1] for i in range(len(sent)) if i + 1 < len(served) and served[i + 1] is not None
           and (sent[i] != FEED[served[i + 1]]).any()]
print("swapped feed frames:", swapped)
assert all(k in swapped for k in range(0, 6)), "face frames weren't swapped"
assert all(k in swapped for k in range(6, 10)), "dim frames weren't swapped"
held = [k for k in swapped if k >= 10]
assert held == list(range(10, 10 + fc.HOLD_FRAMES)), f"hold over missed faces wrong: {held}"
# What Discord gets should be one of our frames, not OBS's placeholder card.
gs = cv2.resize(got, (movie.shape[1], movie.shape[0]))
best = min(float(np.abs(gs.astype(int) - s.astype(int)).mean()) for s in sent if s.mean() > 5)
print("OBS readback mean abs diff vs sent:", round(best, 2))
assert best < 20, "OBS Virtual Camera isn't showing Facecam's frames"

w.start.click()  # Stop: scan back to you, camera stays on
assert w.start.text() == "Start" and w.worker is not None
before = len(sent)
seen[0] = before
pump(lambda: scan_seen(before), 120, "the scan line on the way out")
pump(lambda: all(orange(f) < 200 for f in sent[-5:]), 300, "the scan out to finish")
n = len(sent)
pump(lambda: len(sent) > n + 10, 120, "frames after Stop")
assert all((f == movie).all() for f in sent[-3:]), "after Stop the camera should send you unchanged"
assert w.status.text().startswith("Showing you"), w.status.text()
w.close()
print("e2e ok")
