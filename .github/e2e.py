"""End-to-end check on a Windows runner: real window, real drops,
real models, real OBS Virtual Camera. Only the webcam is faked (CI has none).

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
assert len(fc.face_files()) == 1 and w.start.isEnabled() and "closer match" in w.status.text(), w.status.text()
drop(w, [QUrl.fromLocalFile(p).toString() for p in photos[1:]])  # multi-select from Explorer
assert len(fc.face_files()) == 3 and w.status.text() == "Ready. Blending 3 photos.", w.status.text()
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

# ── 4. live run: fake webcam → swap → real OBS Virtual Camera → read back ──
FEED = [movie] * 6 + [dark] * 4 + [np.zeros_like(movie)] * 12  # face, dim face, no face


class FakeCam:
    def __init__(self, i): self.n = 0
    def start(self, *a): return True
    def release(self): pass

    def read(self):
        time.sleep(0.03)
        f = (FEED[self.n] if self.n < len(FEED) else movie).copy()
        self.n += 1
        return True, f


import modules.video_capture as vc  # noqa: E402
import pyvirtualcam  # noqa: E402
vc.VideoCapturer = FakeCam
sent = []
real_send = pyvirtualcam.Camera.send
pyvirtualcam.Camera.send = lambda self, f: (sent.append(f.copy()), real_send(self, f))[1]

w.start.click()
assert w.start.text() == "Stop" and not w.cameras.isEnabled()
pump(lambda: w.status.text().startswith("Live"), 300, "Live status")
pump(lambda: len(sent) >= len(FEED) + 3, 600, "whole feed into OBS camera")

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
w.start.click()
pump(lambda: w.worker is None, 30, "stop")
assert w.start.text() == "Start" and w.cameras.isEnabled() and not w.status.text().startswith("Stopped"), w.status.text()

# FEED[0] is the size probe before the loop, so FEED[k] went out as sent[k - 1].
swapped = [k for k in range(1, len(FEED)) if (sent[k - 1] != FEED[k]).any()]
print("swapped feed frames:", swapped)
assert all(k in swapped for k in range(1, 6)), "face frames weren't swapped"
assert all(k in swapped for k in range(6, 10)), "dim frames weren't swapped"
held = [k for k in swapped if k >= 10]
assert held == list(range(10, 10 + fc.HOLD_FRAMES)), f"hold over missed faces wrong: {held}"
# What Discord gets should be one of our frames, not OBS's placeholder card.
gs = cv2.resize(got, (movie.shape[1], movie.shape[0]))
best = min(float(np.abs(gs.astype(int) - s.astype(int)).mean()) for s in sent if s.mean() > 5)
print("OBS readback mean abs diff vs sent:", round(best, 2))
assert best < 20, "OBS Virtual Camera isn't showing Facecam's frames"
w.close()
print("e2e ok")
