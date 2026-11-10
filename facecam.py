"""Facecam for Windows.

Drop a face photo (or a few of the same person, which are blended for a
closer match), pick your camera, press Start. The swapped video goes to
OBS Virtual Camera, which Discord (or Zoom, Meet...) can pick as a camera.

    python facecam.py             open the window
    python facecam.py --selftest  swap two demo frames, no camera needed
"""
import glob
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "engine")  # Deep-Live-Cam, unpacked by install.bat
FACES = os.path.join(HERE, "faces")  # photos of one person, blended; newest last
MAX_FACES = 5
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime  # noqa: E402


# Dim light drops detector confidence below insightface's 0.5 default, so the
# swap blinks on and off. Detect more leniently and coast over short misses.
DET_THRESH = 0.35
HOLD_FRAMES = 8  # ~0.25 s at 30 fps


def lift(f):
    """Brightened copy for face detection only; the swap still uses the real frame."""
    m = f.mean()
    return f if m >= 90 else cv2.convertScaleAbs(f, alpha=min(8.0, 110 / max(m, 1.0)))


def load():
    """Load Deep-Live-Cam's face detector and swapper on the GPU via DirectML."""
    import modules.globals as g
    wanted = ["CPUExecutionProvider"] if os.environ.get("FACECAM_CPU") else \
        ["DmlExecutionProvider", "CPUExecutionProvider"]
    g.execution_providers = [p for p in wanted if p in onnxruntime.get_available_providers()]
    from modules.face_analyser import get_one_face
    from modules.processors.frame import face_swapper as fs
    if fs.get_face_swapper() is None:
        raise RuntimeError("couldn't download the face swap model, check your internet")
    from modules.face_analyser import get_face_analyser
    get_face_analyser().det_model.det_thresh = DET_THRESH
    return get_one_face, fs


def face_files():
    return sorted(glob.glob(os.path.join(FACES, "*.png")))


def blended_face(get_one_face, paths=None):
    """One face from every photo that has one. The swap model only reads the
    identity embedding, so averaging it over several angles and lights gives a
    closer, steadier likeness."""
    imgs = (cv2.imread(p) for p in (paths or face_files()))
    faces = [f for f in (get_one_face(i) for i in imgs if i is not None) if f is not None]
    if not faces:
        return None
    face = faces[0]
    if len(faces) > 1:
        face.embedding = np.mean([f.normed_embedding for f in faces], axis=0)
    return face


def selftest():
    def first_frame(name, at=0):
        cap = cv2.VideoCapture(os.path.join(ENGINE, "media", name))
        cap.set(cv2.CAP_PROP_POS_FRAMES, at)
        ok, f = cap.read()
        assert ok, f"can't read {name}"
        return f
    get_one_face, fs = load()
    src = get_one_face(first_frame("ludwig.gif"))
    tgt = first_frame("movie.gif", 10)
    face = get_one_face(tgt)
    assert src is not None and face is not None, "no face found in demo frames"
    out = fs.process_frame(src, tgt.copy(), face)
    assert out.shape == tgt.shape and (out != tgt).any(), "swap changed nothing"
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, at in enumerate((0, 6)):
            paths.append(os.path.join(d, f"{i}.png"))
            cv2.imwrite(paths[-1], first_frame("ludwig.gif", at))
        blend = blended_face(get_one_face, paths)
        one = get_one_face(cv2.imread(paths[0]))
        assert blend is not None and float(blend.normed_embedding @ one.normed_embedding) > 0.8, "blend broke"
    print("selftest ok")


# ─── UI ────────────────────────────────────────────────────────────────────

from PySide6.QtCore import Qt, QThread, QUrl, Signal  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap  # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout,  # noqa: E402
                               QLabel, QPushButton, QVBoxLayout, QWidget)

ERROR_RED = "#d13438"


def list_cameras():
    """[(directshow_index, name)] minus OBS's own camera, which would feed back into itself."""
    from pygrabber.dshow_graph import FilterGraph
    names = FilterGraph().get_input_devices()
    return [(i, n) for i, n in enumerate(names) if "OBS" not in n]


class Worker(QThread):
    frame = Signal(QImage)
    status = Signal(str, bool)  # message, is_error

    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.reload_face = True

    def run(self):
        try:
            self._run()
        except Exception as e:  # anything here would otherwise vanish with the thread
            self.status.emit(f"Stopped: {e}", True)

    def _run(self):
        import pyvirtualcam
        from modules.video_capture import VideoCapturer
        self.status.emit("Loading models (the first start downloads about 500 MB)…", False)
        get_one_face, fs = load()
        cap = VideoCapturer(self.camera_index)
        if not cap.start(1280, 720, 30):
            return self.status.emit("Camera didn't open. Close other apps using it and try again.", True)
        try:
            ok, f = cap.read()
            if not ok:
                return self.status.emit("Camera opened but sent no picture.", True)
            h, w = f.shape[:2]
            try:
                vcam = pyvirtualcam.Camera(w, h, 30, fmt=pyvirtualcam.PixelFormat.BGR, backend="obs")
            except Exception:
                return self.status.emit("OBS Virtual Camera not found. Install OBS Studio, then press Start again.", True)
            src, last, missed = None, None, 0
            with vcam:
                while self.running:
                    if self.reload_face:
                        self.reload_face = False
                        src = blended_face(get_one_face)
                        if src is None:
                            self.status.emit("No face found in those photos. Try clear, front-facing ones.", True)
                        else:
                            self.status.emit("Live. In Discord, pick OBS Virtual Camera.", False)
                    ok, f = cap.read()
                    if not ok:
                        continue
                    if src is not None:
                        target = get_one_face(lift(f))
                        if target is not None:
                            last, missed = target, 0
                        elif missed < HOLD_FRAMES:
                            target, missed = last, missed + 1
                        if target is not None:
                            f = fs.process_frame(src, f, target)
                    vcam.send(f)
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    self.frame.emit(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())
        finally:
            cap.release()


def rounded(pix: QPixmap, w: int, h: int, r: int = 10, thumbs=()) -> QPixmap:
    """Scale-to-fill, clip to a rounded rect, and draw a strip of face thumbnails."""
    pix = pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                     Qt.TransformationMode.SmoothTransformation)
    out = QPixmap(w, h)
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(0, 0, w, h, r, r)
    p.setClipPath(path)
    p.drawPixmap((w - pix.width()) // 2, (h - pix.height()) // 2, pix)
    p.setClipping(False)
    for i, t in enumerate(thumbs):
        x, y, size = 10 + i * 42, h - 46, 36
        t = t.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                     Qt.TransformationMode.SmoothTransformation)
        box = QPainterPath()
        box.addRoundedRect(x, y, size, size, 7, 7)
        p.save()
        p.setClipPath(box)
        p.drawPixmap(x - (t.width() - size) // 2, y - (t.height() - size) // 2, t)
        p.restore()
        p.setPen(QPen(QColor(255, 255, 255, 230), 1.5))
        p.drawPath(box)
    p.end()
    return out


class Window(QWidget):
    STAGE_W, STAGE_H = 480, 270

    def __init__(self):
        super().__init__()
        self.worker = None
        self.has_face = bool(face_files())
        self.thumbs = [QPixmap(f) for f in face_files()]
        self.setWindowTitle("Facecam")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setAcceptDrops(True)

        self.stage = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.stage.setFixedSize(self.STAGE_W, self.STAGE_H)
        self.stage.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stage.mousePressEvent = lambda e: self.choose()
        self.clear = QPushButton("Clear", self.stage)
        self.clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear.setToolTip("Remove all photos to start on a different person")
        self.clear.setStyleSheet("QPushButton { background: rgba(0,0,0,150); color: white; border: none;"
                                 " border-radius: 11px; padding: 4px 10px; font-weight: 600; }"
                                 "QPushButton:hover { background: rgba(0,0,0,200); }")
        self.clear.adjustSize()
        self.clear.move(self.STAGE_W - self.clear.width() - 10, 10)
        self.clear.clicked.connect(self.clear_faces)

        self.cameras = QComboBox()
        for i, name in list_cameras():
            self.cameras.addItem(name, i)
        self.start = QPushButton("Start")
        self.start.setDefault(True)
        self.start.setMinimumWidth(96)
        self.start.clicked.connect(self.toggle)

        self.status = QLabel(wordWrap=True)

        row = QHBoxLayout()
        row.addWidget(self.cameras, 1)
        row.addWidget(self.start)
        box = QVBoxLayout(self)
        box.setContentsMargins(16, 16, 16, 16)
        box.setSpacing(12)
        box.addWidget(self.stage)
        box.addLayout(row)
        box.addWidget(self.status)
        self.setFixedWidth(self.STAGE_W + 32)

        self.show_idle()
        if self.cameras.count() == 0:
            self.say("No camera found. Plug one in and reopen Facecam.", True)
        else:
            self.say(self.ready_text() if self.has_face else "Pick a face, then press Start")
        self.sync()

    # ── state ──
    def sync(self):
        live = self.worker is not None
        self.start.setText("Stop" if live else "Start")
        self.start.setEnabled(live or (self.has_face and self.cameras.count() > 0))
        self.cameras.setEnabled(not live)

    def ready_text(self):
        n = len(self.thumbs)
        return "Ready. Add more photos of them for a closer match." if n == 1 else f"Ready. Blending {n} photos."

    def say(self, text, error=False):
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {ERROR_RED};" if error else "color: palette(placeholder-text);")

    def strip(self, live):
        return self.thumbs if live or len(self.thumbs) > 1 else ()

    def show_idle(self, hover=False):
        self.clear.setVisible(self.has_face and not hover)
        if self.has_face and not hover:
            self.stage.setStyleSheet("border: none;")
            self.stage.setPixmap(rounded(self.thumbs[-1], self.STAGE_W, self.STAGE_H, thumbs=self.strip(False)))
        else:
            self.stage.setPixmap(QPixmap())
            edge = "2px solid palette(highlight)" if hover else "1px dashed palette(mid)"
            self.stage.setStyleSheet(f"border: {edge}; border-radius: 10px;")
            self.stage.setText(f"<b style='font-size:15px'>{'Drop to add this photo' if self.has_face else 'Drop to use this face'}</b>"
                               if hover else
                               "<b style='font-size:15px'>Drop a face photo</b><br>"
                               "<span style='color:palette(placeholder-text)'>"
                               "Drag one from your browser, or click to choose.<br>"
                               "A few photos of the same person match closer.</span>")

    # ── face photo ──
    def choose(self):
        if self.worker:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose photos of the same person", "",
                                                "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        for path in paths:
            self.use(QImage(path))

    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasImage() or md.hasUrls() or md.hasHtml():
            e.acceptProposedAction()
            if not self.worker:
                self.show_idle(hover=True)

    def dragLeaveEvent(self, e):
        if not self.worker:
            self.show_idle()

    def dropEvent(self, e):
        md = e.mimeData()
        img = QImage(md.imageData()) if md.hasImage() else QImage()
        local = [u.toString() for u in md.urls() if u.isLocalFile()]
        if img.isNull() and len(local) > 1:  # several files from Explorer
            for u in local:
                self.use(self.fetch(u))
            return
        if img.isNull():
            # Browsers mostly hand over the picture's address, sometimes only as HTML.
            url = md.urls()[0].toString() if md.hasUrls() else ""
            m = re.search(r'<img[^>]+src="([^"]+)"', md.html()) if md.hasHtml() else None
            if m and not re.search(r"\.(png|jpe?g|webp|gif|bmp)|^data:", url, re.I):
                url = m.group(1).replace("&amp;", "&")
            img = self.fetch(url)
        self.use(img)

    def fetch(self, url):
        q = QUrl(url)
        if q.isLocalFile():
            path = q.toLocalFile()
            # A network-share path (file://host/..., \\host\..., mixed slashes) would
            # make Windows send the user's login hash to that host, so only accept
            # drive-letter paths like C:\...
            if q.host() or not re.match(r"[A-Za-z]:[\\/]", path):
                return QImage()
            return QImage(path)
        if q.scheme() not in ("http", "https", "data"):
            return QImage()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return QImage.fromData(urllib.request.urlopen(req, timeout=10).read(20_000_000))
        except Exception:
            return QImage()

    def use(self, img: QImage):
        if img.isNull():
            if not self.worker:
                self.show_idle()
            return self.say("That wasn't an image. Try dragging the picture itself.", True)
        os.makedirs(FACES, exist_ok=True)
        img.save(os.path.join(FACES, f"{time.time_ns()}.png"), "PNG")  # one format OpenCV always reads
        for old in face_files()[:-MAX_FACES]:
            os.remove(old)
        self.thumbs = [QPixmap(f) for f in face_files()]
        self.has_face = True
        if self.worker:
            self.worker.reload_face = True
        else:
            self.show_idle()
            self.say(self.ready_text())
        self.sync()

    def clear_faces(self):
        for f in face_files():
            os.remove(f)
        self.thumbs, self.has_face = [], False
        if not self.worker:
            self.show_idle()
            self.say("Pick a face, then press Start")
        self.sync()

    # ── run ──
    def toggle(self):
        if self.worker:
            self.worker.running = False
            return
        self.worker = Worker(self.cameras.currentData())
        self.worker.frame.connect(lambda img: self.stage.setPixmap(
            rounded(QPixmap.fromImage(img), self.STAGE_W, self.STAGE_H, thumbs=self.strip(True))))
        self.worker.status.connect(self.say)
        self.worker.finished.connect(self.ended)
        self.stage.setStyleSheet("border: none;")
        self.worker.start()
        self.sync()

    def ended(self):
        error = self.status.styleSheet().startswith(f"color: {ERROR_RED}")
        self.worker = None
        self.show_idle()
        if not error:
            self.say(self.ready_text() if self.has_face else "Pick a face, then press Start")
        self.sync()

    def closeEvent(self, e):
        if self.worker:
            self.worker.running = False
            self.worker.wait(3000)
        e.accept()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        app = QApplication(sys.argv)
        w = Window()
        w.show()
        sys.exit(app.exec())
