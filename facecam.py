"""Facecam for Windows.

Drop a face photo, pick your camera, press Start. The swapped video goes to
OBS Virtual Camera, which Discord (or Zoom, Meet...) can pick as a camera.

    python facecam.py             open the window
    python facecam.py --selftest  swap two demo frames, no camera needed
"""
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "engine")  # Deep-Live-Cam, unpacked by install.bat
FACE = os.path.join(HERE, "face.png")
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)

import cv2  # noqa: E402
import onnxruntime  # noqa: E402


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
    return get_one_face, fs


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
    print("selftest ok")


# ─── UI ────────────────────────────────────────────────────────────────────

from PySide6.QtCore import Qt, QThread, QUrl, Signal  # noqa: E402
from PySide6.QtGui import QImage, QPixmap, QPainter, QPainterPath  # noqa: E402
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
            src = None
            with vcam:
                while self.running:
                    if self.reload_face:
                        self.reload_face = False
                        img = cv2.imread(FACE)
                        src = get_one_face(img) if img is not None else None
                        if src is None:
                            self.status.emit("No face found in that photo. Try a clear, front-facing one.", True)
                        else:
                            self.status.emit("Live. In Discord, pick OBS Virtual Camera.", False)
                    ok, f = cap.read()
                    if not ok:
                        continue
                    if src is not None:
                        target = get_one_face(f)
                        if target is not None:
                            f = fs.process_frame(src, f, target)
                    vcam.send(f)
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    self.frame.emit(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())
        finally:
            cap.release()


def rounded(pix: QPixmap, w: int, h: int, r: int = 10) -> QPixmap:
    """Scale-to-fill and clip to a rounded rect."""
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
    p.end()
    return out


class Window(QWidget):
    STAGE_W, STAGE_H = 480, 270

    def __init__(self):
        super().__init__()
        self.worker = None
        self.has_face = os.path.exists(FACE)
        self.setWindowTitle("Facecam")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setAcceptDrops(True)

        self.stage = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.stage.setFixedSize(self.STAGE_W, self.STAGE_H)
        self.stage.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stage.mousePressEvent = lambda e: self.choose()

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
            self.say("Ready" if self.has_face else "Pick a face, then press Start")
        self.sync()

    # ── state ──
    def sync(self):
        live = self.worker is not None
        self.start.setText("Stop" if live else "Start")
        self.start.setEnabled(live or (self.has_face and self.cameras.count() > 0))
        self.cameras.setEnabled(not live)

    def say(self, text, error=False):
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {ERROR_RED};" if error else "color: palette(placeholder-text);")

    def show_idle(self, hover=False):
        if self.has_face and not hover:
            self.stage.setStyleSheet("border: none;")
            self.stage.setPixmap(rounded(QPixmap(FACE), self.STAGE_W, self.STAGE_H))
        else:
            self.stage.setPixmap(QPixmap())
            edge = "2px solid palette(highlight)" if hover else "1px dashed palette(mid)"
            self.stage.setStyleSheet(f"border: {edge}; border-radius: 10px;")
            self.stage.setText("<b style='font-size:15px'>Drop to use this face</b>" if hover else
                               "<b style='font-size:15px'>Drop a face photo</b><br>"
                               "<span style='color:palette(placeholder-text)'>"
                               "Drag one from your browser, or click to choose</span>")

    # ── face photo ──
    def choose(self):
        if self.worker:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose a face photo", "",
                                              "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
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
            # A network-share path (file://host/..., \\host\...) would make Windows
            # send the user's login hash to that host, so only plain local paths.
            if q.host() or path.startswith(("\\\\", "//")):
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
            self.show_idle()
            return self.say("That wasn't an image. Try dragging the picture itself.", True)
        img.save(FACE, "PNG")  # one format OpenCV always reads
        self.has_face = True
        if self.worker:
            self.worker.reload_face = True
        else:
            self.show_idle()
            self.say("Ready")
        self.sync()

    # ── run ──
    def toggle(self):
        if self.worker:
            self.worker.running = False
            return
        self.worker = Worker(self.cameras.currentData())
        self.worker.frame.connect(lambda img: self.stage.setPixmap(
            rounded(QPixmap.fromImage(img), self.STAGE_W, self.STAGE_H)))
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
            self.say("Ready")
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
