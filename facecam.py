"""Facecam for Windows.

Drop a face photo (or a few of the same person, which are blended for a
closer match), pick your camera, press Start. The swapped video goes to
OBS Virtual Camera, which Discord (or Zoom, Meet...) can pick as a camera.

    python facecam.py             open the window
    python facecam.py --selftest  swap two demo frames, no camera needed
    python facecam.py --prepare   download the face models unless they're already there
"""
import glob
import math
import os
import re
import shutil
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "engine")  # Deep-Live-Cam, unpacked by install.bat
FACES = os.path.join(HERE, "faces")  # photos of one person, blended; newest last
MAX_FACES = 5
PRESETS = os.path.join(HERE, "presets")  # saved photo sets, one folder per name

# The Start menu shortcut runs pythonw.exe (no console), which sets sys.stdout and
# sys.stderr to None. Anything that writes to them directly, like tqdm's bar
# while the swap model downloads, then dies with "'NoneType' object has no
# attribute 'write'". Send that output to a log file instead, before the engine loads.
LOG = os.path.join(HERE, "facecam.log")
if sys.stdout is None or sys.stderr is None:
    _log = open(LOG, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log
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
    f = f if m >= 90 else cv2.convertScaleAbs(f, alpha=min(8.0, 110 / max(m, 1.0)))
    # A light blur takes out sensor grain, which otherwise shakes the landmarks
    # and, in very dark rooms, stops detection altogether.
    return cv2.GaussianBlur(f, (5, 5), 0)


class Steady:
    """One Euro filter over the 5 face landmarks the swap aligns to. Sensor noise
    in low light makes them shake a few pixels a frame; this smooths hard while
    the head is still and backs off as it moves, so turns don't lag."""
    MIN_CUTOFF = 0.5  # lower: steadier when still
    BETA = 0.05       # higher: less lag on fast moves

    def __init__(self):
        self.x = None

    def __call__(self, x):
        now = time.monotonic()
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), now
            return x
        dt, self.t = max(now - self.t, 1e-3), now
        a = lambda cutoff: 1 / (1 + 1 / (2 * math.pi * cutoff * dt))  # noqa: E731
        self.dx = self.dx + a(1.0) * ((x - self.x) / dt - self.dx)
        self.x = self.x + a(self.MIN_CUTOFF + self.BETA * np.abs(self.dx)) * (x - self.x)
        return self.x

    def reset(self):
        self.x = None


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


def preset_names():
    return sorted(d for d in os.listdir(PRESETS) if os.path.isdir(os.path.join(PRESETS, d))) \
        if os.path.isdir(PRESETS) else []


def preset_dir(name):
    """Folder for a preset name, or None if nothing usable is left after
    stripping the characters Windows forbids (which also rules out ../ tricks)."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", name).strip().strip(".")[:40].strip()
    return os.path.join(PRESETS, name) if name else None


def save_preset(name):
    dest = preset_dir(name)
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    for f in face_files():
        shutil.copy2(f, dest)
    return os.path.basename(dest)


def load_preset(name):
    for f in face_files():
        os.remove(f)
    os.makedirs(FACES, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(PRESETS, name, "*.png")))[-MAX_FACES:]:
        shutil.copy2(f, FACES)


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


def models_present():
    """Swap model (engine/models) and the buffalo_l detector (~/.insightface)."""
    swap = any(os.path.exists(os.path.join(ENGINE, "models", n))
               for n in ("inswapper_128.onnx", "inswapper_128_fp16.onnx"))
    det = os.path.expanduser(os.path.join("~", ".insightface", "models", "buffalo_l"))
    return swap and all(os.path.exists(os.path.join(det, n)) for n in ("det_10g.onnx", "w600k_r50.onnx"))


def prepare():
    """Installer step: fetch the models now, so the first Start doesn't wait on them."""
    if models_present():
        print("Face models already installed.")
        return
    os.environ["FACECAM_CPU"] = "1"  # only downloading; no need to wake the GPU
    load()
    assert models_present(), "models still missing after download"
    print("Face models installed.")


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

    st = Steady()  # still noisy points settle, a real move is followed
    rng = np.random.default_rng(0)
    pts = [st(np.full((5, 2), 100.0) + rng.normal(0, 3, (5, 2))) for _ in range(60)]
    assert np.std(pts[30:], axis=0).mean() < 1.0, "Steady didn't smooth"
    for _ in range(30):
        time.sleep(0.01)
        moved = st(np.full((5, 2), 200.0))
    assert np.abs(moved - 200).max() < 5, "Steady lags a real move"
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
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap  # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout,  # noqa: E402
                               QInputDialog, QLabel, QMenu, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

ASSETS = os.path.join(HERE, "assets").replace("\\", "/")  # Qt style sheets want forward slashes

# Dark charcoal with one signal-orange accent. Facecam runs next to Discord,
# usually in a dim room, so it stays dark whatever the system theme is.
BG, STAGE, SURFACE, HOVER, STROKE = "#1b1b1c", "#121213", "#262628", "#303033", "#3a3a3e"
TEXT, MUTED = "#f2f2f3", "#a4a4aa"
ACCENT, ACCENT_HOVER, ON_ACCENT = "#ff6a1a", "#ff8240", "#170a03"
ERROR_RED, LIVE_RED = "#ff7b72", "#ff453a"

STYLE = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-family: "Segoe UI Variable Text", "Segoe UI"; font-size: 10pt; }}
QPushButton, QComboBox {{ background: {SURFACE}; border: 1px solid {STROKE}; border-radius: 6px;
    padding: 6px 12px; min-height: 20px; }}
QPushButton:hover, QComboBox:hover {{ background: {HOVER}; }}
QPushButton:pressed {{ background: #232325; }}
QPushButton:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QPushButton:disabled, QComboBox:disabled {{ color: #6e6e74; background: #212123; border-color: #2c2c2f; }}
QPushButton#primary {{ background: {ACCENT}; color: {ON_ACCENT}; border: 1px solid {ACCENT}; font-weight: 600; }}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#primary:focus {{ border-color: {TEXT}; }}
QPushButton#primary:disabled {{ background: #3b2a21; color: #8d7466; border-color: #3b2a21; }}
QPushButton#primary[live="true"] {{ background: #3a1d1c; color: #ffb3ab; border-color: #5e2b28; }}
QPushButton#primary[live="true"]:hover {{ background: #4a2322; }}
QPushButton#presets {{ padding-right: 28px; }}
QPushButton::menu-indicator {{ image: url({ASSETS}/chevron.svg); width: 12px; height: 12px;
    subcontrol-origin: padding; subcontrol-position: right center; right: 10px; }}
QComboBox {{ padding-right: 28px; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{ image: url({ASSETS}/chevron.svg); width: 12px; height: 12px; }}
QComboBox QAbstractItemView, QMenu {{ background: {SURFACE}; border: 1px solid {STROKE}; padding: 4px; outline: 0;
    selection-background-color: {HOVER}; selection-color: {TEXT}; }}
QMenu::item {{ padding: 6px 28px 6px 10px; border-radius: 4px; }}
QMenu::item:selected {{ background: {HOVER}; }}
QMenu::item:disabled {{ color: #6e6e74; }}
QMenu::separator {{ height: 1px; background: {STROKE}; margin: 4px 6px; }}
QToolTip {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {STROKE}; padding: 4px 6px; }}
QPushButton#clear {{ background: rgba(0,0,0,150); color: white; border: none; border-radius: 11px;
    padding: 3px 10px; min-height: 16px; font-weight: 600; }}
QPushButton#clear:hover {{ background: rgba(0,0,0,210); }}
QLabel#dot {{ background: {LIVE_RED}; border-radius: 4px; }}
"""


def dark_title_bar(window):
    """Match the native title bar to the window (Windows 10 2004+ / 11)."""
    if sys.platform != "win32":
        return
    import ctypes
    hwnd, dwm = int(window.winId()), ctypes.windll.dwmapi
    dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)), 4)  # immersive dark mode
    color = int(BG[5:7] + BG[3:5] + BG[1:3], 16)  # COLORREF is 0x00BBGGRR
    dwm.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(ctypes.c_int(color)), 4)  # caption colour, 11 only


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
            print(f"Stopped: {e!r}", file=sys.stderr)
            self.status.emit(f"Stopped: {e}. Details are in facecam.log.", True)

    def _run(self):
        import pyvirtualcam
        from modules.video_capture import VideoCapturer
        self.status.emit("Loading models…" if models_present() else
                         "Downloading the face models (about 800 MB, first start only)…", False)
        get_one_face, fs = load()
        cap = VideoCapturer(self.camera_index)
        if not cap.start(1280, 720, 30):
            return self.status.emit(camera_busy_message(), True)
        try:
            ok, f = cap.read()
            if not ok:
                return self.status.emit("Camera opened but sent no picture.", True)
            h, w = f.shape[:2]
            try:
                vcam = pyvirtualcam.Camera(w, h, 30, fmt=pyvirtualcam.PixelFormat.BGR, backend="obs")
            except Exception:
                return self.status.emit("OBS Virtual Camera not found. Install OBS Studio, then press Start again.", True)
            src, last, missed, steady = None, None, 0, Steady()
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
                            target.kps = steady(target.kps)
                            last, missed = target, 0
                        elif missed < HOLD_FRAMES:
                            target, missed = last, missed + 1
                        else:
                            steady.reset()  # face gone; don't glide in from where it was
                        if target is not None:
                            f = fs.process_frame(src, f, target)
                    vcam.send(f)
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    self.frame.emit(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())
        finally:
            cap.release()


def camera_busy_message():
    """Name the app holding the webcam when Windows knows (usually Discord)."""
    try:
        from tools.camera_check import who_has_webcam
        held = who_has_webcam()
        app = held[0][1].removesuffix(".exe") if held else None
    except Exception:
        app = None
    if app:
        return (f"Camera didn't open: {app} is using it. Quit {app}, press Start, "
                f"then set {app}'s camera to OBS Virtual Camera.")
    return "Camera didn't open. Another app may be using it. Close that app and press Start again."


def rounded(pix: QPixmap, w: int, h: int, r: int = 8, thumbs=(), dim=False) -> QPixmap:
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
    if dim:
        p.fillRect(0, 0, w, h, QColor(0, 0, 0, 150))
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
        self.preset = None  # name of the preset on screen, until the photos change
        self.setWindowTitle("Facecam")
        self.setWindowIcon(QIcon(os.path.join(HERE, "assets", "facecam.ico")))
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setAcceptDrops(True)

        self.stage = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.stage.setFixedSize(self.STAGE_W, self.STAGE_H)
        self.stage.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stage.mousePressEvent = lambda e: self.choose()
        self.clear = QPushButton("Clear", self.stage, objectName="clear")
        self.clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear.setToolTip("Remove all photos to start on a different person")
        self.clear.ensurePolished()
        self.clear.adjustSize()
        self.clear.move(self.STAGE_W - self.clear.width() - 10, 10)
        self.clear.clicked.connect(self.clear_faces)

        self.presets = QPushButton("Presets", objectName="presets")
        self.presets.setToolTip("Save these photos as a preset, or switch to a saved one")
        self.presets_menu = QMenu(self)
        self.presets_menu.aboutToShow.connect(self.fill_presets_menu)
        self.presets.setMenu(self.presets_menu)

        self.cameras = QComboBox()
        for i, name in list_cameras():
            self.cameras.addItem(name, i)
        self.start = QPushButton("Start", objectName="primary")
        self.start.setDefault(True)
        self.start.setMinimumWidth(96)
        self.start.clicked.connect(self.toggle)

        self.status = QLabel(wordWrap=True)
        self.dot = QLabel(objectName="dot")  # live indicator
        self.dot.setFixedSize(8, 8)
        self.dot.hide()

        row = QHBoxLayout()
        row.addWidget(self.presets)
        row.addWidget(self.cameras, 1)
        row.addWidget(self.start)
        box = QVBoxLayout(self)
        box.setContentsMargins(16, 16, 16, 16)
        box.setSpacing(12)
        box.addWidget(self.stage)
        box.addLayout(row)
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        status_row.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignVCenter)
        status_row.addWidget(self.status, 1)
        box.addLayout(status_row)
        self.setFixedWidth(self.STAGE_W + 32)

        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setFocus()  # no control starts with a focus ring; Tab still reaches them all
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
        self.start.setProperty("live", live)
        self.start.style().unpolish(self.start)
        self.start.style().polish(self.start)
        self.start.setEnabled(live or (self.has_face and self.cameras.count() > 0))
        self.cameras.setEnabled(not live)

    # ── presets ──
    def fill_presets_menu(self):
        m = self.presets_menu
        m.clear()
        names = preset_names()
        for name in names:
            first = sorted(glob.glob(os.path.join(PRESETS, name, "*.png")))
            a = m.addAction(QIcon(QPixmap(first[0])) if first else QIcon(), name,
                            lambda n=name: self.load_preset(n))
            a.setCheckable(True)
            a.setChecked(name == self.preset)
        if not names:
            m.addAction("No presets yet").setEnabled(False)
        m.addSeparator()
        m.addAction("Save These Photos as a Preset…", self.ask_save_preset).setEnabled(self.has_face)
        if names:
            delete = m.addMenu("Delete")
            for name in names:
                delete.addAction(name, lambda n=name: self.delete_preset(n))

    def ask_save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Name this face:", text=self.preset or "")
        if not ok or not preset_dir(name):
            return
        if os.path.basename(preset_dir(name)) in preset_names() and QMessageBox.question(
                self, "Replace Preset?", f"A preset named \u201c{os.path.basename(preset_dir(name))}\u201d "
                "already exists. Replace its photos with these?") != QMessageBox.StandardButton.Yes:
            return
        self.save_preset(name)

    def save_preset(self, name):
        try:
            self.set_preset(save_preset(name))
            self.say(f"Saved \u201c{self.preset}\u201d.")
        except OSError as e:
            self.say(f"Couldn't save that preset: {e.strerror}. Try another name.", True)

    def load_preset(self, name):
        load_preset(name)
        self.thumbs = [QPixmap(f) for f in face_files()]
        self.has_face = bool(self.thumbs)
        self.set_preset(name)
        if self.worker:
            self.worker.reload_face = True
        else:
            self.show_idle()
            self.say(self.ready_text() if self.has_face else "That preset has no photos.", not self.has_face)
        self.sync()

    def delete_preset(self, name):
        shutil.rmtree(os.path.join(PRESETS, name), ignore_errors=True)
        if name == self.preset:
            self.set_preset(None)

    def set_preset(self, name):
        self.preset = name
        self.presets.setText(name or "Presets")

    def ready_text(self):
        n = len(self.thumbs)
        return "Ready. Add more photos of them for a closer match." if n == 1 else f"Ready. Blending {n} photos."

    def say(self, text, error=False):
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {ERROR_RED};" if error else f"color: {MUTED};")
        self.dot.setVisible(text.startswith("Live"))

    def strip(self, live):
        return self.thumbs if live or len(self.thumbs) > 1 else ()

    def show_idle(self, hover=False):
        self.clear.setVisible(self.has_face and not hover)
        if self.has_face and not hover:
            self.stage.setStyleSheet(f"border: none; background: {BG};")
            self.stage.setPixmap(rounded(self.thumbs[-1], self.STAGE_W, self.STAGE_H, thumbs=self.strip(False)))
        else:
            self.stage.setPixmap(QPixmap())
            edge = f"2px solid {ACCENT}" if hover else f"1px dashed {STROKE}"
            self.stage.setStyleSheet(f"border: {edge}; border-radius: 8px; background: {STAGE};")
            self.stage.setText(
                f"<p style='font-size:12pt; font-weight:600'>"
                f"{'Drop to add this photo' if self.has_face else 'Drop to use this face'}</p>" if hover else
                f"<img src='{ASSETS}/face.svg' width='44' height='44'>"
                f"<p style='font-size:12pt; font-weight:600; margin-top:10px'>Drop a face photo</p>"
                f"<p style='color:{MUTED}; margin-top:4px'>Drag one from your browser, or click to choose.<br>"
                f"A few photos of the same person match closer.</p>")

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
        self.set_preset(None)
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
        self.set_preset(None)
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
        self.stage.setStyleSheet(f"border: none; background: {BG};")
        if self.thumbs:  # dim the photo until the first live frame replaces it
            self.stage.setPixmap(rounded(self.thumbs[-1], self.STAGE_W, self.STAGE_H, thumbs=self.strip(True), dim=True))
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
    elif "--prepare" in sys.argv:
        prepare()
    else:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        app.setStyleSheet(STYLE)
        w = Window()
        w.show()
        dark_title_bar(w)
        sys.exit(app.exec())
