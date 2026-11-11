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
import threading
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


ORANGE = (26, 106, 255)  # BGR #ff6a1a, the app's accent
SCAN_SECONDS = 0.9
OVERLAY = 0.35  # opacity of the orange outline and sweep line
GLOW = 1.6      # strength of the soft light around them
STEP = 1 / 15   # most a scan advances per frame, so slow machines still show all of it


def ease(p):
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def add_outline_points(frame, face):
    """The engine skips the 106-point landmarks to save time; the scan needs them
    for the face outline, so fetch them only while a scan runs."""
    if face is None or getattr(face, "landmark_2d_106", None) is not None:
        return
    from modules.face_analyser import get_face_analyser
    model = get_face_analyser().models.get("landmark_2d_106")
    if model is not None:
        model.get(frame, face)


def face_shape(face, w, h):
    """Closed outline of the face: the hull of the 106 landmarks, raised over the
    forehead (the landmarks stop at the brows). No face: the whole frame."""
    pts = getattr(face, "landmark_2d_106", None) if face is not None else None
    if pts is None:
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.int32)
    pts = pts.astype(np.float32)
    top, chin = pts[:, 1].min(), pts[:, 1].max()
    # The landmarks stop at the brows, so add a dome over the forehead.
    cx, half, height = pts[:, 0].mean(), (pts[:, 0].max() - pts[:, 0].min()) / 2, chin - top
    u = np.linspace(-1, 1, 15)
    dome = np.stack([cx + u * half * 0.9, top + 0.1 * height - 0.42 * height * np.sqrt(1 - u ** 2)], axis=1)
    hull = cv2.convexHull(np.vstack([pts, dome]).astype(np.float32)).reshape(-1, 2)
    for _ in range(3):  # Chaikin: round off the hull's corners
        nxt = np.roll(hull, -1, axis=0)
        hull = np.stack([0.75 * hull + 0.25 * nxt, 0.25 * hull + 0.75 * nxt], axis=1).reshape(-1, 2)
    hull[:, 0] = hull[:, 0].clip(0, w - 1)
    hull[:, 1] = hull[:, 1].clip(0, h - 1)
    return hull.astype(np.int32)


def trace(img, shape, frac, thick):
    """Draw the outline from the forehead down both sides, `frac` of the way to the chin."""
    start = int(np.argmin(shape[:, 1]))
    ring = np.roll(shape, -start, axis=0)
    for path in (np.vstack([ring, ring[:1]]), np.vstack([ring[:1], ring[::-1]])):
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        left = frac * seg.sum() / 2
        for (p0, p1), d in zip(zip(path[:-1], path[1:]), seg):
            if left <= 0:
                break
            end = p1 if left >= d else p0 + (p1 - p0) * (left / d)
            cv2.line(img, tuple(map(int, p0)), tuple(map(int, end)), ORANGE, thick, cv2.LINE_AA)
            left -= d


def scan(a, b, face, p, up=False):
    """One frame of the scan: an orange outline traces the face, then a line sweeps
    down it (up, when turning off), leaving face b behind it inside the outline.
    The rest of the frame eases over to b at the very end so nothing pops.
    Only the area around the face is worked on, so it stays cheap at 720p."""
    h, w = a.shape[:2]
    shape = face_shape(face, w, h)
    x0, y0 = shape.min(axis=0)
    x1, y1 = shape.max(axis=0)
    size = max(x1 - x0, y1 - y0, 1)
    k = int(size * 0.06) | 1               # mask softening
    kg = int(size * 0.2) | 1               # glow spread
    feather = max(6.0, size * 0.03)
    pad = feather + k                      # the softened mask reaches k past the outline
    e = ease(min(max(p, 0.0), 1.0))
    reach = (y1 - y0) + 2 * pad            # start and end a feather past the outline
    line = int(y1 + pad - e * reach) if up else int(y0 - pad + e * reach)
    rest = min(max((p - 0.85) / 0.15, 0.0), 1.0)  # outside the face, at the very end
    out = cv2.addWeighted(a, 1 - rest, b, rest, 0) if rest > 0 else a.copy()

    # Work area: the face plus room for the soft edge and the glow.
    m_ = int(pad + kg)
    rx0, ry0 = max(0, int(x0) - m_), max(0, int(y0) - m_)
    rx1, ry1 = min(w, int(x1) + m_ + 1), min(h, int(y1) + m_ + 1)
    aR, bR = a[ry0:ry1, rx0:rx1].astype(np.float32), b[ry0:ry1, rx0:rx1].astype(np.float32)
    rh, rw = aR.shape[:2]
    local = shape - [rx0, ry0]
    mask = np.zeros((rh, rw), np.uint8)
    cv2.fillPoly(mask, [local], 255)
    soft = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32)[..., None] / 255
    rows = np.arange(ry0, ry1, dtype=np.float32)[:, None, None]
    behind = np.clip(((line - rows) if not up else (rows - line)) / feather + 0.5, 0, 1)
    o = aR + (bR - aR) * np.maximum(behind * soft, rest)

    thick = max(2, int(size / 90))
    draw = min(p / 0.25, 1.0)          # outline traces on over the first quarter
    fade = min(1.0, (1 - p) / 0.2)     # everything lets go over the last fifth
    stroke = np.zeros((rh, rw, 3), np.uint8)  # the orange marks, drawn on black
    if fade > 0 and 0 < e < 1:         # the sweep line, only across the face
        ly = line - ry0
        cv2.line(stroke, (int(x0) - rx0, ly), (int(x1) - rx0, ly), ORANGE, thick + 1, cv2.LINE_AA)
        stroke[cv2.dilate(mask, np.ones((3, 3), np.uint8)) == 0] = 0
    if fade > 0 and draw > 0:
        trace(stroke, local, ease(draw), thick + 1)
    if stroke.any():
        # Blur the glow at quarter size: same look, a fraction of the work.
        small = cv2.resize(stroke, (max(1, rw // 4), max(1, rh // 4)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, ((kg // 4) | 1, (kg // 4) | 1), 0)
        glow = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        o += glow * GLOW * fade        # light adds; dark rooms glow most
        st = stroke.astype(np.float32)
        al = st.max(axis=2, keepdims=True) / 255 * OVERLAY * fade
        o = o * (1 - al) + st * al
    out[ry0:ry1, rx0:rx1] = np.clip(o, 0, 255).astype(np.uint8)
    return out
    stroke = np.zeros_like(out)        # the orange marks, drawn on black
    if 0 < e < 1:                      # the sweep line, only across the face
        cv2.line(stroke, (int(x0), line), (int(x1), line), ORANGE, thick + 1, cv2.LINE_AA)
        stroke[cv2.dilate(mask, np.ones((3, 3), np.uint8)) == 0] = 0
    if draw > 0:
        trace(stroke, shape, ease(draw), thick + 1)
    if stroke.any():
        k = int(size * 0.2) | 1
        glow = cv2.GaussianBlur(stroke, (k, k), 0).astype(np.float32)
        o = out.astype(np.float32) + glow * GLOW * fade  # light adds; dark rooms glow most
        s = stroke.astype(np.float32)
        a_ = s.max(axis=2, keepdims=True) / 255 * OVERLAY * fade
        out = np.clip(o * (1 - a_) + s * a_, 0, 255).astype(np.uint8)
    return out


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


def remove_face(path):
    """Delete a face photo. Windows refuses while the camera thread has it open to
    read (a few milliseconds), so retry briefly instead of failing."""
    for _ in range(40):
        try:
            return os.remove(path)
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(0.025)
    os.remove(path)


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
        remove_face(f)
    os.makedirs(FACES, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(PRESETS, name, "*.png")))[-MAX_FACES:]:
        # The camera thread may be reading FACES; it must never see half a file.
        tmp = os.path.join(FACES, os.path.basename(f) + ".part")
        shutil.copy2(f, tmp)
        os.replace(tmp, tmp[:-5])


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
    # Scan: starts on the old frame, ends on the swap everywhere, touches only the face midway.
    add_outline_points(tgt, face)
    assert face.landmark_2d_106 is not None, "no outline points"
    for up in (False, True):
        assert (scan(tgt, out, face, 0.0, up) == tgt).all(), "scan didn't start on the old frame"
        assert (scan(tgt, out, face, 1.0, up) == out).all(), "scan didn't land on the swap"
    x0, x1 = int(face.bbox[0]), int(face.bbox[2])
    away = max(1, x0 - (x1 - x0))
    assert (scan(tgt, out, face, 0.5)[:, :away] == tgt[:, :away]).all(), "mid-scan touched pixels outside the face"

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
    """Holds the webcam from launch and feeds OBS Virtual Camera. You go out as you
    are until `on` is set; then the face scans in (and back out when it's cleared)."""
    frame = Signal(object)  # RGB numpy frame; the QImage is made on the UI thread
    status = Signal(str, bool)  # message, is_error
    mode = Signal(str)          # "you" or "swapping", once frames are flowing

    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.on = False
        self.reload_face = True
        self.ready = False  # models loaded

    def run(self):
        try:
            self._run()
        except Exception as e:  # anything here would otherwise vanish with the thread
            print(f"Stopped: {e!r}", file=sys.stderr)
            self.status.emit(f"Stopped: {e}. Details are in facecam.log.", True)

    def _load(self):
        try:
            self.fns = load()
            self.ready = True
        except Exception as e:
            print(f"Models: {e!r}", file=sys.stderr)
            self.status.emit(f"Couldn't load the face models: {e}", True)

    def _run(self):
        import pyvirtualcam
        from modules.video_capture import VideoCapturer
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
            # Frames go out from here on; the models load alongside so Start is quick.
            threading.Thread(target=self._load, daemon=True).start()
            src, shown, trans, was_on = None, None, None, False  # trans: [from, to, progress, upward]
            tick = time.monotonic()
            last, missed, steady = None, 0, Steady()
            with vcam:
                self.mode.emit("you")
                while self.running:
                    if self.ready and self.reload_face:
                        self.reload_face = False
                        new = blended_face(self.fns[0])
                        if new is None:
                            if self.on:
                                self.status.emit("No face found in those photos. Try clear, front-facing ones.", True)
                        else:
                            if shown is not None:  # wearing the old face: scan across to the new one
                                trans, shown = [shown, new, 0.0, False], new
                            src = new
                    if self.ready and self.on != was_on:
                        if self.on and src is None:
                            self.on = False
                            self.status.emit("No face found in those photos. Try clear, front-facing ones.", True)
                            self.mode.emit("you")
                        elif self.on:
                            trans, shown = [shown, src, 0.0, False], src
                            self.mode.emit("swapping")
                        elif shown is not None:
                            trans, shown = [shown, None, 0.0, True], None
                            self.mode.emit("you")
                        was_on = self.on
                    ok, f = cap.read()
                    if not ok:
                        continue
                    now = time.monotonic()
                    dt, tick = min(now - tick, STEP), now
                    if trans:
                        trans[2] += dt / SCAN_SECONDS
                    p = trans[2] if trans else 1.0
                    if p >= 1:
                        trans = None
                    faces = (trans[0], trans[1]) if trans else (shown,)
                    target = None
                    if any(x is not None for x in faces):
                        get_one_face, fs = self.fns
                        target = get_one_face(lift(f))
                        if target is not None:
                            target.kps = steady(target.kps)
                            last, missed = target, 0
                        elif missed < HOLD_FRAMES:
                            target, missed = last, missed + 1
                        else:
                            steady.reset()  # face gone; don't glide in from where it was

                    def look(face, f=f, t=target):
                        # process_frame writes into the frame it's given, so hand it a copy.
                        return f if face is None or t is None else self.fns[1].process_frame(face, f.copy(), t)

                    if trans:
                        add_outline_points(f, target)
                        out = scan(look(trans[0]), look(trans[1]), target, p, trans[3])
                    else:
                        out = look(shown)
                    vcam.send(out)
                    # Plain array, not a QImage: Qt objects made on this thread and freed
                    # on the UI thread crash PySide's cleanup under load.
                    self.frame.emit(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
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
        self.live = False      # worker is sending frames
        self.swapping = False  # wearing the face (Start pressed)
        self.reopen = False    # camera changed: start a new worker when this one ends
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
        self.cameras.currentIndexChanged.connect(self.switch_camera)
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
            self.open_camera()  # you go out on the virtual camera from the start
        self.sync()

    # ── state ──
    def sync(self):
        self.start.setText("Stop" if self.swapping else "Start")
        self.start.setProperty("live", self.swapping)
        self.start.style().unpolish(self.start)
        self.start.style().polish(self.start)
        self.start.setEnabled(self.swapping or (self.has_face and self.cameras.count() > 0))
        self.start.setToolTip("Scan back to your own face" if self.swapping else "Scan their face onto yours")

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
        if not self.live:
            self.show_idle()
        if not self.swapping:
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
        if self.live:
            return ("Camera on, showing you. Drop a photo to wear their face." if n == 0 else
                    "Showing you. Press Start to switch faces." if n == 1 else
                    f"Showing you. Press Start to wear a blend of {n} photos.")
        if n == 0:
            return "Pick a face, then press Start"
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
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose photos of the same person", "",
                                                "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        for path in paths:
            self.use(QImage(path))

    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasImage() or md.hasUrls() or md.hasHtml():
            e.acceptProposedAction()
            if not self.live:
                self.show_idle(hover=True)

    def dragLeaveEvent(self, e):
        if not self.live:
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
            if not self.live:
                self.show_idle()
            return self.say("That wasn't an image. Try dragging the picture itself.", True)
        os.makedirs(FACES, exist_ok=True)
        path = os.path.join(FACES, f"{time.time_ns()}.png")
        img.save(path + ".part", "PNG")  # PNG: one format OpenCV always reads
        os.replace(path + ".part", path)  # the camera thread never sees half a file
        for old in face_files()[:-MAX_FACES]:
            remove_face(old)
        self.thumbs = [QPixmap(f) for f in face_files()]
        self.has_face = True
        self.set_preset(None)
        if self.worker:
            self.worker.reload_face = True
        if not self.live:
            self.show_idle()
        if not self.swapping:
            self.say(self.ready_text())
        self.sync()

    def clear_faces(self):
        for f in face_files():
            remove_face(f)
        self.thumbs, self.has_face = [], False
        self.set_preset(None)
        if self.swapping:
            self.swapping = self.worker.on = False
        if not self.live:
            self.show_idle()
        self.say(self.ready_text())
        self.sync()

    # ── run ──
    def open_camera(self):
        self.worker = Worker(self.cameras.currentData())
        self.worker.frame.connect(self.show_frame)
        self.worker.status.connect(self.say)
        self.worker.mode.connect(self.on_mode)
        self.worker.finished.connect(self.ended)
        self.worker.on = self.swapping
        self.worker.start()

    def show_frame(self, rgb):
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.stage.setPixmap(rounded(QPixmap.fromImage(img), self.STAGE_W, self.STAGE_H, thumbs=self.strip(True)))

    def switch_camera(self):
        if self.worker:
            self.reopen = True
            self.worker.running = False
        else:
            self.open_camera()

    def on_mode(self, mode):
        self.live = True
        self.stage.setStyleSheet(f"border: none; background: {BG};")
        if mode == "you" and self.swapping:  # the worker refused (no face in the photos)
            self.swapping = False
        if not self.status.styleSheet().startswith(f"color: {ERROR_RED}") or mode == "swapping":
            self.say("Live. In Discord, pick OBS Virtual Camera." if mode == "swapping" else self.ready_text())
        self.sync()

    def toggle(self):
        if not self.worker:  # camera didn't open earlier; try again
            self.open_camera()
            return
        self.swapping = self.worker.on = not self.swapping
        if self.swapping and not self.worker.ready:
            self.say("Loading the face models…")
        self.sync()

    def ended(self):
        self.worker, self.live = None, False
        if self.reopen:
            self.reopen = False
            return self.open_camera()
        self.swapping = False
        self.show_idle()
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
