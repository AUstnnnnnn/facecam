"""Deep-Live-Cam sidecar for Facecam.app.

  bridge.py list                      -> one "index<TAB>name" line per camera
  bridge.py prepare                   -> download the face models unless they're there
  bridge.py run <camera_index> <dir>   -> webcam -> OBS Virtual Camera, your own face until "on"
<dir> holds photos of one person; their face fingerprints are averaged.
stdin while running:  "on" / "off" scan the swap in and out, "face <dir>" reloads the photos
(and scans across to them if the swap is on).
stdout: "status <text>" lines for the UI.
"""
import glob, math, os, sys, threading, time
import cv2
import numpy as np

SUPPORT = os.environ.get("FACECAM_SUPPORT") or os.path.expanduser("~/Library/Application Support/Facecam")
DLC = os.path.join(SUPPORT, "engine")  # Deep-Live-Cam, unpacked by setup.sh
sys.path.insert(0, DLC)
os.chdir(DLC)


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


def say(*a):
    print("status", *a, flush=True)


def list_cameras():
    from cv2_enumerate_cameras import enumerate_cameras
    for c in enumerate_cameras():
        if "OBS" not in c.name:  # our own output, would loop back
            print(f"{c.index}\t{c.name}", flush=True)


def load():
    import modules.globals as g
    g.execution_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    from modules.face_analyser import get_one_face
    from modules.processors.frame import face_swapper as fs
    fs.get_face_swapper()
    from modules.face_analyser import get_face_analyser
    get_face_analyser().det_model.det_thresh = DET_THRESH
    return get_one_face, fs


def face_from(path, get_one_face):
    """One photo, or a folder of photos of the same person blended into one face."""
    paths = sorted(glob.glob(os.path.join(path, "*.png"))) if os.path.isdir(path) else [path]
    faces = [f for f in (get_one_face(i) for i in map(cv2.imread, paths) if i is not None) if f is not None]
    if not faces:
        return None
    face = faces[0]
    if len(faces) > 1:
        # The swap model only reads the identity embedding, so averaging it over
        # several angles and lights gives a closer, steadier likeness.
        face.embedding = np.mean([f.normed_embedding for f in faces], axis=0)
    return face


def prepare():
    """Setup step: fetch the models now, so the first Start doesn't wait on them."""
    import modules.globals as g
    g.execution_providers = ["CPUExecutionProvider"]  # only downloading
    from modules.face_analyser import get_face_analyser
    from modules.processors.frame import face_swapper as fs
    assert fs.pre_check() and get_face_analyser() is not None, "models didn't download"
    print("  Face models ready.", flush=True)


def run(index, face_path):
    import pyvirtualcam
    # Tests: FACECAM_TEST_VIDEO stands in for the webcam (looped), FACECAM_TEST_DUMP
    # gets every 3rd output frame as a PNG.
    video, dump = os.environ.get("FACECAM_TEST_VIDEO"), os.environ.get("FACECAM_TEST_DUMP")
    cap = cv2.VideoCapture(video or index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, frame = cap.read()
    if not ok:
        say("Camera didn't open"); sys.exit(1)
    h, w = frame.shape[:2]

    # Latest-frame grabber so slow swaps never build up camera lag.
    latest = {"f": frame}
    def grab():
        while True:
            ok, f = cap.read()
            if ok: latest["f"] = f
            elif video: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if video: time.sleep(1 / 30)
    threading.Thread(target=grab, daemon=True).start()

    try:
        cam = pyvirtualcam.Camera(w, h, 30, fmt=pyvirtualcam.PixelFormat.BGR, backend="obs")
    except Exception as e:
        say(f"Virtual camera unavailable: {e}"); sys.exit(1)

    # Your own face goes out straight away; the models load meanwhile so Start is instant.
    st = {"src": None, "on": False, "shown": None, "trans": None, "ready": False}
    lock = threading.Lock()

    def begin(to, up=False):
        """Scan from whatever is showing now to `to` (a face, or None for you)."""
        with lock:
            st["trans"] = [st["shown"], to, 0.0, up]  # from, to, progress, upward
            st["shown"] = to

    def loader():
        st["fns"] = load()
        st["src"] = face_from(face_path, st["fns"][0])
        st["ready"] = True
        say("Ready" if st["src"] is not None else "No face found in those photos")
        if st["on"] and st["src"] is not None:
            begin(st["src"])
    threading.Thread(target=loader, daemon=True).start()

    def stdin_loop():
        for line in sys.stdin:
            cmd = line.strip()
            if cmd == "on":
                st["on"] = True
                if st["ready"] and st["src"] is not None:
                    begin(st["src"]); say("Swapping")
                elif not st["ready"]:
                    say("Loading models…")
            elif cmd == "off":
                st["on"] = False
                if st["shown"] is not None:
                    begin(None, up=True)
                say("Off")
            elif cmd.startswith("face ") and st["ready"]:
                f = face_from(cmd[5:], st["fns"][0])
                if f is None:
                    say("No face found in those photos")
                    continue
                st["src"] = f
                if st["on"]:
                    begin(f); say("Swapping")
                else:
                    say("Ready")
        os._exit(0)  # app went away

    threading.Thread(target=stdin_loop, daemon=True).start()
    say("Live")
    last, missed, steady = None, 0, Steady()
    tick = time.monotonic()
    with cam:
        while True:
            f = latest["f"]
            now = time.monotonic()
            dt, tick = min(now - tick, STEP), now
            with lock:
                shown, trans = st["shown"], st["trans"]
                if trans:
                    trans[2] += dt / SCAN_SECONDS
            p = trans[2] if trans else 1.0
            if p >= 1:
                with lock:
                    if st["trans"] is trans:
                        st["trans"] = None
                trans = None
            faces = (trans[0], trans[1]) if trans else (shown,)
            t = None
            if st["ready"] and any(x is not None for x in faces):
                get_one_face, fs = st["fns"]
                t = get_one_face(lift(f))
                if t is not None:
                    t.kps = steady(t.kps)
                    last, missed = t, 0
                elif missed < HOLD_FRAMES:
                    t, missed = last, missed + 1
                else:
                    steady.reset()  # face gone; don't glide in from where it was

            def look(face):
                # process_frame writes into the frame it's given, so hand it a copy.
                return f if face is None or t is None else st["fns"][1].process_frame(face, f.copy(), t)

            if trans:
                add_outline_points(f, t)
                out = scan(look(trans[0]), look(trans[1]), t, p, trans[3])
            else:
                out = look(shown)
            cam.send(out)
            if dump and cam.frames_sent % 3 == 0:
                cv2.imwrite(os.path.join(dump, f"{cam.frames_sent:05d}.jpg"), out)
            cam.sleep_until_next_frame()


def selftest(src_path, tgt_path, out_path):
    import cv2
    get_one_face, fs = load()
    s = face_from(src_path, get_one_face)
    tgt = cv2.imread(tgt_path)
    t = get_one_face(tgt)
    assert s is not None and t is not None, "faces not found"
    t0 = time.time()
    for _ in range(10):
        out = fs.process_frame(s, tgt.copy(), t)
    print(f"{(time.time()-t0)/10*1000:.0f} ms/frame")
    assert out.shape == tgt.shape and (out != tgt).any(), "swap did nothing"

    st = Steady()  # still noisy points settle, a real move is followed
    rng = np.random.default_rng(0)
    pts = [st(np.full((5, 2), 100.0) + rng.normal(0, 3, (5, 2))) for _ in range(60)]
    assert np.std(pts[30:], axis=0).mean() < 1.0, "Steady didn't smooth"
    for _ in range(30):
        time.sleep(0.01)
        moved = st(np.full((5, 2), 200.0))
    assert np.abs(moved - 200).max() < 5, "Steady lags a real move"
    # Scan: starts on the old frame, ends on the new one everywhere (no pop when it stops).
    a, b = tgt, out
    add_outline_points(tgt, t)
    assert t.landmark_2d_106 is not None, "no outline points"
    for up in (False, True):
        assert (scan(a, b, t, 1.0, up) == b).all(), "scan didn't land on b"
        assert (scan(a, b, t, 0.0, up) == a).all(), "scan didn't start on a"
    mid = scan(a, b, t, 0.5)
    x0, y0, x1, y1 = map(int, t.bbox)
    away = max(1, x0 - (x1 - x0))  # columns well left of the face
    assert (mid[:, :away] == a[:, :away]).all(), "mid-scan touched pixels outside the face"
    strip = [scan(a, b, t, q) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
    cv2.imwrite(out_path.replace(".png", "-scan.png"), np.hstack([cv2.resize(x, None, fx=0.5, fy=0.5) for x in strip]))
    cv2.imwrite(out_path, out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list": list_cameras()
    elif cmd == "run": run(int(sys.argv[2]), sys.argv[3])
    elif cmd == "prepare": prepare()
    elif cmd == "selftest": selftest(*sys.argv[2:5])
