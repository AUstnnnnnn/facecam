"""Deep-Live-Cam sidecar for Facecam.app.

  bridge.py list                      -> one "index<TAB>name" line per camera
  bridge.py prepare                   -> download the face models unless they're there
  bridge.py run <camera_index> <dir>   -> webcam -> swap -> OBS Virtual Camera
<dir> holds photos of one person; their face fingerprints are averaged.
stdin while running:  "face <dir>" reloads the photos.
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
    import cv2, pyvirtualcam
    say("Loading models…")
    get_one_face, fs = load()
    src = {"face": face_from(face_path, get_one_face)}
    if src["face"] is None:
        say("No face found in those photos")

    def stdin_loop():
        for line in sys.stdin:
            if line.startswith("face "):
                f = face_from(line[5:].strip(), get_one_face)
                if f is None:
                    say("No face found in those photos")
                else:
                    src["face"] = f
                    say("Live")
        os._exit(0)  # app went away

    threading.Thread(target=stdin_loop, daemon=True).start()

    cap = cv2.VideoCapture(index)
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
    threading.Thread(target=grab, daemon=True).start()

    try:
        cam = pyvirtualcam.Camera(w, h, 30, fmt=pyvirtualcam.PixelFormat.BGR, backend="obs")
    except Exception as e:
        say(f"Virtual camera unavailable: {e}"); sys.exit(1)
    if src["face"] is not None:
        say("Live")
    last, missed, steady = None, 0, Steady()
    with cam:
        while True:
            f = latest["f"]
            if src["face"] is not None:
                t = get_one_face(lift(f))
                if t is not None:
                    t.kps = steady(t.kps)
                    last, missed = t, 0
                elif missed < HOLD_FRAMES:
                    t, missed = last, missed + 1
                else:
                    steady.reset()  # face gone; don't glide in from where it was
                if t is not None:
                    f = fs.process_frame(src["face"], f, t)
            cam.send(f)
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
    cv2.imwrite(out_path, out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list": list_cameras()
    elif cmd == "run": run(int(sys.argv[2]), sys.argv[3])
    elif cmd == "prepare": prepare()
    elif cmd == "selftest": selftest(*sys.argv[2:5])
