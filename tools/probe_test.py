# pyrefly: ignore [missing-import]
import onnxruntime as ort
ort.preload_dlls(directory="")
ort.print_debug_info()

import os
# InsightFace downloads models to ~/.insightface/models/<pack>/ on first run.
model_path = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l", "det_10g.onnx")
sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(sess.get_providers())







"""
encoder.py — Face analysis core for the Face Recognition System

Backend: InsightFace (ArcFace, CUDA via ONNX Runtime).

v3 changes (smoother live tracking, gender/liveness removed):
  - Gender and liveness have been removed entirely. analyze_frame() now
    only returns bbox + embedding, which means:
      * the genderage model is no longer loaded (allowed_modules cut
        down to just "detection" + "recognition" — one less model
        graph in memory and on the inference path)
      * the per-face Laplacian/FFT liveness heuristic is gone, which
        was the single most expensive per-face extra in the old
        analyze_frame() — this is the biggest win for tracking smoothness
        since it ran on every face, every call.
  - Multi-photo enrollment: a person can have several reference photos;
    their embeddings are averaged (and re-normalized) into one robust
    identity vector, which is far more tolerant of lighting/angle/
    expression variation than a single photo.
  - Face search: given a query embedding, return the top-K closest
    enrolled identities by cosine similarity — useful for a "who is this"
    lookup tool, independent of the live camera feed.

perf-v5 / long-distance pass:
  - det_size fixed at (640,640) — hard ceiling for buffalo_l's SCRFD.
  - DET_THRESH lowered to 0.2 — distant faces score low by nature.
  - Full preprocessing pipeline in _preprocess_frame():
      * Lanczos4 upscale 3x (1080p → ~3240p equivalent input)
      * CLAHE contrast normalisation on L channel
      * Unsharp mask sharpening (strength 0.7)
    Together these give SCRFD the best possible input for detecting
    small/far/underlit faces. Each stage is independently tunable.
  - get_embedding() no longer calls app.get() a second time — it's now
    a thin wrapper around analyze_frame()'s detection pass, so a caller
    doing "find biggest face -> get embedding" isn't running the
    detector twice per frame.
  - analyze_frame() short-circuits to [] before touching cv2 if the
    frame is empty/None, avoiding a confusing exception path in a tight
    capture loop.
  - NMS/detection-score threshold exposed as a module constant
    (DET_THRESH) instead of buried in FaceAnalysis defaults, so it's one
    obvious place to tune if tracking is flickering on borderline faces.

Embeddings are 512-dimensional ArcFace vectors (L2-normalized), so
recognition uses cosine similarity instead of Euclidean distance.
"""

import os

# ── VRAM / CUDA init tweaks (must be set before any torch/onnxruntime import) ─
# Prevents PyTorch (present in shared site-packages as torch 2.6.0+cu124) from
# reserving a full CUDA context on import, which squats 500-800 MB of VRAM
# before InsightFace even loads — that's what causes ORT to silently fall back
# to CPU at det_size=(640,640) on the RTX 3050's 4 GB budget.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("ORT_DISABLE_TORCH_INTEROP", "1")

import re
import pickle
import logging
import subprocess
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

log = logging.getLogger("FaceRecog.Encoder")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "face_cache.pkl")

# ── Detector resolution — auto-selected based on available VRAM ─────────────
# SCRFD cost scales ~with det_size^2. On the RTX 3050 (4 GB VRAM), running at
# (640,640) causes ORT to silently fall back to CPU because the model + tensor
# don't fit after the CUDA context overhead. We probe free VRAM at startup and
# pick the largest det_size that safely fits:
#
#   >= 1500 MB free  →  (640, 640)   best long-distance detection
#   >= 800  MB free  →  (480, 480)   good mid-range
#   >= 400  MB free  →  (320, 320)   near-camera / kiosk distance
#   <  400  MB free  →  (160, 160)   fallback, short range only
#
# Override by setting DET_SIZE manually below if you want a fixed value.
def _probe_free_vram_mb() -> int:
    """Return free VRAM in MB on device 0, or 0 on any failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=3, stderr=subprocess.DEVNULL
        ).decode().strip().splitlines()
        return int(out[0].strip())
    except Exception:
        return 0


def _pick_det_size() -> tuple[int, int]:
    free = _probe_free_vram_mb()
    if free >= 1500:
        size = (640, 640)
    elif free >= 800:
        size = (480, 480)
    elif free >= 400:
        size = (320, 320)
    else:
        size = (160, 160)
    log.info(f"VRAM probe: {free} MB free → det_size={size}")
    return size


# Set DET_SIZE = (W, H) here to override auto-selection.
# (640, 640) is the hard ceiling for buffalo_l's SCRFD — max detection range.
# Leave as None to let _pick_det_size() choose based on free VRAM at startup.
DET_SIZE: tuple[int, int] | None = (640, 640)

# Minimum detection confidence. 0.25 catches small/distant faces that score
# low by nature. Raise toward 0.5 if you get ghost/false-positive boxes.
DET_THRESH = 0.25

# Pre-upscale factor applied to each frame before detection.
# A face that's 20px wide at 1080p becomes 40px at 2x — SCRFD finds it.
# Bbox coordinates are scaled back down after detection so overlays are correct.
# 2.0 = good range boost on a 1080p feed with minimal CPU overhead.
UPSCALE_FACTOR = 2.0


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Upscale frame before detection so distant faces have enough pixels for SCRFD."""
    if UPSCALE_FACTOR == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(
        frame,
        (int(w * UPSCALE_FACTOR), int(h * UPSCALE_FACTOR)),
        interpolation=cv2.INTER_LINEAR,
    )

# ── InsightFace app (lazy-loaded singleton) ─────────────────────────────────
_app = None


def _get_app():
    """Lazy-load the InsightFace FaceAnalysis app (downloads model on first use)."""
    global _app
    if _app is None:
        try:
            # Resolve det_size — auto-probe VRAM if not manually overridden
            det_size = DET_SIZE if DET_SIZE is not None else _pick_det_size()

            try:
                # pyrefly: ignore [missing-import]
                import onnxruntime as _ort
                if hasattr(_ort, "preload_dlls"):
                    _ort.preload_dlls()
            except Exception as dll_exc:
                log.debug(f"onnxruntime.preload_dlls() skipped/failed: {dll_exc}")

            # Add pip-installed CUDA DLLs to the Windows DLL search path and PATH
            if os.name == 'nt':
                import site
                for site_dir in site.getsitepackages():
                    nvidia_dir = os.path.join(site_dir, 'nvidia')
                    if os.path.isdir(nvidia_dir):
                        for pkg in os.listdir(nvidia_dir):
                            bin_dir = os.path.join(nvidia_dir, pkg, 'bin')
                            if os.path.isdir(bin_dir):
                                os.add_dll_directory(bin_dir)
                                os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
            # pyrefly: ignore [missing-import]
            from insightface.app import FaceAnalysis
            _app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            _app.prepare(ctx_id=0, det_size=det_size, det_thresh=DET_THRESH)

            # Check active providers — ORT silently drops to CPU if CUDA EP
            # fails (VRAM OOM, DLL mismatch, etc.), so we verify explicitly.
            try:
                active = _app.det_model.session.get_providers()
            except Exception:
                active = ["unknown"]

            # Log VRAM after model load so you can see actual consumption
            free_after = _probe_free_vram_mb()

            if "CUDAExecutionProvider" in active:
                log.info(
                    f"InsightFace loaded — GPU (CUDA) active: {active} | "
                    f"det_size={det_size} | VRAM free after load: {free_after} MB"
                )
            else:
                log.warning(
                    f"InsightFace is running on CPU — providers: {active}. "
                    f"det_size={det_size} | VRAM free after load: {free_after} MB. "
                    "CUDAExecutionProvider did not load — likely VRAM OOM or "
                    "cuDNN/onnxruntime-gpu version mismatch."
                )
        except Exception as e:
            log.error(f"InsightFace init failed: {e}")
            _app = None
    return _app


def analyze_frame(bgr_frame: np.ndarray) -> list[dict]:
    """
    Run face detection + recognition on a frame and return a list of
    per-face dicts:
        {
          "bbox": (x1, y1, x2, y2),
          "embedding": np.ndarray (512,) unit vector,
        }
    Returns [] if no faces / InsightFace unavailable / empty frame.
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return []
    app = _get_app()
    if app is None:
        return []

    # Run full preprocessing pipeline (upscale → CLAHE → sharpen) for maximum
    # long-distance detection range, then scale bboxes back to original coords.
    detect_frame = _preprocess_frame(bgr_frame)
    faces = app.get(detect_frame)
    return [
        {
            "bbox": tuple(int(v / UPSCALE_FACTOR) for v in face.bbox),
            "embedding": face.normed_embedding,
        }
        for face in faces
    ]


def get_embedding(bgr_frame: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    """
    Detect the largest face in bgr_frame and return (embedding, bbox).
    Returns (None, None) if no face is found.

    perf-v5: thin wrapper over analyze_frame() so detection only ever
    runs once per frame, even when a caller wants "just the biggest
    face" rather than the full list.
    """
    faces = analyze_frame(bgr_frame)
    if not faces:
        return None, None
    face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
    return face["embedding"], face["bbox"]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-normalized vectors."""
    return float(np.dot(a, b))


# ── Public API: enrollment / gallery loading ─────────────────────────────────

def _has_image_files(folder: str) -> bool:
    if not os.path.isdir(folder):
        return False
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            return True
    return False


def _average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average a list of unit-vector embeddings and re-normalize to unit length."""
    stacked = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = np.linalg.norm(stacked)
    if norm > 1e-8:
        stacked = stacked / norm
    return stacked


def load_known_faces(folder: str, use_cache: bool = True):
    """
    Returns (encodings_list, names_list).

    Each entry in encodings_list is the AVERAGED embedding across every
    enrollment photo on disk for that name (multi-photo support) —
    filenames are grouped by their base name with optional "_2", "_3", ...
    suffixes, e.g. "Dara_Sok.jpg", "Dara_Sok_2.jpg", "Dara_Sok_3.jpg" all
    belong to "Dara Sok".

    Encodings are 512-d ArcFace unit vectors. Caches to disk so startup is
    fast after first run.
    """
    if use_cache and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            if data.get("names"):
                log.info(f"Loaded {len(data['names'])} identities from cache.")
                return data["encodings"], data["names"]
            if not _has_image_files(folder):
                log.info("Cache empty; no images found.")
                return data.get("encodings", []), data.get("names", [])
        except Exception:
            log.warning("Cache corrupt — rebuilding")

    encodings, names = [], []

    if not os.path.isdir(folder):
        os.makedirs(folder)
        return encodings, names

    app = _get_app()
    if app is None:
        log.error("InsightFace not available — cannot encode faces")
        return encodings, names

    # Group photos by base identity name, stripping "_2"/"_3"/... suffixes
    groups: dict[str, list[str]] = {}
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(filename)[0]
        base = re_strip_multi_suffix(stem)
        name = base.replace("_", " ")
        groups.setdefault(name, []).append(os.path.join(folder, filename))

    for name, paths in groups.items():
        embs = []
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                log.warning(f"Could not read image: {path}")
                continue
            emb, _ = get_embedding(img)
            if emb is not None:
                embs.append(emb)
            else:
                log.warning(f"No face found in {path}")
        if embs:
            final_emb = _average_embeddings(embs) if len(embs) > 1 else embs[0]
            encodings.append(final_emb)
            names.append(name)
            log.info(f"Encoded: {name} ({len(embs)} photo(s) averaged)")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

    log.info(f"{len(names)} identities encoded and cached.")
    return encodings, names


def re_strip_multi_suffix(stem: str) -> str:
    """Strip a trailing '_2', '_3', ... multi-photo suffix from a filename stem."""
    return re.sub(r'_\d+$', '', stem)


def search_face(query_embedding: np.ndarray, gallery_embeddings: list[np.ndarray],
                 gallery_names: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    """
    Compare a query embedding against a gallery of enrolled identity
    embeddings and return the top_k closest matches as
    [(name, similarity), ...] sorted descending by similarity.
    """
    if not gallery_embeddings:
        return []
    gallery = np.array(gallery_embeddings)
    sims = gallery @ query_embedding
    order = np.argsort(sims)[::-1][:top_k]
    return [(gallery_names[i], float(sims[i])) for i in order]


def delete_face(folder: str, name: str):
    """Delete ALL of a person's enrollment photo files (including multi-
    photo "_2", "_3", ... variants) from known_faces/ and invalidate cache."""
    filename_base = name.replace(" ", "_")
    deleted = False
    for filename in os.listdir(folder) if os.path.isdir(folder) else []:
        stem = os.path.splitext(filename)[0]
        if stem == filename_base or re_strip_multi_suffix(stem) == filename_base:
            path = os.path.join(folder, filename)
            try:
                os.remove(path)
                log.info(f"Deleted face image: {path}")
                deleted = True
            except Exception as e:
                log.warning(f"Could not delete {path}: {e}")
    if not deleted:
        log.warning(f"No image file(s) found for '{name}' in {folder}")
    invalidate_cache()


def next_multi_photo_path(folder: str, name: str) -> str:
    """
    Compute the next available filename for an additional enrollment
    photo of `name` (e.g. Dara_Sok.jpg exists -> returns .../Dara_Sok_2.jpg).
    """
    base = name.replace(" ", "_")
    primary = os.path.join(folder, f"{base}.jpg")
    if not os.path.exists(primary):
        return primary
    i = 2
    while True:
        candidate = os.path.join(folder, f"{base}_{i}.jpg")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def invalidate_cache():
    """Call this after registering, adding a photo, or deleting a face so the cache is rebuilt."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        log.info("Face cache invalidated")













"""
Face Recognition System
=========================
Author: JJ — RUPP Computer Architecture Final Project

PERFORMANCE PATCH (perf-v1) — see PERF_NOTES.md for the full writeup.

Summary of what changed vs the v15 baseline and why:

  1. REMOVED the per-frame CPU correlation tracker (_track_loop /
     _create_tracker / cv2.legacy.TrackerMOSSE / TrackerKCF).
     This was the single biggest source of GIL contention: a pure-Python
     + pure-CPU tracker.update() call running on EVERY rendered frame,
     for EVERY tracked face, fighting _video_loop and _detect_loop for
     the same GIL. SmoothBox already has velocity-based prediction
     (vt/vr/vb/vl + PREDICTION_WEIGHT) designed to carry a box between
     detection cycles — the tracker was doing redundant work against
     its own prediction system. Boxes now coast purely on lerp/velocity
     prediction between detections, which is computationally free
     (a handful of float ops) compared to a CV tracker update.

  2. _video_loop now sleeps briefly when no new camera frame is
     available, instead of busy-spinning. A tight no-sleep loop never
     voluntarily yields the GIL for long stretches, which was starving
     the detect thread and the Tkinter main loop.

  3. _poll_frame no longer rebuilds a brand-new PIL.Image + CTkImage on
     every single 8ms tick regardless of whether a new frame arrived.
     It now only does that expensive resize+convert work when a frame
     is actually available, and caches the target display size so we
     don't recompute the aspect-fit math every tick either.

  4. Liveness's FFT-based heuristic (encoder._estimate_liveness) used to
     run on EVERY detected face on EVERY detect cycle (every 5th camera
     frame), even though the UI badge update was already throttled by
     ATTRIBUTE_REFRESH_INTERVAL. The actual computation is now gated by
     that same per-name cooldown, so the expensive FFT only runs ~10x/sec
     per visible identity instead of ~8x/sec PER FACE regardless of cooldown.

  5. encoder.analyze_frame() gained an optional `want_liveness` flag
     (default True) so identity-only passes can skip the FFT path
     entirely. app.py now does a lightweight pass when nobody is due for
     an attribute refresh.

PERFORMANCE PATCH (perf-v2) — UI smoothness pass on top of perf-v1.

  6. _poll_frame was handing CTkImage a full 800x600 PIL image every
     tick and letting CTkImage resize it down internally. Confirmed by
     reading customtkinter's ctk_image.py: CTkImage builds a fresh
     internal cache every call (we create a new CTkImage instance every
     frame) and ALWAYS calls PIL .resize(target_size) — Pillow only
     short-circuits that into a cheap .copy() when the source size
     already equals the target size, otherwise it's a full resample
     pass over every pixel, on the Tkinter main thread, every single
     displayed frame. We now resize with cv2 (SIMD-accelerated, far
     cheaper than PIL for this) to the EXACT target size before handing
     it to PIL/CTkImage, turning that resample into a no-op copy, and we
     do it BEFORE cvtColor so the color conversion also runs on the
     smaller image instead of the full frame.

  7. _identify_face rebuilt np.array(self.known_encodings) from a Python
     list on EVERY detected face, on EVERY detect cycle. Now cached as
     self._known_matrix, rebuilt only inside reload_known_faces() (i.e.
     only when someone is registered/edited/deleted) — removes per-face
     GIL/CPU work that was competing with the video thread for nothing.

  8. Greeting toast: two recognitions within the 4s display window used
     to stack two independent after(4000, _reset_greeting) calls, where
     the first one's reset could stomp the second toast mid-display.
     Now the pending reset timer is cancelled before scheduling a new one.

Everything else — InsightFace identity matching, multi-photo enrollment,
Khmer TTS greeting flow, the recognition event log, Face Search tab — is
UNCHANGED in behavior.

PERFORMANCE PATCH (perf-v5) — camera ceiling diagnosis + render decoupling.

  9. Measured this rig's actual camera (camera_speed_test.py, 8 modes
     tested) and found it has a hard ~30fps ceiling: 1920x1080@30 was the
     FASTEST mode measured, not the slowest — every lower resolution
     tested was paradoxically slower (720p ~7.7fps, 480p ~23fps). This is
     a hardware/driver limit, not something software can raise.
     ThreadedCamera now explicitly requests 1920x1080@30 (the proven-best
     mode) instead of blindly asking for 120fps and hoping.

  10. DISPLAY_WIDTH/HEIGHT changed from 800x600 (4:3) to 960x540 (16:9)
      to match the camera's native aspect ratio — the old setting was
      silently squashing every frame.

  11. _video_loop split into two threads: _video_loop (pure camera
      capture, naturally bounded by the camera's own ~30fps hardware
      pace) and _render_loop (lerp + draw, running on its own
      RENDER_TARGET_FPS timer, independent of when the next real camera
      frame arrives). Box positions are now interpolated forward on
      every render tick rather than only once per captured frame, so
      on-screen motion looks smoother than the raw 30fps camera feed —
      same principle as motion interpolation in video players. This does
      NOT increase real camera FPS (impossible on this hardware); it
      only smooths the perceived motion of a 30fps source.

PyQt6 PORT — all customtkinter / tkinter / PIL.ImageTk dependencies removed.
  - messagebox / filedialog  → QMessageBox / QFileDialog
  - CTkImage / ImageTk       → QPixmap (via cv2 + bytes)
  - label.configure(...)     → label.setText() / label.setStyleSheet()
  - live_log.configure(...)  → QTextEdit native API
  - _poll_frame display      → QLabel.setPixmap() with QPixmap.loadFromData()
  - mainloop()               → QApplication.exec()
"""

# pyrefly: ignore [missing-import]
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import QApplication, QMessageBox, QFileDialog
# pyrefly: ignore [missing-import]
from PyQt6.QtGui import QPixmap
# pyrefly: ignore [missing-import]
from PyQt6.QtCore import Qt, QTimer
# pyrefly: ignore [missing-import]
from PIL import Image
import threading
import queue
import time
import os
import re
import sys
import logging
import ctypes
from datetime import datetime, date
# pyrefly: ignore [missing-import]
from gui import FaceRecognitionGUI, UI_FONT_FAMILIES
# pyrefly: ignore [missing-import]
import database
# pyrefly: ignore [missing-import]
import encoder
# pyrefly: ignore [missing-import]
from tts import speak_greeting, shutdown_tts

# ── Windows high-res timer + priority ─────────────────────────────────────────
_HIGH_RES_TIMER_SET = False
if sys.platform == 'win32':
    try:
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)
    except Exception:
        pass
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
        _HIGH_RES_TIMER_SET = True
    except Exception:
        pass

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("FaceRecog")

cv2.ocl.setUseOpenCL(False)
log.info("OpenCL disabled — using CUDA via InsightFace for GPU acceleration")

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")

CONFIDENCE_THRESHOLD  = 0.50
DETECT_SCALE          = 0.50
DETECT_EVERY_N_FRAMES = 1
DISPLAY_WIDTH         = 960
DISPLAY_HEIGHT        = 540
LERP_SPEED_MIN        = 0.45
LERP_SPEED_MAX        = 0.97
VELOCITY_DAMPING      = 0.70
PREDICTION_WEIGHT     = 0.015   # seconds of velocity look-ahead
FADE_OUT_TIME         = 1.0
SCAN_LINE_SPEED       = 3.0
MAX_DISPLAY_WIDTH     = 1280
MAX_DISPLAY_HEIGHT    = 720
RENDER_TARGET_FPS     = 120
_RENDER_INTERVAL      = 1.0 / RENDER_TARGET_FPS
_CAMERA_IDLE_SLEEP    = 0.001


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ndarray_to_qpixmap(bgr_frame: np.ndarray, w: int = 0, h: int = 0) -> QPixmap:
    """Convert a BGR numpy frame to QPixmap, optionally resizing first.
    Uses raw QImage byte copy — no PNG compression round-trip."""
    # pyrefly: ignore [missing-import]
    from PyQt6.QtGui import QImage
    if w and h and (bgr_frame.shape[1] != w or bgr_frame.shape[0] != h):
        interp = cv2.INTER_LINEAR if w > bgr_frame.shape[1] else cv2.INTER_AREA
        bgr_frame = cv2.resize(bgr_frame, (w, h), interpolation=interp)
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    h_px, w_px, ch = rgb.shape
    # Make sure the array is contiguous so QImage can read raw bytes safely
    rgb = np.ascontiguousarray(rgb)
    qimg = QImage(rgb.data, w_px, h_px, w_px * ch, QImage.Format.Format_RGB888)
    # .copy() detaches from the numpy buffer (which may be freed after return)
    return QPixmap.fromImage(qimg.copy())


def _pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    """Convert a PIL RGB image to QPixmap.
    Uses raw QImage byte copy — no PNG compression round-trip."""
    # pyrefly: ignore [missing-import]
    from PyQt6.QtGui import QImage
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    data = pil_img.tobytes("raw", "RGB")
    qimg = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ── SmoothBox ─────────────────────────────────────────────────────────────────
class SmoothBox:
    __slots__ = (
        'name', 'confidence',
        'dt', 'dr', 'db', 'dl',
        'tt', 'tr', 'tb', 'tl',
        'vt', 'vr', 'vb', 'vl',
        'last_update', 'birth_time',
        'last_detect_t',
    )

    def __init__(self, top, right, bottom, left, name, confidence):
        self.name = name
        self.confidence = confidence
        self.dt = float(top);    self.dr = float(right)
        self.db = float(bottom); self.dl = float(left)
        self.tt = float(top);    self.tr = float(right)
        self.tb = float(bottom); self.tl = float(left)
        self.vt = 0.0; self.vr = 0.0; self.vb = 0.0; self.vl = 0.0
        self.last_update = time.time()
        self.birth_time  = time.time()
        self.last_detect_t = time.time()

    def set_target(self, top, right, bottom, left, name, confidence):
        now = time.time()
        elapsed = max(now - self.last_detect_t, 1e-3)
        raw_vt = (float(top)    - self.tt) / elapsed
        raw_vr = (float(right)  - self.tr) / elapsed
        raw_vb = (float(bottom) - self.tb) / elapsed
        raw_vl = (float(left)   - self.tl) / elapsed
        self.vt = raw_vt * VELOCITY_DAMPING
        self.vr = raw_vr * VELOCITY_DAMPING
        self.vb = raw_vb * VELOCITY_DAMPING
        self.vl = raw_vl * VELOCITY_DAMPING
        self.tt, self.tr = float(top), float(right)
        self.tb, self.tl = float(bottom), float(left)
        self.name = name
        self.confidence = confidence
        self.last_update = now
        self.last_detect_t = now

    def lerp_dt(self, dt: float):
        frame_alpha_min = 1.0 - (1.0 - LERP_SPEED_MIN) ** (dt * 60.0)
        frame_alpha_max = 1.0 - (1.0 - LERP_SPEED_MAX) ** (dt * 60.0)
        self.tt += self.vt * dt
        self.tr += self.vr * dt
        self.tb += self.vb * dt
        self.tl += self.vl * dt
        pred_tt = self.tt + self.vt * PREDICTION_WEIGHT
        pred_tr = self.tr + self.vr * PREDICTION_WEIGHT
        pred_tb = self.tb + self.vb * PREDICTION_WEIGHT
        pred_tl = self.tl + self.vl * PREDICTION_WEIGHT
        dist = abs(pred_tt - self.dt) + abs(pred_tr - self.dr) + \
               abs(pred_tb - self.db) + abs(pred_tl - self.dl)
        norm = min(dist / 100.0, 1.0)
        a = frame_alpha_min + (frame_alpha_max - frame_alpha_min) * norm
        self.dt += (pred_tt - self.dt) * a
        self.dr += (pred_tr - self.dr) * a
        self.db += (pred_tb - self.db) * a
        self.dl += (pred_tl - self.dl) * a
        damping = VELOCITY_DAMPING ** (dt * 60.0)
        self.vt *= damping; self.vr *= damping
        self.vb *= damping; self.vl *= damping

    def lerp(self):
        self.lerp_dt(1.0 / 60.0)

    def ints(self):
        return int(self.dt), int(self.dr), int(self.db), int(self.dl)

    def staleness(self):
        return min((time.time() - self.last_update) / FADE_OUT_TIME, 1.0)

    def is_stale(self, timeout=None):
        timeout = timeout or FADE_OUT_TIME
        return time.time() - self.last_update > timeout

    def center(self):
        return (self.dl + self.dr) / 2.0, (self.dt + self.db) / 2.0


def _iou(at, ar, ab, al, bt, br, bb, bl):
    it = max(at, bt); ib = min(ab, bb)
    il = max(al, bl); ir = min(ar, br)
    inter = max(0, ib - it) * max(0, ir - il)
    if inter == 0:
        return 0.0
    area1 = (ab - at) * (ar - al)
    area2 = (bb - bt) * (br - bl)
    return inter / (area1 + area2 - inter)


def _center_dist(at, ar, ab, al, bt, br, bb, bl):
    cx1, cy1 = (al + ar) / 2.0, (at + ab) / 2.0
    cx2, cy2 = (bl + br) / 2.0, (bt + bb) / 2.0
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def _draw_corner_tracers(frame, left, top, right, bottom, color,
                          thickness=2, corner_len=20, alpha=1.0):
    cl = max(corner_len, int((right - left) * 0.2))
    cl = min(cl, (right - left) // 2, (bottom - top) // 2)
    overlay = frame.copy() if alpha < 1.0 else frame
    cv2.line(overlay, (left, top),     (left + cl, top),     color, thickness)
    cv2.line(overlay, (left, top),     (left, top + cl),     color, thickness)
    cv2.line(overlay, (right, top),    (right - cl, top),    color, thickness)
    cv2.line(overlay, (right, top),    (right, top + cl),    color, thickness)
    cv2.line(overlay, (left, bottom),  (left + cl, bottom),  color, thickness)
    cv2.line(overlay, (left, bottom),  (left, bottom - cl),  color, thickness)
    cv2.line(overlay, (right, bottom), (right - cl, bottom), color, thickness)
    cv2.line(overlay, (right, bottom), (right, bottom - cl), color, thickness)
    thin_color = tuple(int(c * 0.35) for c in color)
    cv2.rectangle(overlay, (left, top), (right, bottom), thin_color, 1)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _draw_scan_line(frame, left, top, right, bottom, color, thickness=2):
    t = time.time() * SCAN_LINE_SPEED
    frac = (np.sin(t) + 1.0) / 2.0
    y = int(top + (bottom - top) * frac)
    fade_color = tuple(int(c * 0.55) for c in color)
    cv2.line(frame, (left + 2, y), (right - 2, y), fade_color, thickness)


def _sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


# ── ThreadedCamera ────────────────────────────────────────────────────────────
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        try:
            fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])
            log.info(
                f"Camera negotiated: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                f"@ {self.cap.get(cv2.CAP_PROP_FPS):.1f} fps requested, "
                f"FOURCC={fourcc_str!r}"
            )
        except Exception as e:
            log.debug(f"Camera property log failed: {e}")
        self.running = True
        self.fps = 0
        self._frame_count = 0
        self._last_time = time.time()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
                if self.ret:
                    self._frame_count += 1
                    now = time.time()
                    if now - self._last_time >= 1.0:
                        self.fps = self._frame_count
                        self._frame_count = 0
                        self._last_time = now
            else:
                time.sleep(0.01)

    def read(self):
        if self.ret and self.frame is not None:
            return self.ret, self.frame.copy()
        return self.ret, None

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, propId):
        return self.cap.get(propId)


# ── App ───────────────────────────────────────────────────────────────────────
class FaceRecognitionApp(FaceRecognitionGUI):
    def __init__(self):
        super().__init__()
        self.cap                   = None
        self._requested_camera_idx = None
        self.running               = False
        self.known_encodings       = []
        self.known_names           = []
        self._known_matrix         = None
        self.known_khmer_names     = {}
        self.today_logged          = set()
        self._pending_recognition  = {}
        self._enc_lock             = threading.Lock()
        self._fps_counter          = 0
        self._fps_timer            = time.perf_counter()
        self._recognition_queue    = queue.Queue()
        self._greeted_session      = set()
        self._last_good_frame      = None
        self._last_frame_lock      = threading.Lock()
        self._latest_captured_frame = None
        self._latest_frame_lock    = threading.Lock()
        self._frame_queue          = queue.Queue(maxsize=1)
        self._pending_frame        = None
        self._pending_frame_lock   = threading.Lock()
        self._detect_event         = threading.Event()
        self._frame_count          = 0
        self._boxes                = []
        self._boxes_lock           = threading.Lock()
        self._last_lerp_time       = time.perf_counter()
        self._search_frame         = None
        self._last_display_size    = None
        self._last_render_size     = None
        self._display_target       = (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        self._display_target_lock  = threading.Lock()

        database.init_db()
        self.reload_known_faces()
        self._build_ui()
        self._start_threads()
        self._process_recognition_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Thread startup ────────────────────────────────────────────────────────
    def _start_threads(self):
        opened_idx = 1
        for idx in [1, 0, 2]:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.release()
                    self.cap = ThreadedCamera(idx)
                    opened_idx = idx
                    log.info(f"Camera opened at index {idx}")
                    break
                else:
                    cap.release()

        if not self.cap or not self.cap.isOpened():
            self.status_bar.setText("● Camera not found")
            self.status_bar.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            log.error("No camera found at indices 0-3")
            return

        self.camera_var.set(f"Camera {opened_idx}")
        self.running = True
        threading.Thread(target=self._video_loop,  name="VideoThread",  daemon=True).start()
        threading.Thread(target=self._render_loop, name="RenderThread", daemon=True).start()
        threading.Thread(target=self._detect_loop, name="DetectThread", daemon=True).start()

        # PERF FIX: _poll_frame used to re-arm itself via self.after(1/2, ...)
        # every tick — at ~500-1000 calls/sec that created and (even after
        # the after() leak fix) tore down hundreds of QTimer objects per
        # second for no reason. One persistent repeating QTimer does the
        # same job with a single long-lived object.
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_frame)
        self._poll_timer.start(1)

    def _change_camera(self, choice):
        try:
            idx = int(choice.split(" ")[1])
            self._requested_camera_idx = idx
        except Exception as e:
            log.error(f"Failed to parse camera choice {choice}: {e}")

    # ── _video_loop ───────────────────────────────────────────────────────────
    def _video_loop(self):
        while self.running:
            if self._requested_camera_idx is not None:
                if self.cap:
                    self.cap.release()
                self.cap = ThreadedCamera(self._requested_camera_idx)
                self._requested_camera_idx = None

            if not self.cap or not self.cap.isOpened():
                time.sleep(0.01)
                continue

            ret, cpu_frame = self.cap.read()
            if not ret or cpu_frame is None:
                time.sleep(_CAMERA_IDLE_SLEEP)
                continue

            cpu_frame = cv2.resize(cpu_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            with self._last_frame_lock:
                self._last_good_frame = cpu_frame

            with self._latest_frame_lock:
                self._latest_captured_frame = cpu_frame

            self._frame_count += 1
            if self._frame_count % DETECT_EVERY_N_FRAMES == 0:
                with self._pending_frame_lock:
                    self._pending_frame = cpu_frame.copy()
                self._detect_event.set()

    # ── _render_loop ──────────────────────────────────────────────────────────
    def _render_loop(self):
        next_tick = time.perf_counter()
        while self.running:
            with self._latest_frame_lock:
                cpu_frame = self._latest_captured_frame

            if cpu_frame is None:
                time.sleep(_CAMERA_IDLE_SLEEP)
                next_tick = time.perf_counter()
                continue

            fh, fw = cpu_frame.shape[:2]

            now_perf = time.perf_counter()
            dt = min(now_perf - self._last_lerp_time, 0.05)
            self._last_lerp_time = now_perf

            with self._boxes_lock:
                self._boxes = [b for b in self._boxes if not b.is_stale()]
                snapshot = list(self._boxes)

            for sb in snapshot:
                sb.lerp_dt(dt)

            frame = cpu_frame.copy()

            for sb in snapshot:
                staleness = sb.staleness()
                alpha = max(0.0, 1.0 - staleness)
                if alpha < 0.05:
                    continue

                top, right, bottom, left = sb.ints()
                top    = max(0, min(top,    fh - 1))
                bottom = max(0, min(bottom, fh - 1))
                left   = max(0, min(left,   fw - 1))
                right  = max(0, min(right,  fw - 1))
                if right <= left or bottom <= top:
                    continue

                if sb.name != "Unknown":
                    base_color = (0, 255, 130)
                    label = f"{sb.name}  {int(sb.confidence * 100)}%"
                else:
                    base_color = (50, 150, 255)
                    label = "Unknown"

                color = tuple(int(c * alpha) for c in base_color)
                thickness = 2 if alpha > 0.5 else 1

                _draw_corner_tracers(frame, left, top, right, bottom,
                                     color, thickness=thickness, alpha=alpha)
                if alpha > 0.7:
                    _draw_scan_line(frame, left, top, right, bottom,
                                    color, thickness=1)

                font_scale = 0.58
                (lw, lh), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                label_y = top - 10 if top - 10 > lh + 4 else bottom + lh + 10

                rx1 = max(0, left)
                ry1 = max(0, label_y - lh - 6)
                rx2 = min(fw, left + lw + 10)
                ry2 = min(fh, label_y + 4)
                if rx2 > rx1 and ry2 > ry1:
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, -1)

                cv2.putText(frame, label, (left + 5, label_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), 1, cv2.LINE_AA)

            # Resize to display target and convert to RGB numpy for the queue
            with self._display_target_lock:
                disp_w, disp_h = self._display_target

            if (disp_w, disp_h) != (frame.shape[1], frame.shape[0]):
                interp = cv2.INTER_LINEAR if disp_w > frame.shape[1] else cv2.INTER_AREA
                out = cv2.resize(frame, (disp_w, disp_h), interpolation=interp)
            else:
                out = frame
            # Convert to contiguous RGB array — _poll_frame reads this directly
            # into QImage with zero compression, no PIL wrapper needed
            rgb = np.ascontiguousarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

            # Keep only the freshest frame in the queue
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(rgb)
            except queue.Full:
                pass

            next_tick += _RENDER_INTERVAL
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()

    # ── _poll_frame — Qt display update (driven by self._poll_timer) ──────────
    def _poll_frame(self):
        if not self.running:
            return
        try:
            # Compute display target size from the video_label geometry
            cw = self.video_label.width()
            ch = self.video_label.height()
            if cw > 100 and ch > 100:
                if self._last_display_size != (cw, ch):
                    available_w = cw - 16
                    available_h = ch - 52
                    img_aspect = DISPLAY_WIDTH / DISPLAY_HEIGHT
                    test_h = available_w / img_aspect
                    if test_h <= available_h:
                        new_w, new_h = int(available_w), int(test_h)
                    else:
                        new_h = int(available_h)
                        new_w = int(available_h * img_aspect)
                    new_w = max(1, new_w)
                    new_h = max(1, new_h)
                    if new_w > MAX_DISPLAY_WIDTH or new_h > MAX_DISPLAY_HEIGHT:
                        cap_scale = min(MAX_DISPLAY_WIDTH / new_w,
                                        MAX_DISPLAY_HEIGHT / new_h)
                        new_w = max(1, int(new_w * cap_scale))
                        new_h = max(1, int(new_h * cap_scale))
                    self._last_display_size = (cw, ch)
                    self._last_render_size  = (new_w, new_h)
                    with self._display_target_lock:
                        self._display_target = (new_w, new_h)

            # Grab pre-converted RGB array and push to QLabel as QPixmap
            rgb = self._frame_queue.get_nowait()
            try:
                while True:
                    rgb = self._frame_queue.get_nowait()
            except queue.Empty:
                pass

            # pyrefly: ignore [missing-import]
            from PyQt6.QtGui import QImage
            h_px, w_px, ch = rgb.shape
            qimg = QImage(rgb.data, w_px, h_px, w_px * ch, QImage.Format.Format_RGB888)
            px = QPixmap.fromImage(qimg.copy())
            self.video_label.setPixmap(px)

            self._fps_counter += 1
            now = time.time()
            if now - self._fps_timer >= 1.0:
                detections = len(self._boxes)
                cam_fps = self.cap.fps if hasattr(self.cap, 'fps') else 0
                status_text = (
                    f"● LIVE  |  App: {self._fps_counter} FPS  "
                    f"|  Cam: {cam_fps} FPS  |  Faces: {detections}  "
                    f"|  Today: {len(self.today_logged)}"
                )
                self.status_bar.setText(status_text)
                self.status_bar.setStyleSheet(
                    f"color: {self.colors['success']}; background: transparent;")
                self._fps_counter = 0
                self._fps_timer   = now

            return

        except queue.Empty:
            pass
        except Exception as e:
            log.debug(f"Frame display error: {e}")

    # ── _detect_loop ──────────────────────────────────────────────────────────
    def _detect_loop(self):
        while self.running:
            triggered = self._detect_event.wait(timeout=0.1)
            if not triggered or not self.running:
                continue
            self._detect_event.clear()

            with self._pending_frame_lock:
                frame = self._pending_frame
            if frame is None:
                continue

            try:
                analyzed = encoder.analyze_frame(frame)
            except Exception as e:
                log.debug(f"InsightFace analyze error: {e}")
                continue

            if not analyzed:
                continue

            fh, fw = frame.shape[:2]
            new_detections = []

            for face in analyzed:
                emb = face["embedding"]
                x1, y1, x2, y2 = face["bbox"]
                x1 = max(0, min(x1, fw - 1)); y1 = max(0, min(y1, fh - 1))
                x2 = max(0, min(x2, fw - 1)); y2 = max(0, min(y2, fh - 1))
                top, right, bottom, left = y1, x2, y2, x1

                with self._enc_lock:
                    name, conf = self._identify_face(emb)

                new_detections.append((top, right, bottom, left, name, conf))

                if name != "Unknown":
                    if name not in self._greeted_session:
                        self._greeted_session.add(name)
                        khmer_name = self.known_khmer_names.get(name, "")
                        speak_greeting(name, khmer_name)
                        display = name.title()
                        self.after(0, lambda d=display: self.show_greeting_toast(d))

                    self._queue_recognition(name, conf)

            # Merge detections into box list
            with self._boxes_lock:
                existing = self._boxes.copy()
                used = set()
                matched = []

                for det in new_detections:
                    dt, dr, db, dl, dname, dconf = det
                    best_idx, best_score = -1, -1.0

                    for j, sb in enumerate(existing):
                        if j in used:
                            continue
                        iou = _iou(dt, dr, db, dl, sb.tt, sb.tr, sb.tb, sb.tl)
                        cdist = _center_dist(dt, dr, db, dl,
                                             sb.tt, sb.tr, sb.tb, sb.tl)
                        max_dist = (DISPLAY_WIDTH ** 2 + DISPLAY_HEIGHT ** 2) ** 0.5
                        cdist_score = max(0.0, 1.0 - cdist / (max_dist * 0.3))
                        score = iou * 0.7 + cdist_score * 0.3
                        if score > best_score and score > 0.2:
                            best_score, best_idx = score, j

                    if best_idx >= 0:
                        existing[best_idx].set_target(dt, dr, db, dl, dname, dconf)
                        matched.append(existing[best_idx])
                        used.add(best_idx)
                    else:
                        matched.append(SmoothBox(dt, dr, db, dl, dname, dconf))

                for j, sb in enumerate(existing):
                    if j not in used and not sb.is_stale():
                        matched.append(sb)

                self._boxes = matched

    # ── Recognition logging ───────────────────────────────────────────────────
    def _queue_recognition(self, name, confidence):
        if name in self.today_logged:
            return
        now = time.time()
        self._pending_recognition = {
            k: v for k, v in self._pending_recognition.items()
            if now - v['last_seen'] < 3.0}
        if name not in self._pending_recognition:
            self._pending_recognition[name] = {
                'count': 1, 'last_seen': now, 'confidence': confidence}
        else:
            self._pending_recognition[name].update({
                'count': self._pending_recognition[name]['count'] + 1,
                'last_seen': now,
                'confidence': confidence,
            })
        if self._pending_recognition[name]['count'] >= 3:
            self.today_logged.add(name)
            payload = dict(self._pending_recognition[name])
            payload['name'] = name
            payload['date'] = str(date.today())
            self._recognition_queue.put(payload)
            del self._pending_recognition[name]

    def _process_recognition_queue(self):
        try:
            while True:
                payload = self._recognition_queue.get_nowait()
                now = datetime.now().strftime("%H:%M:%S")
                database.log_recognition(
                    payload['name'], payload['date'], now,
                    confidence=payload.get('confidence', 0.0),
                )
                self._append_live_log(f"[{now}]  {payload['name']}\n", payload['date'])
                log.info(f"Recognition logged: {payload['name']} at {now}")
        except queue.Empty:
            pass
        finally:
            self._process_id = self.after(100, self._process_recognition_queue)

    # ── Face Recognition ──────────────────────────────────────────────────────
    def _identify_face(self, embedding: np.ndarray):
        if self._known_matrix is None:
            return "Unknown", 0.0
        sims = self._known_matrix @ embedding
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= CONFIDENCE_THRESHOLD:
            return self.known_names[best_idx], best_sim
        return "Unknown", best_sim

    def _append_live_log(self, entry, today):
        """Append text to the QTextEdit live log (called from main thread via after())."""
        try:
            self.live_log.append(entry.rstrip())
            # Scroll to bottom
            sb = self.live_log.verticalScrollBar()
            sb.setValue(sb.maximum())
            count = len(self.today_logged)
            self.stat_today.setText(str(count))
            self.live_log_count.setText(f"{count} seen today")
        except Exception:
            pass

    # ── Registration (multi-photo) ────────────────────────────────────────────
    def _capture_register(self):
        with self._last_frame_lock:
            frame = self._last_good_frame
        if frame is not None:
            self._reg_captured_frame = frame.copy()
            self._reg_staged_frames.append(self._reg_captured_frame)
            self._show_reg_preview(self._reg_captured_frame)
            self.refresh_photo_strip(len(self._reg_staged_frames))
            self.reg_status.setText("✓ Captured!")
            self.reg_status.setStyleSheet("color: #10B981; background: transparent;")
        else:
            self.reg_status.setText("✗ Camera not ready — start Live Camera first")
            self.reg_status.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")

    def _upload_photo(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Photo", "",
            "Image files (*.jpg *.jpeg *.png);;All files (*.*)")
        if not file_path:
            return
        frame = self._read_image_unicode_safe(file_path)
        if frame is None:
            self.reg_status.setText("✗ Failed to load image")
            self.reg_status.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            return
        self._reg_captured_frame = frame
        self._reg_staged_frames.append(frame)
        self._show_reg_preview(frame)
        self.refresh_photo_strip(len(self._reg_staged_frames))
        self.reg_status.setText("✓ Photo uploaded!")
        self.reg_status.setStyleSheet("color: #10B981; background: transparent;")

    @staticmethod
    def _read_image_unicode_safe(file_path):
        try:
            frame_data = np.fromfile(file_path, dtype=np.uint8)
            return cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        except Exception as e:
            log.error(f"Failed to read image {file_path}: {e}")
            return None

    def _show_reg_preview(self, frame):
        """Show a BGR numpy frame in the registration preview QLabel."""
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil.thumbnail((400, 300))
        px = _pil_to_qpixmap(pil)
        self.reg_preview.setPixmap(px)
        self.reg_preview.setText("")

    def _save_registration(self):
        fname = self.reg_fname.text().strip().upper()
        lname = self.reg_lname.text().strip().upper()
        raw_name = f"{fname} {lname}".strip() if fname and lname else fname or lname
        name = _sanitize_name(raw_name)
        khmer_name = self.reg_khmer_name.text().strip()
        person_id = self.reg_id.text().strip() or "N/A"

        if not name or len(name) < 2:
            self.reg_status.setText("✗ Enter a valid name")
            self.reg_status.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            return
        if not self._reg_staged_frames:
            self.reg_status.setText("✗ Capture or upload at least one photo")
            self.reg_status.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            return

        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        base = name.replace(' ', '_')
        primary_path = os.path.join(KNOWN_FACES_DIR, f"{base}.jpg")
        cv2.imwrite(primary_path, self._reg_staged_frames[0])
        database.register_person(name, person_id, primary_path, khmer_name)

        for extra_frame in self._reg_staged_frames[1:]:
            extra_path = encoder.next_multi_photo_path(KNOWN_FACES_DIR, name)
            cv2.imwrite(extra_path, extra_frame)
            database.add_face_photo(name, extra_path)

        encoder.invalidate_cache()

        with self._enc_lock:
            self.reload_known_faces()

        photo_count = len(self._reg_staged_frames)
        self.reg_status.setText(
            f"✓ {name} registered with {photo_count} photo(s)!")
        self.reg_status.setStyleSheet("color: #10B981; background: transparent;")

        self.reg_fname.clear()
        self.reg_lname.clear()
        self.reg_khmer_name.clear()
        self.reg_id.clear()
        self._reg_captured_frame = None
        self._reg_staged_frames = []
        self.refresh_photo_strip(0)
        self.reg_preview.setPixmap(QPixmap())
        self.reg_preview.setText("No photo yet\nCapture or upload a photo")
        self.stat_registered.setText(str(len(self.known_names)))
        log.info(f"Registered new person: {name} ({photo_count} photo(s))")

    def reload_known_faces(self):
        self.known_encodings, self.known_names = encoder.load_known_faces(KNOWN_FACES_DIR)
        self._known_matrix = np.array(self.known_encodings) if self.known_encodings else None
        db_people = {row[0] for row in database.get_all_people()}
        for name in self.known_names:
            if name not in db_people:
                path = os.path.join(KNOWN_FACES_DIR, f"{name.replace(' ', '_')}.jpg")
                database.register_person(name, "N/A", path)
        self.known_khmer_names = {
            row[0]: (row[3] or "") for row in database.get_all_people()
        }

    # ── Add Photo (Manage Faces) ──────────────────────────────────────────────
    def _add_photo_to_selected(self):
        selected = self.manage_tree.selection()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select a person first.")
            return
        item = self.manage_tree.item(selected[0])
        name = item["values"][0]

        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Add a photo for {name}", "",
            "Image files (*.jpg *.jpeg *.png);;All files (*.*)")
        if not file_path:
            return
        frame = self._read_image_unicode_safe(file_path)
        if frame is None:
            QMessageBox.critical(self, "Error", "Failed to load that image.")
            return

        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        new_path = encoder.next_multi_photo_path(KNOWN_FACES_DIR, name)
        cv2.imwrite(new_path, frame)
        database.add_face_photo(name, new_path)
        encoder.invalidate_cache()

        with self._enc_lock:
            self.reload_known_faces()

        self._refresh_manage()
        QMessageBox.information(
            self, "Photo Added", f"Added a new enrollment photo for {name}.")
        log.info(f"Added enrollment photo for {name}: {new_path}")

    # ── Edit ──────────────────────────────────────────────────────────────────
    def _edit_selected_face(self):
        selected = self.manage_tree.selection()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select a face to edit.")
            return
        item = self.manage_tree.item(selected[0])
        name, khmer_name, person_id, _photos, _registered = item["values"]

        def _on_save(new_name_raw, new_khmer_name, new_person_id):
            new_name = _sanitize_name(new_name_raw.upper())
            if not new_name or len(new_name) < 2:
                return False, "Enter a valid name."

            renamed = new_name != name
            if renamed and new_name in set(self.known_names):
                return False, f"'{new_name}' already exists."

            try:
                if renamed:
                    old_base = name.replace(' ', '_')
                    new_base = new_name.replace(' ', '_')
                    new_primary_path = None
                    listing = os.listdir(KNOWN_FACES_DIR) if os.path.isdir(KNOWN_FACES_DIR) else []
                    for filename in listing:
                        stem = os.path.splitext(filename)[0]
                        ext  = os.path.splitext(filename)[1]
                        if stem == old_base:
                            new_path = os.path.join(KNOWN_FACES_DIR, f"{new_base}{ext}")
                            os.replace(os.path.join(KNOWN_FACES_DIR, filename), new_path)
                            new_primary_path = new_path
                        elif encoder.re_strip_multi_suffix(stem) == old_base:
                            suffix = stem[len(old_base):]
                            new_path = os.path.join(KNOWN_FACES_DIR, f"{new_base}{suffix}{ext}")
                            os.replace(os.path.join(KNOWN_FACES_DIR, filename), new_path)

                    if new_primary_path is None:
                        new_primary_path = os.path.join(KNOWN_FACES_DIR, f"{new_base}.jpg")

                    database.update_person(
                        name, new_name, new_person_id, new_khmer_name,
                        photo_path=new_primary_path)
                    encoder.invalidate_cache()
                    self.today_logged = {
                        k.replace(f"{name}_", f"{new_name}_", 1) if k.startswith(f"{name}_") else k
                        for k in self.today_logged
                    }
                    # _greeting_cooldown was removed in the session-based approach;
                    # just remove from the greeted set so they can be greeted as their new name.
                    self._greeted_session.discard(name)
                else:
                    database.update_person(
                        name, new_name, new_person_id, new_khmer_name)

                with self._enc_lock:
                    self.reload_known_faces()

            except Exception as e:
                log.error(f"Failed to update person '{name}': {e}")
                return False, "Update failed — see log for details."

            self._refresh_manage()
            log.info(f"Edited person: {name} -> {new_name}")
            return True, ""

        self._open_edit_dialog(name, khmer_name, person_id, _on_save)

    # ── Face Search ───────────────────────────────────────────────────────────
    def _upload_search_photo(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select a photo to search", "",
            "Image files (*.jpg *.jpeg *.png);;All files (*.*)")
        if not file_path:
            return
        frame = self._read_image_unicode_safe(file_path)
        if frame is None:
            self.search_status.setText("✗ Failed to load image")
            self.search_status.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            return
        self._search_frame = frame
        self._show_search_preview(frame)
        self.search_status.setText("✓ Photo loaded — tap Find Matches")
        self.search_status.setStyleSheet("color: #10B981; background: transparent;")

    def _capture_search(self):
        with self._last_frame_lock:
            frame = self._last_good_frame
        if frame is None:
            self.search_status.setText(
                "✗ Camera not ready — start Live Camera first")
            self.search_status.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            return
        self._search_frame = frame.copy()
        self._show_search_preview(self._search_frame)
        self.search_status.setText("✓ Captured — tap Find Matches")
        self.search_status.setStyleSheet("color: #10B981; background: transparent;")

    def _show_search_preview(self, frame):
        """Show a BGR numpy frame in the search preview QLabel."""
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil.thumbnail((400, 300))
        px = _pil_to_qpixmap(pil)
        self.search_preview.setPixmap(px)
        self.search_preview.setText("")

    def _run_face_search(self):
        if self._search_frame is None:
            self.search_status.setText("✗ Upload or capture a photo first")
            self.search_status.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            return

        emb, bbox = encoder.get_embedding(self._search_frame)
        if emb is None:
            self.search_status.setText("✗ No face detected in that photo")
            self.search_status.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            self.render_search_results([])
            return

        with self._enc_lock:
            matches = encoder.search_face(
                emb, self.known_encodings, self.known_names, top_k=5)

        results = [(name, sim, self.known_khmer_names.get(name, ""))
                   for name, sim in matches]
        self.render_search_results(results)
        if results:
            self.search_status.setText(f"Found {len(results)} candidate match(es)")
            self.search_status.setStyleSheet(
                f"color: {self.colors['muted']}; background: transparent;")
        else:
            self.search_status.setText(
                "No enrolled faces to compare against — register someone first")
            self.search_status.setStyleSheet(
                f"color: {self.colors['muted']}; background: transparent;")

    def _refresh_manage(self):
        people = database.get_all_people()
        for row in self.manage_tree.get_children():
            self.manage_tree.delete(row)
        for name, person_id, registered_at, khmer_name, photo_count in people:
            self.manage_tree.insert(
                "", "end", values=(name, khmer_name or "", person_id,
                                   photo_count, registered_at))
        count = len(people)
        self.manage_count_label.setText(
            f"{count} {'person' if count == 1 else 'people'}")
        self.stat_registered.setText(str(len(self.known_names)))

    def _delete_selected_face(self):
        selected = self.manage_tree.selection()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select a face to delete.")
            return
        item = self.manage_tree.item(selected[0])
        name = item["values"][0]
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete '{name}', all their enrollment photos, and all their "
            f"recognition history?\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        encoder.delete_face(KNOWN_FACES_DIR, name)
        database.delete_person(name)
        with self._enc_lock:
            self.reload_known_faces()
        self._refresh_manage()
        log.info(f"Deleted face and records: {name}")

    # ── Shutdown ──────────────────────────────────────────────────────────────
    def _on_close(self):
        log.info("Shutting down Face Recognition System")
        self.running = False
        self._detect_event.set()
        shutdown_tts()
        if getattr(self, '_poll_timer', None):
            self._poll_timer.stop()
        if hasattr(self, '_process_id'):
            self.after_cancel(self._process_id)
        if getattr(self, '_greeting_reset_timer', None):
            self._greeting_reset_timer.stop()
        if self.cap:
            self.cap.release()
        if _HIGH_RES_TIMER_SET:
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    log.info("Starting Face Recognition System (PyQt6 port)")
    app = QApplication(sys.argv)
    win = FaceRecognitionApp()
    win.show()
    sys.exit(app.exec())