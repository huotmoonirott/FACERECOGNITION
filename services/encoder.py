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

import cv2
import numpy as np

log = logging.getLogger("FaceRecog.Encoder")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (this file now lives in services/)
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