"""
userinterface.app
==================
FaceRecognitionApp: assembles the GUI (FaceRecognitionGUI) with the
pipeline/registration/search/management mixins into the final
application class, and owns process-wide startup (Windows high-res
timer + priority, logging) and shutdown.
"""

import sys
import time
import queue
import threading
import ctypes
import logging

# pyrefly: ignore [missing-import]
import cv2

from services import database
from .gui import FaceRecognitionGUI
from services.tts import shutdown_tts

from core.config import DISPLAY_WIDTH, DISPLAY_HEIGHT
from core.tracking import LostTrackBuffer
from core.pipeline import VideoPipelineMixin
from .registration import RegistrationMixin
from .face_search import FaceSearchMixin
from .management import ManagementMixin

# ── Windows high-res timer + priority ────────────────────────────────────────
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


class FaceRecognitionApp(FaceRecognitionGUI, VideoPipelineMixin,
                          RegistrationMixin, FaceSearchMixin, ManagementMixin):
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
        self._lost_tracks          = LostTrackBuffer()
        self._box_embeddings       = {}   # track_id -> latest embedding (for re-ID save)
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


    # ── Shutdown ─────────────────────────────────────────────────────────────
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
