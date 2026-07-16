"""
core.camera_stream
===================
ThreadedCamera: a background-thread camera reader so capture never blocks
the video/render/detect loops. Requests 1920x1080@30 MJPG, which was
measured (see tools/camera_speed_test.py) to be the fastest mode this
rig's camera actually supports.
"""

import threading
import time
import logging

# pyrefly: ignore [missing-import]
import cv2

log = logging.getLogger("FaceRecog")

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

