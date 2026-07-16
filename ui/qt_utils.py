"""
userinterface.qt_utils
=======================
Small conversion/formatting helpers shared across the pipeline,
registration, and search mixins: numpy/PIL -> QPixmap conversion and
filesystem-safe name sanitization.
"""

import re

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PyQt6.QtGui import QPixmap
# pyrefly: ignore [missing-import]
from PIL import Image


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


def _sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name
