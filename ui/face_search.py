"""
userinterface.face_search
==========================
FaceSearchMixin: "who is this?" reverse lookup — upload or capture a
photo, embed it, and rank it against the enrolled gallery.
"""

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import QFileDialog
# pyrefly: ignore [missing-import]
from PIL import Image

from services import encoder
from .qt_utils import _pil_to_qpixmap


class FaceSearchMixin:
    """Face Search tab logic: upload/capture a probe photo and rank it
    against the enrolled gallery."""

    def _upload_search_photo(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select a photo to search", "",
            "Image files (*.jpg *.jpeg *.png);;All files (*.*)")
        if not file_path:
            return
        frame = self._read_image_unicode_safe(file_path)
        if frame is None:
            self.search_status.setText("Failed to load image")
            self.search_status.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            return
        self._search_frame = frame
        self._show_search_preview(frame)
        self.search_status.setText("Photo loaded — tap Find Matches")
        self.search_status.setStyleSheet("color: #10B981; background: transparent;")

    def _capture_search(self):
        with self._last_frame_lock:
            frame = self._last_good_frame
        if frame is None:
            self.search_status.setText(
                "Camera not ready — start Live Camera first")
            self.search_status.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            return
        self._search_frame = frame.copy()
        self._show_search_preview(self._search_frame)
        self.search_status.setText("Captured — tap Find Matches")
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
            self.search_status.setText("Upload or capture a photo first")
            self.search_status.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            return

        emb, bbox = encoder.get_embedding(self._search_frame)
        if emb is None:
            self.search_status.setText("No face detected in that photo")
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

