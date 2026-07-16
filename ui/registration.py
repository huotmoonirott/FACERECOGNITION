"""
userinterface.registration
===========================
RegistrationMixin: multi-photo face enrollment (capture/upload/save),
reloading the in-memory known-faces gallery after any change, adding
extra photos to an existing person, and the edit-person flow (rename,
re-key photo files, propagate the rename into recognition history).
"""

import os

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import QFileDialog, QMessageBox
# pyrefly: ignore [missing-import]
from PyQt6.QtGui import QPixmap
# pyrefly: ignore [missing-import]
from PIL import Image

import logging
log = logging.getLogger("FaceRecog")

from services import database
from services import encoder

from core.config import KNOWN_FACES_DIR
from .qt_utils import _pil_to_qpixmap, _sanitize_name


class RegistrationMixin:
    """Enrollment tab logic: capture/upload staged photos, save a new
    person, add photos to / edit an existing person."""

    def _capture_register(self):
        with self._last_frame_lock:
            frame = self._last_good_frame
        if frame is not None:
            self._reg_captured_frame = frame.copy()
            self._reg_staged_frames.append(self._reg_captured_frame)
            self._show_reg_preview(self._reg_captured_frame)
            self.refresh_photo_strip(len(self._reg_staged_frames))
            self.reg_status.setText("Captured")
            self.reg_status.setStyleSheet("color: #10B981; background: transparent;")
        else:
            self.reg_status.setText("Camera not ready — start Live Camera first")
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
            self.reg_status.setText("Failed to load image")
            self.reg_status.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            return
        self._reg_captured_frame = frame
        self._reg_staged_frames.append(frame)
        self._show_reg_preview(frame)
        self.refresh_photo_strip(len(self._reg_staged_frames))
        self.reg_status.setText("Photo uploaded")
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
            self.reg_status.setText("Enter a valid name")
            self.reg_status.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            return
        if not self._reg_staged_frames:
            self.reg_status.setText("Capture or upload at least one photo")
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
        photo_word = "photo" if photo_count == 1 else "photos"
        self.reg_status.setText(
            f"{name} registered with {photo_count} {photo_word}")
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

        # FIX: Only trust the NAME from the tree widget — it's the unique
        # key (matches the 'people.name' UNIQUE column). Everything else
        # (khmer_name, person_id) is fetched fresh from the DB via
        # database.get_person(), instead of being unpacked positionally
        # from item["values"]. Positional unpacking silently breaks the
        # moment the tree's column order doesn't exactly match the DB's
        # column order (e.g. management.py inserting columns in a
        # different sequence than get_all_people() returns them) — that
        # mismatch is what was causing khmer_name to go missing/scrambled
        # whenever you opened the Edit dialog.
        name = item["values"][0]

        row = database.get_person(name)
        if row is None:
            QMessageBox.critical(self, "Error", f"'{name}' not found in database.")
            return
        _db_name, person_id, khmer_name, _photo_path, _registered_at = row

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
                    # today_logged holds plain names (see _queue_recognition:
                    # `self.today_logged.add(name)`), never "name_..."
                    # compound keys, so the old prefix-replace comprehension
                    # here (`k.startswith(f"{name}_")`) could never match
                    # anything and silently left the OLD name sitting in
                    # today_logged forever. That meant a renamed person who
                    # was already logged today would get re-logged (and
                    # re-counted in the "Today" stat) under their new name
                    # the next time they were recognized. Just swap the
                    # entry itself.
                    if name in self.today_logged:
                        self.today_logged.discard(name)
                        self.today_logged.add(new_name)
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