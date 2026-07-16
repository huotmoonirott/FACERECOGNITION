"""
userinterface.management
=========================
ManagementMixin: the "Manage Faces" tab — list all enrolled people and
delete a person (their photos, embeddings, and recognition history).
"""

import logging
log = logging.getLogger("FaceRecog")

# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import QMessageBox

from services import database
from services import encoder

from core.config import KNOWN_FACES_DIR


class ManagementMixin:
    """Manage Faces tab logic: list/refresh and delete enrolled people."""

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