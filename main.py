"""
Face Recognition System — entry point
=======================================
Author: JJ — RUPP Computer Architecture Final Project

Launches the PyQt6 application. Logic is split across core/ (recognition
engine), services/ (database/InsightFace/TTS), and userinterface/ (PyQt6
GUI + app assembly) — see each package's __init__.py, and
docs/PERF_NOTES.md for the performance-tuning history (perf-v1/v2/v5
patches, camera ceiling diagnosis, PyQt6 port notes).
"""

import sys
import logging

# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import QApplication

from ui import FaceRecognitionApp

if __name__ == "__main__":
    logging.getLogger("FaceRecog").info("Starting Face Recognition System (PyQt6 port)")
    app = QApplication(sys.argv)
    win = FaceRecognitionApp()
    win.show()
    sys.exit(app.exec())
