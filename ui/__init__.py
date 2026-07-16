"""
userinterface — the PyQt6 GUI: FaceRecognitionGUI (widgets/layout/theme),
the tab mixins (registration/face_search/management), and app.py, which
assembles FaceRecognitionGUI + core.pipeline.VideoPipelineMixin + the tab
mixins into the final FaceRecognitionApp class.
"""

from .app import FaceRecognitionApp

__all__ = ["FaceRecognitionApp"]
