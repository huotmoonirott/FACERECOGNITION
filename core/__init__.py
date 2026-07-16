"""
core — the recognition engine: config constants, per-face tracking
(SmoothBox/LostTrackBuffer), the threaded camera reader, and the
capture/detect/track/render pipeline (VideoPipelineMixin). No PyQt6
widget/layout code lives here — only the display glue (QPixmap/QTimer)
needed to hand frames to the UI.
"""
