# Performance Notes

Face Recognition System
=========================
Author: JJ — RUPP Computer Architecture Final Project

PERFORMANCE PATCH (perf-v1) — see PERF_NOTES.md for the full writeup.

Summary of what changed vs the v15 baseline and why:

  1. REMOVED the per-frame CPU correlation tracker (_track_loop /
     _create_tracker / cv2.legacy.TrackerMOSSE / TrackerKCF).
     This was the single biggest source of GIL contention: a pure-Python
     + pure-CPU tracker.update() call running on EVERY rendered frame,
     for EVERY tracked face, fighting _video_loop and _detect_loop for
     the same GIL. SmoothBox already has velocity-based prediction
     (vt/vr/vb/vl + PREDICTION_WEIGHT) designed to carry a box between
     detection cycles — the tracker was doing redundant work against
     its own prediction system. Boxes now coast purely on lerp/velocity
     prediction between detections, which is computationally free
     (a handful of float ops) compared to a CV tracker update.

  2. _video_loop now sleeps briefly when no new camera frame is
     available, instead of busy-spinning. A tight no-sleep loop never
     voluntarily yields the GIL for long stretches, which was starving
     the detect thread and the Tkinter main loop.

  3. _poll_frame no longer rebuilds a brand-new PIL.Image + CTkImage on
     every single 8ms tick regardless of whether a new frame arrived.
     It now only does that expensive resize+convert work when a frame
     is actually available, and caches the target display size so we
     don't recompute the aspect-fit math every tick either.

  4. Liveness's FFT-based heuristic (encoder._estimate_liveness) used to
     run on EVERY detected face on EVERY detect cycle (every 5th camera
     frame), even though the UI badge update was already throttled by
     ATTRIBUTE_REFRESH_INTERVAL. The actual computation is now gated by
     that same per-name cooldown, so the expensive FFT only runs ~10x/sec
     per visible identity instead of ~8x/sec PER FACE regardless of cooldown.

  5. encoder.analyze_frame() gained an optional `want_liveness` flag
     (default True) so identity-only passes can skip the FFT path
     entirely. app.py now does a lightweight pass when nobody is due for
     an attribute refresh.

PERFORMANCE PATCH (perf-v2) — UI smoothness pass on top of perf-v1.

  6. _poll_frame was handing CTkImage a full 800x600 PIL image every
     tick and letting CTkImage resize it down internally. Confirmed by
     reading customtkinter's ctk_image.py: CTkImage builds a fresh
     internal cache every call (we create a new CTkImage instance every
     frame) and ALWAYS calls PIL .resize(target_size) — Pillow only
     short-circuits that into a cheap .copy() when the source size
     already equals the target size, otherwise it's a full resample
     pass over every pixel, on the Tkinter main thread, every single
     displayed frame. We now resize with cv2 (SIMD-accelerated, far
     cheaper than PIL for this) to the EXACT target size before handing
     it to PIL/CTkImage, turning that resample into a no-op copy, and we
     do it BEFORE cvtColor so the color conversion also runs on the
     smaller image instead of the full frame.

  7. _identify_face rebuilt np.array(self.known_encodings) from a Python
     list on EVERY detected face, on EVERY detect cycle. Now cached as
     self._known_matrix, rebuilt only inside reload_known_faces() (i.e.
     only when someone is registered/edited/deleted) — removes per-face
     GIL/CPU work that was competing with the video thread for nothing.

  8. Greeting toast: two recognitions within the 4s display window used
     to stack two independent after(4000, _reset_greeting) calls, where
     the first one's reset could stomp the second toast mid-display.
     Now the pending reset timer is cancelled before scheduling a new one.

Everything else — InsightFace identity matching, multi-photo enrollment,
Khmer TTS greeting flow, the recognition event log, Face Search tab — is
UNCHANGED in behavior.

PERFORMANCE PATCH (perf-v5) — camera ceiling diagnosis + render decoupling.

  9. Measured this rig's actual camera (camera_speed_test.py, 8 modes
     tested) and found it has a hard ~30fps ceiling: 1920x1080@30 was the
     FASTEST mode measured, not the slowest — every lower resolution
     tested was paradoxically slower (720p ~7.7fps, 480p ~23fps). This is
     a hardware/driver limit, not something software can raise.
     ThreadedCamera now explicitly requests 1920x1080@30 (the proven-best
     mode) instead of blindly asking for 120fps and hoping.

  10. DISPLAY_WIDTH/HEIGHT changed from 800x600 (4:3) to 960x540 (16:9)
      to match the camera's native aspect ratio — the old setting was
      silently squashing every frame.

  11. _video_loop split into two threads: _video_loop (pure camera
      capture, naturally bounded by the camera's own ~30fps hardware
      pace) and _render_loop (lerp + draw, running on its own
      RENDER_TARGET_FPS timer, independent of when the next real camera
      frame arrives). Box positions are now interpolated forward on
      every render tick rather than only once per captured frame, so
      on-screen motion looks smoother than the raw 30fps camera feed —
      same principle as motion interpolation in video players. This does
      NOT increase real camera FPS (impossible on this hardware); it
      only smooths the perceived motion of a 30fps source.

PyQt6 PORT — all customtkinter / tkinter / PIL.ImageTk dependencies removed.
  - messagebox / filedialog  → QMessageBox / QFileDialog
  - CTkImage / ImageTk       → QPixmap (via cv2 + bytes)
  - label.configure(...)     → label.setText() / label.setStyleSheet()
  - live_log.configure(...)  → QTextEdit native API
  - _poll_frame display      → QLabel.setPixmap() with QPixmap.loadFromData()
  - mainloop()               → QApplication.exec()
