# RECOGFACE

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-41cd52)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**RUPP Computer Architecture Final Project — JJ**

A real-time face recognition desktop app built to explore concurrency,
hardware limits, and identity under uncertainty — not just a webcam
feed with a bounding box. PyQt6 UI, InsightFace (buffalo_l) for
detection/embedding, OpenCV for capture/rendering, SQLite for
people/photos/recognition history, and Khmer TTS greetings.

## What it does

- **Live recognition** — detects and identifies enrolled faces in
  real time, with confidence-gated identity locking so a lookalike or
  a bad-angle frame can't flip a confirmed name.
- **Survives occlusion** — a track stays alive through a turned head
  or a brief walk-off-frame, using a growing-uncertainty motion model
  plus a short-term re-identification buffer.
- **Speaks back** — greets recognized people by name using a Khmer
  neural TTS voice.
- **Full enrollment workflow** — register, search ("who is this?"),
  and manage enrolled identities from dedicated tabs.

## ⚠️ Data & privacy

This repo ships with an **empty** `known_faces/` and no database — on
purpose. Enrolled faces are personal biometric data belonging to real
people, so no sample photos, no `face_recognition.db`, and no
recognition history are committed here (see `.gitignore`). Only enroll
people who've actually agreed to it, and double-check `git status`
before your first commit if you're working from an existing local
setup.

## Running it

```bash
pip install -r requirements.txt
python main.py
```

Requires a webcam. GPU (CUDA) is used automatically if available and
falls back to CPU otherwise — see `requirements.txt` for the
`onnxruntime` vs `onnxruntime-gpu` choice.

### First run

There's nothing to enroll yet — `known_faces/` and the database are
created fresh on first launch. Open the **Register Face** tab, add a
name (and optionally a Khmer name), and capture a few photos. The
**Live Camera** tab will start recognizing that person immediately.

## Project layout

```
main.py                        entry point — QApplication + FaceRecognitionApp

core/                          the recognition engine — no widget/layout code
  config.py                      every tunable constant (thresholds, timeouts, sizes)
  tracking.py                    SmoothBox (velocity-predicted box), LostTrackBuffer
                                  (re-ID), IoU/center-distance helpers, tracer drawing
  camera_stream.py               ThreadedCamera — background-thread camera reader
  pipeline.py                    VideoPipelineMixin — capture/detect/track/render
                                  loops, recognition queue, identity matching

services/                      I/O and ML backends — no PyQt6 dependency
  database.py                    SQLite schema + queries (people/face_photos/events)
  encoder.py                      InsightFace loading, embedding, liveness heuristic
  tts.py                          Khmer greeting TTS (edge-tts)

ui/                             PyQt6 GUI
  gui.py                          FaceRecognitionGUI — all widget/layout/theme code
  icons.py                        legacy icon loader (inactive — gui.py renders
                                   icons natively now; kept for reference)
  qt_utils.py                     numpy/PIL <-> QPixmap conversion, name sanitizing
  registration.py                 RegistrationMixin — enroll/add-photo/edit-person
  face_search.py                  FaceSearchMixin — "who is this?" reverse lookup
  management.py                   ManagementMixin — Manage Faces tab (list/delete)
  app.py                           FaceRecognitionApp — combines FaceRecognitionGUI +
                                   core.pipeline.VideoPipelineMixin + the tab mixins;
                                   owns process startup (Windows timer/priority,
                                   logging) and shutdown

known_faces/                   enrolled reference photos (gitignored — empty on clone)
face_recognition.db            SQLite database (gitignored — created on first run)
docs/PERF_NOTES.md             perf-tuning history (perf-v1/v2/v5, PyQt6 port notes)
tools/                          standalone dev scripts, not imported by the app
  camera_speed_test.py            measures real camera FPS at several resolutions
  build_icons.py                  regenerates icon assets (legacy)
  list_cameras.py                 lists available camera indices
  probe_test.py                   ad-hoc test script
```

## Why it's split this way

Three layers, each with a clear direction of dependency:

- **`core/`** — the recognition engine: tracking math, camera reading, the
  capture → detect → identify → track → render loop. Pure logic plus
  the minimal Qt calls needed to hand a finished frame to the UI
  (QPixmap/QTimer). No widget or layout code.
- **`services/`** — I/O and ML backends (SQLite, InsightFace, TTS). No
  PyQt6 dependency at all — importable and testable headless.
- **`ui/`** — the PyQt6 GUI itself: widget layout/theming (`gui.py`),
  plus one mixin per tab (`registration.py`, `face_search.py`,
  `management.py`) that wires button clicks to `core`/`services` calls.
  `app.py` assembles `FaceRecognitionGUI` + `core.pipeline.VideoPipelineMixin`
  + the tab mixins into the single `FaceRecognitionApp` class — same
  runtime object, same shared `self` state, only the source layout
  changed.
- **`main.py`** — just builds `QApplication`, constructs
  `FaceRecognitionApp`, runs the event loop.

The original build was a ~2,000-line monolith. Splitting it out was a
pure refactor — every constant, method body, and comment moved as-is
into its new home; no behavior changed, only where it lives.
`services/database.py` and `services/encoder.py` still resolve
`face_recognition.db` / `face_cache.pkl` to the project root, so
existing local data (if any) is picked up unchanged.

## Notable engineering details

See `docs/PERF_NOTES.md` for the full history, but the short version:

- **Identity confidence isn't one threshold.** A match has to clear an
  absolute similarity floor, beat the runner-up by a margin, hold for
  several consecutive frames before "Unknown" becomes a name, and hold
  even longer before a *locked* identity can be swapped for a
  different one.
- **Blind tracking uses a growing search radius**, not a fixed one —
  the longer a face is occluded, the wider the area the tracker will
  accept as "still the same person" on the next sighting, capped so it
  can't run away entirely.
- **Performance work was mostly concurrency work.** The biggest win
  wasn't a faster algorithm — it was removing a redundant per-frame
  CPU tracker that was starving the GIL, and splitting capture,
  detection, and rendering into independent threads so none of them
  blocks on the others.
- **The camera's hardware ceiling was measured, not assumed** — across
  8 tested modes, 1080p@30 turned out to be the fastest, not the
  slowest, overturning the obvious assumption that lower resolution
  is always faster.

## License

MIT — see `LICENSE`.
