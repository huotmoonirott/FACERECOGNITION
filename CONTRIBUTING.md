# Contributing

This started as a solo RUPP Computer Architecture final project, but
issues and PRs are welcome.

## Setup

```bash
git clone <your-repo-url>
cd RECOGFACE
pip install -r requirements.txt
python main.py
```

## Before opening a PR

- Don't commit anything under `known_faces/`, `*.db`, or `*.pkl` —
  they're gitignored on purpose (see the Data & Privacy note in the
  README). Double-check `git status` before committing.
- Keep the layer boundaries: `core/` and `services/` should stay free
  of PyQt6 imports. UI-only code belongs in `ui/`.
- Tunable constants (thresholds, timeouts, sizes) belong in
  `core/config.py`, not scattered inline.
- If you change tracking/identity behavior, sanity-check it against a
  live camera, not just a single screenshot — most of the tuning here
  (margin gating, confirm streaks, blind-track radius) only shows
  problems over multiple frames.

## Reporting bugs

Include your OS, whether you're on CPU or GPU (CUDAExecutionProvider
vs CPUExecutionProvider — the app logs which one loaded), and the
relevant section of `core/config.py` if it's a tracking/recognition
issue rather than a crash.
