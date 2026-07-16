"""
tts.py — Smart Attendance System
==================================
Author: JJ — RUPP Computer Architecture Final Project

Standalone TTS (Text-to-Speech) module.
Backend: edge-tts (Microsoft Neural TTS — km-KH-PisethNeural)

Usage:
    from tts import speak_greeting, shutdown_tts

API:
    speak_greeting(name: str, khmer_name: str = "") → None
        Queue a non-blocking Khmer greeting for the recognised person.
        Greets are serialised so they never overlap.
        If khmer_name is given (non-empty), it is spoken directly —
        this is the per-student name entered in the Register/Manage UI.
        Otherwise falls back to KHMER_NAME_MAP, then phonetic romanisation.

    shutdown_tts() → None
        Signal the TTS worker thread to exit cleanly (call on app close).

Dependencies:
    pip install edge-tts pygame

Design:
    • One background daemon thread (_tts_worker) drains a Queue.
    • All audio is synthesised to a temp MP3 then played via pygame.mixer.
    • asyncio.run() is called per phrase on the worker thread — keeps the
      main thread fully synchronous / Tkinter-safe.
"""

import asyncio
import logging
import os
import queue
import tempfile
import threading
import time

log = logging.getLogger("SmartAttendance.TTS")

# ── Backend availability check ────────────────────────────────────────────────
try:
    # pyrefly: ignore [missing-import]
    import edge_tts          # pip install edge-tts
    _EDGE_TTS_OK = True
except ImportError:
    _EDGE_TTS_OK = False
    log.warning("edge-tts not installed — TTS disabled. Run: pip install edge-tts")

try:
    # pyrefly: ignore [missing-import]
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False
    log.warning("pygame not installed — TTS disabled. Run: pip install pygame")

TTS_AVAILABLE = _EDGE_TTS_OK and _PYGAME_OK

# ── Voice config ──────────────────────────────────────────────────────────────
# Microsoft Neural Khmer voices:
#   km-KH-PisethNeural   — male   (clear, authoritative)
#   km-KH-SreymomNeural  — female (warm, friendly) ← swap if preferred
EDGE_VOICE = "km-KH-PisethNeural"
EDGE_RATE  = "+0%"    # speaking rate offset, e.g. "-10%" to slow down
EDGE_PITCH = "+0Hz"   # pitch offset

# ── Khmer name map (legacy fallback) ──────────────────────────────────────────
# Keys   : uppercase English label stored in known_faces/
# Values : Khmer Unicode string spoken by the TTS voice
#
# NOTE: Per-student Khmer names are now editable in the Register/Manage UI
# and stored in the database (people.khmer_name). This map is only a
# fallback for entries that predate that feature and don't have a Khmer
# name set in the DB yet — for everyone else it's unused. Left empty here
# on purpose: real students' names don't belong in source code. Add your
# own entries locally if you need this fallback; format shown below.
KHMER_NAME_MAP: dict[str, str] = {
    # "SAMPLE NAME": "ឈ្មោះគំរូ",
}


# ── Greeting template (Khmer) ─────────────────────────────────────────────────
# "Hello {name}, welcome."
def _build_phrase(khmer_name: str) -> str:
    return f"សួស្តី {khmer_name} សូមស្វាគមន៍"


# ── Fallback: romanised phonetic name → Khmer TTS can still read it ──────────
def _apply_khmer_phonetics(name: str) -> str:
    """
    Map common Khmer romanisation patterns so the neural voice
    won't mispronounce them when a name isn't in KHMER_NAME_MAP.
    """
    n = name.lower()
    n = n.replace("ph", "p")
    n = n.replace("kh", "k")
    n = n.replace("chh", "ch")
    n = n.replace("th", "t")
    n = n.replace("nh", "ny")
    n = n.replace("ng", "ng ")
    n = n.replace("ou", "oo")
    n = n.replace("uo", "oo")
    n = n.replace("srey", "sray")
    n = n.replace("oeu", "u")
    n = n.replace("ea", "e ah")
    return " ".join(n.split()).title()


def _resolve_phrase(name: str, khmer_name: str = "") -> str:
    """
    Return the full Khmer greeting phrase for a given English name.

    Priority:
      1. khmer_name passed in directly (per-student value from the DB,
         entered via the Register/Manage UI) — used as-is if non-empty.
      2. KHMER_NAME_MAP lookup by uppercase English name (legacy/manual map).
      3. Phonetic romanisation fallback.
    """
    if khmer_name and khmer_name.strip():
        return _build_phrase(khmer_name.strip())

    k_name = KHMER_NAME_MAP.get(name.upper())
    if k_name:
        return _build_phrase(k_name)

    # Fallback: speak romanised name with phonetic correction
    phonetic = _apply_khmer_phonetics(name)
    return _build_phrase(phonetic)


# ── edge-tts synthesis → pygame playback ──────────────────────────────────────
async def _synthesise(phrase: str, path: str) -> None:
    """Async: synthesise `phrase` and save MP3 to `path`."""
    communicate = edge_tts.Communicate(
        text=phrase,
        voice=EDGE_VOICE,
        rate=EDGE_RATE,
        pitch=EDGE_PITCH,
    )
    await communicate.save(path)


def _speak_blocking(phrase: str) -> None:
    """
    Synthesise and play `phrase` synchronously on the worker thread.
    Blocks until playback finishes.
    """
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        # Synthesise (network call to MS Neural TTS CDN)
        asyncio.run(_synthesise(phrase, path))

        # Playback via pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        pygame.mixer.music.unload()
    except Exception as exc:
        log.error(f"edge-tts playback error: {exc}")
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

# ── Worker thread ─────────────────────────────────────────────────────────────
_tts_queue: queue.Queue = queue.Queue(maxsize=0)


def _tts_worker() -> None:
    """
    Daemon thread: drains _tts_queue one phrase at a time.
    Sentinel value `None` causes a clean exit.
    """
    if not TTS_AVAILABLE:
        return
    while True:
        item = _tts_queue.get()
        if item is None:          # shutdown signal
            break
        _name, phrase = item
        try:
            _speak_blocking(phrase)
        except Exception as exc:
            log.debug(f"TTS worker error: {exc}")


_tts_thread = threading.Thread(
    target=_tts_worker,
    name="TTSThread",
    daemon=True,
)
_tts_thread.start()


# ── Public API ────────────────────────────────────────────────────────────────
def speak_greeting(name: str, khmer_name: str = "") -> None:
    """
    Queue a non-blocking Khmer voice greeting for the recognised person.
    Safe to call from any thread (including the detect loop).

    Args:
        name: The English label of the recognised person
              (e.g. "SAMPLE NAME").
        khmer_name: Optional Khmer-script name for this student, as entered
              in the Register/Manage UI and stored in the database. If
              given, it's used directly for the greeting. If blank, falls
              back to KHMER_NAME_MAP / phonetic romanisation as before.
    """
    if not TTS_AVAILABLE:
        return
    phrase = _resolve_phrase(name, khmer_name)
    try:
        _tts_queue.put_nowait((name, phrase))
    except queue.Full:
        pass  # silently drop if queue is somehow saturated


def shutdown_tts() -> None:
    """
    Signal the TTS worker to exit cleanly.
    Call this from the app's on-close handler.
    """
    _tts_queue.put(None)