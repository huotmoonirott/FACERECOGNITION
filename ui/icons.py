"""
icons.py — loads the real vector-style PNG icons (see build_icons.py) and
exposes them as cached CTkImage objects at whatever pixel size a widget
needs.

Why this exists: emoji glyphs (📷🔍➕👤 etc.) render using the OS's color
emoji font (Segoe UI Emoji on Windows), which looks like clip-art and
varies between machines/OS versions. Real apps use flat, single-color
line icons instead. This module loads pre-rendered PNGs (one master per
icon/color, high-res) and resizes them down with high-quality LANCZOS
resampling to the exact size each button/label needs, which is what gives
a crisp, "really designed" look instead of a blurry/jagged glyph.

Usage:
    from icons import IconStore
    icons = IconStore(base_dir=os.path.dirname(__file__))
    img = icons.get("camera", "text", size=20)   # -> ctk.CTkImage
    ctk.CTkLabel(parent, image=img, text="")
"""
import os
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import customtkinter as ctk

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icons")


class IconStore:
    """Loads master icon PNGs lazily and caches resized CTkImage instances.

    Cache key is (icon_name, color_name, size) so the same icon at the same
    size/color is only ever resized once per process.
    """

    def __init__(self, assets_dir: str = ASSETS_DIR):
        self.assets_dir = assets_dir
        self._masters = {}   # (icon, color) -> PIL.Image (RGBA, high-res)
        self._cache = {}     # (icon, color, size) -> CTkImage

    def _load_master(self, icon: str, color: str):
        key = (icon, color)
        img = self._masters.get(key)
        if img is not None:
            return img
        path = os.path.join(self.assets_dir, f"{icon}__{color}.png")
        if not os.path.isfile(path):
            # Fail soft: return None so callers can fall back to a text
            # glyph rather than crashing the whole UI over one missing icon.
            self._masters[key] = False
            return False
        img = Image.open(path).convert("RGBA")
        self._masters[key] = img
        return img

    def get(self, icon: str, color: str, size: int = 20):
        """Return a CTkImage for `icon` tinted `color` at `size` px square."""
        key = (icon, color, size)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        master = self._load_master(icon, color)
        if master is False:
            return None
        resized = master.resize((size, size), Image.LANCZOS)
        cimg = ctk.CTkImage(light_image=resized, dark_image=resized, size=(size, size))
        self._cache[key] = cimg
        return cimg

    def available(self, icon: str, color: str) -> bool:
        return self._load_master(icon, color) is not False