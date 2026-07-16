"""
build_icons.py — generates real vector-style PNG icons for the Face
Recognition GUI, replacing emoji glyphs.

Run this ONCE (on any machine with cairosvg installed, e.g. this build
sandbox) to produce assets/icons/*.png. The finished PNGs are then loaded
by gui.py via Pillow only — no cairosvg/cairo dependency is needed at
runtime on the user's Windows machine.

Icon paths are simple, hand-written 24x24 line icons in the
Lucide/Feather style (round caps/joins, 2px stroke on a 24-unit grid),
written from scratch for this project — not copied from any icon
library's source files.
"""
import os
# pyrefly: ignore [missing-import]
import cairosvg

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icons")
os.makedirs(OUT_DIR, exist_ok=True)

STROKE = 1.8  # slightly lighter than the 2.0 "standard" for crisper small sizes

def svg_wrap(body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{{COLOR}}" stroke-width="{STROKE}" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )

ICONS = {
    # Sidebar / nav
    "camera": svg_wrap(
        '<path d="M4 8.5A1.5 1.5 0 0 1 5.5 7h2l1-2h7l1 2h2A1.5 1.5 0 0 1 20 8.5v9A1.5 1.5 0 0 1 18.5 19h-13A1.5 1.5 0 0 1 4 17.5z"/>'
        '<circle cx="12" cy="12.5" r="3.4"/>'
    ),
    "search": svg_wrap(
        '<circle cx="10.5" cy="10.5" r="6.5"/>'
        '<line x1="15.3" y1="15.3" x2="20" y2="20"/>'
    ),
    "user_plus": svg_wrap(
        '<circle cx="9.5" cy="8" r="3.4"/>'
        '<path d="M3.5 19c0-3.3 2.7-5.5 6-5.5s6 2.2 6 5.5"/>'
        '<line x1="18" y1="8" x2="18" y2="13"/>'
        '<line x1="15.5" y1="10.5" x2="20.5" y2="10.5"/>'
    ),
    "users": svg_wrap(
        '<circle cx="9" cy="8" r="3.2"/>'
        '<path d="M3.2 19c0-3.1 2.6-5.2 5.8-5.2s5.8 2.1 5.8 5.2"/>'
        '<circle cx="17" cy="8.6" r="2.4"/>'
        '<path d="M15.6 13.4c2.6.2 4.4 2.1 4.4 5"/>'
    ),
    # Brand / top
    "brain": svg_wrap(
        '<path d="M9 4.5c-2 0-3.4 1.5-3.4 3.2 0 .7.2 1.3.6 1.8-1 .5-1.7 1.6-1.7 2.9 0 1.5 1 2.7 2.4 3.1-.1.4-.2.8-.2 1.2 0 2 1.7 3.3 3.6 3.3"/>'
        '<path d="M15 4.5c2 0 3.4 1.5 3.4 3.2 0 .7-.2 1.3-.6 1.8 1 .5 1.7 1.6 1.7 2.9 0 1.5-1 2.7-2.4 3.1.1.4.2.8.2 1.2 0 2-1.7 3.3-3.6 3.3"/>'
        '<line x1="12" y1="4.5" x2="12" y2="19.8"/>'
    ),
    "wave": svg_wrap(
        '<path d="M5 13c1-3 2-9 4.6-9 1.7 0 1.7 2.4 1.2 3.6"/>'
        '<path d="M10.6 7.2c.3-.8 1-1.5 1.9-1.5 1.3 0 1.8 1.2 1.6 2.4"/>'
        '<path d="M14 8.4c.2-.6.8-1 1.5-1 1.1 0 1.6 1 1.4 2"/>'
        '<path d="M17 9.6c1.1-.1 2 .6 2 1.8 0 2.6-2.2 7.1-6.4 7.1-3.4 0-5.6-2-6.7-4.2"/>'
    ),
    # Status / actions
    "check_circle": svg_wrap(
        '<circle cx="12" cy="12" r="8.4"/>'
        '<path d="M8.4 12.4l2.3 2.3 4.9-5.4"/>'
    ),
    "x_circle": svg_wrap(
        '<circle cx="12" cy="12" r="8.4"/>'
        '<line x1="9" y1="9" x2="15" y2="15"/>'
        '<line x1="15" y1="9" x2="9" y2="15"/>'
    ),
    "alert_circle": svg_wrap(
        '<circle cx="12" cy="12" r="8.4"/>'
        '<line x1="12" y1="7.5" x2="12" y2="13"/>'
        '<circle cx="12" cy="16.2" r="0.15" fill="{COLOR}"/>'
    ),
    "refresh": svg_wrap(
        '<path d="M5 11a7 7 0 0 1 12-4.6M19 5v4.4h-4.4"/>'
        '<path d="M19 13a7 7 0 0 1-12 4.6M5 19v-4.4h4.4"/>'
    ),
    "edit": svg_wrap(
        '<path d="M14.5 5.5l3.8 3.8L8.6 19h-3.8v-3.8z"/>'
        '<line x1="13" y1="7" x2="16.8" y2="10.8"/>'
    ),
    "trash": svg_wrap(
        '<path d="M5.5 7h13"/>'
        '<path d="M9.5 7V5.2A1.2 1.2 0 0 1 10.7 4h2.6a1.2 1.2 0 0 1 1.2 1.2V7"/>'
        '<path d="M7.2 7l.7 11.2A1.5 1.5 0 0 0 9.4 19.6h5.2a1.5 1.5 0 0 0 1.5-1.4L16.8 7"/>'
        '<line x1="10.2" y1="10.5" x2="10.4" y2="16"/>'
        '<line x1="13.8" y1="10.5" x2="13.6" y2="16"/>'
    ),
    "image": svg_wrap(
        '<rect x="4" y="4.5" width="16" height="15" rx="2"/>'
        '<circle cx="9" cy="10" r="1.6"/>'
        '<path d="M4.8 16.5L9 12.3a1.4 1.4 0 0 1 2 0l1.6 1.6"/>'
        '<path d="M14 14.8l1.4-1.4a1.4 1.4 0 0 1 2 0l1.8 1.8"/>'
    ),
    "folder": svg_wrap(
        '<path d="M4 7.2A1.5 1.5 0 0 1 5.5 5.7h4l1.6 2h7.4A1.5 1.5 0 0 1 20 9.2v8.3a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5z"/>'
    ),
    "upload": svg_wrap(
        '<path d="M12 15.5V5"/>'
        '<path d="M8 9l4-4 4 4"/>'
        '<path d="M5 16.5v1.7A1.8 1.8 0 0 0 6.8 20h10.4a1.8 1.8 0 0 0 1.8-1.8v-1.7"/>'
    ),
    "save": svg_wrap(
        '<path d="M5.5 4.5h10l3 3v11a1 1 0 0 1-1 1h-12a1 1 0 0 1-1-1v-13a1 1 0 0 1 1-1z"/>'
        '<rect x="8" y="4.5" width="6" height="4.2"/>'
        '<rect x="7.5" y="13.5" width="9" height="5.5"/>'
    ),
    "zoom": svg_wrap(
        '<circle cx="10.5" cy="10.5" r="6.5"/>'
        '<line x1="15.3" y1="15.3" x2="20" y2="20"/>'
        '<line x1="10.5" y1="8" x2="10.5" y2="13"/>'
        '<line x1="8" y1="10.5" x2="13" y2="10.5"/>'
    ),
    # Window chrome / misc
    "bell": svg_wrap(
        '<path d="M7 17v-5.2a5 5 0 0 1 10 0V17"/>'
        '<path d="M5.5 17h13"/>'
        '<path d="M10.4 19.8a1.8 1.8 0 0 0 3.2 0"/>'
    ),
    "person": svg_wrap(
        '<circle cx="12" cy="8.2" r="3.4"/>'
        '<path d="M5.5 19.5c0-3.3 2.9-5.6 6.5-5.6s6.5 2.3 6.5 5.6"/>'
    ),
    "chevron_right": svg_wrap('<path d="M9 5.5l6.5 6.5L9 18.5"/>'),
    "minimize": svg_wrap('<line x1="5" y1="12" x2="19" y2="12"/>'),
    "maximize": svg_wrap('<rect x="5.5" y="5.5" width="13" height="13" rx="1.5"/>'),
    "close": svg_wrap(
        '<line x1="6" y1="6" x2="18" y2="18"/>'
        '<line x1="18" y1="6" x2="6" y2="18"/>'
    ),
    "dots": svg_wrap(
        '<circle cx="6" cy="12" r="1.1" fill="{COLOR}"/>'
        '<circle cx="12" cy="12" r="1.1" fill="{COLOR}"/>'
        '<circle cx="18" cy="12" r="1.1" fill="{COLOR}"/>'
    ),
}

# We render ONE master PNG per (icon, color) at a single high resolution
# (RENDER_PX). gui.py loads that master with Pillow and resizes it down to
# whatever exact pixel size a given button needs, with LANCZOS resampling.
# Rendering high-res once + resizing down on load gives crisper results
# than baking many discrete sizes, and keeps the asset folder small
# (24 icons x ~9 colors = ~216 files instead of 1500+).
RENDER_PX = 128

COLORS = {
    "text":          "#E8EEFF",
    "muted":         "#6B7A9E",
    "sidebar_text":  "#8089A8",
    "white":         "#FFFFFF",
    "accent":        "#3B82F6",
    "success":       "#10B981",
    "danger":        "#F43F5E",
    "warning":       "#F59E0B",
    "cyan":          "#22D3EE",
    "purple":        "#A855F7",
    "dark":          "#06222B",
}

def main():
    count = 0
    for icon_name, svg_template in ICONS.items():
        for color_name, hex_color in COLORS.items():
            svg = svg_template.replace("{COLOR}", hex_color)
            out_path = os.path.join(OUT_DIR, f"{icon_name}__{color_name}.png")
            cairosvg.svg2png(
                bytestring=svg.encode("utf-8"),
                write_to=out_path,
                output_width=RENDER_PX,
                output_height=RENDER_PX,
            )
            count += 1
    print(f"Generated {count} master icon PNGs in {OUT_DIR}")

if __name__ == "__main__":
    main()