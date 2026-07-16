"""
gui.py — UI for the Face Recognition System (PyQt6 port)
==========================================================
RUPP Computer Architecture Final Project — MJ

Converted from customtkinter → PyQt6/PySide6.

All public attributes/methods that main.py depends on are preserved
with identical names and signatures so main.py requires zero changes:

    self.colors                  dict of hex strings
    self.fonts                   dict of QFont objects (was CTkFont)
    self.icons                   IconStore
    self.font_families           dict role→family string
    self.tab_buttons             dict name→QPushButton
    self.video_label             QLabel (displays camera frames)
    self.status_bar               QLabel
    self.camera_var              SimpleVar (str get/set shim)
    self.camera_selector         QComboBox
    self.clock_label             QLabel
    self.greeting_icon           QLabel (check_circle ring icon)
    self.greeting_label          QLabel
    self.greeting_card           QFrame
    self.live_log                QTextEdit (was CTkTextbox)
    self.live_log_count          QLabel
    self.stat_today              QLabel
    self.stat_registered         QLabel
    self.search_preview          QLabel
    self.search_status           QLabel
    self.search_results_frame    QWidget (scrollable)
    self.reg_fname               QLineEdit
    self.reg_lname               QLineEdit
    self.reg_khmer_name          QLineEdit
    self.reg_id                  QLineEdit
    self.reg_photo_strip         QWidget
    self.reg_photo_count_label   QLabel
    self.reg_preview             QLabel
    self.reg_status              QLabel
    self._reg_captured_frame     np.ndarray | None
    self._reg_staged_frames      list
    self.manage_tree             DarkRowList
    self.manage_count_label      QLabel
    self.topbar_title            QLabel

    show_greeting_toast(name, color)
    render_search_results(matches)
    refresh_photo_strip(count)
    icon(name, color_key, size) → QPixmap | None
    icon_button(...)            → QPushButton
    icon_text_button(...)       → QPushButton
    shadow_card(parent, ...)    → (outer_frame, inner_frame)
    _build_ui()
    _switch_tab(name)
    _refresh_manage()
    _open_edit_dialog(name, khmer_name, person_id, on_save)
    after(ms, fn)               → shim that wraps QTimer.singleShot
    after_cancel(id)            → cancels the QTimer
    protocol("WM_DELETE_WINDOW", fn) → connects closeEvent
    quit() / destroy()          → QApplication.quit / self.close
"""

# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QLineEdit, QTextEdit, QComboBox, QScrollArea, QVBoxLayout,
    QHBoxLayout, QGridLayout, QSizePolicy, QDialog, QMessageBox,
    QFileDialog, QSplitter,
)
# pyrefly: ignore [missing-import]
from PyQt6.QtCore import (
    Qt, QTimer, QSize, pyqtSignal, QObject, QRectF, QPointF,
)
# pyrefly: ignore [missing-import]
from PyQt6.QtGui import (
    QFont, QFontDatabase, QColor, QPalette, QPixmap, QIcon,
    QTextCursor, QPainter, QPen, QBrush, QPainterPath,
)
import sys
from datetime import datetime

# Icons are rendered natively via PyQt6 — no dependency on icons.py or CTkImage.
# self.icon(name, color_key, size) returns a QPixmap drawn with QPainter.

# ── Theme ─────────────────────────────────────────────────────────────────────
UI_COLORS = {
    "app_bg":                "#0A0E18",
    "content_bg":            "#0C1020",
    "sidebar":               "#080D1C",
    "sidebar_text":          "#7A85A8",
    "sidebar_selected_text": "#FFFFFF",
    "sidebar_hover":         "#161D34",
    "panel":                 "#0F1525",
    "panel_alt":             "#141B30",
    "panel_hover":           "#1A2340",
    "topbar":                "#0A0E18",
    "text":                  "#EAEFFC",
    "muted":                 "#6B7A9E",
    "outline":               "#1C2540",
    "shadow":                "#040611",
    "accent":                "#3B82F6",
    "accent_glow":           "#1D4ED8",
    "success":               "#10B981",
    "success_dim":           "#0E3A2C",
    "danger":                "#F43F5E",
    "danger_dim":            "#3D1322",
    "warning":               "#F59E0B",
    "cyan":                  "#22D3EE",
    "purple":                "#A855F7",
    "white":                 "#FFFFFF",
}

UI_FONT_STACKS = {
    "display": ["Segoe UI Semibold", "Segoe UI", "Calibri", "Arial"],
    "body":    ["Segoe UI", "Calibri", "Arial"],
    "mono":    ["Cascadia Code", "Consolas", "Courier New"],
    "khmer":   ["Khmer OS", "Leelawadee UI", "Segoe UI"],
}
UI_FONT_FAMILIES = {k: v[0] for k, v in UI_FONT_STACKS.items()}

NAV_ITEMS = [
    ("Live Camera",   "camera",    "#3B82F6"),
    ("Face Search",   "search",    "#3B82F6"),
    ("Register Face", "user_plus", "#3B82F6"),
    ("Manage Faces",  "users",     "#3B82F6"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _resolve_font_families():
    installed = set(QFontDatabase.families())
    resolved = {}
    for role, candidates in UI_FONT_STACKS.items():
        resolved[role] = next((f for f in candidates if f in installed), candidates[-1])
    return resolved


def _qfont(family: str, size: int, bold: bool = False) -> QFont:
    f = QFont(family, size)
    if bold:
        f.setBold(True)
    return f


def _ss_bg(hex_color: str, radius: int = 0, border: str = "") -> str:
    """Minimal stylesheet fragment for a solid background."""
    r = f"border-radius: {radius}px;" if radius else ""
    b = f"border: {border};" if border else "border: none;"
    return f"background-color: {hex_color}; {r} {b}"


def _make_icon_pixmap(name: str, color: str, size: int) -> "QPixmap":
    """
    Draw a simple icon as a QPixmap using QPainter paths.
    Falls back to a small filled circle if the icon name is unknown.
    All icons are drawn to fit within (size x size) with 1px padding.
    """
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)

    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(color)
    pen = QPen(c)
    pen.setWidthF(max(1.2, size * 0.09))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    m = size * 0.08          # margin
    s = size - 2 * m         # drawing area side
    cx = size / 2.0
    cy = size / 2.0

    def rect(rx, ry, rw, rh):
        return QRectF(m + rx * s, m + ry * s, rw * s, rh * s)

    def line(x1, y1, x2, y2):
        p.drawLine(
            QPointF(m + x1 * s, m + y1 * s),
            QPointF(m + x2 * s, m + y2 * s))

    def circle(ox, oy, r):
        p.drawEllipse(QPointF(m + ox * s, m + oy * s), r * s, r * s)

    # ── icon paths ──────────────────────────────────────────────────────
    if name in ("camera", "video"):
        # lens
        p.drawEllipse(rect(0.15, 0.2, 0.5, 0.6))
        # body rounded rect
        body = QPainterPath()
        body.addRoundedRect(rect(0.0, 0.15, 0.72, 0.7), s * 0.12, s * 0.12)
        p.drawPath(body)
        # viewfinder bump
        path = QPainterPath()
        path.moveTo(m + 0.72 * s, m + 0.32 * s)
        path.lineTo(m + 0.95 * s, m + 0.20 * s)
        path.lineTo(m + 0.95 * s, m + 0.80 * s)
        path.lineTo(m + 0.72 * s, m + 0.68 * s)
        p.drawPath(path)

    elif name == "search":
        circle(0.40, 0.40, 0.28)
        line(0.61, 0.61, 0.90, 0.90)

    elif name in ("user_plus", "person"):
        # head
        circle(0.42, 0.28, 0.18)
        # shoulders arc
        path = QPainterPath()
        path.moveTo(m + 0.05 * s, m + 0.95 * s)
        path.cubicTo(
            QPointF(m + 0.05 * s, m + 0.62 * s),
            QPointF(m + 0.80 * s, m + 0.62 * s),
            QPointF(m + 0.80 * s, m + 0.95 * s))
        p.drawPath(path)
        if name == "user_plus":
            # plus sign at top-right
            line(0.72, 0.10, 0.95, 0.10)
            line(0.835, 0.00, 0.835, 0.20)

    elif name == "users":
        # back person (offset)
        circle(0.62, 0.27, 0.15)
        path2 = QPainterPath()
        path2.moveTo(m + 0.37 * s, m + 0.95 * s)
        path2.cubicTo(
            QPointF(m + 0.37 * s, m + 0.66 * s),
            QPointF(m + 0.92 * s, m + 0.66 * s),
            QPointF(m + 0.92 * s, m + 0.95 * s))
        p.drawPath(path2)
        # front person
        circle(0.33, 0.30, 0.17)
        path3 = QPainterPath()
        path3.moveTo(m + 0.02 * s, m + 0.95 * s)
        path3.cubicTo(
            QPointF(m + 0.02 * s, m + 0.64 * s),
            QPointF(m + 0.65 * s, m + 0.64 * s),
            QPointF(m + 0.65 * s, m + 0.95 * s))
        p.drawPath(path3)

    elif name == "refresh":
        import math
        # arc ~300 degrees
        p.drawArc(rect(0.08, 0.08, 0.84, 0.84),
                  int(60 * 16), int(300 * 16))
        # arrowhead at end of arc
        ax = cx + (s / 2 * 0.84 / 2) * math.cos(math.radians(60))
        ay = cy - (s / 2 * 0.84 / 2) * math.sin(math.radians(60))
        arr = QPainterPath()
        arr.moveTo(ax - size * 0.08, ay - size * 0.08)
        arr.lineTo(ax, ay)
        arr.lineTo(ax + size * 0.08, ay + size * 0.08)
        p.drawPath(arr)

    elif name in ("brain", "eye"):
        if name == "brain":
            # simplified brain: two bumpy lobes
            path = QPainterPath()
            path.moveTo(m + 0.50 * s, m + 0.90 * s)
            path.cubicTo(
                QPointF(m + 0.10 * s, m + 0.90 * s),
                QPointF(m + 0.05 * s, m + 0.55 * s),
                QPointF(m + 0.15 * s, m + 0.40 * s))
            path.cubicTo(
                QPointF(m + 0.05 * s, m + 0.20 * s),
                QPointF(m + 0.25 * s, m + 0.05 * s),
                QPointF(m + 0.45 * s, m + 0.15 * s))
            path.lineTo(m + 0.50 * s, m + 0.15 * s)
            path.cubicTo(
                QPointF(m + 0.75 * s, m + 0.05 * s),
                QPointF(m + 0.95 * s, m + 0.20 * s),
                QPointF(m + 0.85 * s, m + 0.40 * s))
            path.cubicTo(
                QPointF(m + 0.95 * s, m + 0.55 * s),
                QPointF(m + 0.90 * s, m + 0.90 * s),
                QPointF(m + 0.50 * s, m + 0.90 * s))
            p.drawPath(path)
            # center divider
            line(0.50, 0.15, 0.50, 0.90)
        else:
            # eye shape
            path = QPainterPath()
            path.moveTo(m + 0.05 * s, m + 0.50 * s)
            path.cubicTo(
                QPointF(m + 0.25 * s, m + 0.15 * s),
                QPointF(m + 0.75 * s, m + 0.15 * s),
                QPointF(m + 0.95 * s, m + 0.50 * s))
            path.cubicTo(
                QPointF(m + 0.75 * s, m + 0.85 * s),
                QPointF(m + 0.25 * s, m + 0.85 * s),
                QPointF(m + 0.05 * s, m + 0.50 * s))
            p.drawPath(path)
            circle(0.50, 0.50, 0.16)

    elif name == "wave":
        # simple wave / hand wave emoji approximation
        path = QPainterPath()
        path.moveTo(m + 0.30 * s, m + 0.80 * s)
        path.cubicTo(
            QPointF(m + 0.10 * s, m + 0.55 * s),
            QPointF(m + 0.10 * s, m + 0.20 * s),
            QPointF(m + 0.35 * s, m + 0.12 * s))
        path.cubicTo(
            QPointF(m + 0.55 * s, m + 0.05 * s),
            QPointF(m + 0.80 * s, m + 0.15 * s),
            QPointF(m + 0.88 * s, m + 0.35 * s))
        path.cubicTo(
            QPointF(m + 0.95 * s, m + 0.55 * s),
            QPointF(m + 0.85 * s, m + 0.80 * s),
            QPointF(m + 0.65 * s, m + 0.88 * s))
        path.cubicTo(
            QPointF(m + 0.45 * s, m + 0.95 * s),
            QPointF(m + 0.32 * s, m + 0.90 * s),
            QPointF(m + 0.30 * s, m + 0.80 * s))
        p.drawPath(path)
        # fingers hint
        line(0.48, 0.12, 0.52, 0.04)
        line(0.62, 0.10, 0.68, 0.02)

    elif name == "check_circle":
        p.drawEllipse(rect(0.0, 0.0, 1.0, 1.0))
        path = QPainterPath()
        path.moveTo(m + 0.22 * s, m + 0.50 * s)
        path.lineTo(m + 0.42 * s, m + 0.70 * s)
        path.lineTo(m + 0.78 * s, m + 0.30 * s)
        p.drawPath(path)

    elif name in ("upload", "image"):
        if name == "upload":
            # up arrow
            line(0.50, 0.65, 0.50, 0.10)
            path = QPainterPath()
            path.moveTo(m + 0.28 * s, m + 0.32 * s)
            path.lineTo(m + 0.50 * s, m + 0.10 * s)
            path.lineTo(m + 0.72 * s, m + 0.32 * s)
            p.drawPath(path)
            line(0.15, 0.75, 0.15, 0.92)
            line(0.15, 0.92, 0.85, 0.92)
            line(0.85, 0.92, 0.85, 0.75)
        else:
            # image frame
            body = QPainterPath()
            body.addRoundedRect(rect(0.0, 0.0, 1.0, 1.0), s * 0.10, s * 0.10)
            p.drawPath(body)
            circle(0.30, 0.33, 0.12)
            path = QPainterPath()
            path.moveTo(m + 0.05 * s, m + 0.85 * s)
            path.lineTo(m + 0.35 * s, m + 0.55 * s)
            path.lineTo(m + 0.60 * s, m + 0.75 * s)
            path.lineTo(m + 0.75 * s, m + 0.60 * s)
            path.lineTo(m + 0.95 * s, m + 0.85 * s)
            p.drawPath(path)

    elif name == "edit":
        path = QPainterPath()
        path.moveTo(m + 0.70 * s, m + 0.05 * s)
        path.lineTo(m + 0.95 * s, m + 0.30 * s)
        path.lineTo(m + 0.30 * s, m + 0.95 * s)
        path.lineTo(m + 0.05 * s, m + 0.95 * s)
        path.lineTo(m + 0.05 * s, m + 0.70 * s)
        path.closeSubpath()
        p.drawPath(path)
        line(0.65, 0.10, 0.90, 0.35)

    elif name == "trash":
        # lid
        line(0.10, 0.25, 0.90, 0.25)
        line(0.35, 0.25, 0.35, 0.10)
        line(0.65, 0.25, 0.65, 0.10)
        line(0.35, 0.10, 0.65, 0.10)
        # body
        body = QPainterPath()
        body.moveTo(m + 0.18 * s, m + 0.25 * s)
        body.lineTo(m + 0.25 * s, m + 0.92 * s)
        body.lineTo(m + 0.75 * s, m + 0.92 * s)
        body.lineTo(m + 0.82 * s, m + 0.25 * s)
        p.drawPath(body)
        line(0.40, 0.38, 0.40, 0.80)
        line(0.60, 0.38, 0.60, 0.80)

    elif name == "zoom":
        circle(0.38, 0.38, 0.28)
        line(0.59, 0.59, 0.90, 0.90)
        # plus inside
        line(0.26, 0.38, 0.50, 0.38)
        line(0.38, 0.26, 0.38, 0.50)

    else:
        # Unknown icon: draw a small filled dot so spaces aren't blank
        p.setBrush(QBrush(c))
        p.drawEllipse(rect(0.30, 0.30, 0.40, 0.40))

    p.end()
    return px


# ── SimpleVar — replaces ctk.StringVar ───────────────────────────────────────
class SimpleVar:
    """Minimal StringVar shim so main.py's camera_var.get()/set() still work."""
    def __init__(self, value=""):
        self._v = value
    def get(self):
        return self._v
    def set(self, v):
        self._v = v


# ── Styled widgets ────────────────────────────────────────────────────────────
def _styled_frame(parent, bg: str, radius: int = 0,
                  border_color: str = "", border_width: int = 0) -> QFrame:
    f = QFrame(parent)
    border = (f"border: {border_width}px solid {border_color};"
              if border_color and border_width else "border: none;")
    f.setStyleSheet(
        f"QFrame {{ background-color: {bg}; border-radius: {radius}px; {border} }}"
    )
    return f


def _styled_label(parent, text: str, font: QFont,
                  color: str, bg: str = "transparent",
                  align=Qt.AlignmentFlag.AlignLeft,
                  wrap: bool = False) -> QLabel:
    lbl = QLabel(text, parent)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {color}; background: transparent;")
    lbl.setAlignment(align)
    if wrap:
        lbl.setWordWrap(True)
    return lbl


def _styled_button(parent, text: str, font: QFont,
                   fg: str, text_color: str,
                   hover: str = "", radius: int = 22,
                   border_color: str = "", border_width: int = 0) -> QPushButton:
    # Qt treats a bare "&" in button text as a mnemonic marker (it tries to
    # underline whatever follows it), which is what turned "Save & Register"
    # into "Save _Register" on screen. "&&" is Qt's escape for a literal "&".
    btn = QPushButton(text.replace("&", "&&"), parent)
    btn.setFont(font)
    bdr = (f"border: {border_width}px solid {border_color};"
           if border_color and border_width else "border: none;")
    hov = hover or fg
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {fg};
            color: {text_color};
            border-radius: {radius}px;
            {bdr}
            padding: 6px 16px;
        }}
        QPushButton:hover {{
            background-color: {hov};
        }}
        QPushButton:pressed {{
            background-color: {hov};
        }}
    """)
    return btn


def _styled_entry(parent, font: QFont, bg: str,
                  text_color: str, border_color: str,
                  radius: int = 10, placeholder: str = "") -> QLineEdit:
    e = QLineEdit(parent)
    e.setFont(font)
    e.setPlaceholderText(placeholder)
    e.setMinimumHeight(40)
    e.setStyleSheet(f"""
        QLineEdit {{
            background-color: {bg};
            color: {text_color};
            border: none;
            border-radius: {radius}px;
            padding: 4px 10px;
        }}
        QLineEdit:focus {{
            background-color: #141B2E;
            border-bottom: 2px solid #3B82F6;
            border-radius: {radius}px {radius}px 0px 0px;
        }}
    """)
    return e


# ── DarkRowList — replaces the custom Treeview widget ────────────────────────
class DarkRowList(QScrollArea):
    """
    Scrollable row list that mimics the original DarkRowList API:
        get_children() → [item_id, ...]
        delete(item_id)
        insert(_parent, _index, values) → item_id
        selection() → [item_id] or []
        item(item_id) → {"values": [...]}
    """

    def __init__(self, parent, columns, gui, row_height=44, **kwargs):
        super().__init__(parent)
        self.gui = gui
        self.columns = columns      # [(key, width, header_text, anchor), ...]
        self.row_height = row_height

        self._rows = {}             # item_id → {frame, values, cells}
        self._order = []
        self._selected_id = None
        self._next_id = 0

        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { border: none; background: transparent; }"
                           "QScrollBar:vertical { width: 8px; background: #0F1525; }"
                           "QScrollBar::handle:vertical { background: #1C2540; border-radius: 4px; }")

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(3)
        self._layout.addStretch(1)
        self.setWidget(self._container)

        self._build_header()

    def _build_header(self):
        hdr = QFrame(self._container)
        hdr.setFixedHeight(38)
        hdr.setStyleSheet(
            f"QFrame {{ background-color: {self.gui.colors['panel']};"
            f"border-radius: 8px; border: none; }}")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(10, 0, 10, 0)
        hl.setSpacing(0)
        for key, width, text, anchor in self.columns:
            lbl = QLabel(text)
            lbl.setFont(self.gui.fonts["small_bold"])
            lbl.setStyleSheet(f"color: {self.gui.colors['muted']}; background: transparent;")
            lbl.setMinimumWidth(width)
            al = (Qt.AlignmentFlag.AlignCenter if anchor == "center"
                  else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            lbl.setAlignment(al)
            hl.addWidget(lbl, 1)
        # Insert header before the stretch
        self._layout.insertWidget(0, hdr)

    # ── Treeview-compatible API ───────────────────────────────────────────
    def get_children(self):
        return list(self._order)

    def delete(self, item_id):
        row = self._rows.pop(item_id, None)
        if row:
            w = row["frame"]
            self._layout.removeWidget(w)
            w.deleteLater()
            self._order.remove(item_id)
            if self._selected_id == item_id:
                self._selected_id = None

    def insert(self, _parent, _index, values):
        item_id = f"row{self._next_id}"
        self._next_id += 1

        frame = QFrame(self._container)
        frame.setFixedHeight(self.row_height)
        frame.setStyleSheet(
            f"QFrame {{ background-color: {self.gui.colors['panel_alt']};"
            f"border-radius: 8px; border: none; }}")
        hl = QHBoxLayout(frame)
        hl.setContentsMargins(10, 0, 10, 0)
        hl.setSpacing(0)

        cells = []
        for i, (key, width, _text, anchor) in enumerate(self.columns):
            val = values[i] if i < len(values) else ""
            lbl = QLabel(str(val) if val not in (None, "") else "—")
            lbl.setFont(self.gui.fonts["body"])
            lbl.setStyleSheet(f"color: {self.gui.colors['text']}; background: transparent;")
            lbl.setMinimumWidth(width)
            al = (Qt.AlignmentFlag.AlignCenter if anchor == "center"
                  else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            lbl.setAlignment(al)
            hl.addWidget(lbl, 1)
            cells.append(lbl)

        frame.mousePressEvent = lambda e, iid=item_id: self._select(iid)
        # Insert before the trailing stretch (last item in layout)
        insert_pos = self._layout.count() - 1
        self._layout.insertWidget(insert_pos, frame)

        self._rows[item_id] = {"frame": frame, "values": list(values), "cells": cells}
        self._order.append(item_id)
        return item_id

    def selection(self):
        return [self._selected_id] if self._selected_id else []

    def item(self, item_id):
        row = self._rows.get(item_id)
        return {"values": row["values"]} if row else {"values": []}

    def _select(self, item_id):
        if self._selected_id and self._selected_id in self._rows:
            prev = self._rows[self._selected_id]
            prev["frame"].setStyleSheet(
                f"QFrame {{ background-color: {self.gui.colors['panel_alt']};"
                "border-radius: 8px; border: none; }}")
            for c in prev["cells"]:
                c.setStyleSheet(f"color: {self.gui.colors['text']}; background: transparent;")
        self._selected_id = item_id
        row = self._rows.get(item_id)
        if row:
            row["frame"].setStyleSheet(
                f"QFrame {{ background-color: {self.gui.colors['accent']};"
                "border-radius: 8px; border: none; }}")
            for c in row["cells"]:
                c.setStyleSheet("color: #FFFFFF; background: transparent;")


# ── Native icon renderer (no icons.py / CTkImage dependency) ─────────────────
class _IconStoreWrapper:
    """
    Renders icons natively with QPainter.
    color_key is a role string (e.g. "text", "muted", "accent") or a hex string.
    get() always returns a valid QPixmap — never None.
    """
    _HEX_TO_ROLE = {
        "#EAEFFC": "text",   "#6B7A9E": "muted",  "#7A85A8": "sidebar_text",
        "#FFFFFF":  "white",  "#3B82F6": "accent",  "#10B981": "success",
        "#F43F5E":  "danger", "#F59E0B": "warning", "#22D3EE": "cyan",
        "#A855F7":  "purple", "#06222B": "dark",
    }

    def __init__(self):
        self._cache: dict = {}

    def get(self, name: str, color_key: str, size: int = 20) -> QPixmap:
        key = (name, color_key, size)
        if key in self._cache:
            return self._cache[key]
        # resolve color
        hex_color = UI_COLORS.get(color_key, color_key)
        px = _make_icon_pixmap(name, hex_color, size)
        self._cache[key] = px
        return px


# ── Main window ───────────────────────────────────────────────────────────────
class FaceRecognitionGUI(QMainWindow):
    """
    Drop-in PyQt6 replacement for the customtkinter FaceRecognitionGUI.
    Inherits from QMainWindow instead of ctk.CTk.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System — RUPP")
        self.resize(1320, 800)
        self.setMinimumSize(980, 640)
        self._timer_ids: dict[int, QTimer] = {}
        self._timer_counter = 0
        self._close_callback = None
        self._clock_timer: QTimer | None = None
        self._greeting_reset_timer: QTimer | None = None
        # Populated by subclass (FaceRecognitionApp in main.py) before _build_ui
        self.known_names: list = []
        self._init_theme()

    # ── Tkinter compatibility shims ───────────────────────────────────────
    def after(self, ms: int, fn) -> int:
        """Mimics tk.after — schedules fn() once after ms milliseconds.

        PERF FIX: the previous version created a new QTimer(self) on every
        call and never destroyed it once it fired (only the Python-side
        dict entry was dropped). Because the timer is parented to `self`,
        Qt's C++ ownership kept every one of those QTimer objects alive
        forever as children of the main window. Call sites that re-arm
        themselves on every tick (e.g. the old _poll_frame loop, ~500-1000x
        /sec) were leaking hundreds of thousands of dead QObjects per
        minute, which is what caused FPS to degrade the longer the app ran.
        Calling t.deleteLater() once the timer fires actually releases it.
        """
        self._timer_counter += 1
        tid = self._timer_counter
        t = QTimer(self)
        t.setSingleShot(True)

        def _fire():
            self._timer_ids.pop(tid, None)
            t.deleteLater()
            fn()

        t.timeout.connect(_fire)
        t.start(ms)
        self._timer_ids[tid] = t
        return tid

    def after_cancel(self, timer_id: int):
        t = self._timer_ids.pop(timer_id, None)
        if t is not None:
            t.stop()

    def protocol(self, name: str, fn):
        if name == "WM_DELETE_WINDOW":
            self._close_callback = fn

    def closeEvent(self, event):
        if self._close_callback:
            # Clear before invoking: _on_close() ends by calling destroy(),
            # which calls self.close() again — without clearing this first,
            # that second close() would re-enter this branch and call
            # _on_close() a second time, infinitely recursing (stack
            # overflow / crash on window close) instead of actually
            # closing on the second pass.
            cb = self._close_callback
            self._close_callback = None
            cb()
            event.ignore()   # let _on_close call destroy()
        else:
            event.accept()

    def quit(self):
        QApplication.quit()

    def destroy(self):
        self.close()
        QApplication.quit()

    # ── Theme ─────────────────────────────────────────────────────────────
    def _init_theme(self):
        self.colors = UI_COLORS
        fams = _resolve_font_families()
        self.font_families = fams
        self.fonts = {
            "display":    _qfont(fams["display"], 22, bold=True),
            "title":      _qfont(fams["display"], 15, bold=True),
            "heading":    _qfont(fams["body"],    13, bold=True),
            "body":       _qfont(fams["body"],    13),
            "body_bold":  _qfont(fams["body"],    13, bold=True),
            "small":      _qfont(fams["body"],    11),
            "small_bold": _qfont(fams["body"],    11, bold=True),
            "mono":       _qfont(fams["mono"],    12),
            "mono_sm":    _qfont(fams["mono"],    10),
            "stat":       _qfont(fams["display"], 32, bold=True),
            "khmer":      _qfont(fams["khmer"],   14),
            "nav_label":  _qfont(fams["body"],     9, bold=True),
        }
        self.icons = _IconStoreWrapper()

        # Global dark palette
        app = QApplication.instance()
        if app:
            app.setStyle("Fusion")
            pal = QPalette()
            bg = QColor(UI_COLORS["app_bg"])
            txt = QColor(UI_COLORS["text"])
            pal.setColor(QPalette.ColorRole.Window,          bg)
            pal.setColor(QPalette.ColorRole.WindowText,      txt)
            pal.setColor(QPalette.ColorRole.Base,            QColor(UI_COLORS["panel_alt"]))
            pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(UI_COLORS["panel"]))
            pal.setColor(QPalette.ColorRole.Text,            txt)
            pal.setColor(QPalette.ColorRole.ButtonText,      txt)
            pal.setColor(QPalette.ColorRole.Button,          QColor(UI_COLORS["panel"]))
            pal.setColor(QPalette.ColorRole.Highlight,       QColor(UI_COLORS["accent"]))
            pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
            app.setPalette(pal)

    def icon(self, name: str, color_key: str = "text", size: int = 20) -> QPixmap:
        return self.icons.get(name, color_key, size)

    # ── Reusable widget helpers ───────────────────────────────────────────
    def icon_button(self, parent, icon_name, command=None, size=36,
                    icon_size=16, color_key="text", fg_color=None,
                    hover_color=None, border=True) -> QPushButton:
        fg  = fg_color    or self.colors["panel_alt"]
        hov = hover_color or self.colors["panel_hover"]
        btn = QPushButton("", parent)
        btn.setFixedSize(size, size)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {fg}; border-radius: {size//2}px; border: none;
            }}
            QPushButton:hover {{ background-color: {hov}; }}
        """)
        px = self.icon(icon_name, color_key, icon_size)
        if px:
            btn.setIcon(QIcon(px))
            btn.setIconSize(QSize(icon_size, icon_size))
        if command:
            btn.clicked.connect(command)
        return btn

    def icon_text_button(self, parent, icon_name, text, command=None,
                         height=44, icon_size=16, icon_color="white",
                         fg_color=None, hover_color=None,
                         text_color="#FFFFFF",
                         border_width=0, border_color="",
                         width=None, **kwargs) -> QPushButton:
        fg  = fg_color    or self.colors["accent"]
        hov = hover_color or self.colors["accent_glow"]
        radius = height // 2
        bdr = (f"border: {border_width}px solid {border_color};"
               if border_width and border_color else "border: none;")
        btn = QPushButton(f"  {text.replace('&', '&&')}  ", parent)
        btn.setFont(self.fonts["body_bold"])
        btn.setFixedHeight(height)
        if width:
            btn.setFixedWidth(width)
        else:
            btn.setMinimumWidth(0)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {fg}; color: {text_color};
                border-radius: {radius}px; {bdr}
                padding: 0 18px;
                text-align: left;
            }}
            QPushButton:hover {{ background-color: {hov}; }}
        """)
        px = self.icon(icon_name, icon_color, icon_size)
        if px:
            btn.setIcon(QIcon(px))
            btn.setIconSize(QSize(icon_size, icon_size))
        if command:
            btn.clicked.connect(command)
        return btn

    def shadow_card(self, parent, corner_radius=16, fg_color=None,
                    pad=3, **kwargs):
        """Returns (outer_frame, inner_frame) — same API as original."""
        outer = QFrame(parent)
        outer.setStyleSheet(
            f"QFrame {{ background-color: {self.colors['shadow']};"
            f"border-radius: {corner_radius+2}px; border: none; }}")
        inner = QFrame(outer)
        ic = fg_color or self.colors["panel"]
        inner.setStyleSheet(
            f"QFrame {{ background-color: {ic};"
            f"border-radius: {corner_radius}px;"
            f"border: none; }}")
        lo = QVBoxLayout(outer)
        lo.setContentsMargins(pad, pad, pad, pad)
        lo.addWidget(inner)
        return outer, inner

    # ── Build UI ──────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        central.setStyleSheet(f"background-color: {self.colors['app_bg']};")
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar = self._build_sidebar(central)
        root_layout.addWidget(sidebar)

        content = self._build_content(central)
        root_layout.addWidget(content, 1)

    # ── Sidebar ───────────────────────────────────────────────────────────
    def _build_sidebar(self, parent) -> QWidget:
        sb = QFrame(parent)
        sb.setFixedWidth(256)
        sb.setStyleSheet(
            f"QFrame {{ background-color: {self.colors['sidebar']}; border: none; }}")
        layout = QVBoxLayout(sb)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top padding (removed accent stripe — active nav items carry the accent)
        layout.addSpacing(0)

        # Logo block
        logo_frame = QWidget(sb)
        logo_frame.setStyleSheet("background: transparent;")
        ll = QHBoxLayout(logo_frame)
        ll.setContentsMargins(24, 28, 24, 24)
        ll.setSpacing(12)

        badge = QFrame(logo_frame)
        badge.setFixedSize(40, 40)
        badge.setStyleSheet(
            f"background-color: {self.colors['accent']};"
            "border-radius: 20px; border: none;")
        bl = QVBoxLayout(badge)
        bl.setContentsMargins(0, 0, 0, 0)
        badge_icon = QLabel(badge)
        px = self.icon("brain", "white", 20)
        if px:
            badge_icon.setPixmap(px)
        badge_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(badge_icon)
        ll.addWidget(badge)

        text_block = QWidget(logo_frame)
        text_block.setStyleSheet("background: transparent;")
        tbl = QVBoxLayout(text_block)
        tbl.setContentsMargins(0, 0, 0, 0)
        tbl.setSpacing(2)
        t1 = _styled_label(text_block, "Face Recognition",
                           self.fonts["title"], self.colors["text"])
        t2 = _styled_label(text_block, "RUPP  ·  Vision System",
                           self.fonts["small"], self.colors["muted"])
        tbl.addWidget(t1)
        tbl.addWidget(t2)
        ll.addWidget(text_block, 1)
        layout.addWidget(logo_frame)

        # Divider
        div = QFrame(sb)
        div.setFixedHeight(1)
        div.setStyleSheet(f"background-color: {self.colors['outline']}; border: none;")
        layout.addWidget(div)
        layout.addSpacing(4)

        # NAV label
        nav_label = _styled_label(sb, "NAVIGATION", self.fonts["nav_label"],
                                  self.colors["muted"])
        nav_label.setContentsMargins(24, 12, 24, 6)
        layout.addWidget(nav_label)

        # Nav buttons
        self.tab_buttons = {}
        nav_container = QWidget(sb)
        nav_container.setStyleSheet("background: transparent;")
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(12, 0, 12, 0)
        nav_layout.setSpacing(3)
        for name, icon_name, color in NAV_ITEMS:
            btn = self._make_nav_btn(nav_container, name, icon_name, color)
            nav_layout.addWidget(btn)
            self.tab_buttons[name] = btn
        layout.addWidget(nav_container)

        layout.addStretch(1)

        # Bottom divider
        div2 = QFrame(sb)
        div2.setFixedHeight(1)
        div2.setStyleSheet(f"background-color: {self.colors['outline']}; border: none;")
        layout.addWidget(div2)

        # Bottom stats
        bottom = QWidget(sb)
        bottom.setStyleSheet("background: transparent;")
        bot_l = QHBoxLayout(bottom)
        bot_l.setContentsMargins(12, 8, 12, 8)
        bot_l.setSpacing(6)

        today_outer, today_card = self.shadow_card(bottom, corner_radius=10, pad=2)
        today_outer.setMaximumHeight(56)
        today_inner_l = QVBoxLayout(today_card)
        today_inner_l.setContentsMargins(6, 4, 6, 4)
        today_inner_l.setSpacing(0)
        self.stat_today = _styled_label(
            today_card, "0",
            _qfont(self.font_families["display"], 15, bold=True),
            self.colors["accent"],
            align=Qt.AlignmentFlag.AlignCenter)
        today_inner_l.addWidget(self.stat_today)
        today_inner_l.addWidget(_styled_label(
            today_card, "Today", self.fonts["small"], self.colors["muted"],
            align=Qt.AlignmentFlag.AlignCenter))
        bot_l.addWidget(today_outer, 1)

        reg_outer, reg_card = self.shadow_card(bottom, corner_radius=10, pad=2)
        reg_outer.setMaximumHeight(56)
        reg_inner_l = QVBoxLayout(reg_card)
        reg_inner_l.setContentsMargins(6, 4, 6, 4)
        reg_inner_l.setSpacing(0)
        self.stat_registered = _styled_label(
            reg_card, str(len(self.known_names)),
            _qfont(self.font_families["display"], 15, bold=True),
            self.colors["success"],
            align=Qt.AlignmentFlag.AlignCenter)
        reg_inner_l.addWidget(self.stat_registered)
        reg_inner_l.addWidget(_styled_label(
            reg_card, "Enrolled", self.fonts["small"], self.colors["muted"],
            align=Qt.AlignmentFlag.AlignCenter))
        bot_l.addWidget(reg_outer, 1)

        layout.addWidget(bottom)
        return sb

    def _make_nav_btn(self, parent, name: str, icon_name: str, color: str) -> QPushButton:
        px_inactive = self.icon(icon_name, "sidebar_text", 17)
        px_active   = self.icon(icon_name, "white",        17)

        btn = QPushButton(f"   {name}", parent)
        btn.setFont(self.fonts["body_bold"])
        btn.setFixedHeight(44)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if px_inactive:
            btn.setIcon(QIcon(px_inactive))
            btn.setIconSize(QSize(17, 17))

        btn._px_inactive = px_inactive
        btn._px_active   = px_active
        self._nav_btn_set_inactive(btn)
        btn.clicked.connect(lambda checked=False, n=name: self._switch_tab(n))
        return btn

    def _nav_btn_set_active(self, btn):
        c = self.colors
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {c['panel_alt']}; color: {c['sidebar_selected_text']};
                border-radius: 10px; border: none;
                text-align: left; padding-left: 12px;
            }}
        """)
        if btn._px_active:
            btn.setIcon(QIcon(btn._px_active))

    def _nav_btn_set_inactive(self, btn):
        c = self.colors
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent; color: {c['sidebar_text']};
                border-radius: 10px; border: none;
                text-align: left; padding-left: 12px;
            }}
            QPushButton:hover {{ background-color: {c['sidebar_hover']}; }}
        """)
        if btn._px_inactive:
            btn.setIcon(QIcon(btn._px_inactive))

    # ── Content area ──────────────────────────────────────────────────────
    def _build_content(self, parent) -> QWidget:
        content = QWidget(parent)
        content.setStyleSheet(f"background-color: {self.colors['content_bg']};")
        vl = QVBoxLayout(content)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        self.content = content
        self._content_layout = vl

        self._build_topbar(content, vl)

        # Tab container
        self._tab_container = QWidget(content)
        self._tab_container.setStyleSheet("background: transparent;")
        vl.addWidget(self._tab_container, 1)
        tab_l = QVBoxLayout(self._tab_container)
        tab_l.setContentsMargins(0, 0, 0, 0)
        tab_l.setSpacing(0)
        self._tab_layout = tab_l

        self.tabs = {
            "Live Camera":   self._build_camera_tab(self._tab_container),
            "Face Search":   self._build_search_tab(self._tab_container),
            "Register Face": self._build_register_tab(self._tab_container),
            "Manage Faces":  self._build_manage_tab(self._tab_container),
        }
        for w in self.tabs.values():
            tab_l.addWidget(w)
            w.hide()

        self._switch_tab("Live Camera")
        return content

    def _build_topbar(self, parent, parent_layout):
        tb = QFrame(parent)
        tb.setFixedHeight(56)
        tb.setStyleSheet(
            f"QFrame {{ background-color: {self.colors['topbar']}; border: none;"
            f"border-bottom: 1px solid {self.colors['outline']}; }}")
        tbl = QHBoxLayout(tb)
        tbl.setContentsMargins(24, 0, 20, 0)

        self.topbar_title = _styled_label(
            tb, "Live Camera", self.fonts["heading"], self.colors["text"])
        tbl.addWidget(self.topbar_title)
        tbl.addStretch(1)

        self.clock_label = _styled_label(tb, "", self.fonts["small"], self.colors["muted"])
        tbl.addWidget(self.clock_label)
        tbl.addSpacing(16)

        refresh_btn = self.icon_button(tb, "refresh", size=34, icon_size=15,
                                       color_key="text",
                                       command=self._topbar_refresh)
        tbl.addWidget(refresh_btn)
        tbl.addSpacing(8)
        profile_btn = self.icon_button(tb, "person", size=34, icon_size=15,
                                       color_key="text")
        tbl.addWidget(profile_btn)

        parent_layout.addWidget(tb)
        self._update_clock()

    def _topbar_refresh(self):
        name = self.topbar_title.text()
        if name == "Manage Faces":
            self._refresh_manage()

    def _update_clock(self):
        try:
            self.clock_label.setText(
                datetime.now().strftime("%a, %d %b %Y  |  %H:%M:%S"))
        except Exception:
            pass
        if self._clock_timer is None:
            self._clock_timer = QTimer(self)
            self._clock_timer.timeout.connect(self._update_clock)
            self._clock_timer.start(1000)

    def _switch_tab(self, name: str):
        for tab_name, widget in self.tabs.items():
            if tab_name == name:
                widget.show()
            else:
                widget.hide()
        self.topbar_title.setText(name)
        for bn, btn in self.tab_buttons.items():
            if bn == name:
                self._nav_btn_set_active(btn)
            else:
                self._nav_btn_set_inactive(btn)
        if name == "Manage Faces":
            self._refresh_manage()

    # ── Camera Tab ────────────────────────────────────────────────────────
    def _build_camera_tab(self, parent) -> QWidget:
        frame = QWidget(parent)
        frame.setStyleSheet("background: transparent;")
        hl = QHBoxLayout(frame)
        hl.setContentsMargins(20, 20, 20, 20)
        hl.setSpacing(10)

        # Left: video feed
        left = QWidget(frame)
        left.setStyleSheet("background: transparent;")
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)

        self.video_wrapper_outer, self.video_wrapper = self.shadow_card(
            left, corner_radius=18, pad=3)
        vw_l = QVBoxLayout(self.video_wrapper)
        vw_l.setContentsMargins(2, 2, 2, 2)
        vw_l.setSpacing(0)

        self.video_label = QLabel(self.video_wrapper)
        self.video_label.setText("Starting camera…")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            f"background-color: {self.colors['panel_alt']};"
            f"color: {self.colors['muted']}; border: none;")
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        vw_l.addWidget(self.video_label, 1)

        # Status bar row inside video wrapper
        status_row = QWidget(self.video_wrapper)
        status_row.setStyleSheet("background: transparent;")
        sr_l = QHBoxLayout(status_row)
        sr_l.setContentsMargins(12, 0, 12, 12)

        self.status_bar = _styled_label(
            status_row, "● Initializing", self.fonts["small"], self.colors["muted"])
        sr_l.addWidget(self.status_bar)
        sr_l.addStretch(1)

        self.camera_var = SimpleVar("Camera 0")
        self.camera_selector = QComboBox(status_row)
        self.camera_selector.addItems(["Camera 0","Camera 1","Camera 2","Camera 3"])
        self.camera_selector.setCurrentText("Camera 0")
        self.camera_selector.setFixedWidth(120)
        self.camera_selector.setFont(self.fonts["small"])
        self.camera_selector.setStyleSheet(f"""
            QComboBox {{
                background-color: {self.colors['panel_alt']};
                color: {self.colors['text']};
                border: none;
                border-radius: 6px; padding: 2px 8px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background-color: {self.colors['panel']};
                color: {self.colors['text']};
                selection-background-color: {self.colors['accent']};
            }}
        """)
        self.camera_selector.currentTextChanged.connect(self._change_camera)
        sr_l.addWidget(self.camera_selector)
        vw_l.addWidget(status_row)

        left_l.addWidget(self.video_wrapper_outer, 1)
        hl.addWidget(left, 5)

        # Right: info panel
        right = QWidget(frame)
        right.setStyleSheet("background: transparent;")
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(12)

        # Greeting toast card
        self.greeting_card_outer, self.greeting_card = self.shadow_card(
            right, corner_radius=14, pad=2)
        self.greeting_card_outer.setFixedHeight(119)
        gc_l = QVBoxLayout(self.greeting_card)
        gc_l.setContentsMargins(16, 10, 16, 10)
        gc_l.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.greeting_icon = QLabel(self.greeting_card)
        self.greeting_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.greeting_icon.setStyleSheet("background: transparent;")
        self.greeting_icon.setPixmap(self.icon("wave", "muted", 26))
        gc_l.addWidget(self.greeting_icon)

        self.greeting_label = _styled_label(
            self.greeting_card, "Waiting for faces...",
            self.fonts["heading"], self.colors["muted"],
            align=Qt.AlignmentFlag.AlignCenter, wrap=True)
        gc_l.addWidget(self.greeting_label)
        right_l.addWidget(self.greeting_card_outer)

        # Recent activity card
        log_outer, log_card = self.shadow_card(right, corner_radius=14, pad=2)
        log_card_l = QVBoxLayout(log_card)
        log_card_l.setContentsMargins(16, 16, 16, 12)
        log_card_l.setSpacing(8)

        log_hdr = QWidget(log_card)
        log_hdr.setStyleSheet("background: transparent;")
        log_hdr_l = QHBoxLayout(log_hdr)
        log_hdr_l.setContentsMargins(0, 0, 0, 0)
        log_hdr_l.addWidget(_styled_label(
            log_hdr, "Recent Activity", self.fonts["title"], self.colors["text"]))
        log_hdr_l.addStretch(1)
        self.live_log_count = _styled_label(
            log_hdr, "0 seen today", self.fonts["small"], self.colors["accent"])
        log_hdr_l.addWidget(self.live_log_count)
        log_card_l.addWidget(log_hdr)

        self.live_log = QTextEdit(log_card)
        self.live_log.setReadOnly(True)
        self.live_log.setFont(self.fonts["mono"])
        # PERF FIX: QTextEdit keeps unlimited undo history + document blocks
        # by default. On a long-running session this grows forever and
        # makes every .append() progressively slower. Cap it.
        self.live_log.setUndoRedoEnabled(False)
        self.live_log.document().setMaximumBlockCount(500)
        self.live_log.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self.colors['panel_alt']};
                color: {self.colors['text']};
                border: none; border-radius: 10px;
                padding: 8px;
            }}
        """)
        log_card_l.addWidget(self.live_log, 1)

        right_l.addWidget(log_outer, 1)
        hl.addWidget(right, 2)
        return frame

    def show_greeting_toast(self, name: str, color: str = None):
        color = color or self.colors["success"]
        full = name.strip().title()
        self.greeting_label.setText(f"Hello, {full}!\nRecognized")
        self.greeting_label.setStyleSheet(
            f"color: {color}; background: transparent;")
        self.greeting_label.setFont(
            _qfont(self.font_families["body"], 13, bold=True))
        self.greeting_card.setStyleSheet(
            f"QFrame {{ background-color: {self.colors['success_dim']};"
            "border-radius: 14px; border: none; }")
        self.greeting_icon.setPixmap(self.icon("wave", color, 26))

        if self._greeting_reset_timer:
            self._greeting_reset_timer.stop()
        self._greeting_reset_timer = QTimer(self)
        self._greeting_reset_timer.setSingleShot(True)
        self._greeting_reset_timer.timeout.connect(self._reset_greeting)
        self._greeting_reset_timer.start(4000)

    def _reset_greeting(self):
        self._greeting_reset_timer = None
        self.greeting_label.setText("Waiting for faces...")
        self.greeting_label.setStyleSheet(
            f"color: {self.colors['muted']}; background: transparent;")
        self.greeting_label.setFont(
            _qfont(self.font_families["body"], 13, bold=True))
        self.greeting_card.setStyleSheet(
            f"QFrame {{ background-color: {self.colors['panel']};"
            "border-radius: 14px; border: none; }")
        self.greeting_icon.setPixmap(self.icon("wave", "muted", 26))

    # ── Face Search Tab ───────────────────────────────────────────────────
    def _build_search_tab(self, parent) -> QWidget:
        frame = QWidget(parent)
        frame.setStyleSheet("background: transparent;")
        hl = QHBoxLayout(frame)
        hl.setContentsMargins(24, 24, 24, 24)
        hl.setSpacing(12)

        # Left: upload + preview
        left_outer, left = self.shadow_card(frame, corner_radius=16, pad=2)
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(20, 20, 20, 20)
        left_l.setSpacing(8)

        title_row = QWidget(left)
        title_row.setStyleSheet("background: transparent;")
        tr_l = QHBoxLayout(title_row)
        tr_l.setContentsMargins(0, 0, 0, 0)
        icon_lbl = QLabel(title_row)
        px = self.icon("search", "text", 17)
        if px:
            icon_lbl.setPixmap(px)
        icon_lbl.setStyleSheet("background: transparent;")
        tr_l.addWidget(icon_lbl)
        tr_l.addWidget(_styled_label(
            title_row, "Search by Photo", self.fonts["title"], self.colors["text"]))
        tr_l.addStretch(1)
        left_l.addWidget(title_row)

        self.search_preview = QLabel(left)
        self.search_preview.setText("No photo selected\nUpload a photo to search")
        self.search_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.search_preview.setStyleSheet(
            f"background-color: {self.colors['panel_alt']};"
            f"color: {self.colors['muted']}; border-radius: 12px; border: none;")
        self.search_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_l.addWidget(self.search_preview, 1)

        btn_row = QWidget(left)
        btn_row.setStyleSheet("background: transparent;")
        br_l = QHBoxLayout(btn_row)
        br_l.setContentsMargins(0, 0, 0, 0)
        br_l.setSpacing(8)
        upload_btn = self.icon_text_button(
            btn_row, "upload", "Upload Photo",
            command=self._upload_search_photo,
            height=44, icon_color="text",
            fg_color="transparent",
            border_width=1, border_color=self.colors["outline"],
            hover_color=self.colors["sidebar_hover"],
            text_color=self.colors["text"])
        br_l.addWidget(upload_btn, 1)
        capture_btn = self.icon_text_button(
            btn_row, "camera", "Capture Webcam",
            command=self._capture_search,
            height=44,
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_glow"])
        br_l.addWidget(capture_btn, 1)
        left_l.addWidget(btn_row)

        find_btn = self.icon_text_button(
            left, "zoom", "Find Matches",
            command=self._run_face_search,
            height=48, icon_color="dark",
            fg_color=self.colors["cyan"],
            hover_color="#0EA5C4",
            text_color="#06222B")
        left_l.addWidget(find_btn)

        self.search_status = _styled_label(
            left, "", self.fonts["small"], self.colors["muted"],
            align=Qt.AlignmentFlag.AlignCenter)
        left_l.addWidget(self.search_status)
        hl.addWidget(left_outer, 1)

        # Right: results
        right_outer, right = self.shadow_card(frame, corner_radius=16, pad=2)
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(20, 20, 20, 20)
        right_l.setSpacing(8)
        right_l.addWidget(_styled_label(
            right, "Closest Matches", self.fonts["title"], self.colors["text"]))

        scroll = QScrollArea(right)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }"
                             "QScrollBar:vertical { width: 8px; background: #0F1525; }"
                             "QScrollBar::handle:vertical { background: #1C2540; border-radius: 4px; }")
        self.search_results_frame = QWidget()
        self.search_results_frame.setStyleSheet("background: transparent;")
        self._search_results_layout = QVBoxLayout(self.search_results_frame)
        self._search_results_layout.setContentsMargins(4, 4, 4, 4)
        self._search_results_layout.setSpacing(8)
        self._search_results_layout.addStretch(1)
        scroll.setWidget(self.search_results_frame)
        right_l.addWidget(scroll, 1)
        hl.addWidget(right_outer, 1)

        self._render_search_placeholder()
        return frame

    def _render_search_placeholder(self):
        # clear layout except stretch
        while self._search_results_layout.count() > 1:
            item = self._search_results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        lbl = _styled_label(
            self.search_results_frame,
            'Upload or capture a photo, then tap\n"Find Matches" to search\nthe enrolled face gallery.',
            self.fonts["small"], self.colors["muted"],
            align=Qt.AlignmentFlag.AlignCenter, wrap=True)
        self._search_results_layout.insertWidget(0, lbl)

    def render_search_results(self, matches):
        while self._search_results_layout.count() > 1:
            item = self._search_results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not matches:
            lbl = _styled_label(
                self.search_results_frame,
                "No matches found in the enrolled gallery.",
                self.fonts["small"], self.colors["muted"],
                align=Qt.AlignmentFlag.AlignCenter)
            self._search_results_layout.insertWidget(0, lbl)
            return

        for i, (name, sim, khmer_name) in enumerate(matches):
            pct = max(0.0, min(sim, 1.0))
            color = (self.colors["success"] if pct >= 0.5 else
                     self.colors["warning"] if pct >= 0.35 else self.colors["danger"])

            card = QFrame(self.search_results_frame)
            card.setStyleSheet(
                f"QFrame {{ background-color: {self.colors['panel_alt']};"
                "border-radius: 12px; border: none; }}")
            card_l = QHBoxLayout(card)
            card_l.setContentsMargins(14, 12, 14, 12)

            rank_lbl = _styled_label(card, f"#{i+1}", self.fonts["heading"],
                                     self.colors["muted"])
            rank_lbl.setFixedWidth(36)
            card_l.addWidget(rank_lbl)

            name_text = name.title()
            if khmer_name:
                name_text += f"   ({khmer_name})"
            name_col = QWidget(card)
            name_col.setStyleSheet("background: transparent;")
            name_col_l = QVBoxLayout(name_col)
            name_col_l.setContentsMargins(0, 0, 0, 0)
            name_col_l.setSpacing(4)
            name_col_l.addWidget(_styled_label(
                name_col, name_text, self.fonts["body_bold"], self.colors["text"]))

            bar_bg = QFrame(name_col)
            bar_bg.setFixedHeight(8)
            bar_bg.setStyleSheet(
                f"QFrame {{ background-color: {self.colors['outline']};"
                "border-radius: 4px; border: none; }}")
            bar_fill = QFrame(bar_bg)
            bar_fill.setFixedHeight(8)
            bar_fill.setStyleSheet(
                f"background-color: {color}; border-radius: 4px; border: none;")
            bar_fill.setFixedWidth(max(4, int(200 * pct)))
            name_col_l.addWidget(bar_bg)
            card_l.addWidget(name_col, 1)

            pct_lbl = _styled_label(card, f"{pct*100:.1f}%",
                                    self.fonts["small_bold"], color)
            pct_lbl.setFixedWidth(64)
            pct_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_l.addWidget(pct_lbl)

            self._search_results_layout.insertWidget(i, card)

    # ── Register Tab ──────────────────────────────────────────────────────
    def _build_register_tab(self, parent) -> QWidget:
        frame = QWidget(parent)
        frame.setStyleSheet("background: transparent;")
        hl = QHBoxLayout(frame)
        hl.setContentsMargins(24, 24, 24, 24)
        hl.setSpacing(12)

        # Left: form (scrollable)
        scroll = QScrollArea(frame)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }"
                             "QScrollBar:vertical { width: 8px; background: #0F1525; }"
                             "QScrollBar::handle:vertical { background: #1C2540; border-radius: 4px; }")
        left_widget = QWidget()
        left_widget.setStyleSheet("background: transparent;")
        left_l = QVBoxLayout(left_widget)
        left_l.setContentsMargins(0, 8, 0, 8)
        left_l.setSpacing(4)
        scroll.setWidget(left_widget)

        # Title
        tr = QWidget(left_widget)
        tr.setStyleSheet("background: transparent;")
        tr_l = QHBoxLayout(tr)
        tr_l.setContentsMargins(0, 0, 0, 0)
        icon_lbl = QLabel(tr)
        px = self.icon("user_plus", "text", 17)
        if px:
            icon_lbl.setPixmap(px)
        icon_lbl.setStyleSheet("background: transparent;")
        tr_l.addWidget(icon_lbl)
        tr_l.addWidget(_styled_label(
            tr, "Register New Face", self.fonts["title"], self.colors["text"]))
        tr_l.addStretch(1)
        left_l.addWidget(tr)

        # Name row
        name_row = QWidget(left_widget)
        name_row.setStyleSheet("background: transparent;")
        nr_l = QHBoxLayout(name_row)
        nr_l.setContentsMargins(0, 4, 0, 0)
        nr_l.setSpacing(8)

        fn_col = QWidget(name_row)
        fn_col.setStyleSheet("background: transparent;")
        fn_l = QVBoxLayout(fn_col)
        fn_l.setContentsMargins(0, 0, 0, 0)
        fn_l.setSpacing(4)
        fn_l.addWidget(_styled_label(fn_col, "First Name",
                                     self.fonts["small_bold"], self.colors["text"]))
        self.reg_fname = _styled_entry(
            fn_col, self.fonts["body"],
            self.colors["panel_alt"], self.colors["text"],
            self.colors["outline"], radius=10)
        fn_l.addWidget(self.reg_fname)
        nr_l.addWidget(fn_col, 1)

        ln_col = QWidget(name_row)
        ln_col.setStyleSheet("background: transparent;")
        ln_l = QVBoxLayout(ln_col)
        ln_l.setContentsMargins(0, 0, 0, 0)
        ln_l.setSpacing(4)
        ln_l.addWidget(_styled_label(ln_col, "Last Name",
                                     self.fonts["small_bold"], self.colors["text"]))
        self.reg_lname = _styled_entry(
            ln_col, self.fonts["body"],
            self.colors["panel_alt"], self.colors["text"],
            self.colors["outline"], radius=10)
        ln_l.addWidget(self.reg_lname)
        nr_l.addWidget(ln_col, 1)
        left_l.addWidget(name_row)

        # Khmer name
        left_l.addSpacing(8)
        left_l.addWidget(_styled_label(
            left_widget, "Khmer Name",
            self.fonts["small_bold"], self.colors["text"]))
        self.reg_khmer_name = _styled_entry(
            left_widget, self.fonts["khmer"],
            self.colors["panel_alt"], self.colors["text"],
            self.colors["outline"], radius=10)
        left_l.addWidget(self.reg_khmer_name)

        # Person ID
        left_l.addSpacing(8)
        left_l.addWidget(_styled_label(
            left_widget, "Person ID",
            self.fonts["small_bold"], self.colors["text"]))
        self.reg_id = _styled_entry(
            left_widget, self.fonts["body"],
            self.colors["panel_alt"], self.colors["text"],
            self.colors["outline"], radius=10)
        left_l.addWidget(self.reg_id)

        # Divider
        div = QFrame(left_widget)
        div.setFixedHeight(1)
        div.setStyleSheet(f"background-color: {self.colors['outline']}; border: none;")
        left_l.addSpacing(8)
        left_l.addWidget(div)
        left_l.addSpacing(4)

        # Photos
        left_l.addWidget(_styled_label(
            left_widget, "Photos",
            self.fonts["small_bold"], self.colors["text"]))

        photo_btn_row = QWidget(left_widget)
        photo_btn_row.setStyleSheet("background: transparent;")
        pb_l = QHBoxLayout(photo_btn_row)
        pb_l.setContentsMargins(0, 0, 0, 0)
        pb_l.setSpacing(8)
        capture_btn = self.icon_text_button(
            photo_btn_row, "camera", "Capture Webcam",
            command=self._capture_register,
            height=44,
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_glow"])
        pb_l.addWidget(capture_btn, 1)
        upload_btn = self.icon_text_button(
            photo_btn_row, "upload", "Upload Photo",
            command=self._upload_photo,
            height=44, icon_color="text",
            fg_color="transparent",
            border_width=1, border_color=self.colors["outline"],
            hover_color=self.colors["sidebar_hover"],
            text_color=self.colors["text"])
        pb_l.addWidget(upload_btn, 1)
        left_l.addWidget(photo_btn_row)

        # Photo strip
        self.reg_photo_strip = QWidget(left_widget)
        self.reg_photo_strip.setStyleSheet("background: transparent;")
        self.reg_photo_strip.setFixedHeight(52)
        strip_l = QHBoxLayout(self.reg_photo_strip)
        strip_l.setContentsMargins(0, 0, 0, 0)
        strip_l.setSpacing(6)
        strip_l.addStretch(1)
        left_l.addWidget(self.reg_photo_strip)

        self.reg_photo_count_label = _styled_label(
            left_widget, "No photos added yet",
            self.fonts["small"], self.colors["muted"])
        left_l.addWidget(self.reg_photo_count_label)

        save_btn = self.icon_text_button(
            left_widget, "check_circle", "Save & Register",
            command=self._save_registration,
            height=48,
            fg_color=self.colors["success"],
            hover_color="#059669")
        left_l.addWidget(save_btn)

        self.reg_status = _styled_label(
            left_widget, "", self.fonts["small"], self.colors["muted"],
            align=Qt.AlignmentFlag.AlignCenter)
        left_l.addWidget(self.reg_status)
        left_l.addStretch(1)

        hl.addWidget(scroll, 1)

        # Right: preview
        right_outer, right = self.shadow_card(frame, corner_radius=16, pad=2)
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(20, 20, 20, 20)
        right_l.setSpacing(12)
        right_l.addWidget(_styled_label(
            right, "Latest Capture", self.fonts["title"], self.colors["text"],
            align=Qt.AlignmentFlag.AlignCenter))

        self.reg_preview = QLabel(right)
        self.reg_preview.setText("No photo yet\nCapture or upload a photo")
        self.reg_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reg_preview.setStyleSheet(
            f"background-color: {self.colors['panel_alt']};"
            f"color: {self.colors['muted']}; border-radius: 12px; border: none;")
        self.reg_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_l.addWidget(self.reg_preview, 1)
        hl.addWidget(right_outer, 1)

        self._reg_captured_frame = None
        self._reg_staged_frames = []
        return frame

    def refresh_photo_strip(self, count: int):
        # Clear existing widgets in the strip layout (keep stretch)
        strip_l = self.reg_photo_strip.layout()
        while strip_l.count() > 1:
            item = strip_l.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i in range(count):
            chip = QFrame(self.reg_photo_strip)
            chip.setFixedSize(46, 46)
            chip.setStyleSheet(
                f"QFrame {{ background-color: {self.colors['panel_alt']};"
                "border-radius: 8px; border: none; }}")
            cl = QVBoxLayout(chip)
            cl.setContentsMargins(0, 0, 0, 0)
            img_lbl = QLabel(chip)
            px = self.icon("image", "muted", 18)
            if px:
                img_lbl.setPixmap(px)
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setStyleSheet("background: transparent;")
            cl.addWidget(img_lbl)
            strip_l.insertWidget(i, chip)
        if count == 0:
            self.reg_photo_count_label.setText("No photos added yet")
        else:
            self.reg_photo_count_label.setText(
                f"{count} photo{'s' if count != 1 else ''} staged for this person")

    # ── Manage Faces Tab ──────────────────────────────────────────────────
    def _build_manage_tab(self, parent) -> QWidget:
        frame = QWidget(parent)
        frame.setStyleSheet("background: transparent;")
        vl = QVBoxLayout(frame)
        vl.setContentsMargins(24, 20, 24, 20)
        vl.setSpacing(0)

        hdr_outer, hdr = self.shadow_card(frame, corner_radius=16, pad=2)
        hdr_l = QHBoxLayout(hdr)
        hdr_l.setContentsMargins(20, 16, 20, 16)
        hdr_l.setSpacing(8)

        title_row = QWidget(hdr)
        title_row.setStyleSheet("background: transparent;")
        tr_l = QHBoxLayout(title_row)
        tr_l.setContentsMargins(0, 0, 0, 0)
        tr_l.setSpacing(8)
        icon_lbl = QLabel(title_row)
        px = self.icon("users", "text", 17)
        if px:
            icon_lbl.setPixmap(px)
        icon_lbl.setStyleSheet("background: transparent;")
        tr_l.addWidget(icon_lbl)
        tr_l.addWidget(_styled_label(
            title_row, "Enrolled People", self.fonts["title"], self.colors["text"]))
        hdr_l.addWidget(title_row)

        self.manage_count_label = _styled_label(
            hdr, "0 people", self.fonts["small"], self.colors["muted"])
        hdr_l.addWidget(self.manage_count_label, 1)

        for btn_cfg in [
            ("refresh",  "Refresh",         self.colors["accent"],   self.colors["accent_glow"], "#FFFFFF", self._refresh_manage),
            ("image",    "Add Photo",        self.colors["cyan"],     "#0EA5C4",                 "#06222B", self._add_photo_to_selected),
            ("edit",     "Edit Selected",    self.colors["purple"],   "#7E22CE",                 "#FFFFFF", self._edit_selected_face),
            ("trash",    "Delete Selected",  self.colors["danger"],   "#BE123C",                 "#FFFFFF", self._delete_selected_face),
        ]:
            icon_n, label, fg, hov, tc, cmd = btn_cfg
            b = self.icon_text_button(
                hdr, icon_n, label, command=cmd,
                height=36,
                fg_color=fg, hover_color=hov, text_color=tc,
                icon_color="dark" if tc == "#06222B" else "white")
            hdr_l.addWidget(b)

        vl.addWidget(hdr_outer)
        vl.addSpacing(12)

        tf_outer, tf = self.shadow_card(frame, corner_radius=16, pad=2)
        tf_l = QVBoxLayout(tf)
        tf_l.setContentsMargins(6, 6, 6, 6)
        self.manage_tree = DarkRowList(
            tf, gui=self,
            columns=[
                ("name",       180, "Name",          "w"),
                ("khmer_name", 140, "Khmer Name",     "w"),
                ("person_id",  110, "Person ID",      "center"),
                ("photos",      70, "Photos",         "center"),
                ("registered", 170, "Registered At",  "center"),
            ])
        tf_l.addWidget(self.manage_tree)
        vl.addWidget(tf_outer, 1)
        return frame

    # ── Edit dialog ────────────────────────────────────────────────────────
    def _open_edit_dialog(self, name: str, khmer_name: str, person_id: str, on_save):
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Person")
        dialog.setFixedSize(420, 420)
        dialog.setStyleSheet(
            f"QDialog {{ background-color: {self.colors['content_bg']}; }}")

        outer, wrap = self.shadow_card(dialog, corner_radius=16, pad=2)
        dlg_l = QVBoxLayout(dialog)
        dlg_l.setContentsMargins(16, 16, 16, 16)
        dlg_l.addWidget(outer)

        wrap_l = QVBoxLayout(wrap)
        wrap_l.setContentsMargins(20, 20, 20, 20)
        wrap_l.setSpacing(6)

        tr = QWidget(wrap)
        tr.setStyleSheet("background: transparent;")
        tr_l = QHBoxLayout(tr)
        tr_l.setContentsMargins(0, 0, 0, 0)
        icon_lbl = QLabel(tr)
        px = self.icon("edit", "text", 16)
        if px:
            icon_lbl.setPixmap(px)
        icon_lbl.setStyleSheet("background: transparent;")
        tr_l.addWidget(icon_lbl)
        tr_l.addWidget(_styled_label(tr, "Edit Person", self.fonts["title"],
                                     self.colors["text"]))
        tr_l.addStretch(1)
        wrap_l.addWidget(tr)

        wrap_l.addWidget(_styled_label(wrap, "Name", self.fonts["small_bold"],
                                       self.colors["text"]))
        name_entry = _styled_entry(wrap, self.fonts["body"],
                                   self.colors["panel_alt"], self.colors["text"],
                                   self.colors["outline"])
        name_entry.setText(name)
        wrap_l.addWidget(name_entry)

        wrap_l.addWidget(_styled_label(wrap, "Khmer Name", self.fonts["small_bold"],
                                       self.colors["text"]))
        khmer_entry = _styled_entry(wrap, self.fonts["khmer"],
                                    self.colors["panel_alt"], self.colors["text"],
                                    self.colors["outline"])
        khmer_entry.setText(khmer_name or "")
        wrap_l.addWidget(khmer_entry)

        wrap_l.addWidget(_styled_label(wrap, "Person ID", self.fonts["small_bold"],
                                       self.colors["text"]))
        id_entry = _styled_entry(wrap, self.fonts["body"],
                                 self.colors["panel_alt"], self.colors["text"],
                                 self.colors["outline"])
        id_entry.setText(person_id or "")
        wrap_l.addWidget(id_entry)

        edit_status = _styled_label(wrap, "", self.fonts["small"],
                                    self.colors["danger"])
        wrap_l.addWidget(edit_status)

        btn_row = QWidget(wrap)
        btn_row.setStyleSheet("background: transparent;")
        br_l = QHBoxLayout(btn_row)
        br_l.setContentsMargins(0, 8, 0, 0)
        br_l.setSpacing(8)

        cancel_btn = _styled_button(
            btn_row, "Cancel", self.fonts["body_bold"],
            "transparent", self.colors["text"],
            hover=self.colors["sidebar_hover"],
            radius=21,
            border_color=self.colors["outline"], border_width=1)
        br_l.addWidget(cancel_btn, 1)

        save_btn = _styled_button(
            btn_row, "Save Changes", self.fonts["body_bold"],
            self.colors["success"], "#FFFFFF",
            hover="#059669", radius=21)
        br_l.addWidget(save_btn, 1)
        wrap_l.addWidget(btn_row)

        cancel_btn.clicked.connect(dialog.reject)

        def _confirm():
            new_name = name_entry.text().strip()
            new_khmer = khmer_entry.text().strip()
            new_id = id_entry.text().strip() or "N/A"
            if not new_name:
                edit_status.setText("Name cannot be empty.")
                return
            ok, msg = on_save(new_name, new_khmer, new_id)
            if ok:
                dialog.accept()
            else:
                edit_status.setText(msg or "Could not save changes.")

        save_btn.clicked.connect(_confirm)
        dialog.exec()

    # NOTE: _change_camera / _capture_register / _upload_photo /
    # _save_registration / _upload_search_photo / _capture_search /
    # _run_face_search / _add_photo_to_selected / _edit_selected_face /
    # _delete_selected_face are intentionally NOT stubbed here.
    #
    # FaceRecognitionApp is built as
    #   class FaceRecognitionApp(FaceRecognitionGUI, VideoPipelineMixin,
    #                             RegistrationMixin, FaceSearchMixin, ManagementMixin)
    # Python's MRO resolves an attribute to the FIRST base class that
    # defines it. FaceRecognitionGUI (this class) is leftmost, so a
    # `pass`-stub method defined here would permanently shadow the real
    # implementation in VideoPipelineMixin/RegistrationMixin/
    # FaceSearchMixin/ManagementMixin — the mixin's method would never
    # be reached, no matter what the subclass looks like. (This file
    # previously defined no-op `pass` stubs for all of the methods
    # above "for main.py", which silently made every Capture/Upload/
    # Save/Search/Add Photo/Edit/Delete button and the camera dropdown
    # do nothing at all.) These methods are declared only by the mixins
    # that actually implement them.


# ── Entry point (for standalone testing) ─────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FaceRecognitionGUI()
    win.known_names = []
    win._build_ui()
    win.show()
    sys.exit(app.exec())