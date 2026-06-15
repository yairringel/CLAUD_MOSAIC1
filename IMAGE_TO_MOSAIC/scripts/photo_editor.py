"""Photo Editor — image-to-mosaic part 1.

Load any image, then either:
  - resize and save it as a PNG (Save Source...), or
  - pick a prompt from prompts/ and send the image + prompt through the Gemini
    image-generation API to produce a mosaic version (Generate),
    then Save Result... as a PNG.

Requires:
  GEMINI_API_KEY env var (free tier of Google AI Studio is fine).
  google-genai, Pillow, PyQt5.

Usage:
  python IMAGE_TO_MOSAIC/scripts/photo_editor.py
"""
from __future__ import annotations

import os
import sys
import traceback
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
import math as _math
from PyQt5.QtCore import QEvent, QPointF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QDialog, QDialogButtonBox,
    QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QRadioButton, QScrollArea,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent.parent       # IMAGE_TO_MOSAIC/
PROJECT_ROOT = ROOT.parent                          # CLAUDE_MOSAIC1.0/
OUTPUT_DIR = ROOT / "output"
INPUT_DIR = ROOT / "input"
PROMPTS_DIR = ROOT / "prompts"
BG_FILL_PROMPT_FILE = PROMPTS_DIR / "background_fill_roman_mosaic.txt"

DEFAULT_MODEL = "gemini-3-pro-image-preview"        # Nano Banana Pro
GENERATION_IMAGE_SIZE = "4K"                         # matches the prompt text
GENERATION_ASPECT = "1:1"


def closest_aspect_ratio(w: int, h: int) -> str:
    """Pick the Nano Banana aspect-ratio string closest to the actual image ratio."""
    target = w / h
    candidates = {
        "1:1": 1.0, "16:9": 16 / 9, "9:16": 9 / 16,
        "4:3": 4 / 3, "3:4": 3 / 4, "2:1": 2.0, "1:2": 0.5,
    }
    return min(candidates.items(), key=lambda kv: abs(kv[1] - target))[0]


def auto_image_size(w: int, h: int) -> str:
    """Pick the smallest Nano Banana output size that comfortably fits the source."""
    longest = max(w, h)
    if longest <= 1100:
        return "1K"
    if longest <= 2100:
        return "2K"
    return "4K"

KEY_PATH_MEMO = ROOT / ".key_path"   # remembers last chosen key-file path (not the key itself)


def read_key_from_file(path: Path) -> str | None:
    """Read a Gemini API key from a file. Supports plain (key only) or KEY=value lines."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("GEMINI_API_KEY"):
            _, _, value = line.partition("=")
            return value.strip().strip('"').strip("'") or None
        if "=" not in line:
            return line.strip('"').strip("'")
    return None


def load_api_key() -> str | None:
    """Return the Gemini API key from any supported source.

    Looked-up locations, in order:
      1. The file path stored in IMAGE_TO_MOSAIC/.key_path (set via "Choose Key File...")
      2. GEMINI_API_KEY env var
      3. IMAGE_TO_MOSAIC/.env  or  CLAUDE_MOSAIC1.0/.env  (GEMINI_API_KEY=... line)
      4. IMAGE_TO_MOSAIC/gemini.key   (plain key)
    Both `.env` and `*.key` are covered by the project .gitignore.
    """
    if KEY_PATH_MEMO.is_file():
        memo = Path(KEY_PATH_MEMO.read_text(encoding="utf-8").strip())
        key = read_key_from_file(memo)
        if key:
            return key

    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()

    for env_path in (ROOT / ".env", PROJECT_ROOT / ".env", ROOT / "gemini.key"):
        key = read_key_from_file(env_path)
        if key:
            return key

    return None


# ---------------------------------------------------------------------------
# Save-size dialog (shared by Source + Result saves)
# ---------------------------------------------------------------------------

class SaveSizeDialog(QDialog):
    def __init__(self, parent, src_w: int, src_h: int):
        super().__init__(parent)
        self.setWindowTitle("Save PNG — choose size")
        self.src_w = src_w
        self.src_h = src_h
        self._suppress = False

        form = QVBoxLayout(self)
        form.addWidget(QLabel(f"Original size: {src_w} × {src_h} px"))

        row_w = QHBoxLayout()
        row_w.addWidget(QLabel("Width:"))
        self.w_spin = QSpinBox(); self.w_spin.setRange(1, 100000); self.w_spin.setValue(src_w)
        row_w.addWidget(self.w_spin)
        form.addLayout(row_w)

        row_h = QHBoxLayout()
        row_h.addWidget(QLabel("Height:"))
        self.h_spin = QSpinBox(); self.h_spin.setRange(1, 100000); self.h_spin.setValue(src_h)
        row_h.addWidget(self.h_spin)
        form.addLayout(row_h)

        self.lock_aspect = QCheckBox("Lock aspect ratio"); self.lock_aspect.setChecked(True)
        form.addWidget(self.lock_aspect)

        self.w_spin.valueChanged.connect(self._on_w_changed)
        self.h_spin.valueChanged.connect(self._on_h_changed)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        form.addWidget(buttons)

    def _on_w_changed(self, value: int) -> None:
        if self._suppress or not self.lock_aspect.isChecked():
            return
        self._suppress = True
        self.h_spin.setValue(max(1, round(value * self.src_h / self.src_w)))
        self._suppress = False

    def _on_h_changed(self, value: int) -> None:
        if self._suppress or not self.lock_aspect.isChecked():
            return
        self._suppress = True
        self.w_spin.setValue(max(1, round(value * self.src_w / self.src_h)))
        self._suppress = False

    def chosen_size(self) -> tuple[int, int]:
        return self.w_spin.value(), self.h_spin.value()


# ---------------------------------------------------------------------------
# Gemini worker thread
# ---------------------------------------------------------------------------

class GenerationWorker(QThread):
    finished_ok = pyqtSignal(bytes)        # image bytes
    failed      = pyqtSignal(str)          # error message
    progress    = pyqtSignal(str)          # status text

    def __init__(self, source_image: Image.Image, prompt_text: str, model_id: str,
                 aspect_ratio: str = GENERATION_ASPECT,
                 image_size: str = GENERATION_IMAGE_SIZE):
        super().__init__()
        self.source_image = source_image
        self.prompt_text = prompt_text
        self.model_id = model_id
        self.aspect_ratio = aspect_ratio
        self.image_size = image_size

    def run(self) -> None:
        try:
            api_key = load_api_key()
            if not api_key:
                self.failed.emit(
                    "No Gemini API key found.\n\n"
                    "Set GEMINI_API_KEY env var, or save the key in "
                    "IMAGE_TO_MOSAIC/.env  (line: GEMINI_API_KEY=...)  "
                    "or in IMAGE_TO_MOSAIC/gemini.key (plain key).",
                )
                return

            self.progress.emit("Preparing image...")
            buf = BytesIO()
            img = self.source_image
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            img.save(buf, "PNG")
            img_bytes = buf.getvalue()

            self.progress.emit(f"Calling {self.model_id}...")
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.model_id,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    self.prompt_text,
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=self.aspect_ratio,
                        image_size=self.image_size,
                    ),
                ),
            )

            result_bytes = None
            for candidate in response.candidates or []:
                for part in (candidate.content.parts if candidate.content else []) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline is not None and getattr(inline, "data", None):
                        result_bytes = inline.data
                        break
                if result_bytes is not None:
                    break

            if result_bytes is None:
                self.failed.emit("No image returned (safety filter or empty response).")
                return
            self.finished_ok.emit(result_bytes)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Background editor dialog
# ---------------------------------------------------------------------------

class BackgroundDialog(QDialog):
    """Choose between a solid-color background or a transparent (cutout) result."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Edit background")
        self.setModal(True)
        self.resize(360, 200)
        self._color = QColor(255, 255, 255)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose what to do with the background of the source image:"))

        self.color_radio = QRadioButton("Replace with a solid color")
        self.color_radio.setChecked(True)
        layout.addWidget(self.color_radio)

        color_row = QHBoxLayout()
        color_row.addSpacing(20)
        color_row.addWidget(QLabel("Color:"))
        self.color_btn = QPushButton("       ")
        self._update_color_button()
        self.color_btn.clicked.connect(self._pick_color)
        color_row.addWidget(self.color_btn)
        color_row.addStretch(1)
        layout.addLayout(color_row)

        self.transparent_radio = QRadioButton("Cut out the subject (transparent background)")
        layout.addWidget(self.transparent_radio)

        # Color picker only enabled when "solid color" is selected.
        self.color_radio.toggled.connect(lambda checked: self.color_btn.setEnabled(checked))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_color(self) -> None:
        chosen = QColorDialog.getColor(self._color, self, "Background color")
        if chosen.isValid():
            self._color = chosen
            self._update_color_button()

    def _update_color_button(self) -> None:
        hex_str = f"#{self._color.red():02x}{self._color.green():02x}{self._color.blue():02x}"
        self.color_btn.setStyleSheet(
            f"background-color: {hex_str}; border: 1px solid #888; min-width: 60px;"
        )
        self.color_btn.setText(hex_str)

    def selected_action(self) -> str:
        return "color" if self.color_radio.isChecked() else "transparent"

    def selected_color(self) -> QColor:
        return self._color


# ---------------------------------------------------------------------------
# Image pane (used twice: Source + Result)
# ---------------------------------------------------------------------------

class SquareOverlayLabel(QLabel):
    """QLabel with a movable + rotatable BLACK square overlay drawn on top of
    its pixmap. Square state lives in IMAGE pixel coordinates so it survives
    zoom changes.

    Mouse behavior:
      - Press inside the square + drag → moves the square.
      - Wheel while hovering inside the square → rotates the square ±5° per click
        (event is consumed, so wheel-zoom does NOT fire).
      - Wheel outside the square → event is ignored, propagates to the scroll
        area's viewport so the existing wheel-zoom still works.
    """

    ROTATE_STEP_DEG = 5.0

    def __init__(self, pane, placeholder: str):
        super().__init__(placeholder)
        self._pane = pane
        # Square state in image-pixel coords. sq_size = 0 hides the overlay
        # and makes this label behave like a plain QLabel.
        self.sq_size = 0
        self.sq_cx = 0.0          # square CENTER in image coords
        self.sq_cy = 0.0
        self.sq_rotation = 0.0    # degrees clockwise
        self._dragging = False
        self._drag_start_mouse_img = None
        self._drag_start_center = None
        self.setMouseTracking(True)

    # ----- public API ------------------------------------------------------

    def set_square_size(self, size: int) -> None:
        """Set side length in image pixels; 0 hides the overlay."""
        new_size = max(0, int(size))
        # On first appearance (size goes 0 → >0), park the square at the
        # top-left so the user has something to grab onto.
        if self.sq_size <= 0 and new_size > 0:
            self.sq_cx = new_size / 2
            self.sq_cy = new_size / 2
            self.sq_rotation = 0.0
        self.sq_size = new_size
        self.update()

    def reset_square_for_new_image(self) -> None:
        """Recenter to top-left on image load so the square doesn't stay where
        it was on the previous (possibly differently-sized) image."""
        if self.sq_size > 0:
            self.sq_cx = self.sq_size / 2
            self.sq_cy = self.sq_size / 2
            self.sq_rotation = 0.0
        self.update()

    def get_square_state(self):
        """Return (size, cx, cy, rotation_deg) — used by Generate to burn the
        square into the PIL image at the right position/rotation/size."""
        return (self.sq_size, self.sq_cx, self.sq_cy, self.sq_rotation)

    # ----- geometry helpers -----------------------------------------------

    def _z(self) -> float:
        return max(0.0001, self._pane.zoom)

    def _pixmap_offset(self):
        """Where the pixmap's (0,0) lands inside this QLabel.

        ImagePane forces the label's alignment to top-left, so the pixmap is
        always at (0, 0) of the label widget. We keep this helper (returning
        a constant) so the math below stays explicit about the assumption.
        """
        return (0, 0)

    def _square_corners_img(self):
        """Return [(x, y), ...] of the four corners in image coords."""
        s = self.sq_size / 2.0
        cosA = _math.cos(_math.radians(self.sq_rotation))
        sinA = _math.sin(_math.radians(self.sq_rotation))
        corners = []
        for dx, dy in ((-s, -s), (s, -s), (s, s), (-s, s)):
            rx = dx * cosA - dy * sinA
            ry = dx * sinA + dy * cosA
            corners.append((self.sq_cx + rx, self.sq_cy + ry))
        return corners

    def _is_point_in_square(self, img_x: float, img_y: float) -> bool:
        if self.sq_size <= 0:
            return False
        s = self.sq_size / 2.0
        cosA = _math.cos(_math.radians(-self.sq_rotation))
        sinA = _math.sin(_math.radians(-self.sq_rotation))
        rx = (img_x - self.sq_cx) * cosA - (img_y - self.sq_cy) * sinA
        ry = (img_x - self.sq_cx) * sinA + (img_y - self.sq_cy) * cosA
        return abs(rx) <= s and abs(ry) <= s

    def _label_to_img(self, lx: float, ly: float):
        z = self._z()
        ox, oy = self._pixmap_offset()
        return ((lx - ox) / z, (ly - oy) / z)

    # ----- paint -----------------------------------------------------------

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.sq_size <= 0 or self._pane._base_qimg is None:
            return
        z = self._z()
        ox, oy = self._pixmap_offset()
        polygon = QPolygonF(
            [QPointF(ox + ix * z, oy + iy * z) for ix, iy in self._square_corners_img()],
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Hollow black frame only — image content inside the frame stays visible
        # so the user can see exactly what region the square is covering.
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0), 3))
        painter.drawPolygon(polygon)

    # ----- mouse: drag to move --------------------------------------------

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.sq_size <= 0:
            super().mousePressEvent(event); return
        ix, iy = self._label_to_img(event.pos().x(), event.pos().y())
        if not self._is_point_in_square(ix, iy):
            super().mousePressEvent(event); return
        self._dragging = True
        self._drag_start_mouse_img = (ix, iy)
        self._drag_start_center = (self.sq_cx, self.sq_cy)
        self.setCursor(Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging and self._pane._base_qimg is not None:
            ix, iy = self._label_to_img(event.pos().x(), event.pos().y())
            dx = ix - self._drag_start_mouse_img[0]
            dy = iy - self._drag_start_mouse_img[1]
            new_cx = self._drag_start_center[0] + dx
            new_cy = self._drag_start_center[1] + dy
            # Soft clamp so the centre stays inside the image (the square's
            # half-size can spill off-frame).
            W = self._pane._base_qimg.width()
            H = self._pane._base_qimg.height()
            self.sq_cx = max(0, min(W, new_cx))
            self.sq_cy = max(0, min(H, new_cy))
            self.update()
            event.accept()
            return
        # Hover cursor hint when over the square
        if self.sq_size > 0:
            ix, iy = self._label_to_img(event.pos().x(), event.pos().y())
            if self._is_point_in_square(ix, iy):
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ----- wheel: rotate when hovering over the square --------------------

    def wheelEvent(self, event):
        if self.sq_size > 0:
            pos = event.position() if hasattr(event, "position") else event.posF()
            ix, iy = self._label_to_img(pos.x(), pos.y())
            if self._is_point_in_square(ix, iy):
                delta = event.angleDelta().y()
                step = self.ROTATE_STEP_DEG if delta > 0 else -self.ROTATE_STEP_DEG
                self.sq_rotation = (self.sq_rotation + step) % 360
                self.update()
                event.accept()
                return
        # Not over the square — let the wheel event propagate so the scroll
        # area's existing wheel-zoom path fires.
        event.ignore()


class ImagePane(QFrame):
    """Image viewer with mouse-wheel zoom centered on the cursor.

    Zoom range is 0.05x–20x. Default zoom = 1.0 (1:1 pixel mapping).
    When zoomed beyond the viewport, scroll bars appear automatically.
    """

    ZOOM_STEP = 1.15
    ZOOM_MIN = 0.05
    ZOOM_MAX = 20.0

    def __init__(self, title: str, placeholder: str):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.placeholder_text = placeholder
        self.pil_image: Image.Image | None = None
        self._base_qimg: QImage | None = None     # full-resolution Qt image
        self.zoom = 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight:bold; padding:2px;")
        layout.addWidget(self.title_label)

        # Custom label that draws + manipulates the black tile-size square on
        # top of its pixmap. Behaves like a plain QLabel when sq_size == 0,
        # so the result pane (where set_reference_square_size is never called)
        # is unaffected.
        self.image_label = SquareOverlayLabel(self, placeholder)
        # Pin the pixmap to the label's top-left so its position is always (0, 0).
        # With Qt.AlignCenter, when the scroll area's setWidgetResizable(True)
        # made the label bigger than the pixmap (zoom out / small image), Qt
        # centered the pixmap and our overlay math drifted as zoom changed.
        # Top-left removes the drift entirely.
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.image_label.setStyleSheet("background:#222; color:#bbb;")
        self.image_label.setMinimumSize(300, 300)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.image_label)
        self.scroll.setWidgetResizable(True)
        # The viewport receives mouse wheel events; intercept them to zoom.
        self.scroll.viewport().installEventFilter(self)
        layout.addWidget(self.scroll, 1)

        self.info_label = QLabel("—")
        self.info_label.setStyleSheet("color:#666;")
        layout.addWidget(self.info_label)

    def set_pil_image(self, img: Image.Image, info: str = "") -> None:
        self.pil_image = img
        rgba = img.convert("RGBA")
        # Pillow buffer must outlive QImage; tobytes copies, so this is safe.
        self._base_qimg = QImage(
            rgba.tobytes("raw", "RGBA"), rgba.width, rgba.height,
            QImage.Format_RGBA8888,
        ).copy()
        self.zoom = 1.0
        # New image → recenter the reference square so it's not stranded at
        # coordinates that made sense for the previous image.
        self.image_label.reset_square_for_new_image()
        self._apply_zoom()
        self.info_label.setText(
            info or f"{img.width} × {img.height} px  |  mode {img.mode}  |  zoom 100%",
        )

    def clear(self) -> None:
        self.pil_image = None
        self._base_qimg = None
        self.zoom = 1.0
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(self.placeholder_text)
        self.info_label.setText("—")

    def set_reference_square_size(self, size: int) -> None:
        """Show / resize / hide the black tile-size reference square on this
        pane. 0 hides it. The square is movable (drag) and rotatable (wheel
        while hovering over it) once visible."""
        self.image_label.set_square_size(int(size))

    def get_reference_square_state(self):
        """Return (size, cx, cy, rotation_deg) of the on-screen square. Used by
        Generate to burn the same square into the image sent to Gemini."""
        return self.image_label.get_square_state()

    def _apply_zoom(self) -> None:
        if self._base_qimg is None:
            return
        new_w = max(1, int(round(self._base_qimg.width() * self.zoom)))
        new_h = max(1, int(round(self._base_qimg.height() * self.zoom)))
        scaled = self._base_qimg.scaled(
            new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        # No square overlay painted here — SquareOverlayLabel.paintEvent draws
        # the square on top of the pixmap so drag/rotate are zero-cost.
        self.image_label.setPixmap(QPixmap.fromImage(scaled))
        self.image_label.resize(scaled.size())
        if self.pil_image is not None:
            self.info_label.setText(
                f"{self.pil_image.width} × {self.pil_image.height} px  |  "
                f"mode {self.pil_image.mode}  |  zoom {int(self.zoom * 100)}%",
            )

    def eventFilter(self, obj, event):
        if obj is self.scroll.viewport() and event.type() == QEvent.Wheel:
            if self._base_qimg is None:
                return True   # eat the event so nothing scrolls on an empty pane
            self._wheel_zoom(event)
            return True
        return super().eventFilter(obj, event)

    def _wheel_zoom(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self.ZOOM_STEP if delta > 0 else (1.0 / self.ZOOM_STEP)
        old_zoom = self.zoom
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        # Cursor position inside the viewport.
        pos = event.position() if hasattr(event, "position") else event.posF()
        vx, vy = pos.x(), pos.y()

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        # Image-space coordinate of the point currently under the cursor.
        img_x = (hbar.value() + vx) / old_zoom
        img_y = (vbar.value() + vy) / old_zoom

        self.zoom = new_zoom
        self._apply_zoom()

        # Reposition the scroll so that the same image point stays under the cursor.
        new_h = int(round(img_x * new_zoom - vx))
        new_v = int(round(img_y * new_zoom - vy))
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), new_h)))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), new_v)))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class PhotoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Editor — Image to Mosaic")
        self.resize(1500, 900)

        self.current_source_path: Path | None = None
        self.current_prompt_path: Path | None = None
        self.current_prompt_text: str = ""
        self.worker: GenerationWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # Toolbar
        bar = QHBoxLayout()
        self.load_btn       = QPushButton("Load Image...")
        self.background_btn = QPushButton("Background...")
        self.expand_btn     = QPushButton("Expand 15%")
        self.prompt_btn     = QPushButton("Choose Prompt...")
        self.key_btn        = QPushButton("Choose Key File...")
        self.preview_btn    = QPushButton("Preview Prompt")
        self.generate_btn   = QPushButton("Generate →")
        self.save_src_btn   = QPushButton("Save Source...")
        self.save_res_btn   = QPushButton("Save Result...")
        for b in (self.load_btn, self.background_btn, self.expand_btn, self.prompt_btn,
                  self.key_btn, self.preview_btn, self.generate_btn,
                  self.save_src_btn, self.save_res_btn):
            bar.addWidget(b)
        bar.addStretch(1)

        self.prompt_label = QLabel("Prompt: (none chosen)")
        bar.addWidget(self.prompt_label)
        root_layout.addLayout(bar)

        # Second row: AI parameters (injected into the prompt)
        bar2 = QHBoxLayout()
        bar2.addWidget(QLabel("Red square (px):"))
        self.square_size_input = QSpinBox()
        self.square_size_input.setRange(0, 5000)
        self.square_size_input.setValue(60)
        self.square_size_input.setSingleStep(2)
        self.square_size_input.setToolTip(
            "Side length of the red reference square drawn on the source's "
            "top-left, in SOURCE-image pixels. 0 hides it.",
        )
        self.square_size_input.valueChanged.connect(self._on_square_size_changed)
        bar2.addWidget(self.square_size_input)

        self.use_red_square_chk = QCheckBox("Use red square as tile size")
        self.use_red_square_chk.setToolTip(
            "When checked: Generate burns the red square into the image it sends "
            "to Gemini and adds a prompt clause telling the model 'every tessera "
            "should be approximately the size of the red square'.",
        )
        bar2.addWidget(self.use_red_square_chk)

        self.keep_aspect_chk = QCheckBox("Keep original image ratio")
        self.keep_aspect_chk.setToolTip(
            "If checked, Generate sends the request with the source image's "
            "aspect ratio (snapped to the nearest one Nano Banana supports — "
            "1:1, 4:3, 3:4, 16:9, 9:16, 2:1, 1:2). "
            "If unchecked, output is forced to 1:1 square (default).",
        )
        bar2.addWidget(self.keep_aspect_chk)

        bar2.addWidget(QLabel("Max tiles across:"))
        self.max_tiles_input = QSpinBox()
        self.max_tiles_input.setRange(0, 500)
        self.max_tiles_input.setValue(0)
        self.max_tiles_input.setSingleStep(1)
        self.max_tiles_input.setToolTip(
            "Hard upper bound on the number of tiles across the output image "
            "width. 0 = no ceiling (don't add the clause). When > 0, a 'MAX N "
            "TILES ACROSS — make tiles bigger if you'd exceed this' clause is "
            "appended to the prompt. Use this when the model keeps producing "
            "tiles smaller than you want."
        )
        bar2.addWidget(self.max_tiles_input)

        self.enlarge_mosaic_chk = QCheckBox("Enlarge mosaic")
        self.enlarge_mosaic_chk.setToolTip(
            "When checked: the input is treated as an EXISTING mosaic. The "
            "selected prompt file is IGNORED — the request becomes a simple "
            "'re-render this same mosaic with bigger stones' instruction. "
            "The red-square option still works (it defines the new stone "
            "size). Keep-aspect still works. Background-fill, mm-tile-size, "
            "and per-feature rules are skipped (not relevant when the input "
            "is already a mosaic)."
        )
        self.enlarge_mosaic_chk.toggled.connect(lambda _=False: self._update_button_states())
        bar2.addWidget(self.enlarge_mosaic_chk)

        self.fill_bg_chk = QCheckBox("Fill empty background with matching Roman mosaic")
        self.fill_bg_chk.setToolTip(
            "Use when the source image has its background removed (cutout / solid color). "
            "The AI will fill the empty area with a Roman mosaic background that complements "
            "the subject — same tessera scale, classical palette, edge-to-edge."
        )
        bar2.addWidget(self.fill_bg_chk)
        bar2.addStretch(1)
        root_layout.addLayout(bar2)

        # Third row: key status
        bar3 = QHBoxLayout()
        self.key_label = QLabel("Key: (none)")
        self.key_label.setStyleSheet("color:#888;")
        bar3.addWidget(self.key_label)
        bar3.addStretch(1)
        root_layout.addLayout(bar3)

        # Split panes
        splitter = QSplitter(Qt.Horizontal)
        self.source_pane = ImagePane("Source", "Load an image to begin")
        self.result_pane = ImagePane("Result", "Result will appear here\nafter Generate")
        splitter.addWidget(self.source_pane)
        splitter.addWidget(self.result_pane)
        splitter.setSizes([750, 750])
        root_layout.addWidget(splitter, 1)

        # Status bar
        self.statusBar().showMessage("Ready.")

        # Wire up
        self.load_btn.clicked.connect(self.load_image)
        self.background_btn.clicked.connect(self.edit_background)
        # Hook so the overlay refreshes the moment the spin value changes.
        # (Already connected above; this line is a no-op intentionally left for clarity.)
        self.expand_btn.clicked.connect(self.expand_borders)
        self.prompt_btn.clicked.connect(self.choose_prompt)
        self.key_btn.clicked.connect(self.choose_key_file)
        self.preview_btn.clicked.connect(self.preview_prompt)
        self.generate_btn.clicked.connect(self.generate)
        self.save_src_btn.clicked.connect(lambda: self.save_pane(self.source_pane, "source"))
        self.save_res_btn.clicked.connect(lambda: self.save_pane(self.result_pane, "result"))

        self._refresh_key_status()
        self._update_button_states()

    # ----- API key --------------------------------------------------------

    def _refresh_key_status(self) -> None:
        """Update the key label to show whether a key is currently discoverable."""
        if KEY_PATH_MEMO.is_file():
            memo = KEY_PATH_MEMO.read_text(encoding="utf-8").strip()
            key = read_key_from_file(Path(memo))
            if key:
                self.key_label.setText(f"Key: loaded from {Path(memo).name}")
                self.key_label.setStyleSheet("color:#2a7;")
                return
            self.key_label.setText(f"Key: ⚠ file missing ({memo})")
            self.key_label.setStyleSheet("color:#c33;")
            return
        if load_api_key():
            self.key_label.setText("Key: loaded (env var / .env / gemini.key)")
            self.key_label.setStyleSheet("color:#2a7;")
            return
        self.key_label.setText("Key: (none — click 'Choose Key File...')")
        self.key_label.setStyleSheet("color:#888;")

    def choose_key_file(self) -> None:
        start_dir = str(ROOT)
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Gemini API key file", start_dir,
            "Key files (*.key *.txt *.env);;All files (*.*)",
        )
        if not path:
            return
        key = read_key_from_file(Path(path))
        if not key:
            QMessageBox.warning(
                self, "No key found",
                f"Could not read a key from {path}.\n\n"
                "Expected either a plain key, or a line: GEMINI_API_KEY=...",
            )
            return
        try:
            KEY_PATH_MEMO.write_text(str(Path(path).resolve()), encoding="utf-8")
        except Exception as e:
            QMessageBox.warning(self, "Could not save key path", f"{type(e).__name__}: {e}")
        self._refresh_key_status()
        self.statusBar().showMessage(f"Key file set: {Path(path).name}")

    # ----- state ----------------------------------------------------------

    def _update_button_states(self) -> None:
        has_src    = self.source_pane.pil_image is not None
        has_result = self.result_pane.pil_image is not None
        has_prompt = bool(self.current_prompt_text) or self.enlarge_mosaic_chk.isChecked()
        running    = self.worker is not None and self.worker.isRunning()
        self.background_btn.setEnabled(has_src and not running)
        self.expand_btn.setEnabled(has_src and not running)
        self.save_src_btn.setEnabled(has_src and not running)
        self.save_res_btn.setEnabled(has_result and not running)
        self.generate_btn.setEnabled(has_src and has_prompt and not running)

    # ----- actions --------------------------------------------------------

    def load_image(self) -> None:
        start_dir = str(INPUT_DIR) if INPUT_DIR.exists() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load image", start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*.*)",
        )
        if not path:
            return
        try:
            img = Image.open(path); img.load()
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"{type(e).__name__}: {e}")
            return
        self.current_source_path = Path(path)
        self.source_pane.set_pil_image(
            img, f"{Path(path).name}  |  {img.width} × {img.height} px  |  mode {img.mode}",
        )
        # Re-apply the current spin value so the red square shows up immediately
        # on the newly loaded image (set_pil_image preserves reference_square_size
        # but it was the previous load's value — re-set so the user sees the
        # correct rectangle on the new dimensions).
        self.source_pane.set_reference_square_size(self.square_size_input.value())
        self.statusBar().showMessage(f"Loaded {Path(path).name}")
        self._update_button_states()

    def _on_square_size_changed(self, value: int) -> None:
        """Spin-box callback: update the red overlay on the source pane."""
        self.source_pane.set_reference_square_size(value)

    def choose_prompt(self) -> None:
        start_dir = str(PROMPTS_DIR) if PROMPTS_DIR.exists() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose prompt file", start_dir,
            "Text files (*.txt);;All files (*.*)",
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Read failed", f"{type(e).__name__}: {e}")
            return
        if not text.strip():
            QMessageBox.warning(self, "Empty prompt", "The selected file is empty.")
            return
        self.current_prompt_path = Path(path)
        self.current_prompt_text = text
        self.prompt_label.setText(f"Prompt: {Path(path).name}  ({len(text)} chars)")
        self.statusBar().showMessage(f"Prompt loaded: {Path(path).name}")
        self._update_button_states()

    def edit_background(self) -> None:
        """Use the AI to replace the source image's background with a solid color or transparency."""
        if self.source_pane.pil_image is None:
            return
        if load_api_key() is None:
            QMessageBox.critical(
                self, "API key missing",
                "No Gemini API key found. Click 'Choose Key File...' first.",
            )
            return

        dlg = BackgroundDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return

        src = self.source_pane.pil_image
        if dlg.selected_action() == "color":
            color = dlg.selected_color()
            hex_str = f"#{color.red():02x}{color.green():02x}{color.blue():02x}"
            prompt_text = (
                "Edit this image: keep the MAIN SUBJECT exactly as it is, with its outline "
                "and inner detail preserved precisely. Replace the ENTIRE BACKGROUND with a "
                f"single solid uniform color: {hex_str}. The new background must be "
                "perfectly flat — no gradients, no texture, no shadows, no other objects. "
                "Output ONLY the result image, same composition."
            )
        else:
            prompt_text = (
                "Cut out the MAIN SUBJECT of this image cleanly along its true outline. "
                "Output the subject with a FULLY TRANSPARENT background — alpha channel = 0 "
                "on every non-subject pixel, alpha = 255 on the subject. The subject's "
                "interior pixels must keep their original colors. Output a PNG with an "
                "alpha channel."
            )

        self.worker = GenerationWorker(
            src, prompt_text, DEFAULT_MODEL,
            aspect_ratio=closest_aspect_ratio(src.width, src.height),
            image_size=auto_image_size(src.width, src.height),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_background_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()

        self.statusBar().showMessage("Processing background...")
        self.background_btn.setText("Processing...")
        self._update_button_states()

    def _on_background_done(self, img_bytes: bytes) -> None:
        try:
            img = Image.open(BytesIO(img_bytes)); img.load()
        except Exception as e:
            QMessageBox.critical(self, "Bad response", f"Could not decode image: {e}")
            return
        self.source_pane.set_pil_image(
            img, f"(background edited)  |  {img.width} × {img.height} px",
        )
        self.statusBar().showMessage(
            f"Background updated  |  {img.width} × {img.height} px",
        )

    # ---- Expand borders by 15% on each side (outpainting) -------------------

    EXPAND_FRACTION = 0.15       # added to each side, so final = W * (1 + 2*frac)

    def expand_borders(self) -> None:
        """Pad the source image by 15% on each side, then ask Gemini to outpaint
        the new padded area. Original pixels are preserved at full resolution in
        the deep interior; near the original-rectangle edge they are softly
        blended into the AI-painted area so the seam is invisible.
        """
        if self.source_pane.pil_image is None:
            return
        if load_api_key() is None:
            QMessageBox.critical(
                self, "API key missing",
                "No Gemini API key found. Click 'Choose Key File...' first.",
            )
            return

        src = self.source_pane.pil_image.convert("RGB")
        W, H = src.width, src.height
        pad_x = max(1, int(round(W * self.EXPAND_FRACTION)))
        pad_y = max(1, int(round(H * self.EXPAND_FRACTION)))
        new_w = W + 2 * pad_x
        new_h = H + 2 * pad_y

        # Build the padded canvas: original centred, mid-gray sentinel around it.
        padded = Image.new("RGB", (new_w, new_h), (128, 128, 128))
        padded.paste(src, (pad_x, pad_y))

        self._expand_orig = src
        self._expand_new_size = (new_w, new_h)
        self._expand_offset = (pad_x, pad_y)
        # Feather width: ~ 1/3 of the padding. Large enough to dissolve the seam,
        # small enough that most of the original stays untouched at full resolution.
        self._expand_feather = max(8, min(pad_x, pad_y) // 3)

        prompt_text = (
            "You are given a rectangular image whose outer border (about 15% "
            "on every side) is a SOLID GRAY frame. The central rectangle "
            "contains a real photograph.\n"
            "Replace the gray frame with content that continues the "
            "photograph outward, so the entire output looks like one "
            "uninterrupted scene.\n"
            "CRITICAL constraints:\n"
            "- The central rectangle stays visually identical: do not alter "
            "content, colors, faces, or composition there.\n"
            "- The new outer band MUST visually continue what is visible at "
            "the edge of the central rectangle — same background, same "
            "textures, same lighting, same perspective, same style. NO hard "
            "edge or visible seam between the original photograph and the "
            "new outpainted area: textures, gradients, and shapes that meet "
            "the boundary must flow smoothly across it.\n"
            "- Do NOT introduce new subjects, objects, faces, text, or "
            "decorative elements in the outpainted band. It is a passive "
            "continuation of the existing scene.\n"
            "- No gray pixels remain anywhere in the output."
        )

        self.worker = GenerationWorker(
            padded, prompt_text, DEFAULT_MODEL,
            aspect_ratio=closest_aspect_ratio(new_w, new_h),
            image_size=auto_image_size(new_w, new_h),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_expanded)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()

        self.statusBar().showMessage(
            f"Expanding to {new_w} × {new_h} (+15% per side)...",
        )
        self.expand_btn.setText("Expanding...")
        self._update_button_states()

    def _on_expanded(self, img_bytes: bytes) -> None:
        """Composite: AI supplies the outer band; the original is feather-blended
        in over the centre so its rectangular edge dissolves into the new band."""
        try:
            ai = Image.open(BytesIO(img_bytes)); ai.load()
        except Exception as e:
            QMessageBox.critical(self, "Bad response", f"Could not decode image: {e}")
            return

        new_w, new_h = self._expand_new_size
        orig = self._expand_orig
        ox, oy = self._expand_offset
        feather = self._expand_feather

        # Bring the AI output up to the target size (LANCZOS for quality).
        if ai.size != (new_w, new_h):
            ai = ai.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
        else:
            ai = ai.convert("RGB")

        # Build the feathered alpha mask for the ORIGINAL:
        #   - 255 (fully opaque, use original) in the interior, away from edges
        #   - 0 (fully transparent, use AI) at the original rectangle's border
        #   - Gaussian-blurred ramp in between → invisible seam
        W, H = orig.size
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        # Solid white interior, leaving a `feather`-wide transparent ring.
        draw.rectangle([feather, feather, W - feather, H - feather], fill=255)
        # Soft-blur the rectangle edge so the transition is smooth.
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather * 0.6))

        ai.paste(orig, (ox, oy), mask=mask)

        self.source_pane.set_pil_image(
            ai,
            f"(expanded +15%)  |  {new_w} x {new_h} px  |  "
            f"original {W} x {H} preserved at centre, {feather}-px feathered edge",
        )
        self.statusBar().showMessage(
            f"Expanded by 15% on each side: {new_w} x {new_h} px  "
            f"(feather: {feather} px)",
        )

    def preview_prompt(self) -> None:
        """Show the exact text that would be sent to Gemini if you clicked Generate now."""
        if not self.enlarge_mosaic_chk.isChecked() and not self.current_prompt_text:
            QMessageBox.information(self, "No prompt", "Pick a prompt file first (Choose Prompt...).")
            return
        final_text = self._build_final_prompt()
        dlg = QDialog(self)
        dlg.setWindowTitle("Preview — final prompt sent to Gemini")
        dlg.resize(900, 700)
        layout = QVBoxLayout(dlg)
        editor = QPlainTextEdit()
        editor.setPlainText(final_text)
        editor.setReadOnly(True)
        editor.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 11pt;")
        layout.addWidget(editor)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec_()

    def _build_final_prompt(self) -> str:
        """Compose the final prompt text sent to Gemini.

        Two paths:
        - Enlarge mode (checkbox on): use the built-in 'rebuild this mosaic
          with bigger stones' prompt, plus red-square addendum if applicable,
          plus aspect-rewrite if keep-aspect is on. Skip mm-size block,
          background fill, and per-feature rules.
        - Normal mode: use the loaded prompt file with the usual injections.
        """
        if self.enlarge_mosaic_chk.isChecked():
            text = self._enlarge_prompt_base()
            if self.keep_aspect_chk.isChecked():
                text = self._rewrite_square_output_clause(text)
            if self.use_red_square_chk.isChecked() and self.square_size_input.value() > 0:
                text = text + self._red_square_addendum(self.square_size_input.value())
        else:
            text = self._inject_size_overrides(self.current_prompt_text or "")
            if self.use_red_square_chk.isChecked() and self.square_size_input.value() > 0:
                text = text + self._red_square_addendum(self.square_size_input.value())

        if self.max_tiles_input.value() > 0:
            text = text + self._max_tiles_addendum(self.max_tiles_input.value())
        return text

    @staticmethod
    def _max_tiles_addendum(max_n: int) -> str:
        """Hard-ceiling clause: cap the number of tiles across the output."""
        return (
            "\n\n"
            "==== HARD TILE-COUNT CEILING ====\n"
            f"The output mosaic must contain NO MORE THAN {max_n} tiles "
            f"across the image width, and NO MORE THAN {max_n} tiles across "
            f"the image height. This is a HARD UPPER BOUND, not a target — "
            f"fewer is fine, more is a failure.\n"
            f"If your default rendering would exceed {max_n} tiles across, "
            f"MAKE EACH TILE BIGGER until the count drops to {max_n} or "
            f"fewer. Apply uniformly across the whole image — subject, "
            f"background, hair, everything — do not shrink tiles in detailed "
            f"areas just to stay under the ceiling.\n"
            f"Self-check: count one row of tiles across the widest part of "
            f"the output. If that count exceeds {max_n}, you have failed and "
            f"must regenerate with bigger tiles."
        )

    @staticmethod
    def _enlarge_prompt_base() -> str:
        """Minimal prompt for the 'enlarge mosaic' mode — the input image is
        treated as an EXISTING mosaic to be rebuilt with larger stones."""
        return (
            "The supplied input image IS ALREADY A MOSAIC made of many small "
            "tesserae (stone tiles). Your task: RE-RENDER the same picture as "
            "a mosaic that uses LARGER STONES — fewer, bigger tiles across "
            "the image — while keeping subject, pose, composition, palette, "
            "framing, and orientation faithful to the input.\n"
            "\n"
            "Tile interiors:\n"
            "- Every tessera is filled with a SINGLE SOLID UNIFORM COLOR from "
            "edge to edge. No inner texture, no veining, no gradient, no "
            "highlights, no 3D shine. Color variation happens BETWEEN tiles, "
            "never within one.\n"
            "- Pick each tile's color as the dominant / average color of the "
            "corresponding region in the input.\n"
            "\n"
            "Grout (the dark separator between tiles):\n"
            "- A DARK CONTINUOUS line on every side of every tile (charcoal "
            "to pure black, #000–#222). Visibly DARKER than every tile in "
            "the mosaic.\n"
            "- AT LEAST 4 pixels wide in the output. Unbroken — no gaps, no "
            "fades. No two tiles touch.\n"
            "\n"
            "Boundaries and lighting:\n"
            "- Tile-to-grout edges are SHARP. No anti-aliasing softness, no "
            "glow.\n"
            "- Lighting is perfectly flat and even. No directional shadows, "
            "glare, or specular hot spots.\n"
            "\n"
            "Coverage:\n"
            "- The mosaic fills the frame edge-to-edge. Every output pixel "
            "is either tile interior or grout — no blank background.\n"
            "\n"
            "Camera:\n"
            "- Strict orthographic top-down view, perpendicular to the "
            "surface. No perspective, no tilt.\n"
            "\n"
            "Output resolution:\n"
            "- Render at 4K (3840 × 3840 px). Do not downscale.\n"
        )

    @staticmethod
    def _pil_with_black_square(img: Image.Image, side_px: int,
                                cx: float, cy: float, rotation_deg: float) -> Image.Image:
        """Return a copy of `img` with a HOLLOW BLACK FRAME burned in (3-px line,
        no fill). cx/cy are the frame's CENTER in input-image coordinates;
        rotation_deg rotates clockwise around that center. The image content
        inside the frame stays visible so the model can see which region the
        user is marking.
        """
        from PIL import Image as _PILImage
        from PIL import ImageDraw as _ImageDraw
        side_px = max(1, int(side_px))
        frame_width = 3
        # Build a transparent sprite the size of the frame's bounding box (the
        # diagonal of the square in worst case, when rotated 45°). Then draw
        # the hollow square inside the sprite and rotate the whole sprite.
        diag = int(_math.ceil(side_px * _math.sqrt(2))) + 4
        sprite = _PILImage.new("RGBA", (diag, diag), (0, 0, 0, 0))
        sd = _ImageDraw.Draw(sprite)
        # Center the unrotated square inside the sprite.
        s_off = (diag - side_px) // 2
        sd.rectangle(
            [s_off, s_off, s_off + side_px, s_off + side_px],
            outline=(0, 0, 0, 255),
            width=frame_width,
        )
        if rotation_deg:
            # PIL's positive angle is COUNTERCLOCKWISE, so negate to match the
            # clockwise convention used by SquareOverlayLabel + Qt.
            sprite = sprite.rotate(-rotation_deg, resample=_PILImage.BICUBIC, expand=False)
        out = img.convert("RGBA").copy()
        paste_x = int(round(cx - diag / 2))
        paste_y = int(round(cy - diag / 2))
        out.alpha_composite(sprite, dest=(paste_x, paste_y))
        return out.convert("RGB")

    @staticmethod
    def _red_square_addendum(side_px: int) -> str:
        """Prompt clause for the hollow black frame burned into the input image.

        (Legacy method name kept; the marker is now a hollow black frame that
        the model treats as the bounding outline of a single mosaic tile.)
        """
        return (
            "\n\n"
            "==== BLACK FRAME = TILE SIZE REFERENCE + TILE OUTLINE ====\n"
            f"A HOLLOW BLACK SQUARE FRAME (thin black outline only, no fill) has been drawn on the supplied input image. The frame is {side_px} × {side_px} pixels in size, it MAY be rotated to any angle, and it may sit anywhere in the picture (not necessarily a corner). The image content INSIDE the frame is visible — that visible content is the part of the picture the frame is marking.\n"
            "This black frame serves TWO purposes simultaneously:\n"
            "1. SIZE REFERENCE: the side length of the frame defines the target size of each individual mosaic tile (tessera) in the output. Every tessera in the output should be approximately the SAME size as the frame (give or take ~10%), applied uniformly across subject, background, hair, everything.\n"
            "2. TILE OUTLINE: the frame marks the EXACT outline of ONE single mosaic tile in the output. In the output mosaic, include a single tessera at the same position, rotation, and size as the frame. The tile's COLOR is the dominant / average color of what's visible inside the frame in the input image (NOT black — black is only the frame line, not the tile color).\n"
            "Do NOT split this tile across multiple smaller tiles. Do NOT render the black frame line itself in the output (the frame is a marker, not content). Do NOT omit the marked tile.\n"
            "Self-check: in the output, can you find ONE tessera at the same location and orientation as the input's frame, sized like the frame, with the color of the image content that was visible inside the frame? AND is every other tessera in the mosaic roughly the same size as that one? If either is no, you have failed."
        )

    def _inject_size_overrides(self, prompt_text: str) -> str:
        """Build the final prompt by appending the size block and optional
        background-fill block. When the red-square checkbox is on, the size
        block is skipped (the red square is the sole size signal). When the
        keep-aspect checkbox is on, any 'Square 4K (3840 × 3840 px)' clause
        in the base prompt is rewritten to aspect-preserving language.
        """
        text = prompt_text

        # Strip / rewrite the hardcoded "Square 4K" line when the user wants
        # to keep the source aspect ratio. The model is told 4K total quality
        # without the square dimension constraint.
        if self.keep_aspect_chk.isChecked():
            text = self._rewrite_square_output_clause(text)

        # Background fill addon (still useful regardless of which size signal is in play).
        if self.fill_bg_chk.isChecked():
            try:
                file_text = BG_FILL_PROMPT_FILE.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                QMessageBox.warning(
                    self, "Background fill file missing",
                    f"Expected file not found:\n{BG_FILL_PROMPT_FILE}\n\n"
                    "Generating without the background-fill instructions.",
                )
                file_text = ""
            except Exception as e:
                QMessageBox.warning(
                    self, "Background fill file unreadable",
                    f"{type(e).__name__}: {e}\n\n"
                    "Generating without the background-fill instructions.",
                )
                file_text = ""
            if file_text:
                text = text + "\n\n" + file_text

        # Subject rendering rules — always appended. Handles small per-feature
        # cases where the model otherwise produces poor mosaic output on
        # portraits (e.g. teeth get split into multiple stones with random
        # colors). Add new per-feature rules to this list as they come up.
        text = text + (
            "\n\n"
            "==== SUBJECT RENDERING RULES ====\n"
            "- TEETH: each visible human tooth in the input must be rendered as "
            "exactly ONE single mosaic tile (one stone), and that stone is "
            "SOLID WHITE. Do not split a tooth across multiple tiles. Do not "
            "use yellow, gray, or any non-white color for teeth. A row of "
            "visible teeth becomes a row of separate white stones, one per tooth."
        )
        return text

    @staticmethod
    def _rewrite_square_output_clause(text: str) -> str:
        """Replace 'Square 4K (3840 × 3840 px)' style phrases in the user
        prompt with aspect-preserving language so the model isn't told to
        produce a square when the user has asked to keep the source ratio."""
        import re
        replacement = (
            "4K resolution (longest side 3840 px), preserving the input "
            "image's aspect ratio (not square)"
        )
        # Match the common phrasings the prompt files use: "Square 4K", optional
        # " resolution", optional parenthetical "(3840 × 3840 px)".
        pattern = re.compile(
            r"Square\s+4K(?:\s+resolution)?(?:\s*\(\s*3840\s*[×x]\s*3840\s*px\s*\))?",
            re.IGNORECASE,
        )
        text = pattern.sub(replacement, text)
        # Also strip standalone "Square (3840 × 3840 px)" if it appears.
        pattern2 = re.compile(
            r"Square\s*\(\s*3840\s*[×x]\s*3840\s*px\s*\)", re.IGNORECASE,
        )
        text = pattern2.sub(replacement, text)
        return text

    def generate(self) -> None:
        if self.source_pane.pil_image is None:
            return
        if not self.enlarge_mosaic_chk.isChecked() and not self.current_prompt_text:
            return
        if load_api_key() is None:
            QMessageBox.critical(
                self, "API key missing",
                "No Gemini API key found.\n\n"
                "Options:\n"
                "  1. Set the GEMINI_API_KEY env var and restart this app.\n"
                "  2. Create IMAGE_TO_MOSAIC/.env with one line:\n"
                "       GEMINI_API_KEY=your_key_here\n"
                "  3. Create IMAGE_TO_MOSAIC/gemini.key with just the key.\n\n"
                "Both .env and *.key are already in .gitignore — they won't "
                "be committed.",
            )
            return

        prompt_with_sizes = self._build_final_prompt()
        src_for_aspect = self.source_pane.pil_image

        # If the "Use red square as tile size" checkbox is on, burn the BLACK
        # square (now movable + rotatable on the source pane) into the image we
        # send to Gemini. The square is the user's tile-size reference AND it
        # becomes a real mosaic stone in the output — see _red_square_addendum.
        # (The addendum text itself is already in prompt_with_sizes via
        # _build_final_prompt; here we only bake the marker into the image.)
        if self.use_red_square_chk.isChecked() and self.square_size_input.value() > 0:
            sq_size, sq_cx, sq_cy, sq_rot = self.source_pane.get_reference_square_state()
            if sq_size > 0:
                src_for_aspect = self._pil_with_black_square(
                    src_for_aspect, sq_size, sq_cx, sq_cy, sq_rot,
                )
                try:
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    debug_path = OUTPUT_DIR / "_last_generate_input.png"
                    src_for_aspect.save(debug_path, "PNG")
                    self.statusBar().showMessage(
                        f"Black square baked in (size={sq_size}, rot={sq_rot:.0f}°) — "
                        f"image sent saved at {debug_path.name}",
                    )
                except Exception:
                    pass    # debug-save is non-critical

        if self.keep_aspect_chk.isChecked():
            aspect = closest_aspect_ratio(src_for_aspect.width, src_for_aspect.height)
        else:
            aspect = GENERATION_ASPECT
        self.worker = GenerationWorker(
            src_for_aspect, prompt_with_sizes, DEFAULT_MODEL,
            aspect_ratio=aspect,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_generated)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()

        self.statusBar().showMessage("Generating...")
        self.generate_btn.setText("Generating...")
        self._update_button_states()

    def _on_progress(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    def _on_generated(self, img_bytes: bytes) -> None:
        try:
            img = Image.open(BytesIO(img_bytes)); img.load()
        except Exception as e:
            QMessageBox.critical(self, "Bad response", f"Could not decode image: {e}")
            return
        self.result_pane.set_pil_image(img, f"Generated  |  {img.width} × {img.height} px")
        self.statusBar().showMessage(
            f"Generated {img.width} × {img.height} px via {DEFAULT_MODEL}.",
        )

    def _on_failed(self, msg: str) -> None:
        QMessageBox.critical(self, "Generation failed", msg)
        self.statusBar().showMessage("Generation failed.")

    def _on_worker_done(self) -> None:
        self.worker = None
        self.generate_btn.setText("Generate →")
        self.background_btn.setText("Background...")
        self.expand_btn.setText("Expand to Circle")
        self._update_button_states()

    def save_pane(self, pane: ImagePane, kind: str) -> None:
        img = pane.pil_image
        if img is None:
            return

        dlg = SaveSizeDialog(self, img.width, img.height)
        if dlg.exec_() != QDialog.Accepted:
            return
        target_w, target_h = dlg.chosen_size()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if kind == "source" and self.current_source_path:
            stem = self.current_source_path.stem
        elif kind == "result" and self.current_source_path:
            stem = self.current_source_path.stem + "_mosaic"
        else:
            stem = kind
        default_name = f"{stem}_{target_w}x{target_h}.png"

        out_path_str, _ = QFileDialog.getSaveFileName(
            self, f"Save {kind} as...", str(OUTPUT_DIR / default_name),
            "PNG image (*.png)",
        )
        if not out_path_str:
            return
        out_path = Path(out_path_str)
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")

        try:
            out = img
            if (out.width, out.height) != (target_w, target_h):
                out = out.resize((target_w, target_h), Image.LANCZOS)
            if out.mode not in ("RGB", "RGBA", "L", "LA"):
                out = out.convert("RGBA")
            out.save(out_path, "PNG")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        self.statusBar().showMessage(
            f"Saved {kind}: {out_path.name}  ({target_w} × {target_h} px)",
        )
        QMessageBox.information(
            self, "Saved",
            f"Saved {out_path.name}\n\n{target_w} × {target_h} px\n{out_path}",
        )


def main() -> int:
    app = QApplication(sys.argv)
    win = PhotoEditor()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
