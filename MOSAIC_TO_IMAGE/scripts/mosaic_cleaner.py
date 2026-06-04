"""Mosaic Cleaner — turn a real-world mosaic photo into a clean, idealized
mosaic image that downstream polygon-detection tools can parse easily.

Workflow:
  1. Load a photo of a real mosaic (input/...)
  2. Process — sends the image + prompt to Gemini Nano Banana Pro; the model
     straightens perspective, flattens tile interiors, darkens grout, etc.
  3. Save Result — drops the cleaned image into output/

The API key is read from:
  1. MOSAIC_TO_IMAGE/.key_path  (set via "Choose Key File...")
  2. GEMINI_API_KEY env var
  3. IMAGE_TO_MOSAIC/.key_path  (sibling project — if you already set it up there)
  4. IMAGE_TO_MOSAIC/.env  /  IMAGE_TO_MOSAIC/gemini.key
"""
from __future__ import annotations

import os
import sys
import traceback
from io import BytesIO
from pathlib import Path

from PIL import Image
from PyQt5.QtCore import QEvent, QPoint, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSplitter,
    QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent.parent           # MOSAIC_TO_IMAGE/
PROJECT_ROOT = ROOT.parent                              # CLAUDE_MOSAIC1.0/
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
PROMPT_FILE = ROOT / "prompts" / "real_to_clean_mosaic.txt"
FILL_GAPS_PROMPT_FILE = ROOT / "prompts" / "fill_missing_tiles.txt"
KEY_PATH_MEMO = ROOT / ".key_path"
SIBLING_KEY_PATH_MEMO = PROJECT_ROOT / "IMAGE_TO_MOSAIC" / ".key_path"

DEFAULT_MODEL = "gemini-3-pro-image-preview"


# ---------------------------------------------------------------------------
# API key loading — searches multiple locations including the sibling project
# ---------------------------------------------------------------------------

def read_key_from_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("GEMINI_API_KEY"):
            _, _, value = line.partition("=")
            return value.strip().strip('"').strip("'") or None
        if "=" not in line:
            return line.strip('"').strip("'")
    return None


def _resolve_via_memo(memo_path: Path) -> str | None:
    if not memo_path.is_file():
        return None
    target = Path(memo_path.read_text(encoding="utf-8").strip())
    return read_key_from_file(target)


def load_api_key() -> str | None:
    key = _resolve_via_memo(KEY_PATH_MEMO)
    if key:
        return key
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()
    key = _resolve_via_memo(SIBLING_KEY_PATH_MEMO)
    if key:
        return key
    for p in (
        ROOT / ".env", ROOT / "gemini.key",
        PROJECT_ROOT / "IMAGE_TO_MOSAIC" / ".env",
        PROJECT_ROOT / "IMAGE_TO_MOSAIC" / "gemini.key",
        PROJECT_ROOT / ".env",
    ):
        key = read_key_from_file(p)
        if key:
            return key
    return None


def closest_aspect_ratio(w: int, h: int) -> str:
    target = w / h
    candidates = {
        "1:1": 1.0, "16:9": 16 / 9, "9:16": 9 / 16,
        "4:3": 4 / 3, "3:4": 3 / 4, "2:1": 2.0, "1:2": 0.5,
    }
    return min(candidates.items(), key=lambda kv: abs(kv[1] - target))[0]


def auto_image_size(w: int, h: int) -> str:
    longest = max(w, h)
    if longest <= 1100:
        return "1K"
    if longest <= 2100:
        return "2K"
    return "4K"


# ---------------------------------------------------------------------------
# Background API-call worker
# ---------------------------------------------------------------------------

class CleanWorker(QThread):
    finished_ok = pyqtSignal(bytes)
    failed      = pyqtSignal(str)
    progress    = pyqtSignal(str)

    def __init__(self, source_image: Image.Image, prompt_text: str,
                 aspect_ratio: str, image_size: str):
        super().__init__()
        self.source_image = source_image
        self.prompt_text = prompt_text
        self.aspect_ratio = aspect_ratio
        self.image_size = image_size

    def run(self) -> None:
        try:
            api_key = load_api_key()
            if not api_key:
                self.failed.emit(
                    "No Gemini API key found.\n\n"
                    "Click 'Choose Key File...' to pick one, or set the "
                    "GEMINI_API_KEY env var.",
                )
                return

            self.progress.emit("Preparing image...")
            buf = BytesIO()
            img = self.source_image
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            img.save(buf, "PNG")
            img_bytes = buf.getvalue()

            self.progress.emit(f"Calling {DEFAULT_MODEL}...")
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
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
            for candidate in response.candidates or []:
                for part in (candidate.content.parts if candidate.content else []) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline is not None and getattr(inline, "data", None):
                        self.finished_ok.emit(inline.data)
                        return
            self.failed.emit("No image returned (safety filter or empty response).")
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Crop overlay label — QLabel that draws a movable, size-locked-square crop
# frame on top of the pixmap. Frame coordinates are stored in IMAGE pixel
# space so they survive zoom changes; we only multiply by `zoom` for display.
# ---------------------------------------------------------------------------

class CropOverlayLabel(QLabel):
    HANDLE_PX = 12   # corner handle size (label space)

    def __init__(self, pane, placeholder: str):
        super().__init__(placeholder)
        self._pane = pane
        self.crop_active = False
        self._frame = None         # (x, y, side) in IMAGE coords; None = no frame
        self._drag_mode = None     # "move" / "resize_tl"/"tr"/"bl"/"br" / None
        self._drag_start_mouse_img = None
        self._drag_start_frame = None
        self.setMouseTracking(True)

    def reset_frame(self) -> None:
        self._frame = None
        self._drag_mode = None
        self.unsetCursor()
        self.update()

    def set_crop_active(self, active: bool) -> None:
        self.crop_active = active
        if active:
            if self._frame is None and self._pane._base_qimg is not None:
                W = self._pane._base_qimg.width()
                H = self._pane._base_qimg.height()
                side = int(min(W, H) * 0.7)
                self._frame = ((W - side) // 2, (H - side) // 2, side)
        else:
            self._drag_mode = None
            self.unsetCursor()
        self.update()

    def get_frame_image_rect(self):
        if not self._frame:
            return None
        x, y, s = self._frame
        return (x, y, x + s, y + s)

    def _z(self) -> float:
        return max(0.0001, self._pane.zoom)

    def _pixmap_offset(self):
        """Where the pixmap's (0,0) lands inside the QLabel.

        ImagePane forces the label's alignment to top-left, so the pixmap
        sits at (0, 0). We keep this helper (returning a constant) so the
        math below documents the assumption.
        """
        return (0, 0)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.crop_active or self._frame is None or self._pane._base_qimg is None:
            return
        x, y, s = self._frame
        z = self._z()
        ox, oy = self._pixmap_offset()
        lx0 = int(round(ox + x * z))
        ly0 = int(round(oy + y * z))
        lside = max(1, int(round(s * z)))
        lx1 = lx0 + lside
        ly1 = ly0 + lside

        painter = QPainter(self)
        # Dim everything outside the frame so the user clearly sees the selection.
        dim = QColor(0, 0, 0, 130)
        painter.fillRect(0, 0, self.width(), ly0, dim)
        painter.fillRect(0, ly1, self.width(), max(0, self.height() - ly1), dim)
        painter.fillRect(0, ly0, lx0, lside, dim)
        painter.fillRect(lx1, ly0, max(0, self.width() - lx1), lside, dim)
        # Frame border
        painter.setPen(QPen(QColor(255, 220, 0), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(lx0, ly0, lside, lside)
        # Corner handles
        hs = self.HANDLE_PX
        painter.setBrush(QColor(255, 220, 0))
        for cx, cy in [(lx0, ly0), (lx1, ly0), (lx0, ly1), (lx1, ly1)]:
            painter.drawRect(cx - hs // 2, cy - hs // 2, hs, hs)

    def _img_pos(self, label_pos: QPoint):
        z = self._z()
        ox, oy = self._pixmap_offset()
        return ((label_pos.x() - ox) / z, (label_pos.y() - oy) / z)

    def _hit_test(self, label_pos: QPoint):
        if not self._frame or not self.crop_active:
            return None
        x, y, s = self._frame
        z = self._z()
        ox, oy = self._pixmap_offset()
        lx0 = ox + x * z; ly0 = oy + y * z
        lx1 = ox + (x + s) * z; ly1 = oy + (y + s) * z
        px = label_pos.x(); py = label_pos.y()
        hs = self.HANDLE_PX
        # Corner handles take priority over the interior
        for name, (cx, cy) in [
            ("resize_tl", (lx0, ly0)), ("resize_tr", (lx1, ly0)),
            ("resize_bl", (lx0, ly1)), ("resize_br", (lx1, ly1)),
        ]:
            if abs(px - cx) <= hs and abs(py - cy) <= hs:
                return name
        if lx0 < px < lx1 and ly0 < py < ly1:
            return "move"
        return None

    def _update_hover_cursor(self, pos: QPoint) -> None:
        mode = self._hit_test(pos)
        if mode == "move":
            self.setCursor(Qt.OpenHandCursor)
        elif mode in ("resize_tl", "resize_br"):
            self.setCursor(Qt.SizeFDiagCursor)
        elif mode in ("resize_tr", "resize_bl"):
            self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.unsetCursor()

    def mousePressEvent(self, event):
        if not self.crop_active or event.button() != Qt.LeftButton:
            super().mousePressEvent(event); return
        mode = self._hit_test(event.pos())
        if mode is None:
            super().mousePressEvent(event); return
        self._drag_mode = mode
        self._drag_start_mouse_img = self._img_pos(event.pos())
        self._drag_start_frame = self._frame
        if mode == "move":
            self.setCursor(Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        if not self.crop_active:
            super().mouseMoveEvent(event); return
        if self._drag_mode is None:
            self._update_hover_cursor(event.pos())
            super().mouseMoveEvent(event); return
        if self._pane._base_qimg is None:
            return
        W = self._pane._base_qimg.width()
        H = self._pane._base_qimg.height()
        cur_x, cur_y = self._img_pos(event.pos())
        x0, y0, side0 = self._drag_start_frame
        x1 = x0 + side0
        y1 = y0 + side0

        if self._drag_mode == "move":
            sx, sy = self._drag_start_mouse_img
            dx = cur_x - sx
            dy = cur_y - sy
            nx = max(0, min(W - side0, x0 + dx))
            ny = max(0, min(H - side0, y0 + dy))
            self._frame = (int(round(nx)), int(round(ny)), side0)
        elif self._drag_mode == "resize_br":
            anchor_x, anchor_y = x0, y0
            new_size = max(cur_x - anchor_x, cur_y - anchor_y)
            new_size = max(20, min(new_size, W - anchor_x, H - anchor_y))
            self._frame = (anchor_x, anchor_y, int(round(new_size)))
        elif self._drag_mode == "resize_tl":
            anchor_x, anchor_y = x1, y1
            new_size = max(anchor_x - cur_x, anchor_y - cur_y)
            new_size = max(20, min(new_size, anchor_x, anchor_y))
            self._frame = (
                int(round(anchor_x - new_size)),
                int(round(anchor_y - new_size)),
                int(round(new_size)),
            )
        elif self._drag_mode == "resize_tr":
            anchor_x, anchor_y = x0, y1
            new_size = max(cur_x - anchor_x, anchor_y - cur_y)
            new_size = max(20, min(new_size, W - anchor_x, anchor_y))
            self._frame = (anchor_x, int(round(anchor_y - new_size)), int(round(new_size)))
        elif self._drag_mode == "resize_bl":
            anchor_x, anchor_y = x1, y0
            new_size = max(anchor_x - cur_x, cur_y - anchor_y)
            new_size = max(20, min(new_size, anchor_x, H - anchor_y))
            self._frame = (int(round(anchor_x - new_size)), anchor_y, int(round(new_size)))
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag_mode is not None and event.button() == Qt.LeftButton:
            self._drag_mode = None
            self._update_hover_cursor(event.pos())
            event.accept()
            return
        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# Image pane with wheel-zoom (same as photo_editor.py / mosaic_to_csv.py)
# ---------------------------------------------------------------------------

class ImagePane(QFrame):
    ZOOM_STEP = 1.15
    ZOOM_MIN = 0.05
    ZOOM_MAX = 20.0

    def __init__(self, title: str, placeholder: str):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.placeholder_text = placeholder
        self.pil_image: Image.Image | None = None
        self._base_qimg: QImage | None = None
        self.zoom = 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight:bold; padding:2px;")
        layout.addWidget(self.title_label)

        # Custom label that knows how to draw / manipulate a square crop frame on top
        # of the pixmap. It handles its own mouse events directly (not via filter).
        self.image_label = CropOverlayLabel(self, placeholder)
        # Pin pixmap to the top-left of the label so its position is always (0, 0).
        # This avoids the rounding/centering coord mismatch between Qt's drawing
        # and our crop math when the label is larger than the scaled pixmap.
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.image_label.setStyleSheet("background:#222; color:#bbb;")
        self.image_label.setMinimumSize(300, 300)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.image_label)
        self.scroll.setWidgetResizable(True)
        self.scroll.viewport().installEventFilter(self)
        layout.addWidget(self.scroll, 1)

        self.info_label = QLabel("—")
        self.info_label.setStyleSheet("color:#666;")
        layout.addWidget(self.info_label)

    def set_crop_mode(self, enabled: bool) -> None:
        """Show or hide the movable + resizable square crop frame on this pane."""
        self.image_label.set_crop_active(enabled)

    def get_crop_rect(self):
        """Return (x0, y0, x1, y1) in image coords if a valid frame is set, else None."""
        return self.image_label.get_frame_image_rect()

    def set_pil_image(self, img: Image.Image, info: str = "") -> None:
        self.pil_image = img
        rgba = img.convert("RGBA")
        self._base_qimg = QImage(
            rgba.tobytes("raw", "RGBA"), rgba.width, rgba.height,
            QImage.Format_RGBA8888,
        ).copy()
        self.zoom = 1.0
        # New image -> any old crop frame is meaningless. Reset and leave crop mode off.
        self.image_label.set_crop_active(False)
        self.image_label.reset_frame()
        self._apply_zoom()
        self.info_label.setText(
            info or f"{img.width} x {img.height} px  |  mode {img.mode}  |  zoom 100%",
        )

    def clear(self) -> None:
        self.pil_image = None
        self._base_qimg = None
        self.zoom = 1.0
        self.image_label.set_crop_active(False)
        self.image_label.reset_frame()
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(self.placeholder_text)
        self.info_label.setText("—")

    def _apply_zoom(self) -> None:
        if self._base_qimg is None:
            return
        new_w = max(1, int(round(self._base_qimg.width() * self.zoom)))
        new_h = max(1, int(round(self._base_qimg.height() * self.zoom)))
        scaled = self._base_qimg.scaled(
            new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(QPixmap.fromImage(scaled))
        self.image_label.resize(scaled.size())
        if self.pil_image is not None:
            self.info_label.setText(
                f"{self.pil_image.width} x {self.pil_image.height} px  |  "
                f"mode {self.pil_image.mode}  |  zoom {int(self.zoom * 100)}%",
            )

    def eventFilter(self, obj, event):
        # Wheel zoom on the scroll viewport
        if obj is self.scroll.viewport() and event.type() == QEvent.Wheel:
            if self._base_qimg is None:
                return True
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
        pos = event.position() if hasattr(event, "position") else event.posF()
        vx, vy = pos.x(), pos.y()
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        img_x = (hbar.value() + vx) / old_zoom
        img_y = (vbar.value() + vy) / old_zoom
        self.zoom = new_zoom
        self._apply_zoom()
        new_h_v = int(round(img_x * new_zoom - vx))
        new_v_v = int(round(img_y * new_zoom - vy))
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), new_h_v)))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), new_v_v)))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MosaicCleaner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mosaic Cleaner — real mosaic photo to clean polygon-friendly mosaic")
        self.resize(1500, 950)

        self.source_path: Path | None = None
        self.worker: CleanWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # Toolbar
        bar = QHBoxLayout()
        self.load_btn       = QPushButton("Load Image...")
        self.crop_btn       = QPushButton("Crop")
        self.crop_btn.setCheckable(True)
        self.crop_btn.setToolTip(
            "Show a square crop frame on the source. Drag the inside to move, "
            "drag a corner to resize (stays square). Click 'Apply Crop' to commit.",
        )
        self.crop_apply_btn = QPushButton("Apply Crop")
        self.crop_apply_btn.setToolTip("Commit the current crop frame")
        self.fill_gaps_chk  = QCheckBox("Fill missing/damaged tiles")
        self.fill_gaps_chk.setToolTip(
            "When checked, the API call ALSO invents new tiles to fill gaps, cracks, "
            "and missing pieces so the output looks complete and intact.",
        )
        self.key_btn        = QPushButton("Choose Key File...")
        self.process_btn    = QPushButton("Process →")
        self.save_btn       = QPushButton("Save Result...")
        for b in (self.load_btn, self.crop_btn, self.crop_apply_btn,
                  self.fill_gaps_chk, self.key_btn, self.process_btn, self.save_btn):
            bar.addWidget(b)
        bar.addStretch(1)
        self.key_label = QLabel("Key: (none)")
        self.key_label.setStyleSheet("color:#888;")
        bar.addWidget(self.key_label)
        root_layout.addLayout(bar)

        # Split panes
        splitter = QSplitter(Qt.Horizontal)
        self.source_pane = ImagePane("Real mosaic photo", "Load a mosaic photo to begin")
        self.result_pane = ImagePane("Cleaned mosaic", "Result will appear here after Process")
        splitter.addWidget(self.source_pane)
        splitter.addWidget(self.result_pane)
        splitter.setSizes([750, 750])
        root_layout.addWidget(splitter, 1)

        self.statusBar().showMessage("Ready.")

        self.load_btn.clicked.connect(self.load_image)
        self.crop_btn.clicked.connect(self.toggle_crop_mode)
        self.crop_apply_btn.clicked.connect(self.apply_crop)
        self.key_btn.clicked.connect(self.choose_key_file)
        self.process_btn.clicked.connect(self.process)
        self.save_btn.clicked.connect(self.save_result)

        self._refresh_key_status()
        self._update_buttons()

    # ----- state ----------------------------------------------------------

    def _update_buttons(self) -> None:
        has_src    = self.source_pane.pil_image is not None
        has_result = self.result_pane.pil_image is not None
        running    = self.worker is not None and self.worker.isRunning()
        self.process_btn.setEnabled(has_src and not running)
        self.save_btn.setEnabled(has_result and not running)
        self.load_btn.setEnabled(not running)
        self.key_btn.setEnabled(not running)
        self.crop_btn.setEnabled(has_src and not running)
        # Apply Crop only matters when a crop frame is currently visible.
        self.crop_apply_btn.setEnabled(
            has_src and not running and self.crop_btn.isChecked(),
        )
        if not has_src and self.crop_btn.isChecked():
            self.crop_btn.setChecked(False)
            self.source_pane.set_crop_mode(False)

    def _refresh_key_status(self) -> None:
        if KEY_PATH_MEMO.is_file():
            target = KEY_PATH_MEMO.read_text(encoding="utf-8").strip()
            if read_key_from_file(Path(target)):
                self.key_label.setText(f"Key: from {Path(target).name}")
                self.key_label.setStyleSheet("color:#2a7;")
                return
            self.key_label.setText(f"Key: ⚠ memo broken ({target})")
            self.key_label.setStyleSheet("color:#c33;")
            return
        if load_api_key():
            self.key_label.setText("Key: loaded (sibling project / env / fallback)")
            self.key_label.setStyleSheet("color:#2a7;")
            return
        self.key_label.setText("Key: (none — click 'Choose Key File...')")
        self.key_label.setStyleSheet("color:#888;")

    # ----- actions --------------------------------------------------------

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
                f"Could not read a key from {path}.\n"
                "Expected: a plain key on its own line, or a 'GEMINI_API_KEY=...' line.",
            )
            return
        try:
            KEY_PATH_MEMO.write_text(str(Path(path).resolve()), encoding="utf-8")
        except Exception as e:
            QMessageBox.warning(self, "Could not save key path", f"{type(e).__name__}: {e}")
        self._refresh_key_status()
        self.statusBar().showMessage(f"Key file set: {Path(path).name}")

    def toggle_crop_mode(self) -> None:
        """Crop button: show or hide the square crop frame on the source pane."""
        enabled = self.crop_btn.isChecked()
        self.source_pane.set_crop_mode(enabled)
        if enabled:
            self.statusBar().showMessage(
                "Crop frame visible — drag the inside to move, corners to resize. "
                "Click 'Apply Crop' to commit.",
            )
        else:
            self.statusBar().showMessage("Crop mode cancelled.")
        self._update_buttons()

    def apply_crop(self) -> None:
        """Commit the current square crop frame — replace the source image with it."""
        src = self.source_pane.pil_image
        if src is None:
            return
        rect = self.source_pane.get_crop_rect()
        if rect is None:
            QMessageBox.information(self, "No crop frame",
                                    "Toggle the Crop button first to place a frame.")
            return
        x0, y0, x1, y1 = rect
        # Defensive clamp.
        x0 = max(0, min(x0, src.width))
        y0 = max(0, min(y0, src.height))
        x1 = max(0, min(x1, src.width))
        y1 = max(0, min(y1, src.height))
        if x1 - x0 < 2 or y1 - y0 < 2:
            QMessageBox.warning(self, "Crop too small",
                                "The crop frame is too small. Resize it and try again.")
            return
        cropped = src.crop((x0, y0, x1, y1))
        self.source_pane.set_pil_image(
            cropped,
            f"(cropped: {src.width}x{src.height} -> {cropped.width}x{cropped.height})  "
            f"|  {cropped.width} x {cropped.height} px",
        )
        # set_pil_image already exits crop mode and resets the frame.
        self.crop_btn.setChecked(False)
        self.statusBar().showMessage(
            f"Cropped to {cropped.width} x {cropped.height} px (square)",
        )
        self._update_buttons()

    def load_image(self) -> None:
        start_dir = str(INPUT_DIR) if INPUT_DIR.exists() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load mosaic photo", start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*.*)",
        )
        if not path:
            return
        try:
            img = Image.open(path); img.load()
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"{type(e).__name__}: {e}")
            return
        self.source_path = Path(path)
        self.source_pane.set_pil_image(
            img, f"{Path(path).name}  |  {img.width} x {img.height} px",
        )
        self.result_pane.clear()
        self.statusBar().showMessage(f"Loaded {Path(path).name}")
        self._update_buttons()

    def process(self) -> None:
        if self.source_pane.pil_image is None:
            return
        if load_api_key() is None:
            QMessageBox.critical(
                self, "API key missing",
                "No Gemini API key found. Click 'Choose Key File...' first.",
            )
            return
        try:
            prompt_text = PROMPT_FILE.read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(
                self, "Prompt file missing",
                f"Could not read {PROMPT_FILE}\n\n{type(e).__name__}: {e}",
            )
            return

        # If the user wants gaps/damage filled, append the addendum from its own file
        # so they can edit the instructions without touching code.
        if self.fill_gaps_chk.isChecked():
            try:
                addendum = FILL_GAPS_PROMPT_FILE.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                QMessageBox.warning(
                    self, "Fill-gaps prompt missing",
                    f"Expected file not found:\n{FILL_GAPS_PROMPT_FILE}\n\n"
                    "Processing without the fill-gaps instructions.",
                )
                addendum = ""
            except Exception as e:
                QMessageBox.warning(
                    self, "Fill-gaps prompt unreadable",
                    f"{type(e).__name__}: {e}\n\nProcessing without it.",
                )
                addendum = ""
            if addendum:
                prompt_text = prompt_text.rstrip() + "\n\n" + addendum

        src = self.source_pane.pil_image
        # Always request 4K — polygon detection needs maximum pixel detail per tile.
        self.worker = CleanWorker(
            src, prompt_text,
            aspect_ratio=closest_aspect_ratio(src.width, src.height),
            image_size="4K",
        )
        self.worker.progress.connect(lambda m: self.statusBar().showMessage(m))
        self.worker.finished_ok.connect(self._on_processed)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()

        self.process_btn.setText("Processing...")
        self.statusBar().showMessage("Cleaning mosaic via Gemini...")
        self._update_buttons()

    def _on_processed(self, img_bytes: bytes) -> None:
        try:
            img = Image.open(BytesIO(img_bytes)); img.load()
        except Exception as e:
            QMessageBox.critical(self, "Bad response", f"Could not decode image: {e}")
            return
        self.result_pane.set_pil_image(
            img, f"Cleaned  |  {img.width} x {img.height} px",
        )
        self.statusBar().showMessage(
            f"Cleaned mosaic ready  |  {img.width} x {img.height} px",
        )

    def _on_failed(self, msg: str) -> None:
        QMessageBox.critical(self, "Process failed", msg)
        self.statusBar().showMessage("Process failed.")

    def _on_worker_done(self) -> None:
        self.worker = None
        self.process_btn.setText("Process →")
        self._update_buttons()

    def save_result(self) -> None:
        img = self.result_pane.pil_image
        if img is None:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = self.source_path.stem if self.source_path else "mosaic"
        default_name = f"{stem}_clean.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save cleaned mosaic as...", str(OUTPUT_DIR / default_name),
            "PNG image (*.png)",
        )
        if not path:
            return
        out_path = Path(path)
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")
        try:
            out = img
            if out.mode not in ("RGB", "RGBA", "L", "LA"):
                out = out.convert("RGBA")
            out.save(out_path, "PNG")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        self.statusBar().showMessage(f"Saved {out_path.name}")
        QMessageBox.information(
            self, "Saved",
            f"Saved {out_path.name}\n\n{img.width} x {img.height} px\n{out_path}",
        )


def main() -> int:
    app = QApplication(sys.argv)
    win = MosaicCleaner()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
