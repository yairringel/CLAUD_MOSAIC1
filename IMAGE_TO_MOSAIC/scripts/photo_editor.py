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

from PIL import Image
from PyQt5.QtCore import QEvent, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
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

        self.image_label = QLabel(placeholder)
        self.image_label.setAlignment(Qt.AlignCenter)
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
        self.prompt_btn     = QPushButton("Choose Prompt...")
        self.key_btn        = QPushButton("Choose Key File...")
        self.preview_btn    = QPushButton("Preview Prompt")
        self.generate_btn   = QPushButton("Generate →")
        self.save_src_btn   = QPushButton("Save Source...")
        self.save_res_btn   = QPushButton("Save Result...")
        for b in (self.load_btn, self.background_btn, self.prompt_btn, self.key_btn,
                  self.preview_btn, self.generate_btn, self.save_src_btn, self.save_res_btn):
            bar.addWidget(b)
        bar.addStretch(1)

        self.prompt_label = QLabel("Prompt: (none chosen)")
        bar.addWidget(self.prompt_label)
        root_layout.addLayout(bar)

        # Second row: AI parameters (injected into the prompt)
        bar2 = QHBoxLayout()
        bar2.addWidget(QLabel("Tile size (mm):"))
        self.tile_size_spin = QDoubleSpinBox()
        self.tile_size_spin.setRange(0.5, 200.0)
        self.tile_size_spin.setSingleStep(0.5)
        self.tile_size_spin.setDecimals(1)
        self.tile_size_spin.setValue(8.0)
        bar2.addWidget(self.tile_size_spin)

        bar2.addWidget(QLabel("Gap size (mm):"))
        self.gap_size_spin = QDoubleSpinBox()
        self.gap_size_spin.setRange(0.1, 50.0)
        self.gap_size_spin.setSingleStep(0.1)
        self.gap_size_spin.setDecimals(1)
        self.gap_size_spin.setValue(1.0)
        bar2.addWidget(self.gap_size_spin)

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
        has_prompt = bool(self.current_prompt_text)
        running    = self.worker is not None and self.worker.isRunning()
        self.background_btn.setEnabled(has_src and not running)
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
        self.statusBar().showMessage(f"Loaded {Path(path).name}")
        self._update_button_states()

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

    def preview_prompt(self) -> None:
        """Show the exact text that would be sent to Gemini if you clicked Generate now."""
        if not self.current_prompt_text:
            QMessageBox.information(self, "No prompt", "Pick a prompt file first (Choose Prompt...).")
            return
        final_text = self._inject_size_overrides(self.current_prompt_text)
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

    def _inject_size_overrides(self, prompt_text: str) -> str:
        """Append authoritative override blocks: tile/gap size, plus optional background fill.

        The mm values are also translated into a tile-count-across-image ratio
        (assuming the generated image represents a 300 mm physical tile) so the
        model has a concrete spatial target it can actually act on — image
        generators don't have physical-unit awareness, but they CAN count
        tesserae across a frame.
        """
        tile_mm = self.tile_size_spin.value()
        gap_mm = self.gap_size_spin.value()
        # Assumed physical size of the generated mosaic. Lets us convert mm -> count.
        assumed_canvas_mm = 300.0
        tiles_across = max(2, int(round(assumed_canvas_mm / (tile_mm + gap_mm))))
        size_override = (
            "\n\n"
            "==== SIZE OVERRIDE (final, authoritative) ====\n"
            f"Target tessera size: {tile_mm:g} mm across.\n"
            f"Target grout gap:    {gap_mm:g} mm between adjacent tesserae.\n"
            f"Concrete spatial target: the generated image represents a "
            f"{assumed_canvas_mm:g} mm × {assumed_canvas_mm:g} mm physical "
            f"mosaic, so EXACTLY ABOUT {tiles_across} tesserae must fit "
            f"across the image width, and the same count across the height.\n"
            f"Apply this size UNIFORMLY to every tile in the output. If any "
            f"prior instruction in this prompt mentions a different tile or "
            f"grout size (e.g. '5-10 mm' or '1-3 mm'), IGNORE those numbers "
            f"and use {tile_mm:g} mm / {gap_mm:g} mm with the {tiles_across}-"
            f"across-image count as the authoritative scale."
        )

        bg_fill = ""
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
                bg_fill = "\n\n" + file_text
        return prompt_text + size_override + bg_fill

    def generate(self) -> None:
        if self.source_pane.pil_image is None or not self.current_prompt_text:
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

        prompt_with_sizes = self._inject_size_overrides(self.current_prompt_text)
        self.worker = GenerationWorker(
            self.source_pane.pil_image, prompt_with_sizes, DEFAULT_MODEL,
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
