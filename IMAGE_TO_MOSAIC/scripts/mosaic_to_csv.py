"""Mosaic to CSV — detect tesserae in a mosaic image, save them as polygons.

Algorithm = csv_experiments attempt 07 (adaptive Gaussian threshold + CC + Douglas-Peucker).

Pipeline:
  1. Grayscale
  2. cv2.adaptiveThreshold (Gaussian) — tile pixels (255) / grout pixels (0)
  3. Connected components on the tile pixels
  4. Per component: outer contour → Douglas-Peucker simplification → mean color
  5. Filter components smaller than min_area, and components touching the image border

CSV format (compatible with polygon_viewer.py and frame1_*.csv):
  coordinates,color_r,color_g,color_b,color_a,color_hex

Usage:
  python IMAGE_TO_MOSAIC/scripts/mosaic_to_csv.py
"""
from __future__ import annotations

import csv
import os
import sys
import traceback
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import QEvent, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent.parent       # IMAGE_TO_MOSAIC/
PROJECT_ROOT = ROOT.parent
OUTPUT_DIR = ROOT / "output"
INPUT_DIR = ROOT / "input"
PROMPTS_DIR = ROOT / "prompts"
SOLID_WHITE_PROMPT_FILE = PROMPTS_DIR / "solid_white_mask.txt"
KEY_PATH_MEMO = ROOT / ".key_path"

GEMINI_MODEL = "gemini-3-pro-image-preview"   # Nano Banana Pro


# ---------------------------------------------------------------------------
# API key loading — same scheme as photo_editor.py / mosaic_cleaner.py
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


def load_api_key() -> str | None:
    if KEY_PATH_MEMO.is_file():
        memo = Path(KEY_PATH_MEMO.read_text(encoding="utf-8").strip())
        key = read_key_from_file(memo)
        if key:
            return key
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()
    for p in (ROOT / ".env", PROJECT_ROOT / ".env", ROOT / "gemini.key"):
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


# ---------------------------------------------------------------------------
# Solid-white-transform worker — sends the source image to Gemini and gets
# back a black/white image where each tile is a separate solid white shape.
# The classical adaptive_threshold pipeline then has a very high-contrast
# input to detect from, while polygon mean-colors are sampled from the
# ORIGINAL image (never from the white-tile transform).
# ---------------------------------------------------------------------------

class SolidWhiteWorker(QThread):
    finished_ok = pyqtSignal(bytes)        # raw PNG bytes from the model
    failed      = pyqtSignal(str)
    progress    = pyqtSignal(str)

    def __init__(self, source_rgb: np.ndarray, prompt_text: str):
        super().__init__()
        self.source_rgb = source_rgb
        self.prompt_text = prompt_text

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
            h, w = self.source_rgb.shape[:2]
            buf = BytesIO()
            Image.fromarray(self.source_rgb, mode="RGB").save(buf, "PNG")
            img_bytes = buf.getvalue()

            self.progress.emit(f"Calling {GEMINI_MODEL} ({w} × {h})...")
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    self.prompt_text,
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=closest_aspect_ratio(w, h),
                        image_size="4K",
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


def ai_bytes_to_binary_mask(png_bytes: bytes, target_w: int, target_h: int) -> np.ndarray:
    """Decode the AI's binary PNG and resize to match the original image dims.

    Returns an HxWx3 uint8 RGB array (because detect_tiles expects RGB) where
    the AI's white tiles → (255,255,255) and the black grout/background → (0,0,0).
    Using nearest-neighbor preserves the binary nature when resizing.
    """
    pil = Image.open(BytesIO(png_bytes)).convert("L")
    if pil.size != (target_w, target_h):
        pil = pil.resize((target_w, target_h), Image.NEAREST)
    arr = np.array(pil)
    # Re-binarize in case the model returned any gray AA pixels.
    arr = ((arr >= 128).astype(np.uint8)) * 255
    # Stack to 3 channels so cv2.cvtColor + adaptiveThreshold get the inputs they expect.
    return np.stack([arr, arr, arr], axis=-1)


# ---------------------------------------------------------------------------
# Image pane with wheel-zoom centered on the cursor
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

        self.image_label = QLabel(placeholder)
        self.image_label.setAlignment(Qt.AlignCenter)
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

    def set_pil_image(self, img: Image.Image, info: str = "") -> None:
        self.pil_image = img
        rgba = img.convert("RGBA")
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
        new_h = int(round(img_x * new_zoom - vx))
        new_v = int(round(img_y * new_zoom - vy))
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), new_h)))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), new_v)))


# ---------------------------------------------------------------------------
# Detection pipeline (csv_experiments attempt 07)
# ---------------------------------------------------------------------------

def detect_tiles(rgb: np.ndarray, block_size: int, C: int,
                 min_area: int, epsilon_ratio: float,
                 color_source_rgb: np.ndarray | None = None):
    """Returns list of (polygon_points (N x 2 float), mean_rgb_float_0_1).

    If ``color_source_rgb`` is provided, the threshold + CC + contour extraction
    runs on ``rgb`` (typically the AI-produced black/white image) but mean
    colors are sampled from ``color_source_rgb`` (the original photograph).
    Must be the same HxW as ``rgb``.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # cv2.adaptiveThreshold needs block_size odd and ≥ 3.
    bs = max(3, int(block_size))
    if bs % 2 == 0:
        bs += 1

    tile_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        bs, int(C),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tile_mask, connectivity=4)

    # Sample mean color from the original image when supplied, otherwise from
    # the same image we detected on.
    color_src = color_source_rgb if color_source_rgb is not None else rgb

    tiles = []
    for lid in range(1, num_labels):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x0 = stats[lid, cv2.CC_STAT_LEFT]
        y0 = stats[lid, cv2.CC_STAT_TOP]
        ww = stats[lid, cv2.CC_STAT_WIDTH]
        hh = stats[lid, cv2.CC_STAT_HEIGHT]
        if x0 == 0 or y0 == 0 or x0 + ww == w or y0 + hh == h:
            continue   # drop components touching the image border

        sub_mask = (labels[y0:y0 + hh, x0:x0 + ww] == lid).astype(np.uint8) * 255
        contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = max(0.5, epsilon_ratio * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2).astype(np.float64)
        pts[:, 0] += x0
        pts[:, 1] += y0

        mean_rgb = (color_src[labels == lid].mean(axis=0) / 255.0).tolist()
        tiles.append((pts, tuple(mean_rgb)))

    return tiles


def render_polygons(tiles, w: int, h: int) -> Image.Image:
    """Fill each polygon with its mean color and add a 1-px black outline."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for pts, mean_rgb in tiles:
        color = (int(mean_rgb[0] * 255), int(mean_rgb[1] * 255), int(mean_rgb[2] * 255))
        poly_int = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [poly_int], color=color)
        cv2.polylines(canvas, [poly_int], isClosed=True, color=(0, 0, 0), thickness=1)
    return Image.fromarray(canvas, mode="RGB")


def write_csv(tiles, out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["coordinates", "color_r", "color_g", "color_b", "color_a", "color_hex"])
        for pts, mean_rgb in tiles:
            coords_str = "[" + ", ".join(f"({x}, {y})" for x, y in pts) + "]"
            r, g, b = mean_rgb
            hex_str = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            writer.writerow([coords_str, r, g, b, 1.0, hex_str])


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MosaicToCsv(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mosaic → CSV Polygons")
        self.resize(1500, 950)

        self.source_path: Path | None = None
        self.source_rgb: np.ndarray | None = None
        # Cached AI-produced binary mask (HxWx3 uint8, same dims as source). Set
        # on first Detect with the "Use solid white transform" checkbox on; reused
        # for subsequent Detects with the same source so changing block-size/C/etc.
        # doesn't re-spend API calls. Cleared whenever a new image is loaded.
        self.solid_white_rgb: np.ndarray | None = None
        self.worker: SolidWhiteWorker | None = None
        self.tiles: list = []

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # Toolbar
        bar = QHBoxLayout()
        self.load_btn    = QPushButton("Load Image...")
        self.key_btn     = QPushButton("Choose Key File...")
        self.detect_btn  = QPushButton("Detect")
        self.save_btn    = QPushButton("Save CSV...")
        for b in (self.load_btn, self.key_btn, self.detect_btn, self.save_btn):
            bar.addWidget(b)
        bar.addStretch(1)
        self.tile_count_label = QLabel("Tiles: —")
        bar.addWidget(self.tile_count_label)
        root_layout.addLayout(bar)

        # Parameters — defaults from csv_experiments attempt 07.
        params = QHBoxLayout()
        self.solid_white_chk = QCheckBox("Use solid white transform")
        self.solid_white_chk.setToolTip(
            "When checked, Detect first asks Gemini to convert the source into a "
            "black/white image (every tile a separate white shape on black). The "
            "adaptive-threshold pipeline runs on that high-contrast image; each "
            "polygon's mean color is sampled from the ORIGINAL image, so colors "
            "stay accurate. The AI image is cached per source — re-running Detect "
            "with different parameters does NOT spend another API call.",
        )
        params.addWidget(self.solid_white_chk)

        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(3, 501); self.block_size_spin.setSingleStep(2)
        self.block_size_spin.setValue(51)
        self.C_spin = QSpinBox()
        self.C_spin.setRange(-50, 50); self.C_spin.setValue(5)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 1_000_000); self.min_area_spin.setValue(200)
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.0, 0.2); self.eps_spin.setDecimals(4)
        self.eps_spin.setSingleStep(0.0005); self.eps_spin.setValue(0.02)

        for label, widget in (
            ("Block size:",    self.block_size_spin),
            ("C (offset):",    self.C_spin),
            ("Min tile area:", self.min_area_spin),
            ("Simplify ε:",    self.eps_spin),
        ):
            params.addWidget(QLabel(label))
            params.addWidget(widget)
        params.addStretch(1)
        root_layout.addLayout(params)

        # Split panes
        splitter = QSplitter(Qt.Horizontal)
        self.source_pane = ImagePane("Source mosaic image", "Load a mosaic image to begin")
        self.result_pane = ImagePane("Detected polygons",  "Polygons will appear here after Detect")
        splitter.addWidget(self.source_pane)
        splitter.addWidget(self.result_pane)
        splitter.setSizes([750, 750])
        root_layout.addWidget(splitter, 1)

        self.statusBar().showMessage("Ready.")

        self.load_btn.clicked.connect(self.load_image)
        self.key_btn.clicked.connect(self.choose_key_file)
        self.detect_btn.clicked.connect(self.detect)
        self.save_btn.clicked.connect(self.save_csv)
        self._update_buttons()

    def _update_buttons(self) -> None:
        running = self.worker is not None and self.worker.isRunning()
        self.detect_btn.setEnabled(self.source_rgb is not None and not running)
        self.save_btn.setEnabled(bool(self.tiles) and not running)
        self.load_btn.setEnabled(not running)
        self.key_btn.setEnabled(not running)
        self.solid_white_chk.setEnabled(not running)

    def choose_key_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Gemini API key file", str(ROOT),
            "Key files (*.key *.txt *.env);;All files (*.*)",
        )
        if not path:
            return
        key = read_key_from_file(Path(path))
        if not key:
            QMessageBox.warning(
                self, "No key found",
                f"Could not read a key from {path}.\n"
                "Expected: a plain key on its own line, or 'GEMINI_API_KEY=...'.",
            )
            return
        try:
            KEY_PATH_MEMO.write_text(str(Path(path).resolve()), encoding="utf-8")
        except Exception as e:
            QMessageBox.warning(self, "Could not save key path", f"{type(e).__name__}: {e}")
        self.statusBar().showMessage(f"Key file set: {Path(path).name}")

    def load_image(self) -> None:
        start_dir = str(INPUT_DIR) if INPUT_DIR.exists() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load mosaic image", start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*.*)",
        )
        if not path:
            return
        try:
            pil = Image.open(path); pil.load()
            rgb_pil = pil.convert("RGB")
            self.source_rgb = np.array(rgb_pil)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"{type(e).__name__}: {e}")
            return
        self.source_path = Path(path)
        self.source_pane.set_pil_image(
            rgb_pil,
            f"{Path(path).name}  |  {rgb_pil.width} × {rgb_pil.height} px",
        )
        self.tiles = []
        self.solid_white_rgb = None    # new source → old AI cache is no longer valid
        self.result_pane.clear()
        self.tile_count_label.setText("Tiles: —")
        self.statusBar().showMessage(f"Loaded {Path(path).name}")
        self._update_buttons()

    # ----- Detect dispatch -------------------------------------------------

    def detect(self) -> None:
        """Runs synchronously when the solid-white checkbox is off; dispatches
        through the API worker (then back into _run_classical_detect) when on.
        """
        if self.source_rgb is None:
            return
        if not self.solid_white_chk.isChecked():
            self._run_classical_detect(detection_rgb=self.source_rgb,
                                       color_source_rgb=None)
            return

        # Solid-white path — use cache if we already have it.
        if self.solid_white_rgb is not None:
            self._run_classical_detect(detection_rgb=self.solid_white_rgb,
                                       color_source_rgb=self.source_rgb)
            return

        # No cache yet — need an API call.
        if load_api_key() is None:
            QMessageBox.critical(
                self, "API key missing",
                "No Gemini API key found. Click 'Choose Key File...' first.",
            )
            return
        try:
            prompt_text = SOLID_WHITE_PROMPT_FILE.read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(
                self, "Prompt file missing",
                f"Could not read {SOLID_WHITE_PROMPT_FILE}\n\n{type(e).__name__}: {e}",
            )
            return

        self.worker = SolidWhiteWorker(self.source_rgb, prompt_text)
        self.worker.progress.connect(lambda m: self.statusBar().showMessage(m))
        self.worker.finished_ok.connect(self._on_solid_white_ready)
        self.worker.failed.connect(self._on_solid_white_failed)
        self.worker.finished.connect(self._on_worker_done)
        self.detect_btn.setText("Calling AI...")
        self.statusBar().showMessage("Requesting solid-white transform from Gemini...")
        self.worker.start()
        self._update_buttons()

    def _on_solid_white_ready(self, png_bytes: bytes) -> None:
        h, w = self.source_rgb.shape[:2]
        try:
            self.solid_white_rgb = ai_bytes_to_binary_mask(png_bytes, w, h)
        except Exception as e:
            QMessageBox.critical(self, "Decode failed", f"{type(e).__name__}: {e}")
            return
        self.statusBar().showMessage(
            "AI transform ready — running adaptive-threshold pipeline...",
        )
        QApplication.processEvents()
        self._run_classical_detect(detection_rgb=self.solid_white_rgb,
                                   color_source_rgb=self.source_rgb)

    def _on_solid_white_failed(self, msg: str) -> None:
        QMessageBox.critical(self, "Solid-white transform failed", msg)
        self.statusBar().showMessage("Solid-white transform failed.")

    def _on_worker_done(self) -> None:
        self.worker = None
        self.detect_btn.setText("Detect")
        self._update_buttons()

    def _run_classical_detect(
        self,
        detection_rgb: np.ndarray,
        color_source_rgb: np.ndarray | None,
    ) -> None:
        """Common terminal step: run detect_tiles + render + update UI."""
        self.statusBar().showMessage("Detecting tiles...")
        QApplication.processEvents()
        try:
            tiles = detect_tiles(
                detection_rgb,
                block_size=self.block_size_spin.value(),
                C=self.C_spin.value(),
                min_area=self.min_area_spin.value(),
                epsilon_ratio=self.eps_spin.value(),
                color_source_rgb=color_source_rgb,
            )
        except Exception as e:
            QMessageBox.critical(self, "Detect failed", f"{type(e).__name__}: {e}")
            self.statusBar().showMessage("Detect failed.")
            return

        self.tiles = tiles
        h, w = detection_rgb.shape[:2]
        preview = render_polygons(tiles, w, h)
        suffix = "  (solid-white transform)" if color_source_rgb is not None else ""
        self.result_pane.set_pil_image(
            preview, f"{len(tiles)} polygons  |  {w} × {h} px{suffix}",
        )
        self.tile_count_label.setText(f"Tiles: {len(tiles)}")
        self.statusBar().showMessage(f"Detected {len(tiles)} tiles{suffix}.")
        self._update_buttons()

    def save_csv(self) -> None:
        if not self.tiles:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        default_name = (
            self.source_path.stem if self.source_path else "mosaic"
        ) + f"_{len(self.tiles)}polys.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save polygons CSV", str(OUTPUT_DIR / default_name),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        out_path = Path(path)
        if out_path.suffix.lower() != ".csv":
            out_path = out_path.with_suffix(".csv")
        try:
            write_csv(self.tiles, out_path)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        self.statusBar().showMessage(f"Saved {len(self.tiles)} polygons → {out_path.name}")
        QMessageBox.information(
            self, "Saved",
            f"Saved {len(self.tiles)} polygons to:\n{out_path}",
        )


def main() -> int:
    app = QApplication(sys.argv)
    win = MosaicToCsv()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
