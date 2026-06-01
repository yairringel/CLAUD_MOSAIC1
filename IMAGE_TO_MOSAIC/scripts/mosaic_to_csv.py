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
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent.parent       # IMAGE_TO_MOSAIC/
OUTPUT_DIR = ROOT / "output"
INPUT_DIR = ROOT / "input"


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
                 min_area: int, epsilon_ratio: float):
    """Returns list of (polygon_points (N x 2 float), mean_rgb_float_0_1)."""
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

        mean_rgb = (rgb[labels == lid].mean(axis=0) / 255.0).tolist()
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
        self.tiles: list = []

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # Toolbar
        bar = QHBoxLayout()
        self.load_btn   = QPushButton("Load Image...")
        self.detect_btn = QPushButton("Detect")
        self.save_btn   = QPushButton("Save CSV...")
        for b in (self.load_btn, self.detect_btn, self.save_btn):
            bar.addWidget(b)
        bar.addStretch(1)
        self.tile_count_label = QLabel("Tiles: —")
        bar.addWidget(self.tile_count_label)
        root_layout.addLayout(bar)

        # Parameters — defaults from csv_experiments attempt 07.
        params = QHBoxLayout()
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
        self.detect_btn.clicked.connect(self.detect)
        self.save_btn.clicked.connect(self.save_csv)
        self._update_buttons()

    def _update_buttons(self) -> None:
        self.detect_btn.setEnabled(self.source_rgb is not None)
        self.save_btn.setEnabled(bool(self.tiles))

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
        self.result_pane.clear()
        self.tile_count_label.setText("Tiles: —")
        self.statusBar().showMessage(f"Loaded {Path(path).name}")
        self._update_buttons()

    def detect(self) -> None:
        if self.source_rgb is None:
            return
        self.statusBar().showMessage("Detecting tiles...")
        QApplication.processEvents()
        try:
            tiles = detect_tiles(
                self.source_rgb,
                block_size=self.block_size_spin.value(),
                C=self.C_spin.value(),
                min_area=self.min_area_spin.value(),
                epsilon_ratio=self.eps_spin.value(),
            )
        except Exception as e:
            QMessageBox.critical(self, "Detect failed", f"{type(e).__name__}: {e}")
            self.statusBar().showMessage("Detect failed.")
            return

        self.tiles = tiles
        h, w = self.source_rgb.shape[:2]
        preview = render_polygons(tiles, w, h)
        self.result_pane.set_pil_image(preview, f"{len(tiles)} polygons  |  {w} × {h} px")
        self.tile_count_label.setText(f"Tiles: {len(tiles)}")
        self.statusBar().showMessage(f"Detected {len(tiles)} tiles.")
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
