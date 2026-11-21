"""Color Fix — GUI for manipulating image colors.

  * Load Image ............ open any image (alpha is preserved).
  * Extract Palette ....... k-means the image into N dominant colors.
  * Save Palette .......... write the palette to a .cpal file (JSON).
  * Load Palette .......... read a .cpal file back.
  * Apply Palette ......... transfer the loaded image onto the palette:
        - Snap  : every pixel becomes its nearest palette color (flat).
        - Recolor (keep shading): the image's own color groups are moved
          onto the nearest palette colors, keeping texture / shading.
  * Save Result ........... save the recolored image as PNG.

Palette file (.cpal) format:
  {"version": 1, "source": "<image name>",
   "colors": [{"hex": "#rrggbb", "rgb": [r, g, b], "weight": 0.123}, ...]}

Usage:
  python IMAGE_TO_MOSAIC/scripts/color_fix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QProgressDialog, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent.parent       # IMAGE_TO_MOSAIC/
OUTPUT_DIR = ROOT / "output"
PALETTE_DIR = OUTPUT_DIR / "palettes"
INPUT_DIR = ROOT / "input"
PALETTE_EXT = ".cpal"
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"


# ── color math ──────────────────────────────────────────────────────────
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(N,3) uint8 RGB → (N,3) float32 Lab (OpenCV float scale: L 0..100)."""
    f = rgb.reshape(-1, 1, 3).astype(np.float32) / 255.0
    return cv2.cvtColor(f, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """(N,3) float32 Lab → (N,3) uint8 RGB (clipped)."""
    f = cv2.cvtColor(lab.reshape(-1, 1, 3).astype(np.float32),
                     cv2.COLOR_Lab2RGB).reshape(-1, 3)
    return np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)


def nearest_index(lab: np.ndarray, centers: np.ndarray,
                  chunk: int = 0) -> np.ndarray:
    """Index of the nearest center (Lab distance) for every row of `lab`."""
    if chunk <= 0:      # keep the distance matrix around ~20M floats
        chunk = max(1000, 20_000_000 // max(1, len(centers)))
    out = np.empty(len(lab), np.int32)
    c2 = (centers ** 2).sum(1)
    for s in range(0, len(lab), chunk):
        p = lab[s:s + chunk]
        d = c2[None, :] - 2.0 * p @ centers.T      # |p|² is constant per row
        out[s:s + chunk] = d.argmin(1)
    return out


def match_one_to_one(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Pair each src color with a DISTINCT dst color (greedy, closest
    pairs first) so every palette color gets used. Leftover src colors
    (when src > dst) fall back to their nearest dst color."""
    d = ((src[:, None, :] - dst[None, :, :]) ** 2).sum(2)
    out = np.full(len(src), -1, np.int32)
    used_s, used_d = set(), set()
    for flat in np.argsort(d, axis=None):
        s, t = divmod(int(flat), d.shape[1])
        if s in used_s or t in used_d:
            continue
        out[s] = t
        used_s.add(s); used_d.add(t)
        if len(used_s) == len(src) or len(used_d) == len(dst):
            break
    rest = out < 0
    if rest.any():
        out[rest] = d[rest].argmin(1)
    return out


def kmeans_lab(lab: np.ndarray, k: int, max_samples: int = 200_000):
    """k-means in Lab on a random subsample. Returns (centers, weights)
    where weights are the share of ALL pixels nearest each center."""
    k = max(1, min(k, len(lab)))
    rng = np.random.default_rng(0)
    if k > 64:          # many colors → smaller sample, one attempt (speed)
        max_samples = min(max_samples, 120_000)
    attempts = 3 if k <= 32 else 1
    sample = lab if len(lab) <= max_samples else \
        lab[rng.choice(len(lab), max_samples, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    cv2.setRNGSeed(0)
    _c, _lbl, centers = cv2.kmeans(np.ascontiguousarray(sample, np.float32),
                                   k, None, crit, attempts, cv2.KMEANS_PP_CENTERS)
    idx = nearest_index(lab, centers)
    weights = np.bincount(idx, minlength=k) / max(1, len(idx))
    return centers, weights


def exact_colors(rgb: np.ndarray):
    """All unique colors of (N,3) uint8 RGB with their pixel share,
    most common first. Returns (colors (M,3) uint8, weights (M,))."""
    packed = (rgb[:, 0].astype(np.int32) << 16) | (rgb[:, 1].astype(np.int32) << 8) | rgb[:, 2]
    vals, counts = np.unique(packed, return_counts=True)
    order = np.argsort(-counts, kind='stable')
    vals, counts = vals[order], counts[order]
    cols = np.stack([(vals >> 16) & 255, (vals >> 8) & 255, vals & 255], 1).astype(np.uint8)
    return cols, counts / max(1, len(rgb))


# ── image helpers ───────────────────────────────────────────────────────
def load_rgba(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))


def read_metadata(path: str) -> dict:
    """Collect every piece of metadata PIL exposes for the image:
    PNG text chunks (tEXt / zTXt / iTXt), EXIF, ICC profile, DPI and
    any other simple info entries (e.g. JPEG comment)."""
    meta: dict = {"text": {}, "exif": None, "icc": None, "dpi": None}
    with Image.open(path) as im:
        meta["text"] = {str(k): str(v) for k, v in getattr(im, "text", {}).items()}
        exif = im.info.get("exif")
        if not exif:
            try:
                ex = im.getexif()
                exif = ex.tobytes() if len(ex) else None
            except Exception:
                exif = None
        meta["exif"] = exif
        meta["icc"] = im.info.get("icc_profile")
        meta["dpi"] = im.info.get("dpi")
        # Other simple string entries (JPEG comment, software, …) that
        # aren't already text chunks — keep them as PNG text.
        skip = {"exif", "icc_profile", "dpi", "transparency", "gamma",
                "interlace", "progressive", "progression", "jfif",
                "jfif_version", "jfif_unit", "jfif_density", "adobe",
                "adobe_transform", "chromaticity", "srgb", "aspect"}
        for k, v in im.info.items():
            if k in skip or k in meta["text"]:
                continue
            if isinstance(v, bytes):
                try:
                    v = v.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            if isinstance(v, (str, int, float)):
                meta["text"][str(k)] = str(v)
    return meta


def save_png_with_metadata(arr: np.ndarray, path, meta: dict | None) -> None:
    """Save RGBA array as PNG, copying the source image's metadata."""
    from PIL.PngImagePlugin import PngInfo
    kw = {}
    if meta:
        if meta.get("text"):
            info = PngInfo()
            for k, v in meta["text"].items():
                info.add_itxt(k, v)
            kw["pnginfo"] = info
        if meta.get("exif"):
            kw["exif"] = meta["exif"]
        if meta.get("icc"):
            kw["icc_profile"] = meta["icc"]
        if meta.get("dpi"):
            kw["dpi"] = tuple(float(x) for x in meta["dpi"])
    Image.fromarray(arr, "RGBA").save(str(path), **kw)


def rgba_to_pixmap(arr: np.ndarray) -> QPixmap:
    h, w = arr.shape[:2]
    arr = np.ascontiguousarray(arr)
    qimg = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimg)


def white_pixels(rgba: np.ndarray) -> np.ndarray:
    """Flat boolean mask of pure-white (255,255,255) pixels — the tile
    background that mosaic_aranger turns transparent."""
    return (rgba[:, :, :3] == 255).all(2).reshape(-1)


def opaque_pixels(rgba: np.ndarray) -> np.ndarray:
    """Flat boolean mask of pixels that count for palette work."""
    return (rgba[:, :, 3] > 0).reshape(-1)


# ── widgets ─────────────────────────────────────────────────────────────
class ImageView(QLabel):
    """Label that keeps a pixmap scaled to fit, aspect preserved."""

    def __init__(self, title: str):
        super().__init__()
        self._pm: QPixmap | None = None
        self._title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#2b2b2b; color:#aaa; border:1px solid #444;")
        self.setText(title)

    def set_pixmap(self, pm: QPixmap | None) -> None:
        self._pm = pm
        if pm is None:
            self.setText(self._title)
            self.setPixmap(QPixmap())
        self._refit()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._refit()

    def _refit(self) -> None:
        if self._pm is not None and not self._pm.isNull():
            self.setPixmap(self._pm.scaled(self.size(), Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation))


class PaletteStrip(QWidget):
    """Grid of color swatches with hex + share labels."""

    COLS = 16

    def __init__(self):
        super().__init__()
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(4)

    def set_palette(self, colors: list[dict]) -> None:
        while self._grid.count():
            w = self._grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for i, c in enumerate(colors):
            r, g, b = c["rgb"]
            sw = QLabel()
            sw.setFixedSize(48, 28)
            sw.setStyleSheet(f"background: rgb({r},{g},{b}); border:1px solid #555;")
            sw.setToolTip(f"{c['hex']}  rgb({r}, {g}, {b})  "
                          f"{100 * c.get('weight', 0):.1f}%")
            txt = QLabel(f"{c['hex']}\n{100 * c.get('weight', 0):.1f}%")
            txt.setAlignment(Qt.AlignCenter)
            txt.setStyleSheet("font-size: 9px;")
            cell = QWidget()
            v = QVBoxLayout(cell)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(1)
            v.addWidget(sw, 0, Qt.AlignCenter)
            v.addWidget(txt)
            self._grid.addWidget(cell, i // self.COLS, i % self.COLS)


class ColorFixWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Color Fix")
        self.resize(1300, 850)

        self.image_path: Path | None = None
        self.image: np.ndarray | None = None      # RGBA uint8
        self.image_meta: dict | None = None       # source metadata (copied on save)
        self.result: np.ndarray | None = None     # RGBA uint8
        self.palette: list[dict] = []             # [{'hex','rgb','weight'}]
        self.palette_name = ""

        # Views
        self.src_view = ImageView("Load an image")
        self.dst_view = ImageView("Result")
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.src_view)
        split.addWidget(self.dst_view)

        # Controls
        b_load = QPushButton("Load Image…");       b_load.clicked.connect(self.on_load_image)
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 4096)
        self.k_spin.setValue(256)
        self.k_spin.setToolTip("Max colors to extract. If the image has this many "
                               "unique colors or fewer, ALL of them are extracted exactly.")
        b_extract = QPushButton("Extract Palette"); b_extract.clicked.connect(self.on_extract)
        b_savep = QPushButton("Save Palette…");    b_savep.clicked.connect(self.on_save_palette)
        b_loadp = QPushButton("Load Palette…");    b_loadp.clicked.connect(self.on_load_palette)
        self.mode = QComboBox()
        self.mode.addItems(["Snap to palette (flat)", "Recolor (keep shading)"])
        self.keep_white_chk = QCheckBox("Keep white background")
        self.keep_white_chk.setChecked(True)
        self.keep_white_chk.setToolTip(
            "Pure white (255,255,255) pixels are treated as tile background:\n"
            "left untouched when applying, and ignored when extracting.\n"
            "Needed for tiles used in mosaic_aranger (white = transparent).")
        b_apply = QPushButton("Apply Palette");     b_apply.clicked.connect(self.on_apply)
        b_saver = QPushButton("Save Result…");     b_saver.clicked.connect(self.on_save_result)
        b_folder = QPushButton("Apply to Folder…")
        b_folder.setToolTip("Apply the loaded palette (current mode) to every image "
                            "in a folder; results go to a new subfolder inside it.")
        b_folder.clicked.connect(self.on_apply_folder)

        bar = QHBoxLayout()
        bar.addWidget(b_load)
        bar.addWidget(self._sep())
        bar.addWidget(QLabel("Colors:"))
        bar.addWidget(self.k_spin)
        bar.addWidget(b_extract)
        bar.addWidget(b_savep)
        bar.addWidget(self._sep())
        bar.addWidget(b_loadp)
        bar.addWidget(self.mode)
        bar.addWidget(self.keep_white_chk)
        bar.addWidget(b_apply)
        bar.addWidget(b_saver)
        bar.addWidget(self._sep())
        bar.addWidget(b_folder)
        bar.addStretch(1)

        self.pal_title = QLabel("Palette: (none)")
        self.pal_strip = PaletteStrip()
        pal_scroll = QScrollArea()
        pal_scroll.setWidgetResizable(True)
        pal_scroll.setWidget(self.pal_strip)
        pal_scroll.setFixedHeight(220)

        central = QWidget()
        v = QVBoxLayout(central)
        v.addLayout(bar)
        v.addWidget(split, 1)
        v.addWidget(self.pal_title)
        v.addWidget(pal_scroll)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Load an image to start.")

    @staticmethod
    def _sep() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setFrameShadow(QFrame.Sunken)
        return f

    def _busy(self, on: bool) -> None:
        if on:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _set_palette(self, colors: list[dict], name: str) -> None:
        self.palette = colors
        self.palette_name = name
        self.pal_title.setText(f"Palette: {name}  ({len(colors)} colors)")
        self.pal_strip.set_palette(colors)

    # ── actions ────────────────────────────────────────────────────────
    def on_load_image(self) -> None:
        start = str(INPUT_DIR if INPUT_DIR.exists() else ROOT)
        path, _ = QFileDialog.getOpenFileName(self, "Load Image", start, IMAGE_FILTER)
        if not path:
            return
        try:
            self.image = load_rgba(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Image", f"Could not open image:\n{e}")
            return
        self.image_path = Path(path)
        try:
            self.image_meta = read_metadata(path)
        except Exception:
            self.image_meta = None
        self.result = None
        self.src_view.set_pixmap(rgba_to_pixmap(self.image))
        self.dst_view.set_pixmap(None)
        h, w = self.image.shape[:2]
        self.statusBar().showMessage(f"Loaded {self.image_path.name}  ({w}×{h})")

    def on_extract(self) -> None:
        if self.image is None:
            QMessageBox.information(self, "Extract Palette", "Load an image first.")
            return
        rgb = self.image[:, :, :3].reshape(-1, 3)
        m = opaque_pixels(self.image)
        if self.keep_white_chk.isChecked():
            m &= ~white_pixels(self.image)
        rgb = rgb[m]
        if len(rgb) == 0:
            QMessageBox.warning(self, "Extract Palette", "Image has no opaque pixels.")
            return
        k = self.k_spin.value()
        self._busy(True)
        try:
            cols, weights = exact_colors(rgb)
            n_unique = len(cols)
            exact = n_unique <= k
            if not exact:
                # More unique colors than requested → k-means down to k.
                centers, weights = kmeans_lab(rgb_to_lab(rgb), k)
                cols = lab_to_rgb(centers)
        finally:
            self._busy(False)
        order = np.argsort(-weights, kind='stable')
        colors = []
        for i in order:
            if weights[i] <= 0:
                continue
            r, g, b = (int(x) for x in cols[i])
            colors.append({"hex": f"#{r:02x}{g:02x}{b:02x}",
                           "rgb": [r, g, b], "weight": round(float(weights[i]), 5)})
        self._set_palette(colors, self.image_path.stem if self.image_path else "image")
        if exact:
            msg = f"Extracted ALL {len(colors)} colors (exact)."
        else:
            msg = (f"Image has {n_unique:,} unique colors → reduced to "
                   f"{len(colors)}. Raise 'Colors' to keep more.")
        self.statusBar().showMessage(msg)

    def on_save_palette(self) -> None:
        if not self.palette:
            QMessageBox.information(self, "Save Palette", "Extract or load a palette first.")
            return
        PALETTE_DIR.mkdir(parents=True, exist_ok=True)
        default = PALETTE_DIR / f"{self.palette_name or 'palette'}{PALETTE_EXT}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Palette", str(default), f"Palette (*{PALETTE_EXT})")
        if not path:
            return
        if not path.lower().endswith(PALETTE_EXT):
            path += PALETTE_EXT
        payload = {"version": 1, "source": self.palette_name, "colors": self.palette}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.statusBar().showMessage(f"Saved palette → {path}")

    def on_load_palette(self) -> None:
        start = str(PALETTE_DIR if PALETTE_DIR.exists() else OUTPUT_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Palette", start, f"Palette (*{PALETTE_EXT} *.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            colors = []
            for c in data["colors"]:
                r, g, b = (int(x) for x in c["rgb"])
                colors.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "rgb": [r, g, b],
                               "weight": float(c.get("weight", 0))})
            if not colors:
                raise ValueError("palette has no colors")
        except Exception as e:
            QMessageBox.critical(self, "Load Palette", f"Invalid palette file:\n{e}")
            return
        self._set_palette(colors, Path(path).stem)
        self.statusBar().showMessage(f"Loaded palette {Path(path).name}")

    def on_apply(self) -> None:
        if self.image is None or not self.palette:
            QMessageBox.information(self, "Apply Palette",
                                    "Load an image and a palette first.")
            return
        self._busy(True)
        try:
            self.result = self.apply_palette(self.image, self.palette,
                                             self.mode.currentIndex() == 1,
                                             self.keep_white_chk.isChecked())
        finally:
            self._busy(False)
        self.dst_view.set_pixmap(rgba_to_pixmap(self.result))
        self.statusBar().showMessage(
            f"Applied palette '{self.palette_name}' ({self.mode.currentText()}).")

    @staticmethod
    def apply_palette(rgba: np.ndarray, palette: list[dict],
                      keep_shading: bool, keep_white: bool = True) -> np.ndarray:
        h, w = rgba.shape[:2]
        lab = rgb_to_lab(rgba[:, :, :3].reshape(-1, 3))
        pal_lab = rgb_to_lab(np.array([c["rgb"] for c in palette], np.uint8))
        bg = white_pixels(rgba) if keep_white else np.zeros(h * w, bool)
        if keep_shading:
            # Group the image into as many color clusters as the palette
            # has, move each cluster's center onto its nearest palette
            # color, and shift every pixel by its cluster's offset.
            mask = opaque_pixels(rgba) & ~bg
            centers, _w = kmeans_lab(lab[mask] if mask.any() else lab, len(palette))
            px_cluster = nearest_index(lab, centers)
            target = pal_lab[match_one_to_one(centers, pal_lab)]
            out_lab = lab + (target - centers)[px_cluster]
            out_rgb = lab_to_rgb(out_lab)
        else:
            out_rgb = np.array([c["rgb"] for c in palette], np.uint8)[
                nearest_index(lab, pal_lab)]
        if keep_white:
            # Pure white is the tile's paper background (mosaic_aranger
            # makes it transparent): keep it exactly white, and never let
            # a real tile pixel BECOME pure white (it would vanish there).
            became_white = (out_rgb == 255).all(1) & ~bg
            out_rgb[became_white] = 254
            out_rgb[bg] = 255
        out = np.empty_like(rgba)
        out[:, :, :3] = out_rgb.reshape(h, w, 3)
        out[:, :, 3] = rgba[:, :, 3]
        return out

    def on_apply_folder(self) -> None:
        """Apply the loaded palette to every image in a chosen folder
        (not recursive). Results are saved as PNG in a new subfolder
        '<folder>/<palette>_fixed' (numbered if it already exists)."""
        if not self.palette:
            QMessageBox.information(self, "Apply to Folder", "Load or extract a palette first.")
            return
        start = str(self.image_path.parent if self.image_path else
                    (INPUT_DIR if INPUT_DIR.exists() else ROOT))
        folder = QFileDialog.getExistingDirectory(self, "Choose Image Folder", start)
        if not folder:
            return
        folder = Path(folder)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        files = sorted(f for f in folder.iterdir()
                       if f.is_file() and f.suffix.lower() in exts)
        if not files:
            QMessageBox.information(self, "Apply to Folder", "No images found in that folder.")
            return
        base = f"{self.palette_name or 'palette'}_fixed"
        out_dir = folder / base
        n = 2
        while out_dir.exists():
            out_dir = folder / f"{base}_{n}"
            n += 1
        keep = self.mode.currentIndex() == 1
        if QMessageBox.question(
                self, "Apply to Folder",
                f"Apply palette '{self.palette_name}' ({self.mode.currentText()}) "
                f"to {len(files)} images?\n\nOutput folder:\n{out_dir}",
        ) != QMessageBox.Yes:
            return
        out_dir.mkdir(parents=True)
        prog = QProgressDialog("Applying palette…", "Cancel", 0, len(files), self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        done, failed = 0, []
        for i, f in enumerate(files):
            if prog.wasCanceled():
                break
            prog.setLabelText(f"{i + 1}/{len(files)}: {f.name}")
            prog.setValue(i)
            QApplication.processEvents()
            try:
                res = self.apply_palette(load_rgba(str(f)), self.palette, keep,
                                         self.keep_white_chk.isChecked())
                try:
                    meta = read_metadata(str(f))
                except Exception:
                    meta = None
                save_png_with_metadata(res, out_dir / f"{f.stem}.png", meta)
                done += 1
            except Exception as e:
                failed.append(f"{f.name}: {e}")
        prog.setValue(len(files))
        msg = f"Saved {done} of {len(files)} images to:\n{out_dir}"
        if failed:
            msg += "\n\nFailed:\n" + "\n".join(failed[:10])
        if done + len(failed) < len(files):
            msg += "\n\n(Cancelled before finishing.)"
        self.statusBar().showMessage(f"Folder done: {done}/{len(files)} → {out_dir}")
        QMessageBox.information(self, "Apply to Folder", msg)

    def on_save_result(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "Save Result", "Apply a palette first.")
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = self.image_path.stem if self.image_path else "image"
        default = OUTPUT_DIR / f"{stem}_{self.palette_name or 'palette'}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save Result", str(default),
                                              "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        save_png_with_metadata(self.result, path, self.image_meta)
        self.statusBar().showMessage(f"Saved result → {path}")


def main() -> None:
    app = QApplication(sys.argv)
    win = ColorFixWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
