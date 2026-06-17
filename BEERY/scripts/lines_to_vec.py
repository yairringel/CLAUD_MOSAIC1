"""Lines → Vector — dedicated GUI for editing the orange line layer of a
Voroni project file (.voroni produced by voroni.py).

Features:
  - Load a .voroni project file → shows the dim base + orange lines + frame
  - DRAW new orange lines with a chosen brush width
  - ERASE orange lines with a fixed-radius circular eraser
  - Change the uniform LINE WIDTH (skeletonise + re-inflate)
  - Native cursor-centered wheel zoom via QGraphicsView (anchored under
    cursor); scrollbars appear automatically when the image is zoomed in
    further than the viewport, so pan-by-scrollbar always works
  - Save back to .voroni or export the composite as PNG

Usage:
  python BEERY/scripts/lines_to_vec.py
"""
from __future__ import annotations

import csv
import io
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as _PILImage
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)
from skimage.morphology import skeletonize


SCHEMA_VERSION_SUPPORTED = 1
PROJECT_EXT = ".voroni"
ERASE_BRUSH_RADIUS = 10

# Polygon detection (Save Polygons) parameters.
# Regions below this pixel area are discarded as artifacts (1-2-px
# skeleton spurs / sliver gaps). 10 px² is small enough that legitimate
# tiles never get filtered — only true noise.
MIN_POLYGON_AREA_PX = 10

# Default output directory (BEERY/output) per project conventions.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


# ---------------------------------------------------------------------------
# Editable QGraphicsView — wheel-zoom anchored under cursor, mouse events
# routed to the parent editor's draw/erase handler.
# ---------------------------------------------------------------------------

class EditableView(QGraphicsView):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        # Scrollbars appear automatically once the scaled scene exceeds the
        # viewport — that's the user's pan mechanism when zoomed in.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.NoDrag)
        self._dragging = False
        self._last_scene_pt: tuple[int, int] | None = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.editor.is_editing():
            pt = self._scene_pt(event)
            if self.editor.apply_brush(pt, pt):
                self._dragging = True
                self._last_scene_pt = pt
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            pt = self._scene_pt(event)
            if self._last_scene_pt is not None:
                self.editor.apply_brush(self._last_scene_pt, pt)
            self._last_scene_pt = pt
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            self._last_scene_pt = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _scene_pt(self, event) -> tuple[int, int]:
        sp = self.mapToScene(event.pos())
        return (int(round(sp.x())), int(round(sp.y())))


# ---------------------------------------------------------------------------
# Main editor window.
# ---------------------------------------------------------------------------

class LinesToVecEditor(QMainWindow):
    DARKEN_FACTOR_FALLBACK = 0.35

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lines → Vector (BEERY)")
        self.resize(1400, 900)

        # Project state.
        self.base_rgb: np.ndarray | None = None         # (h, w, 3) uint8, bright base
        self.line_mask: np.ndarray | None = None         # (h, w) uint8, 0/255 line pixels
        self.frame_width_px: int = 3
        self.darken_factor: float = self.DARKEN_FACTOR_FALLBACK
        self.source_path: str | None = None
        self.current_project_path: Path | None = None
        # Display item.
        self.pixmap_item: QGraphicsPixmapItem | None = None
        # Polygon-preview state — when toggled on, the view renders the
        # detected polygon mosaic instead of the line-edit composite.
        self._polygons_view: bool = False
        # Cached detection result; cleared whenever the line mask changes.
        # Each entry is (pts (N x 2 np.int32-castable), mean_rgb tuple in 0-1).
        self._last_tiles: list | None = None

        # ----- UI build -----
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        bar = QHBoxLayout()
        self.load_btn = QPushButton("Load .voroni")
        self.load_btn.clicked.connect(self.load_project)
        bar.addWidget(self.load_btn)

        self.draw_btn = QPushButton("Draw")
        self.draw_btn.setCheckable(True)
        self.draw_btn.setToolTip(
            "Click + drag on the image to add orange line pixels at the "
            "chosen brush width. The dim background is never modified."
        )
        self.draw_btn.clicked.connect(self._on_draw_toggled)
        bar.addWidget(self.draw_btn)

        self.erase_btn = QPushButton("Erase")
        self.erase_btn.setCheckable(True)
        self.erase_btn.setToolTip(
            f"Click + drag on the image to remove orange line pixels with a "
            f"circular eraser of fixed radius {ERASE_BRUSH_RADIUS} px. The "
            f"background is never modified."
        )
        self.erase_btn.clicked.connect(self._on_erase_toggled)
        bar.addWidget(self.erase_btn)

        bar.addWidget(QLabel("Brush:"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 30)
        self.brush_spin.setValue(3)
        self.brush_spin.setSuffix(" px")
        self.brush_spin.setToolTip(
            f"DRAW stroke thickness, in pixels. The eraser uses a fixed "
            f"{ERASE_BRUSH_RADIUS}-px-radius circular brush and ignores "
            f"this value."
        )
        bar.addWidget(self.brush_spin)

        bar.addWidget(QLabel("Line:"))
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 20)
        self.line_width_spin.setValue(2)
        self.line_width_spin.setSuffix(" px")
        self.line_width_spin.setToolTip(
            "Re-skeletonises the current line mask and re-renders it at "
            "this uniform pixel width. Affects ALL existing lines + any "
            "user-drawn additions."
        )
        self.line_width_spin.valueChanged.connect(self._on_line_width_changed)
        bar.addWidget(self.line_width_spin)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setToolTip(
            "Reset the view zoom so the whole image fits in the viewport."
        )
        self.fit_btn.clicked.connect(self._fit_to_view)
        bar.addWidget(self.fit_btn)

        self.save_btn = QPushButton("Save .voroni")
        self.save_btn.setToolTip(
            "Save the current state back as a .voroni project file."
        )
        self.save_btn.clicked.connect(self.save_project)
        bar.addWidget(self.save_btn)

        self.save_png_btn = QPushButton("Save PNG")
        self.save_png_btn.setToolTip(
            "Export the orange lines + 3-px frame on a SOLID-BLACK "
            "background as a PNG, at the source image's resolution. The "
            "underlying base picture is NOT included."
        )
        self.save_png_btn.clicked.connect(self.save_png)
        bar.addWidget(self.save_png_btn)

        self.show_polygons_btn = QPushButton("Show Polygons")
        self.show_polygons_btn.setCheckable(True)
        self.show_polygons_btn.setToolTip(
            "TOGGLE the view between LINES EDIT (default — dim background + "
            "orange lines + frame) and POLYGON PREVIEW (the polygons that "
            "Save Polygons would emit, drawn as a mosaic with each tile's "
            "mean colour + 1-px black outline on a solid-black background). "
            "Re-runs detection each time you turn it on. Draw / Erase are "
            "disabled while the polygon preview is active."
        )
        self.show_polygons_btn.toggled.connect(self._on_show_polygons_toggled)
        bar.addWidget(self.show_polygons_btn)

        self.save_polygons_btn = QPushButton("Save Polygons")
        self.save_polygons_btn.setToolTip(
            "Treat the current orange lines + frame as boundaries on a "
            "solid-black background. Detect every enclosed region between "
            "the lines as a polygon, sample its mean colour from the bright "
            "base image, and save as a CSV in scripts/image_strech.py's "
            "schema (polygon_id, coordinates [JSON array of [x,y] pairs], "
            "color_r/g/b/a, frame_r/g/b/a, group_id). A preview PNG of the "
            "polygon mosaic is saved alongside."
        )
        self.save_polygons_btn.clicked.connect(self.save_polygons)
        bar.addWidget(self.save_polygons_btn)

        bar.addStretch(1)
        layout.addLayout(bar)

        # Graphics view.
        self.scene = QGraphicsScene(self)
        self.view = EditableView(self)
        self.view.setScene(self.scene)
        layout.addWidget(self.view, 1)

        self.statusBar().showMessage("Load a .voroni project file.")
        self._update_buttons()

    # ----- mode state -----------------------------------------------------

    def is_editing(self) -> bool:
        return self.draw_btn.isChecked() or self.erase_btn.isChecked()

    def _on_draw_toggled(self, checked: bool):
        if checked:
            self.erase_btn.setChecked(False)
            self.view.viewport().setCursor(Qt.CrossCursor)
        else:
            self.view.viewport().setCursor(Qt.ArrowCursor)

    def _on_erase_toggled(self, checked: bool):
        if checked:
            self.draw_btn.setChecked(False)
            self.view.viewport().setCursor(Qt.CrossCursor)
        else:
            self.view.viewport().setCursor(Qt.ArrowCursor)

    def _update_buttons(self):
        loaded = self.line_mask is not None
        for btn in (self.draw_btn, self.erase_btn, self.save_btn,
                    self.save_png_btn, self.save_polygons_btn,
                    self.show_polygons_btn, self.fit_btn):
            btn.setEnabled(loaded)
        self.brush_spin.setEnabled(loaded)
        self.line_width_spin.setEnabled(loaded)
        # In polygon-preview mode Draw / Erase make no sense.
        if self._polygons_view:
            self.draw_btn.setEnabled(False)
            self.erase_btn.setEnabled(False)
        if not loaded:
            self.draw_btn.setChecked(False)
            self.erase_btn.setChecked(False)
            self.show_polygons_btn.setChecked(False)
            self.view.viewport().setCursor(Qt.ArrowCursor)

    # ----- editing operations ---------------------------------------------

    def apply_brush(self, pt1: tuple[int, int], pt2: tuple[int, int]) -> bool:
        """Draw or erase a segment from pt1 → pt2 on self.line_mask.
        Coordinates are in image-pixel space (scene coords). cv2.line handles
        any out-of-bounds clipping. Returns True if applied."""
        if self.line_mask is None or not self.is_editing():
            return False
        if self.draw_btn.isChecked():
            width = max(1, int(self.brush_spin.value()))
            cv2.line(
                self.line_mask, pt1, pt2,
                color=255, thickness=width, lineType=cv2.LINE_8,
            )
        elif self.erase_btn.isChecked():
            r = ERASE_BRUSH_RADIUS
            # Circles at endpoints give round caps; thick line in between
            # fills the swept area on fast drags.
            cv2.circle(self.line_mask, pt1, r, color=0,
                       thickness=-1, lineType=cv2.LINE_8)
            cv2.circle(self.line_mask, pt2, r, color=0,
                       thickness=-1, lineType=cv2.LINE_8)
            if pt1 != pt2:
                cv2.line(self.line_mask, pt1, pt2,
                         color=0, thickness=2 * r, lineType=cv2.LINE_8)
        # Any mask edit invalidates the cached polygon detection.
        self._last_tiles = None
        self._refresh_display()
        return True

    def _on_line_width_changed(self, value: int):
        if self.line_mask is None:
            return
        skel_bool = skeletonize(self.line_mask > 0)
        skel = skel_bool.astype(np.uint8) * 255
        if value <= 1:
            self.line_mask = skel
        else:
            inverted = (skel == 0).astype(np.uint8) * 255
            dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)
            self.line_mask = (
                dist <= (value - 1) / 2.0
            ).astype(np.uint8) * 255
        self._last_tiles = None
        self._refresh_display()

    # ----- rendering ------------------------------------------------------

    def _build_composite(self) -> np.ndarray | None:
        """Build the displayed composite: dim base + orange lines + frame."""
        if self.base_rgb is None or self.line_mask is None:
            return None
        comp = (
            self.base_rgb.astype(np.float32) * self.darken_factor
        ).clip(0, 255).astype(np.uint8)
        comp[self.line_mask > 0] = (255, 102, 0)
        fw = self.frame_width_px
        h, w = comp.shape[:2]
        fw_h = min(fw, h)
        fw_w = min(fw, w)
        if fw_h > 0:
            comp[:fw_h, :] = (255, 102, 0)
            comp[-fw_h:, :] = (255, 102, 0)
        if fw_w > 0:
            comp[:, :fw_w] = (255, 102, 0)
            comp[:, -fw_w:] = (255, 102, 0)
        return comp

    def _render_polygon_mosaic(self, tiles: list) -> np.ndarray | None:
        """Render detected polygons as a mosaic on a solid-black canvas at
        source resolution: each polygon filled with its mean colour + a
        1-px black outline. This is the SAME canvas that gets written to
        the Save Polygons preview PNG, so 'Show Polygons' is a true visual
        preview of what will be saved."""
        if self.line_mask is None:
            return None
        h, w = self.line_mask.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for pts, mean_rgb in tiles:
            colour = (
                int(round(mean_rgb[0] * 255)),
                int(round(mean_rgb[1] * 255)),
                int(round(mean_rgb[2] * 255)),
            )
            poly_int = pts.astype(np.int32)
            cv2.fillPoly(canvas, [poly_int], color=colour)
            cv2.polylines(
                canvas, [poly_int], isClosed=True,
                color=(0, 0, 0), thickness=1,
            )
        return canvas

    def _refresh_display(self):
        if self._polygons_view:
            tiles = self._last_tiles if self._last_tiles is not None else []
            comp = self._render_polygon_mosaic(tiles)
        else:
            comp = self._build_composite()
        if comp is None:
            return
        comp = np.ascontiguousarray(comp)
        h, w = comp.shape[:2]
        qimg = QImage(
            comp.data, w, h, comp.strides[0], QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimg)
        if self.pixmap_item is None:
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(0, 0, w, h)
        else:
            self.pixmap_item.setPixmap(pixmap)
        if self._polygons_view:
            n_tiles = len(self._last_tiles) if self._last_tiles is not None else 0
            self.statusBar().showMessage(
                f"{w} × {h} px  |  POLYGON PREVIEW  |  {n_tiles} polygons"
            )
        else:
            n_orange = (
                int((self.line_mask > 0).sum())
                if self.line_mask is not None else 0
            )
            self.statusBar().showMessage(
                f"{w} × {h} px  |  line width: "
                f"{int(self.line_width_spin.value())} px  "
                f"|  brush: {int(self.brush_spin.value())} px  |  "
                f"orange pixels: {n_orange}"
            )

    def _on_show_polygons_toggled(self, checked: bool):
        if checked and self.line_mask is None:
            self.show_polygons_btn.setChecked(False)
            return
        self._polygons_view = checked
        if checked:
            # Force fresh detection each time we enter polygon view so
            # any line edits done since the last detection are reflected.
            tiles, diagnostics = self._detect_polygons()
            self._last_tiles = tiles
            self.draw_btn.setChecked(False)
            self.erase_btn.setChecked(False)
            self._update_buttons()
            self._refresh_display()
            # Diagnostics in the status bar — surfaces zero-polygon cases.
            self.statusBar().showMessage(
                f"POLYGON PREVIEW  |  components: {diagnostics['n_components']}  "
                f"|  kept: {diagnostics['kept']}  "
                f"|  dropped (area < {MIN_POLYGON_AREA_PX} px²): "
                f"{diagnostics['dropped_area']}  "
                f"|  dropped (< 3 vertices): "
                f"{diagnostics['dropped_degenerate']}"
            )
        else:
            self._update_buttons()
            self._refresh_display()

    def _fit_to_view(self):
        if self.pixmap_item is None:
            return
        self.view.resetTransform()
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    # ----- I/O ------------------------------------------------------------

    def load_project(self):
        start_dir = ""
        if self.current_project_path is not None:
            start_dir = str(self.current_project_path.parent)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Voroni project", start_dir,
            f"Voroni project (*{PROJECT_EXT});;All files (*.*)",
        )
        if not path:
            return
        p = Path(path)
        try:
            with open(p, "rb") as f:
                project = pickle.load(f)
        except Exception as e:
            QMessageBox.critical(
                self, "Load failed", f"{type(e).__name__}: {e}",
            )
            return
        if not isinstance(project, dict):
            QMessageBox.critical(
                self, "Invalid project",
                "File does not contain a Voroni project dictionary.",
            )
            return
        sv = project.get("schema_version")
        if sv != SCHEMA_VERSION_SUPPORTED:
            QMessageBox.warning(
                self, "Schema version mismatch",
                f"This GUI supports schema {SCHEMA_VERSION_SUPPORTED}; the "
                f"file declares schema {sv}. Attempting to load anyway — "
                f"some fields may be missing.",
            )

        base_png = project.get("base_image_png")
        line_mask = project.get("line_mask")
        if base_png is None or line_mask is None:
            QMessageBox.critical(
                self, "Invalid project",
                "Missing 'base_image_png' or 'line_mask' in the file.",
            )
            return
        try:
            base_pil = _PILImage.open(io.BytesIO(base_png)).convert("RGB")
        except Exception as e:
            QMessageBox.critical(
                self, "Bad base image", f"{type(e).__name__}: {e}",
            )
            return
        line_mask_arr = np.array(line_mask, dtype=np.uint8).copy()
        # Sanity-check dimensions match.
        if (base_pil.height, base_pil.width) != line_mask_arr.shape[:2]:
            QMessageBox.warning(
                self, "Dimension mismatch",
                f"Base image is {base_pil.width}×{base_pil.height} but "
                f"line mask is {line_mask_arr.shape[1]}×{line_mask_arr.shape[0]}. "
                f"Loading the line mask resized to the base.",
            )
            line_mask_arr = cv2.resize(
                line_mask_arr,
                (base_pil.width, base_pil.height),
                interpolation=cv2.INTER_NEAREST,
            )

        self.base_rgb = np.array(base_pil, dtype=np.uint8)
        self.line_mask = line_mask_arr
        self.frame_width_px = int(project.get("frame_width_px", 3))
        self.darken_factor = float(
            project.get("darken_factor", self.DARKEN_FACTOR_FALLBACK),
        )
        self.source_path = project.get("source_path")
        line_width = int(project.get("line_width_px", 2))
        self.line_width_spin.blockSignals(True)
        self.line_width_spin.setValue(line_width)
        self.line_width_spin.blockSignals(False)
        self.current_project_path = p

        # New project → drop any cached detection from the previous file
        # and exit polygon-preview mode (a stale mosaic would be confusing).
        self._last_tiles = None
        self._polygons_view = False
        self.show_polygons_btn.blockSignals(True)
        self.show_polygons_btn.setChecked(False)
        self.show_polygons_btn.blockSignals(False)
        # Reset the previous pixmap so the new image lands at scene (0, 0).
        if self.pixmap_item is not None:
            self.scene.removeItem(self.pixmap_item)
            self.pixmap_item = None
        self._refresh_display()
        self._fit_to_view()
        self.setWindowTitle(f"Lines → Vector (BEERY) — {p.name}")
        self.statusBar().showMessage(f"Loaded {p.name}")
        self._update_buttons()

    def save_project(self):
        if self.line_mask is None or self.base_rgb is None:
            return
        default_path = (
            self.current_project_path if self.current_project_path is not None
            else Path("untitled.voroni")
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Voroni project", str(default_path),
            f"Voroni project (*{PROJECT_EXT});;All files (*.*)",
        )
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != PROJECT_EXT:
            p = p.with_suffix(PROJECT_EXT)
        base_pil = _PILImage.fromarray(self.base_rgb)
        buf = io.BytesIO()
        base_pil.save(buf, format="PNG", optimize=True)
        project = {
            "schema_version": SCHEMA_VERSION_SUPPORTED,
            "source_path": self.source_path,
            "source_size_px": (self.base_rgb.shape[1], self.base_rgb.shape[0]),
            "base_image_png": buf.getvalue(),
            "line_mask": self.line_mask.copy(),
            "frame_width_px": self.frame_width_px,
            "stretch_pct": 100,
            "line_width_px": int(self.line_width_spin.value()),
            "darken_factor": self.darken_factor,
        }
        try:
            with open(p, "wb") as f:
                pickle.dump(project, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            QMessageBox.critical(
                self, "Save failed", f"{type(e).__name__}: {e}",
            )
            return
        size_mb = p.stat().st_size / (1024 * 1024)
        self.statusBar().showMessage(
            f"Saved Voroni project ({size_mb:.1f} MB) → {p.name}",
        )
        self.current_project_path = p
        self.setWindowTitle(f"Lines → Vector (BEERY) — {p.name}")

    def save_png(self):
        """Save the orange lines + 3-px frame on a SOLID-BLACK background
        as a PNG at the source image's resolution. The base picture is
        NOT included — this is the layer the downstream polygon detector
        needs as input."""
        if self.line_mask is None:
            return
        h, w = self.line_mask.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[self.line_mask > 0] = (255, 102, 0)
        fw = self.frame_width_px
        fw_h = min(fw, h)
        fw_w = min(fw, w)
        if fw_h > 0:
            canvas[:fw_h, :] = (255, 102, 0)
            canvas[-fw_h:, :] = (255, 102, 0)
        if fw_w > 0:
            canvas[:, :fw_w] = (255, 102, 0)
            canvas[:, -fw_w:] = (255, 102, 0)

        default = "lines_on_black.png"
        if self.current_project_path is not None:
            default = (
                self.current_project_path.with_suffix(".png").stem
                + "_lines.png"
            )
        if self.current_project_path is not None:
            start_dir = str(self.current_project_path.parent / default)
        else:
            start_dir = default
        path, _ = QFileDialog.getSaveFileName(
            self, "Save lines (on solid black) PNG", start_dir,
            "PNG image (*.png);;All files (*.*)",
        )
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != ".png":
            p = p.with_suffix(".png")
        try:
            _PILImage.fromarray(canvas).save(p, "PNG")
        except Exception as e:
            QMessageBox.critical(
                self, "Save failed", f"{type(e).__name__}: {e}",
            )
            return
        self.statusBar().showMessage(
            f"Saved lines on black ({w} × {h} px) → {p.name}",
        )

    # ----- polygon extraction --------------------------------------------

    def _detect_polygons(self) -> tuple[list, dict]:
        """Run the boundary→interior→contour→simplify pipeline. Returns
        (tiles, diagnostics). Tiles is a list of (pts, mean_rgb) tuples.
        Diagnostics is a dict with counts, useful for showing the user
        WHY a detection returned zero polygons:
            n_components: total cv2.connectedComponents labels (incl. background)
            interior_regions: number of interior labels (n_components - 1)
            kept: polygons that passed all filters
            dropped_area: regions with area < MIN_POLYGON_AREA_PX
            dropped_degenerate: regions whose simplified contour had < 3 vertices
        """
        if self.line_mask is None or self.base_rgb is None:
            return [], {
                "n_components": 0, "interior_regions": 0,
                "kept": 0, "dropped_area": 0, "dropped_degenerate": 0,
            }

        h, w = self.line_mask.shape[:2]

        # Boundary mask: orange lines + the 3-px frame at fixed positions
        # (same positions the live composite uses).
        boundary = (self.line_mask > 0).astype(np.uint8) * 255
        fw = self.frame_width_px
        fw_h = min(fw, h)
        fw_w = min(fw, w)
        if fw_h > 0:
            boundary[:fw_h, :] = 255
            boundary[-fw_h:, :] = 255
        if fw_w > 0:
            boundary[:, :fw_w] = 255
            boundary[:, -fw_w:] = 255

        interior = (boundary == 0).astype(np.uint8) * 255
        n_labels, labels = cv2.connectedComponents(
            interior, connectivity=8,
        )

        tiles: list = []
        dropped_area = 0
        dropped_degenerate = 0
        for lid in range(1, n_labels):
            region_mask_bool = (labels == lid)
            area = int(region_mask_bool.sum())
            if area < MIN_POLYGON_AREA_PX:
                dropped_area += 1
                continue
            region_uint8 = region_mask_bool.astype(np.uint8) * 255
            # CHAIN_APPROX_SIMPLE collapses runs of collinear pixels along
            # axis-aligned / diagonal segments but PRESERVES every curve
            # point on the pixel boundary. We deliberately do NOT run
            # cv2.approxPolyDP afterwards: Douglas-Peucker simplification
            # draws straight lines between picked vertices, which on
            # concave regions can cut across an empty corner and project
            # outside the actual region — overlapping the neighbour. By
            # using only CHAIN_APPROX_SIMPLE, every polygon edge is a
            # subset of the region's true pixel outline, so adjacent
            # polygons share boundaries but cannot overlap.
            contours, _ = cv2.findContours(
                region_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                dropped_degenerate += 1
                continue
            cnt = max(contours, key=cv2.contourArea)
            pts = cnt.reshape(-1, 2)
            if len(pts) < 3:
                dropped_degenerate += 1
                continue
            mean_rgb_arr = (
                self.base_rgb[region_mask_bool].mean(axis=0) / 255.0
            )
            mean_rgb = (
                float(mean_rgb_arr[0]),
                float(mean_rgb_arr[1]),
                float(mean_rgb_arr[2]),
            )
            tiles.append((pts, mean_rgb))

        diagnostics = {
            "n_components": int(n_labels),
            "interior_regions": int(max(0, n_labels - 1)),
            "kept": len(tiles),
            "dropped_area": dropped_area,
            "dropped_degenerate": dropped_degenerate,
        }
        return tiles, diagnostics

    def save_polygons(self):
        """Detect every enclosed region BETWEEN the orange lines + frame
        (i.e. treat the lines + frame as boundary on a solid-black
        background) and save them as a CSV of polygons in the project's
        format. Each polygon's mean colour is sampled from the bright base
        image. A preview PNG of the polygon mosaic is saved alongside."""
        if self.line_mask is None or self.base_rgb is None:
            return

        h, w = self.line_mask.shape[:2]
        tiles, diagnostics = self._detect_polygons()
        # Cache for Show Polygons reuse + auto-refresh of preview view.
        self._last_tiles = tiles
        if self._polygons_view:
            self._refresh_display()

        if not tiles:
            QMessageBox.warning(
                self, "No polygons",
                "No polygons were detected.\n\n"
                f"Diagnostics:\n"
                f"  - connected interior regions found: "
                f"{diagnostics['interior_regions']}\n"
                f"  - dropped (area < {MIN_POLYGON_AREA_PX} px²): "
                f"{diagnostics['dropped_area']}\n"
                f"  - dropped (< 3 polygon vertices): "
                f"{diagnostics['dropped_degenerate']}\n\n"
                f"If 'interior regions' is 0 or 1, your lines aren't "
                f"closing off any space — they're probably broken/sparse, "
                f"or one line crossing means a single 'inside' region "
                f"covers the whole image. Toggle 'Show Polygons' to see "
                f"what the detector sees, then draw/erase to close gaps.",
            )
            return

        # 4) Output path. Default to BEERY/output/, name from the loaded
        # project file when available.
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        default_name = "polygons.csv"
        if self.current_project_path is not None:
            default_name = (
                self.current_project_path.with_suffix(".csv").name
            )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save polygons CSV",
            str(OUTPUT_DIR / default_name),
            "CSV polygons (*.csv);;All files (*.*)",
        )
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != ".csv":
            p = p.with_suffix(".csv")

        # 5) Write CSV in the schema scripts/image_strech.py reads:
        #    polygon_id, coordinates, color_r, color_g, color_b, color_a,
        #    frame_r, frame_g, frame_b, frame_a, group_id
        # The `coordinates` column is a JSON array of [x, y] pairs —
        # image_strech.py parses it with json.loads, which DOES NOT accept
        # the older semicolon-separated "x,y;x,y" format. Default frame is
        # opaque black (0, 0, 0, 1) and group_id is empty; downstream tools
        # only require `coordinates`, the rest are honoured if present.
        try:
            with open(p, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "polygon_id", "coordinates",
                    "color_r", "color_g", "color_b", "color_a",
                    "frame_r", "frame_g", "frame_b", "frame_a",
                    "group_id",
                ])
                for i, (pts, (r, g, b)) in enumerate(tiles):
                    coords_json = json.dumps([
                        [int(round(float(x))), int(round(float(y)))]
                        for x, y in pts
                    ])
                    writer.writerow([
                        i, coords_json,
                        r, g, b, 1.0,
                        0.0, 0.0, 0.0, 1.0,
                        "",
                    ])
        except Exception as e:
            QMessageBox.critical(
                self, "Save failed", f"{type(e).__name__}: {e}",
            )
            return

        # 6) Preview PNG: each polygon filled with its mean colour, 1-px
        # black outline. Non-fatal if it fails — the CSV is the deliverable.
        preview_path = p.with_suffix(".preview.png")
        try:
            preview = np.zeros((h, w, 3), dtype=np.uint8)
            for pts, (r, g, b) in tiles:
                colour = (
                    int(round(r * 255)),
                    int(round(g * 255)),
                    int(round(b * 255)),
                )
                poly_int = pts.astype(np.int32)
                cv2.fillPoly(preview, [poly_int], color=colour)
                cv2.polylines(
                    preview, [poly_int], isClosed=True,
                    color=(0, 0, 0), thickness=1,
                )
            _PILImage.fromarray(preview).save(preview_path, "PNG")
        except Exception:
            preview_path = None

        self.statusBar().showMessage(
            f"Saved {len(tiles)} polygons → {p.name}",
        )
        msg = (
            f"Saved {len(tiles)} polygons to:\n{p}\n\n"
            f"Format: image_strech.py-compatible CSV (polygon_id, "
            f"coordinates as JSON, color_r/g/b/a, frame_r/g/b/a, "
            f"group_id).\n"
        )
        if preview_path is not None:
            msg += f"\nPreview PNG: {preview_path}"
        QMessageBox.information(self, "Saved", msg)


def main() -> int:
    app = QApplication(sys.argv)
    win = LinesToVecEditor()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
