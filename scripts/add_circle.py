"""Add Circle — GUI for adding an AI-generated decorative circular frame
(meander, laurel wreath, rope, floral vine, ...) to a mosaic image.

Workflow:
  1. Load a mosaic image (Load Image...).
  2. Two concentric orange circles appear on the image — an inner and an
     outer. Drag anywhere in the ring to MOVE both circles together;
     drag near either circle edge to RESIZE that circle. Spinboxes in
     the right-hand panel mirror the state for exact numeric control.
  3. Pick a pattern variant from the dropdown (Meander / Laurel wreath /
     Rope / Floral vine).
  4. Click Generate Frame — the source image with the two orange guide
     circles overlaid is sent to Gemini with the chosen prompt. The
     result is masked to the annular ring geometrically (using our
     canonical inner + outer radii + centre) and composited onto a
     fresh copy of the ORIGINAL source. Base pixels outside the ring
     stay byte-for-byte identical.
  5. Save the composited image as PNG (Save Result...).

Reuses IMAGE_TO_MOSAIC/scripts/photo_editor.py's GenerationWorker + API
key resolver so no key management is duplicated.

Usage:
  python scripts/add_circle.py
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as _PILImage
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap,
)
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QFileDialog,
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent            # scripts/
PROJECT_ROOT = ROOT.parent                         # CLAUDE_MOSAIC1.0/
PROMPTS_DIR = ROOT / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Reuse the API scaffolding from IMAGE_TO_MOSAIC so we don't duplicate key
# handling / worker plumbing.
_IMAGE_TO_MOSAIC_SCRIPTS = PROJECT_ROOT / "IMAGE_TO_MOSAIC" / "scripts"
if str(_IMAGE_TO_MOSAIC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_IMAGE_TO_MOSAIC_SCRIPTS))

from photo_editor import (           # noqa: E402  (path munging above)
    DEFAULT_MODEL,
    GenerationWorker,
    auto_image_size,
    closest_aspect_ratio,
    load_api_key,
)


# ---------------------------------------------------------------------------
# Prompt variants — populated into the toolbar QComboBox.
# ---------------------------------------------------------------------------

PROMPT_OPTIONS: list[tuple[str, Path]] = [
    ("Meander (Greek key)", PROMPTS_DIR / "add_circle_meander.txt"),
    ("Laurel wreath",       PROMPTS_DIR / "add_circle_laurel.txt"),
    ("Rope / cable",        PROMPTS_DIR / "add_circle_rope.txt"),
    ("Floral vine",         PROMPTS_DIR / "add_circle_floral.txt"),
]

# Orange used for the guide circles overlaid on the source before sending
# to Gemini — same colour as voroni.py's Voronoi lines so any downstream
# tool that expects "#FF6600 marker" works uniformly.
GUIDE_COLOUR_BGR = (0, 102, 255)   # OpenCV uses BGR
GUIDE_COLOUR_QT = QColor(255, 102, 0)
GUIDE_OUTLINE_PX = 3                 # thickness of orange guide circles

# Distance (in image pixels) from a circle edge at which a mouse-press
# starts a RESIZE drag instead of a MOVE drag. Scaled by view zoom so the
# hit-zone feels the same at any zoom level.
EDGE_HIT_TOL_IMG_PX = 8.0


# ---------------------------------------------------------------------------
# QGraphicsView with mouse-driven circle move / resize + wheel zoom.
# ---------------------------------------------------------------------------

class CircleCanvasView(QGraphicsView):
    """Displays the base image with two concentric orange circles overlaid.
    Mouse:
      - Press near inner-circle edge  → drag resizes inner diameter.
      - Press near outer-circle edge  → drag resizes outer diameter.
      - Press inside the outer circle → drag moves both circles together.
      - Wheel                         → cursor-centered zoom.
    """

    geometry_changed = pyqtSignal(int, int, int, int)
    # cx, cy, inner_diameter, outer_diameter — all in image pixels.

    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        # Scrollbars whenever the zoomed scene overflows the viewport —
        # so pan-by-scrollbar always works when zoomed in.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.NoDrag)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Scene items. Explicit z-values so rendering order is stable
        # regardless of the order we create/rebuild things:
        #   pixmap (0) < ring fill (1) < outlines (5) < handles (10)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._ring_fill_item: QGraphicsPathItem | None = None
        self._inner_item: QGraphicsEllipseItem | None = None
        self._outer_item: QGraphicsEllipseItem | None = None
        self._inner_handle: QGraphicsRectItem | None = None
        self._outer_handle: QGraphicsRectItem | None = None

        # Toggled by the main window's "Fill ring" checkbox.
        self._ring_fill_visible: bool = False
        # Colour used for the ring fill overlay. Kept in sync with the
        # main window's Background colour picker via set_ring_fill_color().
        self._ring_fill_color: QColor = QColor(255, 255, 255)

        # Canonical circle state (image-pixel space).
        self._cx: int = 0
        self._cy: int = 0
        self._inner_d: int = 0
        self._outer_d: int = 0

        # Image dims for clamping.
        self._img_w: int = 0
        self._img_h: int = 0

        # Drag state.
        self._drag_mode: str | None = None
        # For 'move': anchor at scene coords the user pressed at, plus the
        # initial cx/cy so we translate by delta.
        self._drag_press_scene: tuple[float, float] | None = None
        self._drag_press_center: tuple[int, int] | None = None

    # ---- public setters ---------------------------------------------------

    def set_image(self, pil_img: _PILImage.Image):
        rgba = pil_img.convert("RGBA")
        qimg = QImage(
            rgba.tobytes("raw", "RGBA"),
            pil_img.width, pil_img.height,
            QImage.Format_RGBA8888,
        ).copy()
        pix = QPixmap.fromImage(qimg)
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pix)
        else:
            self._pixmap_item.setPixmap(pix)
        self._img_w = pil_img.width
        self._img_h = pil_img.height
        self._scene.setSceneRect(0, 0, self._img_w, self._img_h)

        # Sensible defaults for a fresh image: centered, outer ~90 % of
        # the shorter side, inner ~60 %.
        short = min(self._img_w, self._img_h)
        self._cx = self._img_w // 2
        self._cy = self._img_h // 2
        self._outer_d = int(short * 0.90)
        self._inner_d = int(short * 0.60)

        self._rebuild_circle_items()
        self._emit_geometry()
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def set_geometry(self, cx: int, cy: int, inner_d: int, outer_d: int):
        """Called by the spinboxes when the user types values directly."""
        if self._pixmap_item is None:
            return
        self._cx = int(cx)
        self._cy = int(cy)
        self._inner_d = max(2, int(inner_d))
        self._outer_d = max(self._inner_d + 2, int(outer_d))
        self._update_circle_items()

    def get_geometry(self) -> tuple[int, int, int, int]:
        return (self._cx, self._cy, self._inner_d, self._outer_d)

    def fit(self):
        if self._pixmap_item is None:
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    # ---- circle rendering -------------------------------------------------

    def _rebuild_circle_items(self):
        """Remove and re-create the ring fill, outline ellipses and resize
        handles after loading a new image."""
        for item in (self._ring_fill_item,
                     self._inner_item, self._outer_item,
                     self._inner_handle, self._outer_handle):
            if item is not None:
                self._scene.removeItem(item)

        # Ring fill (donut). Hidden by default; toggled by the main
        # window's "Fill ring" checkbox. Colour comes from the main
        # window's Background colour picker.
        self._ring_fill_item = QGraphicsPathItem()
        self._ring_fill_item.setBrush(QBrush(self._ring_fill_color))
        self._ring_fill_item.setPen(QPen(Qt.NoPen))
        self._ring_fill_item.setZValue(1)
        self._ring_fill_item.setVisible(self._ring_fill_visible)
        self._scene.addItem(self._ring_fill_item)

        # Circle outlines — cosmetic pen keeps line width constant at any zoom.
        pen = QPen(GUIDE_COLOUR_QT)
        pen.setWidth(GUIDE_OUTLINE_PX)
        pen.setCosmetic(True)
        self._outer_item = self._scene.addEllipse(0, 0, 1, 1, pen, QBrush(Qt.NoBrush))
        self._inner_item = self._scene.addEllipse(0, 0, 1, 1, pen, QBrush(Qt.NoBrush))
        self._outer_item.setZValue(5)
        self._inner_item.setZValue(5)

        # Resize handles — small orange squares at the 3-o'clock point of
        # each circle. ItemIgnoresTransformations keeps them at a fixed
        # screen-pixel size regardless of view zoom, so they're always
        # clickable and never dwarfed by the circle.
        handle_size = 10   # screen pixels
        h = handle_size / 2.0
        for setter_attr in ("_inner_handle", "_outer_handle"):
            handle = QGraphicsRectItem(-h, -h, handle_size, handle_size)
            handle.setBrush(QBrush(GUIDE_COLOUR_QT))
            handle.setPen(QPen(Qt.black))
            handle.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            handle.setZValue(10)
            handle.setCursor(Qt.SizeFDiagCursor)
            self._scene.addItem(handle)
            setattr(self, setter_attr, handle)

        self._update_circle_items()

    def set_ring_fill_visible(self, visible: bool) -> None:
        """Called by the main window when the 'Fill ring' checkbox
        toggles. The fill is a display-only overlay; when the user runs
        Generate Frame with it enabled, we also bake the ring into the
        API input separately, using the currently-chosen background
        colour (see set_ring_fill_color)."""
        self._ring_fill_visible = bool(visible)
        if self._ring_fill_item is not None:
            self._ring_fill_item.setVisible(self._ring_fill_visible)

    def set_ring_fill_color(self, color: QColor) -> None:
        """Update the ring fill overlay's colour. Called from the main
        window's Background colour picker so the display swatch reflects
        the choice live."""
        self._ring_fill_color = QColor(color)
        if self._ring_fill_item is not None:
            self._ring_fill_item.setBrush(QBrush(self._ring_fill_color))

    def _update_circle_items(self):
        if self._inner_item is None or self._outer_item is None:
            return
        r_in = self._inner_d / 2.0
        r_out = self._outer_d / 2.0
        self._inner_item.setRect(
            self._cx - r_in, self._cy - r_in, self._inner_d, self._inner_d,
        )
        self._outer_item.setRect(
            self._cx - r_out, self._cy - r_out, self._outer_d, self._outer_d,
        )
        # Ring-fill path (odd-even fill rule: outer ellipse minus inner
        # ellipse = donut shape).
        if self._ring_fill_item is not None:
            path = QPainterPath()
            path.setFillRule(Qt.OddEvenFill)
            path.addEllipse(
                self._cx - r_out, self._cy - r_out,
                self._outer_d, self._outer_d,
            )
            path.addEllipse(
                self._cx - r_in, self._cy - r_in,
                self._inner_d, self._inner_d,
            )
            self._ring_fill_item.setPath(path)
        # Handles at 3-o'clock of each circle. Position is in scene coords;
        # the handle's own rect is in screen pixels (ItemIgnoresTransformations).
        if self._inner_handle is not None:
            self._inner_handle.setPos(self._cx + r_in, self._cy)
        if self._outer_handle is not None:
            self._outer_handle.setPos(self._cx + r_out, self._cy)
        self._update_scene_rect()

    def _update_scene_rect(self):
        """Expand the scene rect so the outer circle is always inside the
        scrollable area — otherwise circles bigger than the image would
        get clipped by the view's scrolling range and the user couldn't
        pan out to see them."""
        if self._pixmap_item is None:
            return
        r_out = self._outer_d / 2.0
        left = min(0, self._cx - r_out)
        top = min(0, self._cy - r_out)
        right = max(self._img_w, self._cx + r_out)
        bottom = max(self._img_h, self._cy + r_out)
        self._scene.setSceneRect(left, top, right - left, bottom - top)

    def _emit_geometry(self):
        self.geometry_changed.emit(
            self._cx, self._cy, self._inner_d, self._outer_d,
        )

    # ---- mouse ------------------------------------------------------------

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        if self._pixmap_item is None or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        # 1) Handle hit → resize the corresponding circle. itemAt uses the
        # topmost item; handles are at zValue=10 so they always win over
        # the outline + fill.
        hit_item = self.itemAt(event.pos())
        if hit_item is self._inner_handle:
            self._drag_mode = "resize_inner"
            event.accept()
            return
        if hit_item is self._outer_handle:
            self._drag_mode = "resize_outer"
            event.accept()
            return
        # 2) Anywhere on OR inside the outer circle → move both circles
        # together. Grabbing the visible orange outline now moves; only
        # the little handle squares resize.
        sp = self.mapToScene(event.pos())
        sx, sy = float(sp.x()), float(sp.y())
        dist = float(np.hypot(sx - self._cx, sy - self._cy))
        r_out = self._outer_d / 2.0
        # A small grab zone JUST outside the outer circle so the user can
        # click on the line itself even if the mouse is a pixel or two
        # outside the exact circle. Same zoom-invariant tolerance as before.
        grab_zone = EDGE_HIT_TOL_IMG_PX / max(1e-6, self.transform().m11())
        if dist <= r_out + grab_zone:
            self._drag_mode = "move"
            self._drag_press_scene = (sx, sy)
            self._drag_press_center = (self._cx, self._cy)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_mode is None or self._pixmap_item is None:
            super().mouseMoveEvent(event)
            return
        sp = self.mapToScene(event.pos())
        sx, sy = float(sp.x()), float(sp.y())
        if self._drag_mode == "move":
            px, py = self._drag_press_scene    # type: ignore[misc]
            ox, oy = self._drag_press_center   # type: ignore[misc]
            new_cx = int(round(ox + (sx - px)))
            new_cy = int(round(oy + (sy - py)))
            self._cx = max(0, min(self._img_w - 1, new_cx))
            self._cy = max(0, min(self._img_h - 1, new_cy))
        else:
            # Resize — the target radius is the distance from cursor to
            # the (fixed) circle center.
            new_r = float(np.hypot(sx - self._cx, sy - self._cy))
            new_d = max(2, int(round(new_r * 2.0)))
            if self._drag_mode == "resize_inner":
                # Never let inner exceed outer.
                self._inner_d = min(new_d, self._outer_d - 2)
                self._inner_d = max(2, self._inner_d)
            else:
                # Never let outer collapse below inner + 2.
                self._outer_d = max(new_d, self._inner_d + 2)
        self._update_circle_items()
        self._emit_geometry()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag_mode is not None and event.button() == Qt.LeftButton:
            self._drag_mode = None
            self._drag_press_scene = None
            self._drag_press_center = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# Main window.
# ---------------------------------------------------------------------------

class AddCircleEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Add Circle — decorative circular frame")
        self.resize(1400, 900)

        # Loaded state.
        self.source_pil: _PILImage.Image | None = None
        self.source_path: Path | None = None
        self.composite_pil: _PILImage.Image | None = None   # last generated
        self.worker: GenerationWorker | None = None
        # Padding geometry stashed by generate_frame → read by _on_worker_ok
        # so both sides use the same padded coord system:
        # (pad_left, pad_top, padded_w, padded_h,
        #  padded_cx, padded_cy, inner_d, outer_d)
        self._last_pad: tuple | None = None
        # Frame colour choices. Baked into the AI-input canvas (when the
        # Fill ring checkbox is on) and injected as a colour override into
        # every prompt. Defaults are the classic Greek meander palette.
        self.bg_color: QColor = QColor(255, 255, 255)      # cream / white
        self.pattern_color: QColor = QColor(0, 0, 0)        # black

        # ----- UI -----
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # Toolbar.
        bar = QHBoxLayout()
        self.load_btn = QPushButton("Load Image...")
        self.load_btn.clicked.connect(self.load_image)
        bar.addWidget(self.load_btn)

        bar.addWidget(QLabel("Pattern:"))
        self.prompt_combo = QComboBox()
        for label, path in PROMPT_OPTIONS:
            self.prompt_combo.addItem(label, str(path))
        self.prompt_combo.setToolTip(
            "Which decorative pattern to fill the annular ring with. Each "
            "option sends a different prompt file to Gemini."
        )
        bar.addWidget(self.prompt_combo)

        self.generate_btn = QPushButton("Generate Frame")
        self.generate_btn.setToolTip(
            "Overlay the two orange guide circles on the source image, send "
            "that to Gemini with the selected pattern prompt, then mask the "
            "result to the annular ring and composite it back onto a fresh "
            "copy of the original source. Base pixels outside the ring stay "
            "byte-for-byte identical."
        )
        self.generate_btn.clicked.connect(self.generate_frame)
        bar.addWidget(self.generate_btn)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.clicked.connect(self._fit_view)
        bar.addWidget(self.fit_btn)

        self.save_btn = QPushButton("Save Result...")
        self.save_btn.setToolTip(
            "Save the last composited image (original + AI-generated ring) "
            "as a PNG at source resolution."
        )
        self.save_btn.clicked.connect(self.save_result)
        bar.addWidget(self.save_btn)

        bar.addStretch(1)
        root_layout.addLayout(bar)

        # Body split: canvas on the left, control panel on the right.
        body = QHBoxLayout()

        self.view = CircleCanvasView()
        self.view.geometry_changed.connect(self._on_view_geometry_changed)
        body.addWidget(self.view, 1)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel.setFixedWidth(230)
        panel_layout.addWidget(QLabel("<b>Circle geometry</b>"))

        def _add_spin(label_text: str, minimum: int, maximum: int,
                      value: int, suffix: str = " px"):
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            spin.setSuffix(suffix)
            row.addWidget(spin)
            panel_layout.addLayout(row)
            return spin

        self.cx_spin = _add_spin("Center X:", 0, 100000, 0)
        self.cy_spin = _add_spin("Center Y:", 0, 100000, 0)
        self.inner_spin = _add_spin("Inner ⌀:", 2, 100000, 0)
        self.outer_spin = _add_spin("Outer ⌀:", 2, 100000, 0)

        for spin in (self.cx_spin, self.cy_spin,
                     self.inner_spin, self.outer_spin):
            spin.valueChanged.connect(self._on_spinbox_changed)

        self.recenter_btn = QPushButton("Center on image")
        self.recenter_btn.clicked.connect(self._recenter)
        panel_layout.addWidget(self.recenter_btn)

        panel_layout.addWidget(QLabel("<b>Frame colours</b>"))

        # Two colour pickers — background (the ring's fill) and pattern
        # (the strokes / motif inside the ring). Both are injected as an
        # override into the prompt sent to Gemini, so the AI is told
        # exactly which colours to use for the frame regardless of what
        # the base prompt file's colour guidance says.
        self.bg_color_btn = QPushButton()
        self.bg_color_btn.setFixedHeight(30)
        self.bg_color_btn.setToolTip(
            "Ring background colour — the solid colour behind the "
            "pattern strokes inside the annular ring. Also used to "
            "pre-fill the ring when 'Fill ring' is checked."
        )
        self.bg_color_btn.clicked.connect(self._pick_bg_color)
        panel_layout.addWidget(self.bg_color_btn)

        self.pattern_color_btn = QPushButton()
        self.pattern_color_btn.setFixedHeight(30)
        self.pattern_color_btn.setToolTip(
            "Pattern colour — the colour of the strokes, motifs, or "
            "leaves inside the annular ring. The AI is told this is the "
            "ONLY colour to use for pattern strokes."
        )
        self.pattern_color_btn.clicked.connect(self._pick_pattern_color)
        panel_layout.addWidget(self.pattern_color_btn)

        # Sync button labels + swatch styles with the current colours,
        # and push the initial background colour into the view.
        self._refresh_color_buttons()
        self.view.set_ring_fill_color(self.bg_color)

        # Fill the annular ring with the chosen background colour. When
        # checked, the ring is displayed filled on the canvas AND the
        # image sent to Gemini has the ring pre-filled — the AI gets a
        # clean canvas inside the ring instead of the noisy base mosaic.
        self.fill_check = QCheckBox("Fill ring with background colour")
        self.fill_check.setToolTip(
            "Show the annular ring filled with the chosen background "
            "colour on the canvas AND send the image to Gemini with the "
            "ring pre-filled — gives the AI a clean canvas so the "
            "pattern isn't mixed with the underlying mosaic. Uncheck "
            "to send the raw mosaic as the AI's reference inside the "
            "ring."
        )
        self.fill_check.toggled.connect(self.view.set_ring_fill_visible)
        panel_layout.addWidget(self.fill_check)

        # Also ask the AI to repair missing/damaged mosaic content inside
        # the inner circle. When checked, an extra clause is appended to
        # the pattern prompt AND the composite mask is extended to
        # include the inner-circle area, so pixels there are replaced
        # with the AI's inpainted mosaic.
        self.fill_inner_check = QCheckBox("Fill missing mosaic\ninside inner circle")
        self.fill_inner_check.setToolTip(
            "Also ask Gemini to inpaint any missing, damaged, or blank "
            "areas of the mosaic INSIDE the inner circle — in the same "
            "style and palette as the surrounding intact mosaic. When "
            "checked, the composite replaces inner-circle pixels too "
            "(not only the ring). Leaves outside-the-outer-circle "
            "pixels byte-identical to the source in both modes."
        )
        panel_layout.addWidget(self.fill_inner_check)

        panel_layout.addStretch(1)
        body.addWidget(panel)
        root_layout.addLayout(body, 1)

        self.statusBar().showMessage("Load an image to begin.")
        self._update_button_states()

    # ----- I/O ------------------------------------------------------------

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load mosaic image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;All files (*.*)",
        )
        if not path:
            return
        try:
            pil = _PILImage.open(path).convert("RGB")
        except Exception as e:
            QMessageBox.critical(
                self, "Load failed", f"{type(e).__name__}: {e}",
            )
            return
        self.source_pil = pil
        self.source_path = Path(path)
        self.composite_pil = None
        self.view.set_image(pil)
        self._sync_spinboxes_from_view()
        self.setWindowTitle(f"Add Circle — {self.source_path.name}")
        self.statusBar().showMessage(
            f"Loaded {self.source_path.name}  ({pil.width} × {pil.height} px)",
        )
        self._update_button_states()

    def save_result(self):
        if self.composite_pil is None:
            QMessageBox.information(
                self, "Nothing to save",
                "Generate a frame first — Save Result... exports the last "
                "composited image (original + AI-generated ring).",
            )
            return
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        default = "add_circle_result.png"
        if self.source_path is not None:
            default = f"{self.source_path.stem}_circle.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save composited PNG",
            str(OUTPUT_DIR / default),
            "PNG image (*.png);;All files (*.*)",
        )
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != ".png":
            p = p.with_suffix(".png")
        try:
            self.composite_pil.save(p, "PNG")
        except Exception as e:
            QMessageBox.critical(
                self, "Save failed", f"{type(e).__name__}: {e}",
            )
            return
        self.statusBar().showMessage(f"Saved → {p.name}")

    # ----- geometry syncing ----------------------------------------------

    def _on_view_geometry_changed(self, cx: int, cy: int,
                                  inner_d: int, outer_d: int):
        # Update spinboxes without re-firing valueChanged into the view.
        for spin, value in (
            (self.cx_spin, cx),
            (self.cy_spin, cy),
            (self.inner_spin, inner_d),
            (self.outer_spin, outer_d),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _on_spinbox_changed(self, _v: int):
        if self.source_pil is None:
            return
        self.view.set_geometry(
            self.cx_spin.value(), self.cy_spin.value(),
            self.inner_spin.value(), self.outer_spin.value(),
        )

    def _sync_spinboxes_from_view(self):
        cx, cy, inner_d, outer_d = self.view.get_geometry()
        # Center is clamped to image bounds (a center outside the image
        # doesn't make sense for a frame), but the diameters are NOT capped
        # to the image size — the user is free to scale circles far past
        # the image edges. The view's scene rect grows to accommodate.
        w = self.source_pil.width if self.source_pil else 100000
        h = self.source_pil.height if self.source_pil else 100000
        self.cx_spin.setMaximum(max(1, w - 1))
        self.cy_spin.setMaximum(max(1, h - 1))
        # 10× the image's larger side as a soft ceiling — well past any
        # useful decorative-frame size but keeps the spinbox from allowing
        # accidental billion-pixel values.
        big = max(2, 10 * max(w, h))
        self.inner_spin.setMaximum(big)
        self.outer_spin.setMaximum(big)
        self._on_view_geometry_changed(cx, cy, inner_d, outer_d)

    def _recenter(self):
        if self.source_pil is None:
            return
        w, h = self.source_pil.width, self.source_pil.height
        cx = w // 2
        cy = h // 2
        # Preserve current diameters.
        _, _, inner_d, outer_d = self.view.get_geometry()
        self.view.set_geometry(cx, cy, inner_d, outer_d)

    # ----- colour pickers -------------------------------------------------

    @staticmethod
    def _styled_swatch(btn: QPushButton, color: QColor, label_prefix: str):
        """Style a button as a colour swatch — its background matches the
        chosen colour, its text is the hex code + label."""
        text_color = "black" if color.lightness() > 128 else "white"
        btn.setText(f"{label_prefix}: {color.name().upper()}")
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {color.name()}; "
            f"color: {text_color}; border: 1px solid #666; "
            f"padding: 4px 8px; }}"
            f"QPushButton:hover {{ border: 1px solid #000; }}"
        )

    def _refresh_color_buttons(self):
        self._styled_swatch(self.bg_color_btn, self.bg_color, "Background")
        self._styled_swatch(
            self.pattern_color_btn, self.pattern_color, "Pattern",
        )

    def _pick_bg_color(self):
        c = QColorDialog.getColor(
            self.bg_color, self, "Choose ring background colour",
        )
        if not c.isValid():
            return
        self.bg_color = c
        self._refresh_color_buttons()
        # Push into the view so the ring-fill overlay updates live.
        self.view.set_ring_fill_color(c)

    def _pick_pattern_color(self):
        c = QColorDialog.getColor(
            self.pattern_color, self, "Choose pattern colour",
        )
        if not c.isValid():
            return
        self.pattern_color = c
        self._refresh_color_buttons()

    def _fit_view(self):
        self.view.fit()

    def _update_button_states(self):
        loaded = self.source_pil is not None
        for w in (self.generate_btn, self.save_btn, self.fit_btn,
                  self.recenter_btn):
            w.setEnabled(loaded)
        for spin in (self.cx_spin, self.cy_spin,
                     self.inner_spin, self.outer_spin):
            spin.setEnabled(loaded)
        # Save is only useful once a composite exists.
        if not loaded or self.composite_pil is None:
            self.save_btn.setEnabled(loaded and self.composite_pil is not None)

    # ----- generation -----------------------------------------------------

    def generate_frame(self):
        if self.source_pil is None:
            return
        if load_api_key() is None:
            QMessageBox.critical(
                self, "No API key",
                "No Gemini API key found. Set GEMINI_API_KEY in your "
                "environment, or place the key in "
                "IMAGE_TO_MOSAIC/.env or IMAGE_TO_MOSAIC/gemini.key.",
            )
            return

        # Load selected prompt.
        prompt_path_str = self.prompt_combo.currentData()
        if not prompt_path_str:
            QMessageBox.critical(
                self, "No prompt", "No prompt selected in the dropdown.",
            )
            return
        try:
            prompt_text = Path(prompt_path_str).read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(
                self, "Prompt read failed",
                f"Could not read {prompt_path_str}: {e}",
            )
            return

        cx, cy, inner_d, outer_d = self.view.get_geometry()
        r_out_f = outer_d / 2.0
        # Pad the canvas so the outer circle fits fully. Without this, when
        # the outer circle extends past the source image, the ring's
        # pattern gets clipped to the source bounds — the AI has nowhere
        # to draw the pattern and the composite shows no visible frame.
        # With padding, the AI's canvas covers the whole ring; we
        # composite on the padded canvas and the result is a larger
        # output image with the source in the middle and the decorative
        # ring around it.
        src_w = self.source_pil.width
        src_h = self.source_pil.height
        pad_left = max(0, int(np.ceil(r_out_f - cx)))
        pad_top = max(0, int(np.ceil(r_out_f - cy)))
        pad_right = max(0, int(np.ceil(cx + r_out_f - src_w)))
        pad_bottom = max(0, int(np.ceil(cy + r_out_f - src_h)))
        padded_w = src_w + pad_left + pad_right
        padded_h = src_h + pad_top + pad_bottom
        padded_cx = cx + pad_left
        padded_cy = cy + pad_top

        # Stash padded geometry so _on_worker_ok can rebuild the composite
        # in the same coordinate system.
        self._last_pad = (
            pad_left, pad_top, padded_w, padded_h,
            padded_cx, padded_cy, inner_d, outer_d,
        )

        # Build the padded canvas: source at (pad_left, pad_top), black
        # elsewhere. Black is a clean neutral canvas for a decorative
        # circular frame; the AI sees exactly where the ring is.
        base_rgb = np.array(self.source_pil.convert("RGB"), dtype=np.uint8)
        annotated_rgb = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
        annotated_rgb[
            pad_top:pad_top + src_h,
            pad_left:pad_left + src_w,
        ] = base_rgb

        # Optional pre-fill of the ring with the chosen background colour,
        # BEFORE drawing the orange guides — the fill uses the padded
        # geometry so it lines up with the guides.
        if self.fill_check.isChecked():
            Y, X = np.ogrid[:padded_h, :padded_w]
            dist_sq = (X - padded_cx) ** 2 + (Y - padded_cy) ** 2
            r_in = inner_d / 2.0
            r_out = outer_d / 2.0
            ring_mask_pad = (dist_sq >= r_in ** 2) & (dist_sq <= r_out ** 2)
            annotated_rgb[ring_mask_pad] = (
                self.bg_color.red(),
                self.bg_color.green(),
                self.bg_color.blue(),
            )

        # Draw the two orange guide circles at the PADDED centre. These
        # give the AI an unambiguous visual reference for the ring.
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR).copy()
        cv2.circle(
            annotated_bgr, (padded_cx, padded_cy), inner_d // 2,
            GUIDE_COLOUR_BGR, thickness=GUIDE_OUTLINE_PX,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            annotated_bgr, (padded_cx, padded_cy), outer_d // 2,
            GUIDE_COLOUR_BGR, thickness=GUIDE_OUTLINE_PX,
            lineType=cv2.LINE_AA,
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        annotated_pil = _PILImage.fromarray(annotated_rgb)

        # Append the exact PADDED geometry to the prompt so the AI knows
        # where the ring is in the canvas we're about to send.
        prompt_with_geom = (
            prompt_text
            + "\n\nRing geometry in the input image (pixels):\n"
            + f"  - centre: ({padded_cx}, {padded_cy})\n"
            + f"  - inner circle diameter: {inner_d} px "
            + f"(radius {inner_d // 2} px)\n"
            + f"  - outer circle diameter: {outer_d} px "
            + f"(radius {outer_d // 2} px)\n"
            + f"  - image size: {padded_w} × {padded_h} px\n"
        )
        # Colour override — the user's chosen frame colours. The prompt
        # files talk about vector strokes; here we clarify that these
        # colours apply to TILES (see MOSAIC RENDERING below), not
        # continuous strokes.
        bg_hex = self.bg_color.name().upper()
        pattern_hex = self.pattern_color.name().upper()
        prompt_with_geom += (
            "\nCOLOUR OVERRIDE (mandatory — supersedes any colour guidance "
            "elsewhere in this prompt):\n"
            f"  - Background tiles: solid {bg_hex} — every mosaic tile "
            f"that fills the ring's 'background' area is exactly this "
            f"colour.\n"
            f"  - Pattern tiles: solid {pattern_hex} — every mosaic tile "
            f"that belongs to the pattern (stroke, motif element, leaf, "
            f"twist, etc.) is exactly this colour.\n"
            "  - These are the ONLY two tile colours inside the annular "
            "ring. Grout lines between tiles are a separate thin dark "
            "line (see MOSAIC RENDERING below).\n"
        )
        # Mosaic rendering — the whole point of this workflow: the frame
        # must LOOK like the same kind of mosaic as the source, not a
        # graphic-design overlay. This block supersedes the "MECHANICAL
        # / VECTOR" language the pattern prompt files use.
        prompt_with_geom += (
            "\nMOSAIC RENDERING (mandatory — supersedes any style / "
            "vector / mechanical guidance elsewhere in this prompt):\n"
            "  - The entire content of the annular ring must be rendered "
            "as INDIVIDUAL CERAMIC MOSAIC TILES (tesserae) — small "
            "stone-like pieces butted together with thin dark grout lines "
            "between them.\n"
            "  - Tile SIZE must MATCH the tile size visible in the input "
            "mosaic (inside the inner circle, where the original mosaic "
            "lives). Examine the tile pitch there and reproduce that "
            "same pitch throughout the entire ring — same tiles-per-inch "
            "as the source.\n"
            "  - Tiles are roughly SQUARE or slightly RECTANGULAR, with "
            "small hand-cut irregularities in size and edge — not "
            "perfect computer-drawn squares.\n"
            "  - The pattern is formed by TILE PLACEMENT: pattern-"
            "coloured tiles arranged into the pattern shape, surrounded "
            "by background-coloured tiles filling the rest of the ring. "
            "The pattern silhouette is still recognisable (e.g. a "
            "meander still reads as a Greek key) — it is just built out "
            "of tesserae rather than drawn as a vector stroke.\n"
            "  - Grout lines between tiles are thin, dark, and "
            "consistent — matching the grout width and colour visible "
            "in the input mosaic.\n"
            "  - NO painterly strokes, NO smooth continuous curves, NO "
            "clean vector outlines. Every visible mark inside the ring "
            "is a tile.\n"
        )
        # Optional add-on: repair missing / damaged / blank mosaic
        # content inside the inner circle. Appended only when the
        # "Fill missing mosaic inside inner circle" checkbox is on.
        if self.fill_inner_check.isChecked():
            prompt_with_geom += (
                "\nADDITIONAL TASK — mosaic repair INSIDE the inner circle:\n"
                "- INSIDE the inner orange circle, examine the existing "
                "mosaic content. Detect any missing, damaged, blank, or "
                "unfinished tile areas and INPAINT them with ceramic "
                "mosaic tiles in the SAME style, palette, tile shape, "
                "and pattern as the surrounding intact mosaic.\n"
                "- Preserve intact mosaic tiles where they already exist "
                "— only fill in the gaps.\n"
                "- The transition between the decorative ring pattern "
                "(in the annular ring) and the repaired mosaic (inside "
                "the inner circle) must be clean and register exactly "
                "at the inner circle boundary — no overlap, no gap.\n"
            )

        aspect = closest_aspect_ratio(padded_w, padded_h)
        size = auto_image_size(padded_w, padded_h)

        # Fire the worker. Disable Generate until it comes back so we
        # can't queue overlapping API calls.
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating...")
        self.statusBar().showMessage(
            f"Calling Gemini ({self.prompt_combo.currentText()})...",
        )

        self.worker = GenerationWorker(
            source_image=annotated_pil,
            prompt_text=prompt_with_geom,
            model_id=DEFAULT_MODEL,
            aspect_ratio=aspect,
            image_size=size,
        )
        self.worker.finished_ok.connect(self._on_worker_ok)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.progress.connect(
            lambda s: self.statusBar().showMessage(s),
        )
        self.worker.start()

    def _on_worker_ok(self, img_bytes: bytes):
        try:
            ai_pil = _PILImage.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            self._reset_generate_button()
            QMessageBox.critical(
                self, "Bad response", f"Gemini returned unreadable image: {e}",
            )
            return
        if self.source_pil is None or self._last_pad is None:
            self._reset_generate_button()
            return

        # Restore padding + geometry from the values stashed at
        # generate_frame time.
        (pad_left, pad_top, padded_w, padded_h,
         padded_cx, padded_cy, inner_d, outer_d) = self._last_pad

        # Resize AI output to the padded canvas dimensions.
        if ai_pil.size != (padded_w, padded_h):
            ai_pil = ai_pil.resize(
                (padded_w, padded_h), _PILImage.LANCZOS,
            )

        # Rebuild the padded input (source at (pad_left, pad_top), black
        # elsewhere) — this is what everything OUTSIDE the composite
        # mask keeps. Source pixels are preserved byte-for-byte inside
        # their padded location.
        src_rgb = np.array(self.source_pil.convert("RGB"), dtype=np.uint8)
        padded_input = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
        padded_input[
            pad_top:pad_top + src_rgb.shape[0],
            pad_left:pad_left + src_rgb.shape[1],
        ] = src_rgb

        # Ring / inner masks in PADDED coordinates.
        Y, X = np.ogrid[:padded_h, :padded_w]
        dist_sq = (X - padded_cx) ** 2 + (Y - padded_cy) ** 2
        r_in = inner_d / 2.0
        r_out = outer_d / 2.0
        ring_mask = (dist_sq >= r_in ** 2) & (dist_sq <= r_out ** 2)
        if self.fill_inner_check.isChecked():
            inner_mask = dist_sq <= r_in ** 2
            composite_mask = ring_mask | inner_mask
        else:
            composite_mask = ring_mask

        ai_rgb = np.array(ai_pil, dtype=np.uint8)
        # Replace masked pixels with AI pixels. Everything else stays:
        # source pixels inside the source's padded location, black in
        # the padding area outside the outer circle.
        padded_input[composite_mask] = ai_rgb[composite_mask]
        self.composite_pil = _PILImage.fromarray(padded_input)

        # Auto-hide the white ring-fill overlay so it doesn't cover the
        # freshly-drawn pattern.
        if self.fill_check.isChecked():
            self.fill_check.blockSignals(True)
            self.fill_check.setChecked(False)
            self.fill_check.blockSignals(False)
            self.view.set_ring_fill_visible(False)

        # Display the (possibly larger) composite. Re-apply the padded
        # geometry so the overlay circles line up with the composited
        # ring on the new canvas.
        self.view.set_image(self.composite_pil)
        self.view.set_geometry(padded_cx, padded_cy, inner_d, outer_d)
        self._sync_spinboxes_from_view()

        self._reset_generate_button()
        n_ring = int(ring_mask.sum())
        n_total = int(composite_mask.sum())
        pad_note = ""
        if padded_w > src_rgb.shape[1] or padded_h > src_rgb.shape[0]:
            pad_note = (
                f"  [canvas padded to {padded_w} × {padded_h} px "
                f"to fit ring]"
            )
        if n_total > n_ring:
            n_inner = n_total - n_ring
            summary = (
                f"Composited {n_ring:,} ring + {n_inner:,} inner-circle "
                f"pixels from Gemini "
                f"({self.prompt_combo.currentText()} + mosaic repair)."
                f"{pad_note}"
            )
        else:
            summary = (
                f"Composited {n_ring:,} ring pixels from Gemini "
                f"({self.prompt_combo.currentText()}).{pad_note}"
            )
        self.statusBar().showMessage(f"{summary} Save Result... to export.")
        self.save_btn.setEnabled(True)

    def _on_worker_failed(self, msg: str):
        self._reset_generate_button()
        QMessageBox.critical(self, "Generation failed", msg)

    def _reset_generate_button(self):
        self.generate_btn.setEnabled(self.source_pil is not None)
        self.generate_btn.setText("Generate Frame")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    app = QApplication(sys.argv)
    win = AddCircleEditor()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
