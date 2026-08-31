"""mosaic_aranger.py — GUI for arranging mosaic elements on a grid.

Foundation pass:
  * Canvas that draws an N×N grid (default 15×15) fitted to the widget.
  * Left sidebar with a Grid section:
      - "N × N" readout of the current grid size.
      - ↑ / ↓ buttons that increase / decrease N by one.

Later passes will add element placement, drag-and-drop, etc.
"""

import sys
import json
import re
import pickle
from pathlib import Path

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFrame, QLabel, QSpinBox, QPushButton, QScrollArea, QFileDialog,
    QCheckBox, QMessageBox,
)
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QBrush, QPixmap, QImage,
)

PROJECT_EXT = ".aranger"
# V2 (2026): the .aranger file stores INFORMATION ONLY — grid dims,
# view state, and per-placed-tile (source_path, corners_px, label,
# row, col). No image bytes. Load Project re-reads each PNG from
# disk (and re-applies the white-transparent transform). V1 (which
# embedded PNG bytes) still opens via a fallback branch in the
# loader; the resulting resave will drop the embedded bytes.
PROJECT_VERSION = 2


class ArrangerCanvas(QWidget):
    """Central canvas — plain white background with an independently-
    sized cols × rows grid overlay. Cells are square (kept equal by
    fitting the grid inside the widget's usable area). `set_grid_dims`
    triggers a repaint."""

    DEFAULT_COLS = 15
    DEFAULT_ROWS = 15
    MIN_DIM      = 1
    MAX_DIM      = 500

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 600)
        self.setStyleSheet("background-color: white;")
        self.grid_cols = self.DEFAULT_COLS
        self.grid_rows = self.DEFAULT_ROWS

        # View transform — mouse-wheel zooms around the cursor. `pan_x`
        # / `pan_y` are the on-screen offset of the zoom-1 origin; they
        # get shifted by wheelEvent so the world point under the cursor
        # stays put during zoom.
        self.zoom_factor = 1.0
        self.pan_x       = 0.0
        self.pan_y       = 0.0
        self._MIN_ZOOM   = 0.05
        self._MAX_ZOOM   = 40.0

        # Grid line style — pink, matching the convention in the other
        # scripts (image_strech / mosaic_editor).
        self._grid_color     = QColor(255, 105, 180)
        self._grid_thickness = 2
        # When False, paintEvent skips the grid overlay entirely (both
        # the pre-tile and post-tile passes). Toggled from the sidebar.
        self.show_grid = True

        # Tile placement state.
        #   pending_tile: single tile armed for placement — clicking a
        #     grid cell drops it there. Stores the QPixmap plus the
        #     `mosaic_box_corners` metadata (or None for a plain PNG).
        #   pending_batch: a list of {pixmap, corners_px, dr, dc,
        #     label}. Clicking a grid cell drops EVERY item at its
        #     (row + dr, col + dc). Used by the composite / "place
        #     all tiles as an array" mode.
        #   Only one of {pending_tile, pending_batch} is active at a
        #   time; arming one clears the other.
        #   placed_tiles: every dropped tile, ordered by placement
        #     time. Each is {pixmap, corners_px, row, col, label}.
        self.pending_tile:  dict | None  = None
        self.pending_batch: list[dict] | None = None
        self.placed_tiles: list[dict]    = []

        # Select-and-move state. When nothing is armed for placement,
        # clicking a cell that already holds a placed tile SELECTS the
        # topmost one there; the next click on any grid cell moves it.
        # ESC (or clicking the same cell twice) clears the selection.
        self.selected_placed_index: int = -1

        # Widget needs keyboard focus for the ESC-to-release shortcut.
        self.setFocusPolicy(Qt.StrongFocus)

    def set_show_grid(self, on: bool) -> None:
        on = bool(on)
        if on == self.show_grid:
            return
        self.show_grid = on
        self.update()

    def set_grid_dims(self, cols: int, rows: int) -> None:
        cols = max(self.MIN_DIM, min(self.MAX_DIM, int(cols)))
        rows = max(self.MIN_DIM, min(self.MAX_DIM, int(rows)))
        if cols == self.grid_cols and rows == self.grid_rows:
            return
        self.grid_cols = cols
        self.grid_rows = rows
        self.update()

    def paintEvent(self, _ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        # White background — drawn in *screen* space before the
        # zoom / pan transform, so it always fills the widget
        # regardless of view state.
        painter.fillRect(self.rect(), QBrush(Qt.white))

        w = self.width()
        h = self.height()
        cols = max(1, self.grid_cols)
        rows = max(1, self.grid_rows)

        # Square cells that fit inside the widget at zoom = 1 — the
        # LIMITING dimension picks the cell size and the grid is
        # centred. Once we apply the view transform everything scales
        # together.
        cell = min(w / cols, h / rows)
        grid_w = cell * cols
        grid_h = cell * rows
        ox = (w - grid_w) / 2.0
        oy = (h - grid_h) / 2.0

        # Apply view transform: screen = zoom * world + pan.
        painter.translate(self.pan_x, self.pan_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        # Cosmetic pen keeps line thickness constant in SCREEN pixels
        # regardless of zoom — grid lines don't turn into thick slabs
        # when the user zooms in.
        pen = QPen(self._grid_color, self._grid_thickness)
        pen.setCosmetic(True)
        painter.setPen(pen)

        if self.show_grid:
            # Vertical lines (cols + 1 of them, including outer borders).
            for i in range(cols + 1):
                x = ox + i * cell
                painter.drawLine(int(x), int(oy),
                                 int(x), int(oy + grid_h))

            # Horizontal lines (rows + 1).
            for j in range(rows + 1):
                y = oy + j * cell
                painter.drawLine(int(ox),          int(y),
                                 int(ox + grid_w), int(y))

        # ── Placed tiles ──────────────────────────────────────────
        # Draw each dropped tile using its box-corner metadata: the
        # PNG's containing-box corners get aligned to the target grid
        # cell, so the PNG naturally spills over the cell edges by the
        # DXF's offset expansion (same look as mosaic_editor).
        for tile in self.placed_tiles:
            self._paint_placed_tile(painter, tile, cell, ox, oy)

        # ── Selection highlight ─────────────────────────────────────
        # Cyan cosmetic border around the selected placed tile's cell,
        # drawn after the tiles so it's visible on top of the image.
        if 0 <= self.selected_placed_index < len(self.placed_tiles):
            sel = self.placed_tiles[self.selected_placed_index]
            sel_pen = QPen(QColor(0, 200, 255), 3)
            sel_pen.setCosmetic(True)
            painter.setPen(sel_pen)
            painter.setBrush(Qt.NoBrush)
            sx = ox + sel['col'] * cell
            sy = oy + sel['row'] * cell
            painter.drawRect(QRectF(sx, sy, cell, cell))

        # The grid lines below have already been drawn UNDER the tiles.
        # Draw them again on top so the grid stays visible over the
        # dropped tiles too — nice for alignment.
        if self.show_grid:
            painter.setPen(pen)
            for i in range(cols + 1):
                x = ox + i * cell
                painter.drawLine(int(x), int(oy),
                                 int(x), int(oy + grid_h))
            for j in range(rows + 1):
                y = oy + j * cell
                painter.drawLine(int(ox),          int(y),
                                 int(ox + grid_w), int(y))
        painter.end()

    @staticmethod
    def _paint_placed_tile(painter: QPainter, tile: dict,
                            cell: float, ox: float, oy: float) -> None:
        """Blit one placed tile so its containing-box corners map onto
        the target grid cell. When corners_px is None (plain PNG with
        no metadata) the whole PNG is mapped 1:1 into the cell."""
        pm = tile['pixmap']
        if pm is None or pm.isNull():
            return
        row = tile['row']; col = tile['col']
        target_x0 = ox + col * cell
        target_y0 = oy + row * cell

        corners_px = tile.get('corners_px')
        if corners_px and len(corners_px) >= 4:
            # TL / TR / BR / BL — box width & height in PNG-pixel space.
            src_tl_x, src_tl_y = corners_px[0]
            src_tr_x, _        = corners_px[1]
            _,        src_bl_y = corners_px[3]
            src_box_w = max(1.0, src_tr_x - src_tl_x)
            src_box_h = max(1.0, src_bl_y - src_tl_y)
        else:
            src_tl_x, src_tl_y = 0.0, 0.0
            src_box_w = max(1.0, float(pm.width()))
            src_box_h = max(1.0, float(pm.height()))

        # World size of one source PNG pixel once the source box is
        # scaled to fit the target cell.
        sx = cell / src_box_w
        sy = cell / src_box_h

        # Position the whole PNG so its "box top-left" coincides with
        # the target cell's top-left corner. Everything outside the
        # box (the offset bleed) naturally hangs over the cell edges.
        full_x0 = target_x0 - src_tl_x * sx
        full_y0 = target_y0 - src_tl_y * sy
        full_w  = pm.width()  * sx
        full_h  = pm.height() * sy

        painter.drawPixmap(
            QRectF(full_x0, full_y0, full_w, full_h),
            pm,
            QRectF(0, 0, pm.width(), pm.height()),
        )

    # ── mouse wheel: zoom around cursor ───────────────────────────────
    def wheelEvent(self, ev):
        """Roll the wheel forward → zoom in; back → zoom out. The
        world point under the cursor stays fixed on screen."""
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else (1.0 / 1.25)
        new_zoom = max(self._MIN_ZOOM,
                       min(self._MAX_ZOOM, self.zoom_factor * factor))
        if new_zoom == self.zoom_factor:
            return

        # Cursor position in world coords BEFORE the zoom change
        # (screen = zoom * world + pan  →  world = (screen - pan) / zoom).
        cx, cy = ev.x(), ev.y()
        wx = (cx - self.pan_x) / self.zoom_factor
        wy = (cy - self.pan_y) / self.zoom_factor

        # Update zoom, then re-derive pan so (wx, wy) still sits under
        # the cursor: pan = screen - zoom * world.
        self.zoom_factor = new_zoom
        self.pan_x = cx - wx * self.zoom_factor
        self.pan_y = cy - wy * self.zoom_factor
        self.update()

    # ── grid geometry helpers ─────────────────────────────────────────
    def _grid_layout(self) -> tuple:
        """Return `(cell, ox, oy, grid_w, grid_h)` in world (zoom-1)
        coords — matching how paintEvent lays the grid out."""
        w = self.width(); h = self.height()
        cols = max(1, self.grid_cols)
        rows = max(1, self.grid_rows)
        cell = min(w / cols, h / rows)
        grid_w = cell * cols
        grid_h = cell * rows
        ox = (w - grid_w) / 2.0
        oy = (h - grid_h) / 2.0
        return cell, ox, oy, grid_w, grid_h

    def _cell_from_screen(self, sx: int, sy: int):
        """Which grid cell (row, col) contains screen point (sx, sy)?
        Returns None when the point is outside the grid area."""
        wx = (sx - self.pan_x) / self.zoom_factor
        wy = (sy - self.pan_y) / self.zoom_factor
        cell, ox, oy, grid_w, grid_h = self._grid_layout()
        if wx < ox or wx >= ox + grid_w: return None
        if wy < oy or wy >= oy + grid_h: return None
        col = int((wx - ox) / cell)
        row = int((wy - oy) / cell)
        return (row, col)

    # ── tile placement ────────────────────────────────────────────────
    def arm_tile(self, pixmap: QPixmap, corners_px, label: str = "",
                 source_path: str | None = None) -> None:
        """Arm a SINGLE tile for placement. Clears any pending batch
        and any current placed-tile selection (mutually exclusive).

        `source_path` (absolute path to the on-disk PNG) is stashed on
        the pending tile and propagated to each placed_tiles entry
        that results from dropping it, so Save Project can round-trip
        the placement without embedding image bytes."""
        self.pending_batch = None
        self.selected_placed_index = -1
        if pixmap is None or pixmap.isNull():
            self.pending_tile = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return
        self.pending_tile = {
            'pixmap':      pixmap,
            'corners_px':  corners_px,
            'label':       label,
            'source_path': source_path,
        }
        self.setCursor(Qt.CrossCursor)
        self.update()

    def arm_batch(self, items: list) -> None:
        """Arm a BATCH placement — the next canvas click drops every
        item at (target_row + item['dr'], target_col + item['dc']).
        Each item is a dict with keys: pixmap, corners_px, dr, dc,
        label. Clears any pending single tile and any current
        placed-tile selection."""
        self.pending_tile = None
        self.selected_placed_index = -1
        if not items:
            self.pending_batch = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return
        self.pending_batch = list(items)
        self.setCursor(Qt.CrossCursor)
        self.update()

    def clear_pending_tile(self) -> None:
        self.pending_tile  = None
        self.pending_batch = None
        self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, ev):
        """ESC releases: clears any pending placement AND any current
        placed-tile selection. Other keys fall through to the default
        handler."""
        if ev.key() == Qt.Key_Escape:
            changed = (self.pending_tile is not None
                       or self.pending_batch is not None
                       or self.selected_placed_index != -1)
            self.pending_tile  = None
            self.pending_batch = None
            self.selected_placed_index = -1
            self.setCursor(Qt.ArrowCursor)
            if changed:
                self.update()
            return
        super().keyPressEvent(ev)

    def mousePressEvent(self, ev):
        # Ensure the widget has keyboard focus after any click so the
        # ESC-to-release shortcut works right away.
        self.setFocus(Qt.MouseFocusReason)

        if ev.button() != Qt.LeftButton:
            return
        rc = self._cell_from_screen(ev.x(), ev.y())
        if rc is None:
            return
        row, col = rc

        # Batch placement wins over single tile when armed.
        if self.pending_batch is not None:
            for item in self.pending_batch:
                self.placed_tiles.append({
                    'pixmap':      item['pixmap'],
                    'corners_px':  item.get('corners_px'),
                    'label':       item.get('label', ''),
                    'source_path': item.get('source_path'),
                    'row':         row + int(item.get('dr', 0)),
                    'col':         col + int(item.get('dc', 0)),
                })
            self.update()
            return

        if self.pending_tile is not None:
            self.placed_tiles.append({
                'pixmap':      self.pending_tile['pixmap'],
                'corners_px':  self.pending_tile['corners_px'],
                'label':       self.pending_tile['label'],
                'source_path': self.pending_tile.get('source_path'),
                'row':         row,
                'col':         col,
            })
            # Keep armed so the user can drop copies rapidly.
            self.update()
            return

        # ── select-and-move (no placement armed) ───────────────────
        # If a tile is already selected, this click moves it to the
        # clicked cell then RELEASES the selection. Same-cell click on
        # a selected tile just deselects. Otherwise, pick the topmost
        # placed tile sitting on the clicked cell and select it.
        if 0 <= self.selected_placed_index < len(self.placed_tiles):
            tile = self.placed_tiles[self.selected_placed_index]
            if not (tile['row'] == row and tile['col'] == col):
                tile['row'] = row
                tile['col'] = col
            # Either way (moved or same-cell), the paste is done → release.
            self.selected_placed_index = -1
            self.update()
            return

        # Nothing selected → try to pick the top tile at this cell.
        # Iterate in REVERSE so the last-drawn (visually topmost) tile
        # wins when several were dropped on the same cell.
        for idx in range(len(self.placed_tiles) - 1, -1, -1):
            t = self.placed_tiles[idx]
            if t['row'] == row and t['col'] == col:
                self.selected_placed_index = idx
                self.update()
                return
        # Empty cell click — nothing to do.


class SidePanel(QFrame):
    """Left sidebar — currently just the Grid controls (readout + up/
    down arrows). Set up so more sections can be appended above the
    trailing addStretch()."""

    def __init__(self, canvas: ArrangerCanvas, window=None, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.window_ref = window   # MosaicArrangerWindow for save/load
        self.setFrameStyle(QFrame.StyledPanel)
        self.setMinimumWidth(200)
        self.setMaximumWidth(240)
        self.setStyleSheet("background-color: #f0f0f0;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Project section ───────────────────────────────────────────
        # Save / Load buttons at the very top so they're always in a
        # predictable spot regardless of what other sections grow below.
        layout.addWidget(self._section_label("Project"))

        self.save_btn = QPushButton("Save Project")
        self.save_btn.setToolTip(
            f"Save grid dims, view state, and every placed tile "
            f"(image + label + cell) to a *{PROJECT_EXT} file."
        )
        self.save_btn.clicked.connect(self._on_save_project)
        layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load Project")
        self.load_btn.setToolTip(
            f"Load a *{PROJECT_EXT} file. REPLACES the current arrangement."
        )
        self.load_btn.clicked.connect(self._on_load_project)
        layout.addWidget(self.load_btn)

        # ── Grid section ──────────────────────────────────────────────
        # Two direct-input spinboxes for the grid's column × row count.
        # Value changes push straight to the canvas, which clamps to
        # its own min/max — subsequent editing sees the clamped value.
        layout.addWidget(self._section_label("Grid"))

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("Cols:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(canvas.MIN_DIM, canvas.MAX_DIM)
        self.cols_spin.setValue(canvas.grid_cols)
        self.cols_spin.setToolTip(
            "Number of grid columns (cells across the canvas)."
        )
        self.cols_spin.valueChanged.connect(self._on_size_changed)
        row.addWidget(self.cols_spin)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(canvas.MIN_DIM, canvas.MAX_DIM)
        self.rows_spin.setValue(canvas.grid_rows)
        self.rows_spin.setToolTip(
            "Number of grid rows (cells down the canvas)."
        )
        self.rows_spin.valueChanged.connect(self._on_size_changed)
        row.addWidget(self.rows_spin)
        layout.addLayout(row)

        # Show / hide the pink grid overlay. On by default, matches
        # ArrangerCanvas.show_grid so the checkbox reads correctly.
        self.show_grid_chk = QCheckBox("Show Grid")
        self.show_grid_chk.setChecked(canvas.show_grid)
        self.show_grid_chk.setToolTip(
            "Toggle the pink grid overlay. Off = plain canvas + placed "
            "tiles only (grid still exists; tiles still snap to cells)."
        )
        self.show_grid_chk.toggled.connect(self.canvas.set_show_grid)
        layout.addWidget(self.show_grid_chk)

        layout.addStretch(1)

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: bold; color: #303030; margin-top: 4px;")
        return lbl

    def _on_size_changed(self, _v: int) -> None:
        self.canvas.set_grid_dims(
            self.cols_spin.value(),
            self.rows_spin.value(),
        )

    # ── project save / load ──────────────────────────────────────────
    def _on_save_project(self) -> None:
        w = self.window_ref
        if w is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "",
            f"Arranger Project (*{PROJECT_EXT})",
        )
        if not path:
            return
        if not path.lower().endswith(PROJECT_EXT):
            path += PROJECT_EXT
        try:
            w.save_project(path)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return

    def _on_load_project(self) -> None:
        w = self.window_ref
        if w is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "",
            f"Arranger Project (*{PROJECT_EXT})",
        )
        if not path:
            return
        try:
            skipped, warnings = w.load_project(path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        # Reflect the newly-loaded canvas state in the sidebar widgets.
        self.cols_spin.blockSignals(True)
        self.rows_spin.blockSignals(True)
        self.show_grid_chk.blockSignals(True)
        self.cols_spin.setValue(self.canvas.grid_cols)
        self.rows_spin.setValue(self.canvas.grid_rows)
        self.show_grid_chk.setChecked(self.canvas.show_grid)
        self.cols_spin.blockSignals(False)
        self.rows_spin.blockSignals(False)
        self.show_grid_chk.blockSignals(False)
        if skipped:
            body = (f"Loaded, but skipped {skipped} tile(s) whose "
                    "source PNGs were missing or unreadable.")
            if warnings:
                # Cap the listing so a big missing directory doesn't
                # produce a wall-of-text dialog.
                shown = warnings[:12]
                body += "\n\n" + "\n".join(shown)
                if len(warnings) > len(shown):
                    body += f"\n… and {len(warnings) - len(shown)} more."
            QMessageBox.warning(self, "Load complete", body)


class ClickableFrame(QFrame):
    """QFrame that reports LMB clicks via a callback. Used for the
    thumbnail cards in TilesPanel — each card holds a tile preview
    and, when clicked, arms that tile for placement on the canvas."""

    def __init__(self, on_click, parent=None):
        super().__init__(parent)
        self._on_click = on_click
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and callable(self._on_click):
            self._on_click()


class TilesPanel(QFrame):
    """Right sidebar — browses tile PNGs from a chosen directory.
    A `Choose Directory` button at the top opens a folder picker; on
    accept the panel recursively finds every `.png` under it (so
    mosaic_editor's per-tile subdirs are picked up automatically),
    sorts them by filename, and renders a scrollable column of
    thumbnails + captions. Clicking a thumbnail arms it on the
    canvas — the next canvas click drops it into that grid cell."""

    THUMB_MAX_W = 130          # px, thumbnails scaled to fit this width
    THUMB_MAX_H = 130          # px, capped so tall tiles don't blow up
    COMPOSITE_MAX_SIDE = 150   # px, composite thumbnail longest side
    PANEL_WIDTH = 180
    METADATA_KEY = "mosaic_box_corners"

    # Label parse — mosaic_editor names tiles "<row-letter><col-number>",
    # e.g. "A1", "F6". Rows A–Z map to 0–25; columns are 1-indexed.
    _LABEL_RE = re.compile(r"^([A-Z])(\d+)$")

    def __init__(self, canvas: ArrangerCanvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.setFrameStyle(QFrame.StyledPanel)
        self.setFixedWidth(self.PANEL_WIDTH)
        self.setStyleSheet("background-color: #f0f0f0;")

        self.current_dir: Path | None = None
        # Card currently highlighted as "armed" — kept so we can drop
        # the highlight when a new card is selected.
        self._armed_card: ClickableFrame | None = None
        self._armed_path: Path | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        outer.addWidget(self._section_label("Tiles"))

        self.choose_btn = QPushButton("Choose Directory")
        self.choose_btn.setToolTip(
            "Pick a directory. Every .png beneath it (recursively) "
            "will be listed below as a scrollable thumbnail column."
        )
        self.choose_btn.clicked.connect(self._on_choose_dir)
        outer.addWidget(self.choose_btn)

        self.dir_label = QLabel("(no directory chosen)")
        self.dir_label.setWordWrap(True)
        self.dir_label.setStyleSheet(
            "color: #555; font-size: 10px; font-style: italic;"
        )
        outer.addWidget(self.dir_label)

        # Scroll area holds a column of thumbnail rows. `setWidgetResizable`
        # ensures the inner widget stretches with the scroll area's
        # width so thumbnails always fit.
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self.scroll, 1)

        self._inner  = QWidget()
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(8)
        self._inner_layout.addStretch(1)
        self.scroll.setWidget(self._inner)

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-weight: bold; color: #303030; margin-top: 2px;"
        )
        return lbl

    def _on_choose_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose Tiles Directory", ""
        )
        if not path:
            return
        self.current_dir = Path(path)
        self.dir_label.setText(str(self.current_dir))
        self._reload_thumbs()

    def _reload_thumbs(self) -> None:
        """Rebuild the scrollable column from PNGs in `current_dir`."""
        # Clear existing rows (keep the trailing addStretch).
        while self._inner_layout.count() > 1:
            item = self._inner_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        # Clear any armed state left over from the previous directory.
        self._disarm()

        if self.current_dir is None or not self.current_dir.is_dir():
            return

        png_paths = sorted(self.current_dir.rglob("*.png"))
        if not png_paths:
            empty = QLabel("(no .png files found)")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet(
                "color: #888; font-style: italic; padding: 8px;"
            )
            self._inner_layout.insertWidget(0, empty)
            return

        # Parse "<row-letter><col-number>" filenames — only labelled
        # tiles participate in the composite.
        labelled: list = []   # list of (path, dr, dc)
        for p in png_paths:
            m = self._LABEL_RE.match(p.stem.upper())
            if m is None:
                continue
            dr = ord(m.group(1)) - ord('A')
            dc = int(m.group(2)) - 1
            if dc < 0:
                continue
            labelled.append((p, dr, dc))

        # Normalise so the top-left labelled tile is (0, 0) — e.g. if
        # the directory only holds B2..C4, "B2" becomes (0, 0) so the
        # user still clicks a single cell and gets a compact array.
        if labelled:
            min_dr = min(t[1] for t in labelled)
            min_dc = min(t[2] for t in labelled)
            labelled = [(p, dr - min_dr, dc - min_dc)
                        for (p, dr, dc) in labelled]

            composite_card = self._build_composite_row(labelled)
            if composite_card is not None:
                self._inner_layout.insertWidget(
                    self._inner_layout.count() - 1, composite_card
                )

        for png in png_paths:
            card = self._build_thumb_row(png)
            self._inner_layout.insertWidget(
                self._inner_layout.count() - 1, card
            )

    # ── composite: "all tiles arranged by name" ───────────────────────
    def _build_composite_row(self, labelled: list):
        """Build the composite thumbnail card (all labelled tiles laid
        out in one image at their (dr, dc) positions) plus a caption.
        Clicking the card arms a BATCH placement — clicking a canvas
        cell then drops every tile with A1 on that cell and the rest
        offset by their (dr, dc). Returns the card widget or None if
        the composite can't be built."""
        pm_by_dc_dr = {}
        max_dr = 0; max_dc = 0
        for (p, dr, dc) in labelled:
            pm = QPixmap(str(p))
            if pm.isNull():
                continue
            pm_by_dc_dr[(dr, dc)] = pm
            if dr > max_dr: max_dr = dr
            if dc > max_dc: max_dc = dc
        if not pm_by_dc_dr:
            return None

        n_rows = max_dr + 1
        n_cols = max_dc + 1
        # Cell size for the composite thumbnail. Aim to keep the
        # LONGEST side of the whole composite around COMPOSITE_MAX_SIDE.
        cell = max(1, int(self.COMPOSITE_MAX_SIDE
                          / max(n_rows, n_cols)))
        W = cell * n_cols
        H = cell * n_rows

        composite = QImage(W, H, QImage.Format_ARGB32)
        composite.fill(Qt.transparent)
        painter = QPainter(composite)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        for (dr, dc), pm in pm_by_dc_dr.items():
            scaled = pm.scaled(
                cell, cell,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            # Centre each thumbnail in its cell.
            x = dc * cell + (cell - scaled.width())  // 2
            y = dr * cell + (cell - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        painter.end()
        composite_pm = QPixmap.fromImage(composite)

        # Card widget — same click-to-arm mechanic as individual tiles,
        # but the handler goes through _on_composite_clicked.
        card = ClickableFrame(on_click=None)
        card.setFrameStyle(QFrame.StyledPanel)
        card.setStyleSheet(
            "ClickableFrame { background-color: white; }"
        )
        card._on_click = lambda t=list(labelled), c=card: \
                          self._on_composite_clicked(t, c)

        v = QVBoxLayout(card)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)

        img_lbl = QLabel()
        img_lbl.setPixmap(composite_pm)
        img_lbl.setAlignment(Qt.AlignCenter)
        v.addWidget(img_lbl)

        cap = QLabel(f"Place all ({len(pm_by_dc_dr)} tiles)")
        cap.setAlignment(Qt.AlignCenter)
        cap.setStyleSheet(
            "font-size: 10px; color: #333; font-weight: bold;"
        )
        v.addWidget(cap)
        return card

    def _on_composite_clicked(self, labelled: list,
                              card: ClickableFrame) -> None:
        """Arm a batch — every labelled tile becomes a batch item
        with its own (dr, dc). Clicking the composite again disarms."""
        if self._armed_card is card:
            self._disarm()
            return

        items: list = []
        for (p, dr, dc) in labelled:
            pm = QPixmap(str(p))
            if pm.isNull():
                continue
            pm = self._pixmap_with_white_transparent(pm)
            corners_px = self._read_corners_metadata(p)
            items.append({
                'pixmap':      pm,
                'corners_px':  corners_px,
                'dr':          dr,
                'dc':          dc,
                'label':       p.stem,
                'source_path': str(Path(p).resolve()),
            })
        if not items:
            return
        self.canvas.arm_batch(items)

        # Swap highlight to the composite card.
        if self._armed_card is not None:
            self._armed_card.setStyleSheet(
                "ClickableFrame { background-color: white; }"
            )
        self._armed_card = card
        self._armed_path = None    # composite has no single path
        card.setStyleSheet(
            "ClickableFrame { background-color: #b3e5fc; "
            "border: 2px solid #0288d1; }"
        )

    def _build_thumb_row(self, png_path: Path) -> QWidget:
        """One thumbnail row: scaled preview + filename caption. The
        whole card is clickable — pressing it arms this tile for
        placement on the canvas."""
        # Create the card first, then bind the click handler so the
        # lambda can capture the card itself for the highlight swap.
        row = ClickableFrame(on_click=None)
        row.setFrameStyle(QFrame.StyledPanel)
        row.setStyleSheet(
            "ClickableFrame { background-color: white; }"
        )
        row._on_click = lambda p=png_path, r=row: self._on_tile_clicked(p, r)

        v = QVBoxLayout(row)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)

        pm = QPixmap(str(png_path))
        if not pm.isNull():
            thumb = pm.scaled(
                self.THUMB_MAX_W, self.THUMB_MAX_H,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            img_lbl = QLabel()
            img_lbl.setPixmap(thumb)
            img_lbl.setAlignment(Qt.AlignCenter)
            v.addWidget(img_lbl)
        else:
            broken = QLabel("(unreadable)")
            broken.setAlignment(Qt.AlignCenter)
            broken.setStyleSheet("color: red;")
            v.addWidget(broken)

        name_lbl = QLabel(png_path.stem)
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setStyleSheet("font-size: 10px; color: #333;")
        name_lbl.setToolTip(str(png_path))
        v.addWidget(name_lbl)

        return row

    # ── click handling ────────────────────────────────────────────────
    def _on_tile_clicked(self, png_path: Path,
                         card: ClickableFrame | None = None) -> None:
        """Arm this tile on the canvas. Clicking the same tile again
        DISARMS it (toggle). A different tile swaps the highlight."""
        if self._armed_path == png_path:
            self._disarm()
            return

        pm = QPixmap(str(png_path))
        pm = self._pixmap_with_white_transparent(pm)
        corners_px = self._read_corners_metadata(png_path)
        self.canvas.arm_tile(
            pm, corners_px,
            label=png_path.stem,
            source_path=str(Path(png_path).resolve()),
        )

        # Swap highlight.
        if self._armed_card is not None:
            self._armed_card.setStyleSheet(
                "ClickableFrame { background-color: white; }"
            )
        self._armed_card = card
        self._armed_path = png_path
        if card is not None:
            card.setStyleSheet(
                "ClickableFrame { background-color: #ffe58a; "
                "border: 2px solid #ff9500; }"
            )

    def _disarm(self) -> None:
        self.canvas.clear_pending_tile()
        if self._armed_card is not None:
            self._armed_card.setStyleSheet(
                "ClickableFrame { background-color: white; }"
            )
        self._armed_card = None
        self._armed_path = None

    @staticmethod
    def _pixmap_with_white_transparent(pm: QPixmap) -> QPixmap:
        """Return a copy of `pm` with pure-white (R=G=B=255) pixels
        made fully transparent — so the tile's paper-white background
        (drawn outside the DXF polygon by mosaic_editor's Save Tile
        Image) disappears when the tile is dropped on the canvas."""
        if pm is None or pm.isNull():
            return pm
        img = pm.toImage().convertToFormat(QImage.Format_RGBA8888)
        w, h = img.width(), img.height()
        stride = img.bytesPerLine()
        ptr = img.constBits()
        ptr.setsize(h * stride)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, stride)
        # Slice to just the pixel bytes (bytesPerLine may include row
        # padding on some Qt builds). `frombuffer` gives a READ-ONLY
        # view when it wraps Qt's `constBits()` buffer; `.copy()`
        # forces a writable, contiguous (h, w, 4) array we can mutate.
        arr = arr[:, : w * 4].reshape(h, w, 4).copy()
        # Pure-white RGB → alpha 0. Non-white pixels keep whatever
        # alpha they had (255 for opaque source).
        white = (arr[:, :, 0] == 255) & \
                (arr[:, :, 1] == 255) & \
                (arr[:, :, 2] == 255)
        arr[white, 3] = 0
        new_img = QImage(
            arr.tobytes(), w, h, w * 4,
            QImage.Format_RGBA8888,
        ).copy()
        return QPixmap.fromImage(new_img)

    @staticmethod
    def _read_corners_metadata(png_path: Path):
        """Return the `corners_px` list from a PNG's `mosaic_box_corners`
        tEXt chunk, or None if it's absent / unparseable."""
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(str(png_path)) as im:
                raw = im.info.get(TilesPanel.METADATA_KEY)
            if not raw:
                return None
            data = json.loads(raw)
            corners = data.get("corners_px")
            if (isinstance(corners, list) and len(corners) == 4
                    and all(len(p) == 2 for p in corners)):
                return corners
        except Exception:
            pass
        return None


class MosaicArrangerWindow(QMainWindow):
    """Main application window — sidebar on the left, canvas fills the
    rest."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mosaic Arranger")
        self.setGeometry(100, 100, 1200, 800)

        central = QWidget()
        self.setCentralWidget(central)

        main = QHBoxLayout(central)
        main.setSpacing(8)
        main.setContentsMargins(8, 8, 8, 8)

        self.canvas = ArrangerCanvas()
        self.sidebar = SidePanel(self.canvas, window=self)
        self.tiles_panel = TilesPanel(self.canvas)

        main.addWidget(self.sidebar)
        main.addWidget(self.canvas, 1)   # canvas gets the stretch
        main.addWidget(self.tiles_panel)

    # ── project save / load ─────────────────────────────────────────
    def save_project(self, path: str) -> None:
        """Serialize grid dims, view state, and every placed tile to
        `path` — INFORMATION ONLY. Each tile is written as its source
        PNG's absolute path + `corners_px` metadata + label + cell.
        No image bytes are embedded, so a typical .aranger file is a
        few KB regardless of how many tiles were placed.

        Load Project re-reads each source PNG from disk to rebuild the
        pixmap (and re-applies the white-transparent transform)."""
        tiles_out = []
        for tile in self.canvas.placed_tiles:
            src = tile.get('source_path')
            if not src:
                # Tiles placed before source_path tracking existed have
                # no on-disk origin we can point at — skip rather than
                # silently invent one.
                continue
            tiles_out.append({
                'source_path': str(src),
                'corners_px':  tile.get('corners_px'),
                'label':       tile.get('label', ''),
                'row':         int(tile.get('row', 0)),
                'col':         int(tile.get('col', 0)),
            })

        payload = {
            'version':   PROJECT_VERSION,
            'grid_cols': self.canvas.grid_cols,
            'grid_rows': self.canvas.grid_rows,
            'show_grid': self.canvas.show_grid,
            'zoom':      self.canvas.zoom_factor,
            'pan_x':     self.canvas.pan_x,
            'pan_y':     self.canvas.pan_y,
            'tiles':     tiles_out,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_project(self, path: str) -> tuple[int, list[str]]:
        """Load a project from `path`, REPLACING the current arrangement.

        Returns `(skipped_count, warnings)`. `skipped_count` is the
        number of placed tiles that couldn't be rebuilt (missing or
        unreadable source PNG); `warnings` lists their source paths."""
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Not an arranger project file.")
        version = payload.get('version', 0)
        if version > PROJECT_VERSION:
            raise ValueError(
                f"Project file version {version} is newer than this "
                f"arranger (max supported: {PROJECT_VERSION})."
            )

        cols = int(payload.get('grid_cols', self.canvas.DEFAULT_COLS))
        rows = int(payload.get('grid_rows', self.canvas.DEFAULT_ROWS))
        self.canvas.set_grid_dims(cols, rows)
        self.canvas.set_show_grid(bool(payload.get('show_grid', True)))
        self.canvas.zoom_factor = float(payload.get('zoom', 1.0))
        self.canvas.pan_x       = float(payload.get('pan_x', 0.0))
        self.canvas.pan_y       = float(payload.get('pan_y', 0.0))

        new_tiles: list[dict] = []
        warnings: list[str]   = []
        skipped = 0
        for t in payload.get('tiles', []):
            src = t.get('source_path')
            if not src:
                skipped += 1
                continue
            if not Path(src).exists():
                warnings.append(f"Missing tile PNG: {src}")
                skipped += 1
                continue
            pm = QPixmap(str(src))
            if pm.isNull():
                warnings.append(f"Unreadable tile PNG: {src}")
                skipped += 1
                continue
            # Same on-load transform as the tiles panel applies to a
            # freshly-armed tile, so a round-tripped project looks
            # identical to placing everything again by hand.
            pm = TilesPanel._pixmap_with_white_transparent(pm)
            new_tiles.append({
                'pixmap':      pm,
                'corners_px':  t.get('corners_px'),
                'label':       t.get('label', ''),
                'source_path': str(src),
                'row':         int(t.get('row', 0)),
                'col':         int(t.get('col', 0)),
            })
        self.canvas.placed_tiles = new_tiles
        self.canvas.clear_pending_tile()
        # Fresh scene → no placed-tile selection carried over.
        self.canvas.selected_placed_index = -1
        self.canvas.update()
        return skipped, warnings


def main():
    app = QApplication(sys.argv)
    win = MosaicArrangerWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
