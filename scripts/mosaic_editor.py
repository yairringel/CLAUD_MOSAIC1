"""mosaic_editor.py — multi-layer mosaic viewer/editor.

Loads any mix of:
  * .csv files matching the shared save_array schema
    (polygon_id, coordinates, color_r..color_a — colors 0..1).
  * .mosaic ZIP bundles from image_strech.py's Save Mosaic
    (polygons.csv + background.png). The bundle's background image is
    consumed once at load time to sample each polygon's image fill, then
    discarded — matches the user's model of "the source bg is only used
    to bake the polygon fills."
  * Plain image files as a SHARED canvas background behind every layer.
    Only one such background exists at a time; loading a new one
    replaces the previous.

Canvas: mouse-wheel zoom around cursor + right-drag pan, grid overlay
with adjustable cell size / color / thickness. Polygons can be
click-selected, dragged, and deleted individually. Loaded batches
(groups) can be hidden or removed en masse from the Layers panel.
"""

import sys
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFrame, QLabel, QPushButton, QFileDialog, QCheckBox, QSpinBox,
    QDoubleSpinBox, QMessageBox, QScrollArea, QColorDialog, QSizePolicy,
)
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QPixmap, QImage, QPolygonF,
)


# ═══════════════════════════════════════════════════════════════════════
# CSV / .mosaic loaders
# ═══════════════════════════════════════════════════════════════════════

def parse_polygon_csv(csv_text: str):
    """Parse the shared save_array CSV schema. Returns a list of dicts:
    {'points': [(x,y), ...], 'color': QColor}. Colors default to
    solid black if a row is malformed."""
    out = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        try:
            pts = json.loads(row['coordinates'])
        except Exception:
            continue
        if not isinstance(pts, list) or len(pts) < 3:
            continue
        points = [(float(p[0]), float(p[1])) for p in pts
                  if isinstance(p, (list, tuple)) and len(p) >= 2]
        if len(points) < 3:
            continue
        try:
            r = int(round(float(row.get('color_r', 0)) * 255))
            g = int(round(float(row.get('color_g', 0)) * 255))
            b = int(round(float(row.get('color_b', 0)) * 255))
            a = int(round(float(row.get('color_a', 1)) * 255))
        except Exception:
            r, g, b, a = 0, 0, 0, 255
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        a = max(0, min(255, a))
        out.append({'points': points, 'color': QColor(r, g, b, a)})
    return out


def load_mosaic_bundle(path: str):
    """Read a .mosaic ZIP. Returns (polygons, background_rgba_np) where:
      * polygons is the parsed CSV entries (same shape as parse_polygon_csv)
      * background_rgba_np is a numpy uint8 array (H, W, 4) or None
    """
    with zipfile.ZipFile(path, 'r') as zf:
        names = zf.namelist()
        # polygons.csv is required; background.png is optional (though
        # standard bundles always include it).
        csv_name = next((n for n in names if n.lower().endswith('polygons.csv')), None)
        if csv_name is None:
            raise ValueError("Bundle is missing polygons.csv")
        polys = parse_polygon_csv(zf.read(csv_name).decode('utf-8'))
        bg_np = None
        bg_name = next((n for n in names if n.lower().endswith('background.png')), None)
        if bg_name is not None:
            bg_bytes = zf.read(bg_name)
            qimg = QImage.fromData(bg_bytes)
            if not qimg.isNull():
                qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
                w, h = qimg.width(), qimg.height()
                stride = qimg.bytesPerLine()
                ptr = qimg.constBits()
                ptr.setsize(h * stride)
                arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, stride)
                bg_np = arr[:, : w * 4].reshape(h, w, 4).copy()
        return polys, bg_np


def build_image_fill(polygon_points, bg_rgba: np.ndarray):
    """Extract the polygon's pixels from the background bitmap and bake in
    a polygon-shaped alpha mask. Returns (fill_qimage, bbox_tuple) or
    (None, None) if the polygon lies outside the image.

    Mask is built with PIL's ImageDraw.polygon — reliable across Qt/PIL
    versions and doesn't depend on any painter-to-Grayscale8 support."""
    if bg_rgba is None or bg_rgba.size == 0:
        return None, None
    H, W = bg_rgba.shape[:2]
    xs = [p[0] for p in polygon_points]; ys = [p[1] for p in polygon_points]
    x0 = int(np.floor(min(xs))); y0 = int(np.floor(min(ys)))
    x1 = int(np.ceil (max(xs))); y1 = int(np.ceil (max(ys)))
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W, x1); y1 = min(H, y1)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None, None

    crop = bg_rgba[y0:y1, x0:x1].copy()

    # Build a single-channel alpha mask via PIL. Polygon coords are
    # remapped into the crop's local (0..w, 0..h) space. ImageDraw
    # accepts floats and rasterises the whole polygon in one call.
    from PIL import Image as _PILImage, ImageDraw as _ImageDraw
    mask_img = _PILImage.new('L', (w, h), 0)
    poly_local = [(float(px - x0), float(py - y0))
                  for (px, py) in polygon_points]
    _ImageDraw.Draw(mask_img).polygon(poly_local, fill=255)
    mask_arr = np.array(mask_img, dtype=np.uint8)   # (h, w) uint8

    # Multiply the crop's existing alpha channel by the mask. If the
    # source background PNG had no alpha (fully opaque), this simply
    # copies mask into the alpha channel.
    crop[:, :, 3] = (
        (crop[:, :, 3].astype(np.uint16) * mask_arr.astype(np.uint16)) // 255
    ).astype(np.uint8)

    # Convert to a QImage that owns its data (copy() detaches the
    # numpy buffer so it survives after `crop` goes out of scope).
    fill_qimg = QImage(crop.data, w, h, w * 4,
                       QImage.Format_RGBA8888).copy()
    return fill_qimg, (x0, y0, w, h)


# ═══════════════════════════════════════════════════════════════════════
# Canvas
# ═══════════════════════════════════════════════════════════════════════

class MosaicCanvas(QWidget):
    """Zoom + pan + grid + polygon rendering + click/drag/delete."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 500)
        self.setStyleSheet("background-color: #303030;")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        # Per-polygon: {'points', 'fill_type', 'color', 'fill_qimg',
        #               'fill_bbox', 'group_id', 'visible'}
        # fill_type ∈ {'solid', 'image', 'none'}
        self.polygons: list[dict] = []

        # Per-group: {'id', 'name', 'kind', 'visible'}
        self.groups: list[dict]   = []
        self._next_group_id = 1

        self.canvas_background: QPixmap | None = None

        # View transform.
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Grid.
        self.grid_enabled = False
        self.grid_size_percent = 10.0          # % of canvas background width
        self.grid_size_world = 100.0           # falls back when no background
        self.grid_color = QColor(255, 105, 180)
        self.grid_thickness = 2
        self.grid_offset_x = 0.0
        self.grid_offset_y = 0.0

        # Interaction state.
        self.selected_index = -1
        self._dragging_polygon = False
        self._drag_start_world = (0.0, 0.0)
        self._drag_start_points: list[tuple[float, float]] = []
        self._drag_start_bbox: tuple | None = None
        self._panning = False
        self._pan_start_screen = (0, 0)
        self._pan_start_offset = (0.0, 0.0)

    # ── layers management ─────────────────────────────────────────────
    def add_group(self, name: str, kind: str) -> int:
        gid = self._next_group_id
        self._next_group_id += 1
        self.groups.append({'id': gid, 'name': name, 'kind': kind,
                            'visible': True})
        return gid

    def group_visible(self, gid: int) -> bool:
        for g in self.groups:
            if g['id'] == gid:
                return g['visible']
        return True

    def set_group_visible(self, gid: int, visible: bool) -> None:
        for g in self.groups:
            if g['id'] == gid:
                g['visible'] = visible
                break
        self.update()

    def delete_group(self, gid: int) -> int:
        before = len(self.polygons)
        self.polygons = [p for p in self.polygons if p['group_id'] != gid]
        self.groups   = [g for g in self.groups   if g['id']       != gid]
        if self.selected_index >= len(self.polygons):
            self.selected_index = -1
        self.update()
        return before - len(self.polygons)

    # ── loaders ───────────────────────────────────────────────────────
    def load_csv(self, path: str) -> int:
        text = Path(path).read_text(encoding='utf-8')
        polys = parse_polygon_csv(text)
        gid = self.add_group(Path(path).name, 'csv')
        for p in polys:
            self.polygons.append({
                'points':    p['points'],
                'fill_type': ('solid' if p['color'].alpha() > 0 else 'none'),
                'color':     p['color'],
                'fill_qimg': None,
                'fill_bbox': None,
                'group_id':  gid,
            })
        self.selected_index = -1
        self.update()
        return len(polys)

    def load_mosaic(self, path: str) -> int:
        """Load a .mosaic bundle by sampling each polygon's pixels from
        the bundled background.png (via build_image_fill), then
        discarding the source image — matches the user's model of
        'the source background is only used to bake the per-polygon
        fills; only a Load Image background persists behind the whole
        canvas.'"""
        polys, bg_np = load_mosaic_bundle(path)
        gid = self.add_group(Path(path).name, 'mosaic')
        for p in polys:
            fill_qimg, fill_bbox = (None, None)
            if bg_np is not None:
                fill_qimg, fill_bbox = build_image_fill(p['points'], bg_np)
            self.polygons.append({
                'points':    p['points'],
                'fill_type': ('image' if fill_qimg is not None else
                              ('solid' if p['color'].alpha() > 0 else 'none')),
                'color':     p['color'],
                'fill_qimg': fill_qimg,
                'fill_bbox': fill_bbox,
                'group_id':  gid,
            })
        self.selected_index = -1
        self.update()
        return len(polys)

    def load_canvas_background(self, path: str) -> None:
        pm = QPixmap(path)
        if pm.isNull():
            raise ValueError(f"Could not load image: {path}")
        self.canvas_background = pm
        self.update()

    def clear_canvas_background(self) -> None:
        self.canvas_background = None
        self.update()

    # ── coord transforms ──────────────────────────────────────────────
    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        return (x * self.zoom_factor + self.pan_x,
                y * self.zoom_factor + self.pan_y)

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return ((sx - self.pan_x) / self.zoom_factor,
                (sy - self.pan_y) / self.zoom_factor)

    def reset_view(self) -> None:
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    # ── painting ──────────────────────────────────────────────────────
    def paintEvent(self, _ev):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(48, 48, 48))

        # 1. Canvas background — drawn first, behind all polygons.
        if self.canvas_background is not None:
            bg = self.canvas_background
            sx, sy = self.world_to_screen(0, 0)
            sw = bg.width()  * self.zoom_factor
            sh = bg.height() * self.zoom_factor
            painter.drawPixmap(QRectF(sx, sy, sw, sh), bg,
                               QRectF(0, 0, bg.width(), bg.height()))

        # 2. Grid overlay.
        if self.grid_enabled:
            self._draw_grid(painter)

        # 3. Polygons.
        gid_visible = {g['id']: g['visible'] for g in self.groups}
        for i, poly in enumerate(self.polygons):
            if not gid_visible.get(poly['group_id'], True):
                continue
            self._draw_polygon(painter, i, poly)

        painter.end()

    def _draw_grid(self, painter: QPainter) -> None:
        # Determine cell size in world units. If a canvas background is
        # loaded, tie the % input to its width; else fall back to a
        # standalone world unit.
        if self.canvas_background is not None:
            cell = self.canvas_background.width() * (self.grid_size_percent / 100.0)
        else:
            cell = self.grid_size_world
        cell = max(1.0, cell)

        painter.setPen(QPen(self.grid_color, self.grid_thickness))
        painter.setBrush(Qt.NoBrush)

        # Visible-world extent given the widget size and current transform.
        w0, h0 = self.width(), self.height()
        wx0, wy0 = self.screen_to_world(0, 0)
        wx1, wy1 = self.screen_to_world(w0, h0)

        # First / last vertical lines within view.
        first_col = int(np.floor((wx0 - self.grid_offset_x) / cell))
        last_col  = int(np.ceil ((wx1 - self.grid_offset_x) / cell))
        for k in range(first_col, last_col + 1):
            x_w = self.grid_offset_x + k * cell
            sx, _ = self.world_to_screen(x_w, 0)
            painter.drawLine(int(sx), 0, int(sx), h0)

        first_row = int(np.floor((wy0 - self.grid_offset_y) / cell))
        last_row  = int(np.ceil ((wy1 - self.grid_offset_y) / cell))
        for k in range(first_row, last_row + 1):
            y_w = self.grid_offset_y + k * cell
            _, sy = self.world_to_screen(0, y_w)
            painter.drawLine(0, int(sy), w0, int(sy))

    def _draw_polygon(self, painter: QPainter, i: int, poly: dict) -> None:
        pts = poly['points']
        # Screen-space QPolygonF for stroke/select highlighting.
        qpoly = QPolygonF([self._qpointf(*self.world_to_screen(x, y))
                           for x, y in pts])

        if poly['fill_type'] == 'image' and poly['fill_qimg'] is not None:
            # Draw the pre-masked RGBA image at its bbox origin. Use
            # QRectF so we don't truncate sub-pixel positions to int
            # (that was showing up as a systematic offset between the
            # image content and the polygon outline at high zoom).
            x0, y0, w, h = poly['fill_bbox']
            sx, sy = self.world_to_screen(x0, y0)
            sw = w * self.zoom_factor
            sh = h * self.zoom_factor
            painter.drawImage(QRectF(sx, sy, sw, sh), poly['fill_qimg'])
        elif poly['fill_type'] == 'solid':
            painter.setBrush(QBrush(poly['color']))
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(qpoly)
        # 'none' → nothing filled

        # Outline (thicker + red when selected)
        if i == self.selected_index:
            painter.setPen(QPen(QColor(255, 30, 30), 3))
        else:
            painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPolygon(qpoly)

    @staticmethod
    def _qpointf(x: float, y: float):
        from PyQt5.QtCore import QPointF
        return QPointF(x, y)

    # ── zoom / pan ────────────────────────────────────────────────────
    def wheelEvent(self, ev):
        # Zoom around cursor.
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else (1.0 / 1.25)
        # Keep world point under cursor fixed.
        cx, cy = ev.x(), ev.y()
        wx, wy = self.screen_to_world(cx, cy)
        self.zoom_factor = max(0.02, min(100.0, self.zoom_factor * factor))
        # Reset pan so (wx, wy) still maps to (cx, cy).
        self.pan_x = cx - wx * self.zoom_factor
        self.pan_y = cy - wy * self.zoom_factor
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            # Right-drag pans.
            self._panning = True
            self._pan_start_screen = (ev.x(), ev.y())
            self._pan_start_offset = (self.pan_x, self.pan_y)
            self.setCursor(Qt.ClosedHandCursor)
            return
        if ev.button() == Qt.LeftButton:
            wx, wy = self.screen_to_world(ev.x(), ev.y())
            idx = self._pick_polygon_at(wx, wy)
            self.selected_index = idx
            if idx >= 0:
                self._dragging_polygon = True
                self._drag_start_world = (wx, wy)
                self._drag_start_points = list(self.polygons[idx]['points'])
                # Snapshot the fill bbox at drag start too — mouseMove
                # must apply the *total* dx/dy since drag-start to this
                # snapshot, NOT accumulate onto the already-shifted
                # bbox (which would move the image faster than the
                # polygon and it'd drift away).
                fb = self.polygons[idx].get('fill_bbox')
                self._drag_start_bbox = tuple(fb) if fb is not None else None
                self.setCursor(Qt.ClosedHandCursor)
            self.update()

    def mouseMoveEvent(self, ev):
        if self._panning:
            dx = ev.x() - self._pan_start_screen[0]
            dy = ev.y() - self._pan_start_screen[1]
            self.pan_x = self._pan_start_offset[0] + dx
            self.pan_y = self._pan_start_offset[1] + dy
            self.update()
            return
        if self._dragging_polygon and self.selected_index >= 0:
            wx, wy = self.screen_to_world(ev.x(), ev.y())
            dx = wx - self._drag_start_world[0]
            dy = wy - self._drag_start_world[1]
            poly = self.polygons[self.selected_index]
            poly['points'] = [(px + dx, py + dy)
                              for (px, py) in self._drag_start_points]
            # Move fill bbox with the polygon. Always compute from the
            # ORIGINAL bbox snapshotted at drag-start, not from the
            # current (already-shifted) one — otherwise every frame
            # accumulates the delta twice and the image drifts.
            if self._drag_start_bbox is not None:
                x0, y0, w, h = self._drag_start_bbox
                poly['fill_bbox'] = (x0 + dx, y0 + dy, w, h)
            self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        if ev.button() == Qt.LeftButton and self._dragging_polygon:
            self._dragging_polygon = False
            self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace) \
                and self.selected_index >= 0:
            del self.polygons[self.selected_index]
            self.selected_index = -1
            self.update()

    def _pick_polygon_at(self, wx: float, wy: float) -> int:
        # Iterate from top (last drawn) to bottom, respecting group vis.
        gid_visible = {g['id']: g['visible'] for g in self.groups}
        for i in range(len(self.polygons) - 1, -1, -1):
            poly = self.polygons[i]
            if not gid_visible.get(poly['group_id'], True):
                continue
            if self._point_in_polygon(wx, wy, poly['points']):
                return i
        return -1

    @staticmethod
    def _point_in_polygon(x: float, y: float, pts) -> bool:
        n = len(pts)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]; xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside


# ═══════════════════════════════════════════════════════════════════════
# Sidebar / main window
# ═══════════════════════════════════════════════════════════════════════

class LayersPanel(QScrollArea):
    """Vertical scroll list of loaded groups with per-group visibility
    checkbox and delete button. Rebuilt whenever groups change."""

    def __init__(self, canvas: MosaicCanvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)

        self._inner  = QWidget()
        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        self._layout.addStretch()
        self.setWidget(self._inner)

    def rebuild(self) -> None:
        # Remove old rows.
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        for g in self.canvas.groups:
            row = QFrame()
            row.setFrameStyle(QFrame.StyledPanel)
            row.setStyleSheet("background-color: #f5f5f5;")
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(4, 2, 4, 2)
            row_lay.setSpacing(4)

            cb = QCheckBox()
            cb.setChecked(g['visible'])
            cb.setToolTip("Show / hide this layer.")
            cb.toggled.connect(
                lambda checked, gid=g['id']:
                    self._on_visibility_toggled(gid, checked)
            )
            row_lay.addWidget(cb)

            lbl = QLabel(f"[{g['kind']}] {g['name']}")
            lbl.setStyleSheet("font-size: 11px;")
            lbl.setWordWrap(False)
            row_lay.addWidget(lbl, 1)

            del_btn = QPushButton("×")
            del_btn.setFixedSize(22, 22)
            del_btn.setToolTip("Delete every polygon in this layer.")
            del_btn.clicked.connect(
                lambda _=False, gid=g['id']: self._on_delete_group(gid)
            )
            row_lay.addWidget(del_btn)

            self._layout.insertWidget(self._layout.count() - 1, row)

    def _on_visibility_toggled(self, gid: int, checked: bool) -> None:
        self.canvas.set_group_visible(gid, checked)

    def _on_delete_group(self, gid: int) -> None:
        self.canvas.delete_group(gid)
        self.rebuild()


class MosaicEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mosaic Editor")
        self.setGeometry(80, 60, 1300, 850)

        central = QWidget(); self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setContentsMargins(6, 6, 6, 6); main.setSpacing(6)

        # ── canvas ────────────────────────────────────────────────────
        self.canvas = MosaicCanvas()
        canvas_frame = QFrame()
        canvas_frame.setFrameStyle(QFrame.StyledPanel)
        cf_lay = QVBoxLayout(canvas_frame)
        cf_lay.setContentsMargins(0, 0, 0, 0)
        cf_lay.addWidget(self.canvas)

        # ── sidebar ───────────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setFrameStyle(QFrame.StyledPanel)
        sidebar.setMinimumWidth(260)
        sidebar.setMaximumWidth(320)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(6, 6, 6, 6); sb.setSpacing(6)

        sb.addWidget(self._section_label("Load"))
        btn_csv = QPushButton("Load CSV (append)")
        btn_csv.clicked.connect(self.on_load_csv)
        sb.addWidget(btn_csv)

        btn_mosaic = QPushButton("Load .mosaic (append)")
        btn_mosaic.clicked.connect(self.on_load_mosaic)
        sb.addWidget(btn_mosaic)

        btn_img = QPushButton("Load Image (canvas background)")
        btn_img.clicked.connect(self.on_load_image)
        sb.addWidget(btn_img)

        btn_img_clear = QPushButton("Clear Canvas Background")
        btn_img_clear.clicked.connect(self.canvas.clear_canvas_background)
        sb.addWidget(btn_img_clear)

        sb.addSpacing(8)
        sb.addWidget(self._section_label("View"))
        btn_reset_view = QPushButton("Reset Zoom / Pan")
        btn_reset_view.clicked.connect(self.canvas.reset_view)
        sb.addWidget(btn_reset_view)

        sb.addSpacing(8)
        sb.addWidget(self._section_label("Grid"))
        self.grid_cb = QCheckBox("Show grid")
        self.grid_cb.toggled.connect(self.on_grid_toggled)
        sb.addWidget(self.grid_cb)

        # Cell size — a single spinbox: uses % of the canvas
        # background's width when one is loaded, else a plain world unit
        # (100 by default).
        row = QHBoxLayout()
        row.addWidget(QLabel("Cell size:"))
        self.grid_size_spin = QDoubleSpinBox()
        self.grid_size_spin.setRange(0.5, 10000.0)
        self.grid_size_spin.setValue(10.0)
        self.grid_size_spin.setDecimals(1)
        self.grid_size_spin.setSuffix(" %/px")
        self.grid_size_spin.setToolTip(
            "When a canvas background is loaded: % of its width per cell.\n"
            "Otherwise: cell edge in world units."
        )
        self.grid_size_spin.valueChanged.connect(self.on_grid_size_changed)
        row.addWidget(self.grid_size_spin)
        sb.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Thickness:"))
        self.grid_thickness_spin = QSpinBox()
        self.grid_thickness_spin.setRange(1, 20)
        self.grid_thickness_spin.setValue(2)
        self.grid_thickness_spin.valueChanged.connect(
            self.on_grid_thickness_changed
        )
        row.addWidget(self.grid_thickness_spin)
        sb.addLayout(row)

        row = QHBoxLayout()
        self._grid_color_btn = QPushButton("Grid color")
        self._grid_color_btn.clicked.connect(self.on_grid_color)
        self._refresh_grid_color_btn()
        row.addWidget(self._grid_color_btn)
        sb.addLayout(row)

        sb.addSpacing(8)
        sb.addWidget(self._section_label("Layers"))
        self.layers_panel = LayersPanel(self.canvas)
        sb.addWidget(self.layers_panel, 1)

        # ── assemble ──────────────────────────────────────────────────
        main.addWidget(sidebar)
        main.addWidget(canvas_frame, 1)

        self.statusBar().showMessage(
            "Load a CSV / .mosaic / image to start. "
            "Scroll to zoom, right-drag to pan, click to select, Delete to remove."
        )

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: bold; color: #303030; margin-top: 2px;")
        return lbl

    # ── handlers ──────────────────────────────────────────────────────
    def on_load_csv(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Mosaic CSV(s)", "",
            "CSV files (*.csv);;All files (*.*)"
        )
        if not paths:
            return
        total = 0
        for path in paths:
            try:
                n = self.canvas.load_csv(path)
                total += n
            except Exception as e:
                QMessageBox.warning(
                    self, "Load CSV",
                    f"Failed to load {Path(path).name}: {type(e).__name__}: {e}"
                )
        self.layers_panel.rebuild()
        self.statusBar().showMessage(
            f"Loaded {total} polygon(s) from {len(paths)} CSV file(s)."
        )

    def on_load_mosaic(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load .mosaic bundle(s)", "",
            "Mosaic bundle (*.mosaic *.zip);;All files (*.*)"
        )
        if not paths:
            return
        total = 0
        for path in paths:
            try:
                n = self.canvas.load_mosaic(path)
                total += n
            except Exception as e:
                QMessageBox.warning(
                    self, "Load Mosaic",
                    f"Failed to load {Path(path).name}: {type(e).__name__}: {e}"
                )
        self.layers_panel.rebuild()
        self.statusBar().showMessage(
            f"Loaded {total} polygon(s) from {len(paths)} .mosaic bundle(s). "
            f"Source backgrounds were consumed to fill each polygon."
        )

    def on_load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Canvas Background", "",
            "Image files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)"
        )
        if not path:
            return
        try:
            self.canvas.load_canvas_background(path)
        except Exception as e:
            QMessageBox.warning(
                self, "Load Image",
                f"Failed to load image: {type(e).__name__}: {e}"
            )
            return
        self.statusBar().showMessage(
            f"Canvas background: {Path(path).name}"
        )

    def on_grid_toggled(self, checked: bool) -> None:
        self.canvas.grid_enabled = checked
        self.canvas.update()

    def on_grid_size_changed(self, v: float) -> None:
        # Reinterpret automatically based on whether a canvas background
        # is loaded (see MosaicCanvas._draw_grid).
        self.canvas.grid_size_percent = v
        self.canvas.grid_size_world = v
        self.canvas.update()

    def on_grid_thickness_changed(self, v: int) -> None:
        self.canvas.grid_thickness = v
        self.canvas.update()

    def on_grid_color(self) -> None:
        c = QColorDialog.getColor(self.canvas.grid_color, self, "Grid color")
        if c.isValid():
            self.canvas.grid_color = c
            self._refresh_grid_color_btn()
            self.canvas.update()

    def _refresh_grid_color_btn(self) -> None:
        c = self.canvas.grid_color
        self._grid_color_btn.setStyleSheet(
            f"background-color: rgb({c.red()},{c.green()},{c.blue()}); "
            "color: white; border: 1px solid #444;"
        )


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    win = MosaicEditorWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
