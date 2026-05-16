#!/usr/bin/env python3
"""
Randomiser - Polygon Array Randomizer
Load a CSV polygon array (same format as mosaic_cutter.py), subdivide each
polygon edge with extra control points, then randomly perturb every vertex.

No-overlap guarantee:
  Adjacent polygons share edges whose vertices sit at identical world positions.
  Every unique vertex position gets exactly ONE random displacement vector, so
  shared boundaries always move together and polygons can never overlap.
"""

import sys
import csv
import ast
import math
import random

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel,
    QSpinBox, QDoubleSpinBox,
    QGroupBox, QMessageBox,
)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF

from shapely.geometry import Polygon
from shapely.validation import make_valid


# ── Tolerance for snapping nearby vertices (covers float-string roundtrip) ──
SNAP_TOL = 1e-3


# ════════════════════════════ CSV I/O ════════════════════════════

def load_polygons_csv(filename):
    """
    Load polygons from a CSV file using the same column layout as
    mosaic_cutter.py:  coordinates, color_r, color_g, color_b, color_a, color_hex
    """
    polygons, colors = [], []
    with open(filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords_str = row.get('coordinates', row.get('polygon_coords', ''))
            coords_str = coords_str.strip('"\'')
            try:
                coord_list = ast.literal_eval(coords_str)
                coords = [(float(p[0]), float(p[1])) for p in coord_list]
            except Exception:
                continue
            if len(coords) < 3:
                continue

            poly = Polygon(coords)
            if not poly.is_valid:
                poly = make_valid(poly)
            polygons.append(poly)

            # Parse colour (supports separate r/g/b/a columns or combined)
            if 'color_r' in row and 'color_g' in row and 'color_b' in row:
                r = float(row['color_r'])
                g = float(row['color_g'])
                b = float(row['color_b'])
                a = float(row.get('color_a', 1.0))
                r = int(r * 255) if r <= 1.0 else int(r)
                g = int(g * 255) if g <= 1.0 else int(g)
                b = int(b * 255) if b <= 1.0 else int(b)
                a = int(a * 255) if a <= 1.0 else int(a)
                colors.append(QColor(r, g, b, a))
            elif 'color' in row:
                color_str = row['color'].strip('()[]"\'')
                parts = color_str.replace(',', ' ').split()
                if len(parts) >= 3:
                    r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                    if r <= 1.0 and g <= 1.0 and b <= 1.0:
                        r, g, b = int(r * 255), int(g * 255), int(b * 255)
                    colors.append(QColor(int(r), int(g), int(b)))
                else:
                    colors.append(QColor(128, 128, 128))
            else:
                colors.append(QColor(128, 128, 128))

    return polygons, colors


def save_polygons_csv(filename, polygons, colors):
    """Save polygons back to the same CSV format."""
    with open(filename, 'w', newline='') as f:
        fieldnames = ['coordinates', 'color_r', 'color_g', 'color_b', 'color_a', 'color_hex']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for poly, color in zip(polygons, colors):
            if poly is None:
                continue
            geoms = list(poly.geoms) if hasattr(poly, 'geoms') else [poly]
            for g in geoms:
                if not hasattr(g, 'exterior'):
                    continue
                coords = list(g.exterior.coords[:-1])
                writer.writerow({
                    'coordinates': str(coords),
                    'color_r':    color.red()   / 255.0,
                    'color_g':    color.green() / 255.0,
                    'color_b':    color.blue()  / 255.0,
                    'color_a':    color.alpha() / 255.0,
                    'color_hex':  color.name(),
                })


# ════════════════════════ Randomization ══════════════════════════

def _vkey(x, y):
    """Snap-round a vertex to a hashable key."""
    return (round(x / SNAP_TOL), round(y / SNAP_TOL))


def _ekey(p1, p2):
    """Canonical (sorted) key for an edge defined by two vertex keys."""
    k1, k2 = _vkey(*p1), _vkey(*p2)
    return (k1, k2) if k1 < k2 else (k2, k1)


def _subdivide_with_map(poly, edge_ctrl_map):
    """
    Return the polygon exterior as a flat list of (x, y) tuples.
    The number of intermediate vertices on each edge is looked up from
    edge_ctrl_map (keyed by canonical edge key).  Missing edges default to 0.
    """
    coords = list(poly.exterior.coords[:-1])
    n = len(coords)
    out = []
    for i in range(n):
        p1 = coords[i]
        p2 = coords[(i + 1) % n]
        n_ctrl = edge_ctrl_map.get(_ekey(p1, p2), 0)
        out.append(p1)
        for k in range(1, n_ctrl + 1):
            t = k / (n_ctrl + 1)
            out.append((p1[0] + t * (p2[0] - p1[0]),
                        p1[1] + t * (p2[1] - p1[1])))
    return out


def randomize_polygons(polygons, n_ctrl_max, strength):
    """
    Subdivide edges and randomly perturb the resulting vertices.

    Each edge in the mesh is assigned a random number of control points
    (0 … n_ctrl_max).  Because adjacent polygons share the same edge object,
    the subdivision is always identical on both sides — shared boundaries
    deform together and polygons can never overlap.

    Parameters
    ----------
    polygons   : list of shapely.Polygon
    n_ctrl_max : int   – maximum extra vertices per edge (each edge picks
                         its own random value in [0, n_ctrl_max])
    strength   : float – maximum displacement distance (world units)

    Returns
    -------
    list of shapely.Polygon  (None entries where a polygon became invalid)
    """
    # Step 1 – assign a random control-point count to every unique edge.
    #   Keyed by canonical (sorted) pair of vertex keys so shared edges in
    #   adjacent polygons always use the same subdivision count.
    edge_ctrl_map = {}
    for poly in polygons:
        if poly is None or not hasattr(poly, 'exterior'):
            continue
        coords = list(poly.exterior.coords[:-1])
        n = len(coords)
        for i in range(n):
            ek = _ekey(coords[i], coords[(i + 1) % n])
            if ek not in edge_ctrl_map:
                edge_ctrl_map[ek] = random.randint(0, n_ctrl_max)

    # Step 2 – expand each polygon using the per-edge control-point counts
    expanded = []
    for poly in polygons:
        if poly is None or not hasattr(poly, 'exterior'):
            expanded.append(None)
            continue
        expanded.append(_subdivide_with_map(poly, edge_ctrl_map))

    # Step 2 – collect the set of original corner keys so they always get a
    #           non-zero displacement (otherwise polygons with all-shared corners
    #           can appear not to move).
    corner_keys = set()
    for poly in polygons:
        if poly is None or not hasattr(poly, 'exterior'):
            continue
        for x, y in list(poly.exterior.coords[:-1]):
            corner_keys.add(_vkey(x, y))

    # Step 3 – assign ONE random displacement to every unique vertex position.
    #   Corners always receive a displacement with magnitude > 0 so they are
    #   visibly moved even when shared by several polygons.
    displacements = {}
    for coords in expanded:
        if coords is None:
            continue
        for x, y in coords:
            k = _vkey(x, y)
            if k not in displacements:
                angle = random.uniform(0.0, 2.0 * math.pi)
                if k in corner_keys:
                    # Corners: use full strength range (never zero)
                    mag = random.uniform(strength * 0.3, strength)
                else:
                    mag = random.uniform(0.0, strength)
                displacements[k] = (mag * math.cos(angle), mag * math.sin(angle))

    # Step 4 – apply displacements and rebuild Shapely polygons.
    result = []
    for coords in expanded:
        if coords is None:
            result.append(None)
            continue
        moved = []
        for x, y in coords:
            dx, dy = displacements[_vkey(x, y)]
            moved.append((x + dx, y + dy))
        try:
            p = Polygon(moved)
            if not p.is_valid:
                p = make_valid(p)
            # make_valid can return MultiPolygon / GeometryCollection when the
            # displaced ring self-intersects.  Recover the largest simple piece.
            if not isinstance(p, Polygon):
                candidates = []
                geoms = list(p.geoms) if hasattr(p, 'geoms') else []
                for g in geoms:
                    if isinstance(g, Polygon):
                        candidates.append(g)
                    elif hasattr(g, 'geoms'):
                        candidates.extend(x for x in g.geoms if isinstance(x, Polygon))
                p = max(candidates, key=lambda g: g.area) if candidates else None
            result.append(p)
        except Exception:
            result.append(None)

    return result


def scale_polygons(polygons, max_downscale_pct):
    """
    Randomly scale each polygon down toward its centroid.

    Each polygon picks an independent random scale factor in
    [1 - max_downscale_pct/100, 1].  Scaling toward the centroid never
    causes overlaps – it only creates gaps between neighbours.

    Parameters
    ----------
    polygons         : list of shapely.Polygon or None
    max_downscale_pct: float – maximum downscale in percent (0 = no change,
                               100 = collapse to a point)
    Returns
    -------
    list of shapely.Polygon or None
    """
    if max_downscale_pct <= 0:
        return list(polygons)
    result = []
    for poly in polygons:
        if poly is None or not hasattr(poly, 'exterior'):
            result.append(poly)
            continue
        scale = random.uniform(1.0 - max_downscale_pct / 100.0, 1.0)
        cx, cy = poly.centroid.x, poly.centroid.y
        scaled_coords = [
            (cx + (x - cx) * scale, cy + (y - cy) * scale)
            for x, y in poly.exterior.coords[:-1]
        ]
        try:
            p = Polygon(scaled_coords)
            result.append(p if p.is_valid else make_valid(p))
        except Exception:
            result.append(poly)  # fall back to unscaled
    return result


# ═══════════════════════════ Canvas ══════════════════════════════

class RandomiserCanvas(QWidget):
    """Zoomable/pannable polygon viewer."""

    def __init__(self):
        super().__init__()
        self.polygons: list = []
        self.colors:   list = []
        self.scale  = 1.0
        self.pan_x  = 0.0
        self.pan_y  = 0.0
        self._last_mouse = None
        self._panning    = False
        self.setMinimumSize(700, 500)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #f0f0f0;")

    # ── public API ───────────────────────────────────────────────

    def set_polygons(self, polygons, colors):
        self.polygons = [p for p in polygons if p is not None and hasattr(p, 'exterior')]
        self.colors   = [c for p, c in zip(polygons, colors)
                         if p is not None and hasattr(p, 'exterior')]
        self._fit_view()
        self.update()

    # ── internal ─────────────────────────────────────────────────

    def _fit_view(self):
        if not self.polygons:
            return
        xs, ys = [], []
        for p in self.polygons:
            b = p.bounds          # (minx, miny, maxx, maxy)
            xs += [b[0], b[2]]
            ys += [b[1], b[3]]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = self.width()  or 700
        h = self.height() or 500
        margin = 40
        sx = (w - 2 * margin) / max(max_x - min_x, 1e-9)
        sy = (h - 2 * margin) / max(max_y - min_y, 1e-9)
        self.scale = min(sx, sy)
        self.pan_x = margin - min_x * self.scale
        self.pan_y = margin - min_y * self.scale

    def _w2s(self, x, y):
        """World to screen."""
        return x * self.scale + self.pan_x, y * self.scale + self.pan_y

    # ── Qt overrides ─────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        pen = QPen(QColor(0, 0, 0), 1)
        painter.setPen(pen)

        for poly, color in zip(self.polygons, self.colors):
            coords = list(poly.exterior.coords[:-1])
            pts = QPolygonF([QPointF(*self._w2s(x, y)) for x, y in coords])
            painter.setBrush(QBrush(color))
            painter.drawPolygon(pts)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        cx, cy = event.x(), event.y()
        self.pan_x = cx - factor * (cx - self.pan_x)
        self.pan_y = cy - factor * (cy - self.pan_y)
        self.scale *= factor
        self.update()

    def mousePressEvent(self, event):
        mid   = event.button() == Qt.MiddleButton
        alt_l = (event.button() == Qt.LeftButton and
                 event.modifiers() & Qt.AltModifier)
        if mid or alt_l:
            self._panning    = True
            self._last_mouse = event.pos()

    def mouseMoveEvent(self, event):
        if self._panning and self._last_mouse:
            self.pan_x += event.x() - self._last_mouse.x()
            self.pan_y += event.y() - self._last_mouse.y()
            self._last_mouse = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self._panning    = False
        self._last_mouse = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.polygons:
            self._fit_view()
            self.update()


# ═══════════════════════════ Main Window ═════════════════════════

class RandomiserWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polygon Randomiser")
        self.resize(1100, 700)

        self._polygons_orig    = []   # original polygons from CSV
        self._polygons_current = []   # currently displayed polygons
        self._colors           = []
        self._csv_path         = None

        # ── Layout ──────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.canvas = RandomiserCanvas()
        root.addWidget(self.canvas, stretch=1)

        panel = QVBoxLayout()
        panel.setAlignment(Qt.AlignTop)
        panel.setContentsMargins(8, 8, 8, 8)
        panel.setSpacing(10)
        root.addLayout(panel, stretch=0)

        # ── File group ──────────────────────────────────────────
        file_grp = QGroupBox("File")
        file_lay = QVBoxLayout(file_grp)

        self.load_btn = QPushButton("Load CSV…")
        self.load_btn.clicked.connect(self._load_csv)

        self.save_btn = QPushButton("Save CSV…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_csv)

        self.status_lbl = QLabel("No file loaded.")
        self.status_lbl.setWordWrap(True)

        file_lay.addWidget(self.load_btn)
        file_lay.addWidget(self.save_btn)
        file_lay.addWidget(self.status_lbl)
        panel.addWidget(file_grp)

        # ── Randomization group ─────────────────────────────────
        rand_grp = QGroupBox("Randomization")
        rand_lay = QVBoxLayout(rand_grp)

        rand_lay.addWidget(QLabel("Max control points per edge:"))
        self.ctrl_spin = QSpinBox()
        self.ctrl_spin.setRange(0, 30)
        self.ctrl_spin.setValue(2)
        self.ctrl_spin.setToolTip(
            "Each edge independently picks a random number of control points\n"
            "in the range [0, max].  Shared edges always use the same count\n"
            "in both adjacent polygons to prevent overlaps."
        )
        rand_lay.addWidget(self.ctrl_spin)

        rand_lay.addWidget(QLabel("Random strength (world units):"))
        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.0, 1_000_000.0)
        self.strength_spin.setValue(10.0)
        self.strength_spin.setSingleStep(1.0)
        self.strength_spin.setDecimals(2)
        self.strength_spin.setToolTip(
            "Maximum displacement distance applied to each vertex.\n"
            "Set relative to your polygon size (e.g. 5–20 for a 300-unit grid)."
        )
        rand_lay.addWidget(self.strength_spin)

        rand_lay.addWidget(QLabel("Max downscale per polygon (%%):"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 99.0)
        self.scale_spin.setValue(0.0)
        self.scale_spin.setSingleStep(1.0)
        self.scale_spin.setDecimals(1)
        self.scale_spin.setToolTip(
            "Each polygon is independently scaled down by a random amount\n"
            "between 0%% and this value (toward its centroid).\n"
            "0 = no scaling.  Never causes overlaps, only gaps."
        )
        rand_lay.addWidget(self.scale_spin)

        self.rand_btn = QPushButton("Randomize")
        self.rand_btn.setEnabled(False)
        self.rand_btn.clicked.connect(self._do_randomize)

        self.reset_btn = QPushButton("Reset to Original")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self._reset)

        rand_lay.addWidget(self.rand_btn)
        rand_lay.addWidget(self.reset_btn)
        panel.addWidget(rand_grp)

        # ── Help note ───────────────────────────────────────────
        note = QLabel(
            "<small><i>"
            "Zoom: scroll wheel<br>"
            "Pan: middle-click drag<br>"
            "or Alt + left-click drag"
            "</i></small>"
        )
        note.setTextFormat(Qt.RichText)
        panel.addWidget(note)
        panel.addStretch()

    # ════════════════════════ Slots ══════════════════════════════

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open polygon CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            polys, colors = load_polygons_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not read CSV:\n{e}")
            return

        if not polys:
            QMessageBox.warning(self, "Empty File",
                                "No valid polygons found in the selected CSV.")
            return

        self._polygons_orig    = polys
        self._polygons_current = list(polys)
        self._colors           = colors
        self._csv_path         = path

        self.canvas.set_polygons(self._polygons_current, self._colors)
        self.status_lbl.setText(f"{len(polys)} polygons loaded.")
        self.rand_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

    def _do_randomize(self):
        if not self._polygons_orig:
            return
        n_ctrl       = self.ctrl_spin.value()
        strength     = self.strength_spin.value()
        max_ds_pct   = self.scale_spin.value()
        try:
            result = randomize_polygons(self._polygons_orig, n_ctrl, strength)
            result = scale_polygons(result, max_ds_pct)
        except Exception as e:
            QMessageBox.critical(self, "Randomization Error", str(e))
            return

        valid = sum(1 for p in result if p is not None)
        self._polygons_current = result
        self.canvas.set_polygons(self._polygons_current, self._colors)
        self.status_lbl.setText(
            f"Randomized — {valid}/{len(result)} polygons valid.\n"
            f"max ctrl pts/edge: {n_ctrl}, strength: {strength:.2f}, "
            f"max downscale: {max_ds_pct:.1f}%%"
        )

    def _reset(self):
        self._polygons_current = list(self._polygons_orig)
        self.canvas.set_polygons(self._polygons_current, self._colors)
        self.status_lbl.setText(
            f"{len(self._polygons_orig)} polygons (original)."
        )

    def _save_csv(self):
        default = self._csv_path or ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save polygon CSV", default, "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            save_polygons_csv(path, self._polygons_current, self._colors)
            self.status_lbl.setText(
                f"Saved {sum(1 for p in self._polygons_current if p is not None)}"
                f" polygons."
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save CSV:\n{e}")


# ═══════════════════════════ Entry point ═════════════════════════

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = RandomiserWindow()
    win.show()
    sys.exit(app.exec_())
