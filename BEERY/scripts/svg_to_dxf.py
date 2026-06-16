"""SVG → DXF — load an SVG on the left, convert to DXF, preview on the right.

V1: 1:1 conversion. 1 SVG unit = 1 DXF unit. All path geometry is flattened to
polylines (each Move command starts a new polyline) at a configurable
sampling precision, then written as open LWPOLYLINEs. Y-axis is flipped so
the DXF reads right-side-up in CAD viewers (SVG y grows down, DXF y grows up).

Usage:
  python BEERY/scripts/svg_to_dxf.py
"""
from __future__ import annotations

import csv
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5.QtCore import QEvent, QSize, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSplitter,
    QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent.parent       # BEERY/
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"


# ---------------------------------------------------------------------------
# SVG → polylines
# ---------------------------------------------------------------------------

def svg_to_polylines(svg_path: Path,
                     sampling_precision: float = 0.5,
                     ) -> tuple[list[np.ndarray], float, float]:
    """Parse an SVG and flatten all path geometry to polylines.

    Returns (polylines, viewport_w, viewport_h). Each polyline is an (N, 2)
    float array of (x, y) in the SVG's own coordinate system (y grows down).
    A new polyline starts at every Move command and at every disconnected
    subpath. ``sampling_precision`` is the maximum allowed deviation between
    the flattened polyline and the true curve, in SVG units.
    """
    from svgelements import SVG, Path as SvgPath, Move, Close, Shape

    svg = SVG.parse(str(svg_path), reify=True)
    vp_w = float(svg.viewbox.width) if svg.viewbox else float(svg.width or 0)
    vp_h = float(svg.viewbox.height) if svg.viewbox else float(svg.height or 0)

    polylines: list[np.ndarray] = []

    for element in svg.elements():
        # Skip the SVG root itself and any non-geometry element
        if not isinstance(element, Shape):
            continue
        try:
            # Convert any Shape (Rect, Circle, Ellipse, Polygon, Polyline, Path)
            # into a Path so we can iterate path segments uniformly.
            path = SvgPath(element) if not isinstance(element, SvgPath) else element
        except Exception:
            continue

        # `path.npoint(t)` samples the path at parametric t∈[0,1]. We sample
        # each individual segment at a density driven by sampling_precision.
        current_polyline: list[tuple[float, float]] = []

        for segment in path.segments():
            if isinstance(segment, Move):
                # Close out the previous polyline (if any) and start a new one.
                if len(current_polyline) >= 2:
                    polylines.append(np.array(current_polyline, dtype=np.float64))
                current_polyline = []
                p = segment.end
                if p is not None:
                    current_polyline.append((float(p.x), float(p.y)))
                continue
            if isinstance(segment, Close):
                # Close: snap back to subpath start if not already there.
                if current_polyline:
                    sx, sy = current_polyline[0]
                    ex, ey = current_polyline[-1]
                    if (sx - ex) ** 2 + (sy - ey) ** 2 > 1e-9:
                        current_polyline.append((sx, sy))
                continue

            # Sample a curved / line segment. Choose sample count so chord
            # length is ≲ sampling_precision (rough — uses bbox diagonal as
            # an upper bound on segment length).
            try:
                length = float(segment.length(error=sampling_precision))
            except Exception:
                length = 0.0
            n_samples = max(2, int(np.ceil(length / max(sampling_precision, 1e-3))))
            # Drop the first sample to avoid duplicating the previous point.
            ts = np.linspace(0.0, 1.0, n_samples)
            for t in ts[1:]:
                try:
                    p = segment.point(t)
                except Exception:
                    p = segment.end
                if p is None:
                    continue
                current_polyline.append((float(p.x), float(p.y)))

        if len(current_polyline) >= 2:
            polylines.append(np.array(current_polyline, dtype=np.float64))

    return polylines, vp_w, vp_h


def svg_shapes_to_shrunk_polylines(
    svg_path: Path,
    gap: float,
    sampling_precision: float = 0.5,
) -> tuple[list[np.ndarray], float, float, int, int]:
    """For each CLOSED shape in the SVG, shrink it inward by gap/2 and return
    the resulting outlines as polylines.

    The total spacing between two shapes that touched in the SVG is ``gap``
    (gap/2 removed from each side). Open paths cannot be shrunk inward and
    are skipped. Shapes that collapse to nothing under the buffer are dropped.

    Returns (polylines, vp_w, vp_h, skipped_open, dropped_collapsed).
    """
    from svgelements import SVG, Path as SvgPath, Move, Close, Shape
    from shapely.geometry import Polygon as ShapelyPolygon

    svg = SVG.parse(str(svg_path), reify=True)
    vp_w = float(svg.viewbox.width) if svg.viewbox else float(svg.width or 0)
    vp_h = float(svg.viewbox.height) if svg.viewbox else float(svg.height or 0)

    shrink = gap / 2.0
    polylines: list[np.ndarray] = []
    skipped_open = 0
    dropped = 0

    def collect_subpath(pts: list[tuple[float, float]], is_closed: bool) -> None:
        nonlocal skipped_open, dropped, polylines
        if len(pts) < 3:
            if pts:
                skipped_open += 1
            return
        sx, sy = pts[0]
        ex, ey = pts[-1]
        closed = is_closed or ((sx - ex) ** 2 + (sy - ey) ** 2 < 1e-6)
        if not closed:
            skipped_open += 1
            return
        try:
            poly = ShapelyPolygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            shrunk = poly.buffer(-shrink) if shrink > 0 else poly
        except Exception:
            dropped += 1
            return
        if shrunk.is_empty:
            dropped += 1
            return
        geoms = [shrunk] if shrunk.geom_type == "Polygon" else list(shrunk.geoms)
        for g in geoms:
            ext = np.array(list(g.exterior.coords), dtype=np.float64)
            polylines.append(ext)
            for ring in g.interiors:
                polylines.append(np.array(list(ring.coords), dtype=np.float64))

    for element in svg.elements():
        if not isinstance(element, Shape):
            continue
        try:
            path = SvgPath(element) if not isinstance(element, SvgPath) else element
        except Exception:
            continue

        current_pts: list[tuple[float, float]] = []
        subpath_is_closed = False

        for segment in path.segments():
            if isinstance(segment, Move):
                collect_subpath(current_pts, subpath_is_closed)
                current_pts = []
                subpath_is_closed = False
                p = segment.end
                if p is not None:
                    current_pts.append((float(p.x), float(p.y)))
                continue
            if isinstance(segment, Close):
                subpath_is_closed = True
                if current_pts:
                    sx, sy = current_pts[0]
                    ex, ey = current_pts[-1]
                    if (sx - ex) ** 2 + (sy - ey) ** 2 > 1e-9:
                        current_pts.append((sx, sy))
                continue
            try:
                length = float(segment.length(error=sampling_precision))
            except Exception:
                length = 0.0
            n_samples = max(2, int(np.ceil(length / max(sampling_precision, 1e-3))))
            ts = np.linspace(0.0, 1.0, n_samples)
            for t in ts[1:]:
                try:
                    p = segment.point(t)
                except Exception:
                    p = segment.end
                if p is None:
                    continue
                current_pts.append((float(p.x), float(p.y)))

        collect_subpath(current_pts, subpath_is_closed)

    return polylines, vp_w, vp_h, skipped_open, dropped


def svg_to_white_borders(
    svg_path: Path,
    stroke_width: float,
    sampling_precision: float = 0.5,
) -> tuple[list[np.ndarray], float, float, int]:
    """Treat every SVG path as a BLACK line of width ``stroke_width``. The
    union of those strokes is the "black region"; the rest of the viewport is
    the "white region" — typically a set of separate pieces between the
    strokes. Return each white piece's outline as polylines.

    Each closed exterior boundary is emitted; if a piece has holes (e.g. a
    donut-shaped piece), every interior ring is emitted too. The natural
    spacing between adjacent pieces equals ``stroke_width``.

    Returns (polylines, vp_w, vp_h, n_white_regions).
    """
    from shapely.geometry import LineString, box
    from shapely.ops import unary_union

    polylines_in, vp_w, vp_h = svg_to_polylines(svg_path, sampling_precision)
    if vp_w <= 0 or vp_h <= 0:
        return [], vp_w, vp_h, 0

    half = max(stroke_width, 0.0) / 2.0
    if half <= 0:
        # No strokes to subtract — the whole viewport is one white piece.
        viewport_outline = np.array(
            [(0.0, 0.0), (vp_w, 0.0), (vp_w, vp_h), (0.0, vp_h), (0.0, 0.0)],
            dtype=np.float64,
        )
        return [viewport_outline], vp_w, vp_h, 1

    stroke_polys = []
    for pl in polylines_in:
        if len(pl) < 2:
            continue
        try:
            ls = LineString([(float(x), float(y)) for x, y in pl])
        except Exception:
            continue
        if ls.length < 1e-9:
            continue
        stroke_polys.append(ls.buffer(half))

    viewport = box(0.0, 0.0, float(vp_w), float(vp_h))
    if not stroke_polys:
        ext = np.array(list(viewport.exterior.coords), dtype=np.float64)
        return [ext], vp_w, vp_h, 1

    black = unary_union(stroke_polys)
    white = viewport.difference(black)

    out: list[np.ndarray] = []
    if white.is_empty:
        return out, vp_w, vp_h, 0
    geoms = [white] if white.geom_type == "Polygon" else list(white.geoms)
    n_regions = 0
    for g in geoms:
        if g.is_empty or g.geom_type != "Polygon":
            continue
        out.append(np.array(list(g.exterior.coords), dtype=np.float64))
        for ring in g.interiors:
            out.append(np.array(list(ring.coords), dtype=np.float64))
        n_regions += 1
    return out, vp_w, vp_h, n_regions


# ---------------------------------------------------------------------------
# DXF writer + preview renderer
# ---------------------------------------------------------------------------

def write_dxf(polylines: list[np.ndarray], out_path: Path, viewport_h: float,
              units_index: int = 4) -> tuple[int, int]:
    """Write each polyline as an open LWPOLYLINE. Y-axis flipped so the DXF
    reads right-side-up in CAD viewers.

    Hardened for Fusion 360 import:
      - R2018 format with full header (`setup=True`)
      - `$INSUNITS` set (default 4 = millimeters) so Fusion knows the scale
      - Explicit `PATHS` layer rather than a colour-only entity attribute
      - Coordinates rounded to 4 decimals, NaN/Inf rejected, consecutive
        duplicate points removed (Fusion rejects zero-length segments)
      - `doc.audit()` runs before save; file is then re-opened with
        `ezdxf.readfile()` as a final sanity check.

    units_index: AutoCAD $INSUNITS code. 0=unitless, 1=inch, 4=mm (default),
    5=cm, 6=meter.

    Returns (polylines_written, total_points_written) for status reporting.
    """
    import ezdxf

    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = int(units_index)
    doc.header["$LUNITS"] = 2          # decimal length units
    doc.header["$MEASUREMENT"] = 1     # metric drawing
    if "PATHS" not in doc.layers:
        doc.layers.add("PATHS", color=7)

    msp = doc.modelspace()

    polys_written = 0
    points_written = 0
    for pl in polylines:
        if len(pl) < 2:
            continue
        cleaned: list[tuple[float, float]] = []
        for x, y in pl:
            cx = round(float(x), 4)
            cy = round(float(viewport_h - y), 4)
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            if cleaned and cleaned[-1] == (cx, cy):
                continue
            cleaned.append((cx, cy))
        if len(cleaned) < 2:
            continue
        # Detect closed polyline: first ≈ last point. Write as a real closed
        # LWPOLYLINE (CAD-cleaner than an open polyline with a duplicate point).
        is_closed = False
        if len(cleaned) >= 3:
            fx, fy = cleaned[0]
            lx, ly = cleaned[-1]
            if abs(fx - lx) < 1e-4 and abs(fy - ly) < 1e-4:
                is_closed = True
                cleaned = cleaned[:-1]
        msp.add_lwpolyline(
            cleaned, close=is_closed,
            dxfattribs={"layer": "PATHS"},
        )
        polys_written += 1
        points_written += len(cleaned)

    auditor = doc.audit()
    # If audit fixed anything we still save — ezdxf removes/repairs in place.
    if auditor.has_errors:
        print(f"[svg_to_dxf] audit errors (post-fix): {len(auditor.errors)}")

    doc.saveas(str(out_path))

    # Final sanity check: re-open the file we just wrote.
    try:
        ezdxf.readfile(str(out_path))
    except Exception as e:
        raise RuntimeError(
            f"Generated DXF failed re-validation when reopened: {e}"
        ) from e

    return polys_written, points_written


def write_csv(polylines: list[np.ndarray], out_path: Path,
              color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
              simplify_eps: float = 0.5) -> int:
    """Write each polyline as a row in the project's CSV polygon format
    (compatible with mosaic_to_csv.py / polygon_viewer.py / frame*_*.csv).

    Polylines are simplified with Douglas-Peucker so only true corners /
    control points remain — collinear in-between points (from path sampling
    or Shapely buffer output) are dropped. `simplify_eps` is the maximum
    allowed deviation in SVG units (mm); 0.5 keeps every real corner while
    removing sub-millimetre redundancy.

    Coordinates are kept in SVG user units (y grows down — same convention as
    the existing CSV polygon format; no Y-flip). Closed polylines have their
    duplicate trailing point removed before serialisation. Polylines with
    fewer than 3 points after simplification are skipped.

    Returns the number of polygons written.
    """
    import cv2

    r, g, b = color_rgb
    hex_str = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    n_written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "coordinates", "color_r", "color_g", "color_b", "color_a", "color_hex",
        ])
        for pl in polylines:
            if len(pl) < 3:
                continue
            pts = pl
            is_closed = False
            if len(pts) >= 4:
                fx, fy = pts[0]
                lx, ly = pts[-1]
                if abs(fx - lx) < 1e-4 and abs(fy - ly) < 1e-4:
                    pts = pts[:-1]
                    is_closed = True
            if simplify_eps > 0 and len(pts) >= 3:
                contour = pts.astype(np.float32).reshape(-1, 1, 2)
                simplified = cv2.approxPolyDP(
                    contour, float(simplify_eps), closed=is_closed,
                )
                pts = simplified.reshape(-1, 2)
                if len(pts) < 3:
                    continue
            coords_str = "[" + ", ".join(
                f"({float(x):.4f}, {float(y):.4f})" for x, y in pts
            ) + "]"
            writer.writerow([coords_str, r, g, b, 1.0, hex_str])
            n_written += 1
    return n_written


def render_preview(polylines: list[np.ndarray],
                   viewport_w: float, viewport_h: float,
                   max_dim: int = 1600) -> Image.Image:
    """Black 1-px polylines on white, fit to ≤max_dim on the longest side."""
    import cv2

    if viewport_w <= 0 or viewport_h <= 0:
        return Image.new("RGB", (200, 200), (255, 255, 255))
    scale = min(max_dim / viewport_w, max_dim / viewport_h, 1.0)
    if scale <= 0:
        scale = 1.0
    w = max(1, int(np.ceil(viewport_w * scale)))
    h = max(1, int(np.ceil(viewport_h * scale)))
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for pl in polylines:
        if len(pl) < 2:
            continue
        scaled = (pl * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [scaled], isClosed=False,
                      color=(0, 0, 0), thickness=1)
    return Image.fromarray(canvas, mode="RGB")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class SvgToDxf(QMainWindow):
    ZOOM_STEP = 1.15
    ZOOM_MIN = 0.05
    ZOOM_MAX = 20.0

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SVG → DXF (BEERY)")
        self.resize(1500, 900)

        self.svg_path: Path | None = None
        self.polylines: list[np.ndarray] = []
        self.viewport_w: float = 0.0
        self.viewport_h: float = 0.0
        # Zoom state per pane. Base size is captured on Load (left) / Convert
        # (right); wheel events resize the widget to base_size * zoom.
        self.left_zoom: float = 1.0
        self.right_zoom: float = 1.0
        self.left_base_size: QSize | None = None
        self.right_base_qimg: QImage | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # Toolbar
        bar = QHBoxLayout()
        self.load_btn       = QPushButton("Load SVG...")
        self.convert_btn    = QPushButton("Convert 1:1")
        bar.addWidget(self.load_btn)
        bar.addWidget(self.convert_btn)

        bar.addWidget(QLabel("Gap:"))
        self.gap_spin = QDoubleSpinBox()
        self.gap_spin.setRange(0.0, 50.0)
        self.gap_spin.setDecimals(2)
        self.gap_spin.setSingleStep(0.1)
        self.gap_spin.setValue(1.0)
        self.gap_spin.setSuffix(" mm")
        self.gap_spin.setToolTip(
            "Total spacing between adjacent shapes after shrinking. Each "
            "shape is shrunk inward by gap/2, so two touching shapes end up "
            "with this much space between them."
        )
        bar.addWidget(self.gap_spin)
        self.convert_gap_btn = QPushButton("Convert with gap")
        bar.addWidget(self.convert_gap_btn)

        bar.addWidget(QLabel("Line width:"))
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.0, 50.0)
        self.line_width_spin.setDecimals(2)
        self.line_width_spin.setSingleStep(0.1)
        self.line_width_spin.setValue(2.0)
        self.line_width_spin.setSuffix(" mm")
        self.line_width_spin.setToolTip(
            "Treat every SVG path as a black line of this width. 'Convert "
            "white' then extracts the borders of the WHITE regions between "
            "the black strokes — each region becomes one DXF piece, with "
            "natural spacing equal to this line width."
        )
        bar.addWidget(self.line_width_spin)
        self.convert_white_btn = QPushButton("Convert white")
        bar.addWidget(self.convert_white_btn)

        self.shapes_csv_btn = QPushButton("Shapes → CSV...")
        self.shapes_csv_btn.setToolTip(
            "One-click: parse every closed shape in the SVG (rect, circle, "
            "ellipse, polygon, closed path), preview them on the right, and "
            "save directly to the project's CSV polygon format. Open paths "
            "are skipped. No shrinking or gap applied."
        )
        bar.addWidget(self.shapes_csv_btn)

        self.save_btn = QPushButton("Save DXF...")
        bar.addWidget(self.save_btn)
        self.save_csv_btn = QPushButton("Save CSV...")
        self.save_csv_btn.setToolTip(
            "Save the current polygons in the project's CSV polygon format "
            "(coordinates, color_r/g/b/a, color_hex). The same format used "
            "by mosaic_to_csv.py — loadable by polygon_viewer.py and the "
            "other downstream polygon tools."
        )
        bar.addWidget(self.save_csv_btn)
        bar.addStretch(1)
        self.info_label = QLabel("—")
        self.info_label.setStyleSheet("color:#666;")
        bar.addWidget(self.info_label)
        root_layout.addLayout(bar)

        # Split panes
        splitter = QSplitter(Qt.Horizontal)

        # Left: SVG view
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_title = QLabel("Source SVG")
        left_title.setStyleSheet("font-weight:bold; padding:2px;")
        left_layout.addWidget(left_title)
        self.svg_widget = QSvgWidget()
        self.svg_widget.setStyleSheet("background:#fff;")
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidget(self.svg_widget)
        # widgetResizable=False so we control the inner widget's size for zoom.
        self.left_scroll.setWidgetResizable(False)
        self.left_scroll.setAlignment(Qt.AlignCenter)
        self.left_scroll.viewport().installEventFilter(self)
        left_layout.addWidget(self.left_scroll, 1)
        splitter.addWidget(left_container)

        # Right: DXF preview (rendered from the polylines we'll write)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_title = QLabel("DXF preview (1:1)")
        right_title.setStyleSheet("font-weight:bold; padding:2px;")
        right_layout.addWidget(right_title)
        self.preview_label = QLabel("Convert an SVG to see the DXF preview here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background:#222; color:#bbb;")
        self.preview_label.setMinimumSize(400, 400)
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidget(self.preview_label)
        self.right_scroll.setWidgetResizable(False)
        self.right_scroll.setAlignment(Qt.AlignCenter)
        self.right_scroll.viewport().installEventFilter(self)
        right_layout.addWidget(self.right_scroll, 1)
        splitter.addWidget(right_container)

        splitter.setSizes([750, 750])
        root_layout.addWidget(splitter, 1)

        self.statusBar().showMessage("Ready.")

        self.load_btn.clicked.connect(self.load_svg)
        self.convert_btn.clicked.connect(self.convert_1to1)
        self.convert_gap_btn.clicked.connect(self.convert_with_gap)
        self.convert_white_btn.clicked.connect(self.convert_white)
        self.shapes_csv_btn.clicked.connect(self.shapes_to_csv)
        self.save_btn.clicked.connect(self.save_dxf)
        self.save_csv_btn.clicked.connect(self.save_csv)
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_svg = self.svg_path is not None
        has_polys = bool(self.polylines)
        self.convert_btn.setEnabled(has_svg)
        self.convert_gap_btn.setEnabled(has_svg)
        self.convert_white_btn.setEnabled(has_svg)
        self.shapes_csv_btn.setEnabled(has_svg)
        self.save_btn.setEnabled(has_polys)
        self.save_csv_btn.setEnabled(has_polys)

    # ----- zoom -----------------------------------------------------------

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            if obj is self.left_scroll.viewport() and self.left_base_size is not None:
                self._wheel_zoom(
                    event, self.left_scroll,
                    get_zoom=lambda: self.left_zoom,
                    set_zoom=self._set_left_zoom,
                )
                return True
            if obj is self.right_scroll.viewport() and self.right_base_qimg is not None:
                self._wheel_zoom(
                    event, self.right_scroll,
                    get_zoom=lambda: self.right_zoom,
                    set_zoom=self._set_right_zoom,
                )
                return True
        return super().eventFilter(obj, event)

    def _wheel_zoom(self, event, scroll, get_zoom, set_zoom) -> None:
        """Cursor-centered wheel zoom for either pane. Keeps the image point
        under the cursor in the same viewport position after the zoom step."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self.ZOOM_STEP if delta > 0 else (1.0 / self.ZOOM_STEP)
        old_zoom = get_zoom()
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return
        pos = event.position() if hasattr(event, "position") else event.posF()
        vx, vy = pos.x(), pos.y()
        hbar = scroll.horizontalScrollBar()
        vbar = scroll.verticalScrollBar()
        # Image-space coords of the pixel currently under the cursor
        img_x = (hbar.value() + vx) / old_zoom
        img_y = (vbar.value() + vy) / old_zoom
        set_zoom(new_zoom)  # resizes the inner widget
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(),
                      int(round(img_x * new_zoom - vx)))))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(),
                      int(round(img_y * new_zoom - vy)))))

    def _set_left_zoom(self, z: float) -> None:
        self.left_zoom = z
        self._apply_left_zoom()

    def _set_right_zoom(self, z: float) -> None:
        self.right_zoom = z
        self._apply_right_zoom()

    def _apply_left_zoom(self) -> None:
        if self.left_base_size is None:
            return
        w = max(1, int(round(self.left_base_size.width() * self.left_zoom)))
        h = max(1, int(round(self.left_base_size.height() * self.left_zoom)))
        self.svg_widget.setFixedSize(w, h)

    def _apply_right_zoom(self) -> None:
        if self.right_base_qimg is None:
            return
        bw, bh = self.right_base_qimg.width(), self.right_base_qimg.height()
        w = max(1, int(round(bw * self.right_zoom)))
        h = max(1, int(round(bh * self.right_zoom)))
        scaled = self.right_base_qimg.scaled(
            w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(QPixmap.fromImage(scaled))
        self.preview_label.resize(scaled.size())

    # ----- actions --------------------------------------------------------

    def load_svg(self) -> None:
        start_dir = str(INPUT_DIR) if INPUT_DIR.exists() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load SVG", start_dir, "SVG files (*.svg);;All files (*.*)",
        )
        if not path:
            return
        p = Path(path)
        if not p.is_file():
            QMessageBox.critical(self, "Not found", f"File not found: {p}")
            return
        self.svg_path = p
        self.polylines = []
        self.svg_widget.load(str(p))
        # Capture the SVG's intrinsic pixel size for zooming.
        intrinsic = self.svg_widget.renderer().defaultSize()
        if intrinsic.isEmpty():
            intrinsic = QSize(300, 300)
        self.left_base_size = intrinsic
        # Fit-to-viewport initial zoom so the SVG isn't tiny.
        vp = self.left_scroll.viewport().size()
        if intrinsic.width() > 0 and intrinsic.height() > 0:
            fit = min(vp.width() / intrinsic.width(),
                      vp.height() / intrinsic.height())
            self.left_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, fit if fit > 0 else 1.0))
        else:
            self.left_zoom = 1.0
        self._apply_left_zoom()

        self.right_base_qimg = None
        self.right_zoom = 1.0
        self.preview_label.clear()
        self.preview_label.setText("Click Convert 1:1 to render the DXF preview")
        self.preview_label.resize(self.preview_label.minimumSize())
        self.info_label.setText(f"Loaded: {p.name}")
        self.statusBar().showMessage(f"Loaded SVG: {p.name}")
        self._update_buttons()

    def convert_1to1(self) -> None:
        if self.svg_path is None:
            return
        self.statusBar().showMessage("Parsing SVG...")
        QApplication.processEvents()
        try:
            polylines, vp_w, vp_h = svg_to_polylines(self.svg_path)
        except Exception as e:
            QMessageBox.critical(
                self, "Conversion failed",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )
            self.statusBar().showMessage("Conversion failed.")
            return
        self.polylines = polylines
        self.viewport_w = vp_w
        self.viewport_h = vp_h
        if not polylines:
            QMessageBox.warning(
                self, "No geometry",
                "Parsed the SVG but found no path geometry to convert.",
            )
            self.preview_label.setText("No geometry found in this SVG")
            self.statusBar().showMessage("No geometry found.")
            self._update_buttons()
            return

        preview = render_preview(polylines, vp_w, vp_h)
        qimg = QImage(
            preview.tobytes("raw", "RGB"), preview.width, preview.height,
            preview.width * 3, QImage.Format_RGB888,
        ).copy()
        self.right_base_qimg = qimg
        # Fit-to-viewport initial zoom for the new preview.
        vp = self.right_scroll.viewport().size()
        if qimg.width() > 0 and qimg.height() > 0:
            fit = min(vp.width() / qimg.width(), vp.height() / qimg.height())
            self.right_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, fit if fit > 0 else 1.0))
        else:
            self.right_zoom = 1.0
        self._apply_right_zoom()
        total_pts = sum(len(pl) for pl in polylines)
        self.info_label.setText(
            f"{len(polylines)} polylines  |  {total_pts} points  |  "
            f"viewport {vp_w:g} × {vp_h:g}"
        )
        self.statusBar().showMessage(
            f"Converted: {len(polylines)} polylines, {total_pts} points.",
        )
        self._update_buttons()

    def convert_with_gap(self) -> None:
        """Shrink each CLOSED shape inward by gap/2 so adjacent shapes end up
        with the chosen gap between them. Open paths and shapes that collapse
        under the buffer are dropped (reported in the info chip)."""
        if self.svg_path is None:
            return
        gap = float(self.gap_spin.value())
        self.statusBar().showMessage(f"Parsing SVG and shrinking shapes (gap={gap:g})...")
        QApplication.processEvents()
        try:
            polylines, vp_w, vp_h, skipped_open, dropped = (
                svg_shapes_to_shrunk_polylines(self.svg_path, gap=gap)
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Conversion failed",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )
            self.statusBar().showMessage("Conversion failed.")
            return
        self.polylines = polylines
        self.viewport_w = vp_w
        self.viewport_h = vp_h
        if not polylines:
            QMessageBox.warning(
                self, "No geometry",
                f"After shrinking by {gap/2:g} mm, no closed shapes remained "
                f"(open paths skipped: {skipped_open}, "
                f"collapsed-by-gap: {dropped}).",
            )
            self.preview_label.setText("No geometry after shrinking")
            self.statusBar().showMessage("No geometry after shrinking.")
            self._update_buttons()
            return

        preview = render_preview(polylines, vp_w, vp_h)
        qimg = QImage(
            preview.tobytes("raw", "RGB"), preview.width, preview.height,
            preview.width * 3, QImage.Format_RGB888,
        ).copy()
        self.right_base_qimg = qimg
        # Fit-to-viewport initial zoom for the new preview.
        vp = self.right_scroll.viewport().size()
        if qimg.width() > 0 and qimg.height() > 0:
            fit = min(vp.width() / qimg.width(), vp.height() / qimg.height())
            self.right_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, fit if fit > 0 else 1.0))
        else:
            self.right_zoom = 1.0
        self._apply_right_zoom()
        total_pts = sum(len(pl) for pl in polylines)
        self.info_label.setText(
            f"{len(polylines)} shrunk shapes  |  {total_pts} points  |  "
            f"gap {gap:g} mm  |  skipped open: {skipped_open}  |  "
            f"collapsed: {dropped}"
        )
        self.statusBar().showMessage(
            f"Shrunk: {len(polylines)} closed shapes "
            f"(skipped open: {skipped_open}, collapsed: {dropped}).",
        )
        self._update_buttons()

    def convert_white(self) -> None:
        """Inflate every SVG path to a black line of `Line width` mm; emit the
        outlines of the resulting WHITE regions as DXF polylines. Each piece
        gets natural spacing equal to the chosen line width."""
        if self.svg_path is None:
            return
        line_width = float(self.line_width_spin.value())
        self.statusBar().showMessage(
            f"Computing white regions (line width={line_width:g})..."
        )
        QApplication.processEvents()
        try:
            polylines, vp_w, vp_h, n_regions = svg_to_white_borders(
                self.svg_path, stroke_width=line_width,
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Conversion failed",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )
            self.statusBar().showMessage("Conversion failed.")
            return
        self.polylines = polylines
        self.viewport_w = vp_w
        self.viewport_h = vp_h
        if not polylines:
            QMessageBox.warning(
                self, "No white regions",
                f"At line width {line_width:g} mm the strokes cover the "
                f"entire viewport — no white regions remain.",
            )
            self.preview_label.setText("No white regions at this line width")
            self.statusBar().showMessage("No white regions.")
            self._update_buttons()
            return

        preview = render_preview(polylines, vp_w, vp_h)
        qimg = QImage(
            preview.tobytes("raw", "RGB"), preview.width, preview.height,
            preview.width * 3, QImage.Format_RGB888,
        ).copy()
        self.right_base_qimg = qimg
        vp = self.right_scroll.viewport().size()
        if qimg.width() > 0 and qimg.height() > 0:
            fit = min(vp.width() / qimg.width(), vp.height() / qimg.height())
            self.right_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, fit if fit > 0 else 1.0))
        else:
            self.right_zoom = 1.0
        self._apply_right_zoom()
        total_pts = sum(len(pl) for pl in polylines)
        self.info_label.setText(
            f"{n_regions} white regions  |  {len(polylines)} polylines  |  "
            f"{total_pts} points  |  line width {line_width:g} mm"
        )
        self.statusBar().showMessage(
            f"White regions: {n_regions} pieces, {len(polylines)} polylines.",
        )
        self._update_buttons()

    def shapes_to_csv(self) -> None:
        """One-click: parse closed SVG shapes (no shrink, no gap), update the
        preview, and save the polygons directly to a CSV."""
        if self.svg_path is None:
            return
        self.statusBar().showMessage("Parsing SVG shapes...")
        QApplication.processEvents()
        try:
            polylines, vp_w, vp_h, skipped_open, dropped = (
                svg_shapes_to_shrunk_polylines(self.svg_path, gap=0.0)
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Parse failed",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )
            self.statusBar().showMessage("Parse failed.")
            return
        if not polylines:
            QMessageBox.warning(
                self, "No closed shapes",
                f"Found no closed shapes in the SVG "
                f"(open paths skipped: {skipped_open}).",
            )
            self.statusBar().showMessage("No closed shapes found.")
            return

        self.polylines = polylines
        self.viewport_w = vp_w
        self.viewport_h = vp_h

        # Update the right-pane preview so the user sees what was parsed.
        preview = render_preview(polylines, vp_w, vp_h)
        qimg = QImage(
            preview.tobytes("raw", "RGB"), preview.width, preview.height,
            preview.width * 3, QImage.Format_RGB888,
        ).copy()
        self.right_base_qimg = qimg
        vp = self.right_scroll.viewport().size()
        if qimg.width() > 0 and qimg.height() > 0:
            fit = min(vp.width() / qimg.width(), vp.height() / qimg.height())
            self.right_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, fit if fit > 0 else 1.0))
        else:
            self.right_zoom = 1.0
        self._apply_right_zoom()
        total_pts = sum(len(pl) for pl in polylines)
        self.info_label.setText(
            f"{len(polylines)} shapes  |  {total_pts} points  |  "
            f"skipped open: {skipped_open}"
        )
        self._update_buttons()

        # Open the save dialog immediately.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        default_name = self.svg_path.stem + ".csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV polygons", str(OUTPUT_DIR / default_name),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            self.statusBar().showMessage(
                f"Parsed {len(polylines)} shapes (save cancelled).",
            )
            return
        out_path = Path(path)
        if out_path.suffix.lower() != ".csv":
            out_path = out_path.with_suffix(".csv")
        try:
            n = write_csv(self.polylines, out_path)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        self.statusBar().showMessage(f"Saved {n} polygons → {out_path.name}")
        QMessageBox.information(
            self, "Saved",
            f"Saved {n} polygons to:\n{out_path}\n\n"
            f"Format: coordinates, color_r/g/b/a, color_hex "
            f"(compatible with mosaic_to_csv polygon viewers).",
        )

    def save_csv(self) -> None:
        if not self.polylines or self.svg_path is None:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        default_name = self.svg_path.stem + ".csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV polygons", str(OUTPUT_DIR / default_name),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        out_path = Path(path)
        if out_path.suffix.lower() != ".csv":
            out_path = out_path.with_suffix(".csv")
        try:
            n = write_csv(self.polylines, out_path)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        self.statusBar().showMessage(f"Saved {n} polygons → {out_path.name}")
        QMessageBox.information(
            self, "Saved",
            f"Saved {n} polygons to:\n{out_path}\n\n"
            f"Format: coordinates, color_r/g/b/a, color_hex "
            f"(compatible with mosaic_to_csv polygon viewers).",
        )

    def save_dxf(self) -> None:
        if not self.polylines or self.svg_path is None:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        default_name = self.svg_path.stem + ".dxf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save DXF", str(OUTPUT_DIR / default_name),
            "DXF files (*.dxf);;All files (*.*)",
        )
        if not path:
            return
        out_path = Path(path)
        if out_path.suffix.lower() != ".dxf":
            out_path = out_path.with_suffix(".dxf")
        try:
            polys_written, points_written = write_dxf(
                self.polylines, out_path, viewport_h=self.viewport_h,
            )
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        self.statusBar().showMessage(
            f"Saved {polys_written} polylines / {points_written} pts → {out_path.name}",
        )
        QMessageBox.information(
            self, "Saved",
            f"Saved {polys_written} polylines ({points_written} points) to:\n"
            f"{out_path}\n\n"
            f"Format: AutoCAD R2018, $INSUNITS = 4 (millimetres), "
            f"layer 'PATHS'.",
        )


def main() -> int:
    app = QApplication(sys.argv)
    win = SvgToDxf()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
