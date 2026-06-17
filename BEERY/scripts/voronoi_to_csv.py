"""Voronoi → CSV — convert a Voronoi-style image with orange dividing lines
into the project's CSV polygon format.

Loads any image that has solid #FF6600 lines acting as cell dividers (e.g.
the output of `image_to_voronoi.py` with the "Orange dividing lines"
checkbox), runs orange-HSV thresholding + connected components, and saves
each detected cell as one CSV row with its mean colour sampled from the
input image.

Builds on mosaic_to_csv.MosaicToCsv (sibling pattern to vitrage_to_dxf.py):
the UI, image panes, scrolling, save dialog, and parameter spinboxes are
inherited. Only the detection algorithm is replaced — we have crisp orange
lines, not adaptive-threshold grout, so the inherited algorithm doesn't fit.

Usage:
  python BEERY/scripts/voronoi_to_csv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from IMAGE_TO_MOSAIC/scripts as a sibling tree.
ROOT = Path(__file__).resolve().parent.parent       # BEERY/
PROJECT_ROOT = ROOT.parent                          # CLAUDE_MOSAIC1.0/
IMAGE_TO_MOSAIC_SCRIPTS = PROJECT_ROOT / "IMAGE_TO_MOSAIC" / "scripts"
if str(IMAGE_TO_MOSAIC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(IMAGE_TO_MOSAIC_SCRIPTS))

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import (
    QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap,
)
from PyQt5.QtWidgets import (
    QApplication, QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem,
    QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
    QLabel, QMessageBox, QPushButton, QSpinBox,
)
from skimage.morphology import skeletonize

from mosaic_to_csv import MosaicToCsv as _MosaicToCsv, render_polygons


def _skeleton_to_polylines(sk: np.ndarray, simplify_eps: float) -> list[np.ndarray]:
    """Decompose a 1-pixel-wide skeleton into individual polylines. Junctions
    (3+ neighbours) split the skeleton into chains; each chain is walked
    pixel-by-pixel from one endpoint to the other, then bridged back to its
    adjacent junctions so the network stays connected. Same algorithm as
    vitrage_to_dxf — inlined here to avoid a fragile cross-folder import."""
    kern = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nbrs = cv2.filter2D(sk, ddepth=cv2.CV_8U, kernel=kern) * sk
    junctions_mask = ((nbrs >= 3) & (sk > 0)).astype(np.uint8)
    j_y, j_x = np.where(junctions_mask > 0)
    j_set = set(zip(j_x.tolist(), j_y.tolist()))

    chains = sk.copy()
    chains[junctions_mask > 0] = 0
    num, labels, _, _ = cv2.connectedComponentsWithStats(chains, connectivity=8)

    DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, -1), (1, -1), (-1, 1)]

    def adj_junction(pt):
        x, y = pt
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                cand = (x + dx, y + dy)
                if cand in j_set:
                    return cand
        return None

    polylines: list[np.ndarray] = []
    for lid in range(1, num):
        ys, xs = np.where(labels == lid)
        if len(xs) == 0:
            continue
        pixels = set(zip(xs.tolist(), ys.tolist()))
        endpoints = []
        for (x, y) in pixels:
            cnt = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    if (x + dx, y + dy) in pixels:
                        cnt += 1
            if cnt <= 1:
                endpoints.append((x, y))
        start = endpoints[0] if endpoints else next(iter(pixels))
        ordered = [start]
        visited = {start}
        current = start
        while True:
            cx, cy = current
            next_pt = None
            for (dx, dy) in DIRS:
                cand = (cx + dx, cy + dy)
                if cand in pixels and cand not in visited:
                    next_pt = cand
                    break
            if next_pt is None:
                break
            ordered.append(next_pt)
            visited.add(next_pt)
            current = next_pt
        j0 = adj_junction(ordered[0])
        if j0 is not None and j0 != ordered[0]:
            ordered.insert(0, j0)
        j1 = adj_junction(ordered[-1])
        if j1 is not None and j1 not in (ordered[0], ordered[-1]):
            ordered.append(j1)
        if len(ordered) < 2:
            continue
        pts = np.array(ordered, dtype=np.float32).reshape(-1, 1, 2)
        if simplify_eps > 0:
            pts = cv2.approxPolyDP(pts, simplify_eps, closed=False)
        polylines.append(pts.reshape(-1, 2).astype(np.float64))
    return polylines


# Labels in MosaicToCsv's toolbar we want to hide alongside their widgets.
_HIDDEN_LABELS = {"Block size:", "C (offset):"}


# ---------------------------------------------------------------------------
# Vector editor — QGraphicsView-based line drawing with movable control points
# ---------------------------------------------------------------------------

class ControlPointItem(QGraphicsEllipseItem):
    """A draggable yellow circle representing one polyline vertex. Qt's
    ``ItemIsMovable`` flag does all the drag heavy lifting for us — we just
    relay the new position to the parent view via ``itemChange``."""

    def __init__(self, view: "VectorEditorView", poly_idx: int, vert_idx: int,
                 x: float, y: float, radius: float):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.view = view
        self.poly_idx = poly_idx
        self.vert_idx = vert_idx
        self.setPos(x, y)
        self.setBrush(QBrush(QColor(255, 230, 0)))
        self.setPen(QPen(QColor(30, 30, 30), 1))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        # ItemSendsGeometryChanges is required for ItemPositionChange to be
        # delivered (we intercept that to implement snap-to-merge).
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        # Ignore the view's zoom so the dot keeps the same screen size when
        # the user zooms in / out — easier to grab at high zoom.
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(10)
        self.setCursor(Qt.OpenHandCursor)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(QColor(255, 255, 120)))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(QColor(255, 230, 0)))
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        # ItemPositionChange fires BEFORE Qt commits the move — intercept it
        # to snap onto a nearby control point from another polyline. Returning
        # the snapped QPointF makes that the actual new position.
        if change == QGraphicsItem.ItemPositionChange and self.view is not None:
            snapped = self.view._snap_position(self.poly_idx, value)
            if snapped is not None:
                value = snapped
        # ItemPositionHasChanged fires after the move (with the value Qt
        # actually committed) — push it to the view so adjacent line items
        # update in real time.
        elif change == QGraphicsItem.ItemPositionHasChanged and self.view is not None:
            self.view._on_cp_moved(self.poly_idx, self.vert_idx,
                                   float(value.x()), float(value.y()))
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        # Erase mode short-circuits: clicking a CP removes it, no drag.
        if self.view is not None and self.view.mode == "erase":
            self.view._remove_control_point(self.poly_idx, self.vert_idx)
            event.accept()
            return
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)


class VectorEditorView(QGraphicsView):
    """Vector line-editing view. Holds the source image as a background
    pixmap and the polyline geometry as `QGraphicsLineItem` segments plus
    movable `ControlPointItem` vertices. Three modes (set via ``set_mode``):

      - ``"default"``: drag control points; clicks on empty space pan/zoom
      - ``"draw"``:    click + drag draws a new polyline (committed on release)
      - ``"erase"``:   click on a CP removes it; click on a segment removes
                       the whole polyline
    """

    LINE_COLOR = QColor(220, 30, 30)            # red — easy to see on most images
    DRAW_PREVIEW_COLOR = QColor(255, 130, 130)  # pale red for the in-progress stroke
    SNAP_SCREEN_TOL = 14.0                       # screen pixels — snap radius

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))
        # Wheel zoom is centered on the cursor — this is the magic line.
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)

        self.background_item: QGraphicsPixmapItem | None = None
        # polylines[poly_idx] = Nx2 np.float64 array (kept in sync with items).
        self.polylines: list[np.ndarray] = []
        # cp_items[poly_idx][vert_idx] = ControlPointItem
        self.cp_items: list[list[ControlPointItem]] = []
        # line_items[poly_idx][segment_idx] = QGraphicsLineItem connecting
        # vertices [segment_idx] and [segment_idx + 1].
        self.line_items: list[list[QGraphicsLineItem]] = []

        self.mode: str = "default"
        self.line_width: int = 2

        # In-progress free-form draw stroke (scene coords).
        self._draw_path: list[QPointF] = []
        self._draw_preview_item: QGraphicsPathItem | None = None

    # ----- public API ----------------------------------------------------

    def set_background(self, qimage: QImage) -> None:
        if self.background_item is not None:
            self._scene.removeItem(self.background_item)
            self.background_item = None
        pix = QPixmap.fromImage(qimage)
        self.background_item = self._scene.addPixmap(pix)
        self.background_item.setZValue(-10)
        self._scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        self.resetTransform()
        self.fitInView(self.background_item, Qt.KeepAspectRatio)

    def set_polylines(self, polylines: list[np.ndarray]) -> None:
        # Clear previous items.
        for cps in self.cp_items:
            for cp in cps:
                self._scene.removeItem(cp)
        for lines in self.line_items:
            for line in lines:
                self._scene.removeItem(line)
        self.cp_items = []
        self.line_items = []
        self.polylines = [np.asarray(pl, dtype=np.float64).copy() for pl in polylines]
        for pi in range(len(self.polylines)):
            self._build_items_for_polyline(pi)

    def set_line_width(self, w: int) -> None:
        self.line_width = max(1, int(w))
        # Update existing line items' pen widths.
        pen = self._make_line_pen()
        for lines in self.line_items:
            for line in lines:
                line.setPen(pen)

    def set_mode(self, mode: str) -> None:
        assert mode in ("default", "draw", "erase")
        self.mode = mode
        # Cursor + draggability changes based on mode.
        cp_movable = (mode == "default")
        for cps in self.cp_items:
            for cp in cps:
                cp.setFlag(QGraphicsItem.ItemIsMovable, cp_movable)
                cp.setCursor(
                    Qt.OpenHandCursor if cp_movable
                    else (Qt.CrossCursor if mode == "draw" else Qt.PointingHandCursor)
                )
        if mode == "draw":
            self.viewport().setCursor(Qt.CrossCursor)
        elif mode == "erase":
            self.viewport().setCursor(Qt.PointingHandCursor)
        else:
            self.viewport().setCursor(Qt.ArrowCursor)

    # ----- internal: item construction -----------------------------------

    def _make_line_pen(self) -> QPen:
        pen = QPen(self.LINE_COLOR, self.line_width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        # Cosmetic = constant-width regardless of view zoom (so a 2-px line
        # stays a thin line even when zoomed-in 10×).
        pen.setCosmetic(True)
        return pen

    def _cp_radius(self) -> float:
        return float(max(4, self.line_width + 2))

    def _build_items_for_polyline(self, poly_idx: int) -> None:
        pl = self.polylines[poly_idx]
        pen = self._make_line_pen()
        lines = []
        for vi in range(len(pl) - 1):
            line = QGraphicsLineItem(
                float(pl[vi][0]), float(pl[vi][1]),
                float(pl[vi + 1][0]), float(pl[vi + 1][1]),
            )
            line.setPen(pen)
            line.setZValue(0)
            self._scene.addItem(line)
            lines.append(line)
        self.line_items.append(lines)
        cps = []
        r = self._cp_radius()
        for vi in range(len(pl)):
            cp = ControlPointItem(
                self, poly_idx, vi,
                float(pl[vi][0]), float(pl[vi][1]), r,
            )
            cp.setFlag(QGraphicsItem.ItemIsMovable, self.mode == "default")
            self._scene.addItem(cp)
            cps.append(cp)
        self.cp_items.append(cps)

    def _reindex_polyline(self, poly_idx: int) -> None:
        """After insertion / removal in the polylines list, the items in
        cp_items / line_items past ``poly_idx`` need their ``poly_idx``
        attribute updated."""
        for pi in range(poly_idx, len(self.cp_items)):
            for cp in self.cp_items[pi]:
                cp.poly_idx = pi

    def _reindex_vertices(self, poly_idx: int) -> None:
        for vi, cp in enumerate(self.cp_items[poly_idx]):
            cp.vert_idx = vi

    # ----- internal: callbacks from ControlPointItem ---------------------

    def _snap_position(self, dragger_poly_idx: int, proposed):
        """If ``proposed`` (a QPointF in scene coords) is within snap tolerance
        of any control point that belongs to a DIFFERENT polyline, return that
        target point's scene position — Qt will use it as the actual new
        position of the dragged item. Returns None if no snap candidate is in
        range, leaving the move unchanged.

        Tolerance is measured in SCREEN pixels (so the user feels the same
        snap radius regardless of zoom) and converted to scene pixels via
        the view's current scale.
        """
        if not self.cp_items:
            return None
        view_scale = float(self.transform().m11())
        if view_scale <= 1e-6:
            view_scale = 1.0
        tol_scene = self.SNAP_SCREEN_TOL / view_scale
        tol_sq = tol_scene * tol_scene
        px, py = float(proposed.x()), float(proposed.y())
        best_pos = None
        best_d2 = tol_sq
        for pi, cps in enumerate(self.cp_items):
            if pi == dragger_poly_idx:
                continue
            for cp in cps:
                cp_pos = cp.pos()
                dx = float(cp_pos.x()) - px
                dy = float(cp_pos.y()) - py
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_pos = cp_pos
        return best_pos  # QPointF or None

    def _on_cp_moved(self, poly_idx: int, vert_idx: int,
                     x: float, y: float) -> None:
        if not (0 <= poly_idx < len(self.polylines)):
            return
        pl = self.polylines[poly_idx]
        if not (0 <= vert_idx < len(pl)):
            return
        pl[vert_idx, 0] = x
        pl[vert_idx, 1] = y
        # Update adjacent line items.
        lines = self.line_items[poly_idx]
        if vert_idx > 0 and (vert_idx - 1) < len(lines):
            prev = lines[vert_idx - 1]
            prev.setLine(
                float(pl[vert_idx - 1][0]), float(pl[vert_idx - 1][1]),
                x, y,
            )
        if vert_idx < len(pl) - 1 and vert_idx < len(lines):
            nxt = lines[vert_idx]
            nxt.setLine(
                x, y,
                float(pl[vert_idx + 1][0]), float(pl[vert_idx + 1][1]),
            )

    def _remove_control_point(self, poly_idx: int, vert_idx: int) -> None:
        if not (0 <= poly_idx < len(self.polylines)):
            return
        pl = self.polylines[poly_idx]
        if len(pl) <= 2:
            self._remove_polyline(poly_idx)
            return
        # Remove the CP item.
        cp = self.cp_items[poly_idx].pop(vert_idx)
        self._scene.removeItem(cp)
        # Drop the two adjacent line items and add the replacement single line
        # joining the previous and next vertices (if they exist).
        new_pl = np.delete(pl, vert_idx, axis=0)
        for line in self.line_items[poly_idx]:
            self._scene.removeItem(line)
        self.line_items[poly_idx] = []
        pen = self._make_line_pen()
        for vi in range(len(new_pl) - 1):
            line = QGraphicsLineItem(
                float(new_pl[vi][0]), float(new_pl[vi][1]),
                float(new_pl[vi + 1][0]), float(new_pl[vi + 1][1]),
            )
            line.setPen(pen)
            self._scene.addItem(line)
            self.line_items[poly_idx].append(line)
        self.polylines[poly_idx] = new_pl
        self._reindex_vertices(poly_idx)

    def _remove_polyline(self, poly_idx: int) -> None:
        if not (0 <= poly_idx < len(self.polylines)):
            return
        for cp in self.cp_items[poly_idx]:
            self._scene.removeItem(cp)
        for line in self.line_items[poly_idx]:
            self._scene.removeItem(line)
        self.cp_items.pop(poly_idx)
        self.line_items.pop(poly_idx)
        self.polylines.pop(poly_idx)
        self._reindex_polyline(poly_idx)

    # ----- input handling -------------------------------------------------

    def wheelEvent(self, event):
        """Cursor-centered zoom — magic enabled by AnchorUnderMouse."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.20 if delta > 0 else (1.0 / 1.20)
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        scene_pos = self.mapToScene(event.pos())
        if self.mode == "draw":
            self._draw_path = [scene_pos]
            self._refresh_draw_preview()
            event.accept()
            return
        if self.mode == "erase":
            # Click on a CP or a line item.
            item = self.itemAt(event.pos())
            if isinstance(item, ControlPointItem):
                self._remove_control_point(item.poly_idx, item.vert_idx)
                event.accept()
                return
            if isinstance(item, QGraphicsLineItem):
                for pi, lines in enumerate(self.line_items):
                    if item in lines:
                        self._remove_polyline(pi)
                        event.accept()
                        return
            event.accept()
            return
        # Default mode: let Qt handle CP drag (or pass through for empty area).
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.mode == "draw" and (event.buttons() & Qt.LeftButton):
            scene_pos = self.mapToScene(event.pos())
            # Avoid spamming duplicates if the cursor doesn't move.
            if not self._draw_path or self._draw_path[-1] != scene_pos:
                self._draw_path.append(scene_pos)
                self._refresh_draw_preview()
            event.accept()
            return
        if self.mode == "erase" and (event.buttons() & Qt.LeftButton):
            # Continuous erase: process whatever's under the cursor.
            item = self.itemAt(event.pos())
            if isinstance(item, ControlPointItem):
                self._remove_control_point(item.poly_idx, item.vert_idx)
            elif isinstance(item, QGraphicsLineItem):
                for pi, lines in enumerate(self.line_items):
                    if item in lines:
                        self._remove_polyline(pi)
                        break
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self.mode == "draw":
            self._commit_draw_stroke()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _refresh_draw_preview(self) -> None:
        if self._draw_preview_item is not None:
            self._scene.removeItem(self._draw_preview_item)
            self._draw_preview_item = None
        if len(self._draw_path) < 2:
            return
        path = QPainterPath()
        path.moveTo(self._draw_path[0])
        for pt in self._draw_path[1:]:
            path.lineTo(pt)
        item = QGraphicsPathItem(path)
        pen = QPen(self.DRAW_PREVIEW_COLOR, self.line_width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(20)
        self._scene.addItem(item)
        self._draw_preview_item = item

    def _commit_draw_stroke(self) -> None:
        if self._draw_preview_item is not None:
            self._scene.removeItem(self._draw_preview_item)
            self._draw_preview_item = None
        if len(self._draw_path) < 2:
            self._draw_path = []
            return
        # Douglas-Peucker simplification so we get a few real control points
        # rather than one per move event.
        pts = np.array(
            [(p.x(), p.y()) for p in self._draw_path], dtype=np.float32,
        ).reshape(-1, 1, 2)
        eps = max(1.0, float(self.line_width) * 0.75)
        simplified = cv2.approxPolyDP(pts, eps, closed=False).reshape(-1, 2)
        if len(simplified) >= 2:
            self.polylines.append(simplified.astype(np.float64))
            self._build_items_for_polyline(len(self.polylines) - 1)
        self._draw_path = []


class VoronoiToCsv(_MosaicToCsv):
    """MosaicToCsv with orange-line detection instead of adaptive threshold."""

    # HSV bounds for solid #FF6600. OpenCV H is 0-179, S/V 0-255. The high
    # S / V floors reject anti-aliased line edges, orange-tinted skin tones,
    # and warm photographic content.
    HSV_LOW = (8, 200, 180)
    HSV_HIGH = (16, 255, 255)

    def __init__(self) -> None:
        # Set subclass state BEFORE super().__init__(): the parent's __init__
        # calls self._update_buttons() at the end of its setup, and Python's
        # MRO dispatches that to OUR override, which reads these attributes.
        # If we set them after super(), the first call sees no attribute and
        # raises AttributeError → the window never finishes constructing.
        self.orange_mask: np.ndarray | None = None

        # Vector representation of the line drawing: list of Nx2 polylines.
        # Built from the orange mask by Isolate / Thin; then editable via the
        # Draw / Erase tools. Rendered with visible control points.
        self.polylines: list[np.ndarray] | None = None

        # Editing state.
        self.draw_mode: bool = False
        self.erase_mode: bool = False
        self._dragging: bool = False
        # Drag-time scratch polyline (the line being drawn right now, before
        # it's committed on mouse release).
        self._current_stroke: list[tuple[float, float]] = []
        # Active control-point drag (set in default mode when the user
        # left-clicks on a control point). None when not dragging a CP.
        self._cp_drag: tuple[int, int] | None = None

        super().__init__()
        self.setWindowTitle("Voronoi → CSV (BEERY)")
        self.source_pane.title_label.setText("Source Voronoi image (orange-lined)")
        self.result_pane.title_label.setText("Detected cells (CSV preview)")
        self.tile_count_label.setText("Cells: —")

        # Insert the vector editor between the source and result panes so all
        # three are visible at once: Source | Editor | Result.
        self.editor_view = VectorEditorView()
        splitter = self.result_pane.parent()
        if hasattr(splitter, "insertWidget"):
            result_index = splitter.indexOf(self.result_pane)
            splitter.insertWidget(result_index, self.editor_view)
            # Re-balance: source / editor / result.
            total = sum(splitter.sizes()) or 1500
            third = max(300, total // 3)
            splitter.setSizes([third, third, total - 2 * third])

        # Hide the inherited controls that don't apply to orange detection.
        self.solid_white_chk.setChecked(False)
        self.solid_white_chk.setVisible(False)
        self.block_size_spin.setVisible(False)
        self.C_spin.setVisible(False)
        # Key file is only for the AI-mask path, which we don't use.
        self.key_btn.setVisible(False)
        for label in self.findChildren(QLabel):
            if label.text() in _HIDDEN_LABELS:
                label.setVisible(False)

        # Add Isolate orange → [Line width spinbox + Thin button] → Detect → Save.
        self.isolate_btn = QPushButton("Isolate orange")
        self.isolate_btn.setToolTip(
            "Step 1: extract ONLY the solid #FF6600 dividing line. The "
            "right pane shows the isolated line (orange on dark grey) so "
            "you can verify before running Detect."
        )
        self.isolate_btn.clicked.connect(self.isolate_orange)

        # Step 2: skeletonise + re-inflate the mask to a uniform width.
        self.target_width_spin = QSpinBox()
        self.target_width_spin.setRange(1, 30)
        self.target_width_spin.setValue(2)
        self.target_width_spin.setSuffix(" px")
        self.target_width_spin.setToolTip(
            "Target line width in pixels after the Thin line step."
        )
        self.thin_btn = QPushButton("Thin line")
        self.thin_btn.setToolTip(
            "Skeletonise the isolated orange mask to a 1-pixel centerline "
            "and re-inflate it to a UNIFORM width = the value to the left. "
            "Useful when the AI's orange line has varying thickness — this "
            "produces a clean, even line that the polygon-detection step "
            "can extract more reliably."
        )
        self.thin_btn.clicked.connect(self.thin_line)
        self._thin_label = QLabel("Line width:")

        # Draw / Erase toggles — both use the same Line width above. Click
        # + drag on the right pane to draw a new line or erase an existing
        # one. The buttons are mutually exclusive.
        self.draw_btn = QPushButton("Draw")
        self.draw_btn.setCheckable(True)
        self.draw_btn.setToolTip(
            "Toggle Draw mode: click and drag on the line drawing in the "
            "right pane to add new line segments at the chosen Line width."
        )
        self.draw_btn.clicked.connect(self._on_draw_toggled)

        self.erase_btn = QPushButton("Erase")
        self.erase_btn.setCheckable(True)
        self.erase_btn.setToolTip(
            "Toggle Erase mode: click and drag on the line drawing in the "
            "right pane to remove line pixels along the cursor path "
            "(eraser width = the chosen Line width)."
        )
        self.erase_btn.clicked.connect(self._on_erase_toggled)

        # Locate the toolbar QHBoxLayout that holds Load Image, and insert
        # our new widgets right after it in left-to-right order.
        target_layout = None
        central = self.centralWidget()
        if central is not None and central.layout() is not None:
            root_layout = central.layout()
            for i in range(root_layout.count()):
                sub = root_layout.itemAt(i).layout()
                if sub is None:
                    continue
                for j in range(sub.count()):
                    if sub.itemAt(j).widget() is self.load_btn:
                        target_layout = sub
                        insert_at = j + 1
                        for widget in (
                            self.isolate_btn,
                            self._thin_label,
                            self.target_width_spin,
                            self.thin_btn,
                            self.draw_btn,
                            self.erase_btn,
                        ):
                            target_layout.insertWidget(insert_at, widget)
                            insert_at += 1
                        break
                if target_layout is not None:
                    break
        if target_layout is None:
            # Fallback: append to whichever layout the parent uses.
            bar_layout = self.load_btn.parentWidget().layout()
            if bar_layout is not None:
                for widget in (
                    self.isolate_btn, self._thin_label,
                    self.target_width_spin, self.thin_btn,
                    self.draw_btn, self.erase_btn,
                ):
                    bar_layout.addWidget(widget)

        # The vector editor handles its own input — no event filter on the
        # bitmap result pane is needed any more.
        # When the user changes the line-width spin, push it straight into
        # the editor so the displayed line / CP size updates immediately.
        self.target_width_spin.valueChanged.connect(
            lambda w: self.editor_view.set_line_width(int(w)),
        )

        # Voronoi-specific defaults: cells are larger than mosaic tesserae so
        # raise min-area, and the Voronoi edges are mostly straight so a
        # moderate Douglas-Peucker epsilon is enough.
        self.min_area_spin.setRange(0, 1_000_000)
        self.min_area_spin.setValue(500)
        self.eps_spin.setRange(0.0, 0.2)
        self.eps_spin.setDecimals(4)
        self.eps_spin.setSingleStep(0.0005)
        self.eps_spin.setValue(0.005)

    # ----- state ----------------------------------------------------------

    def load_image(self) -> None:
        """Inherit the parent's load flow, then clear cached state from the
        previous image (orange mask + polylines + any drag in progress)."""
        super().load_image()
        self.orange_mask = None
        self.polylines = None
        self._dragging = False
        self._current_stroke = []
        self._cp_drag = None
        if hasattr(self, "draw_btn"):
            self.draw_btn.setChecked(False)
            self.draw_mode = False
        if hasattr(self, "erase_btn"):
            self.erase_btn.setChecked(False)
            self.erase_mode = False
        self._update_buttons()

    def _update_buttons(self) -> None:
        super()._update_buttons()
        running = self.worker is not None and self.worker.isRunning()
        has_src = self.source_rgb is not None and not running
        has_mask = self.orange_mask is not None and not running
        has_polylines = (
            getattr(self, "polylines", None) is not None and not running
        )
        if hasattr(self, "isolate_btn"):
            self.isolate_btn.setEnabled(has_src)
        if hasattr(self, "thin_btn"):
            self.thin_btn.setEnabled(has_mask)
            self.target_width_spin.setEnabled(has_mask)
        if hasattr(self, "draw_btn"):
            # Draw and Erase operate on the polyline list, not the mask, so
            # they activate once Isolate orange has produced a vector layer.
            self.draw_btn.setEnabled(has_polylines)
            self.erase_btn.setEnabled(has_polylines)
            if not has_polylines:
                if self.draw_btn.isChecked():
                    self.draw_btn.setChecked(False)
                if self.erase_btn.isChecked():
                    self.erase_btn.setChecked(False)
                self.draw_mode = False
                self.erase_mode = False
                self._dragging = False
                self._current_stroke = []
                self._cp_drag = None

    # ----- draw / erase ---------------------------------------------------

    def _on_draw_toggled(self, checked: bool) -> None:
        self.draw_mode = checked
        if checked:
            self.erase_btn.setChecked(False)
            self.erase_mode = False
        self._push_mode_to_editor()

    def _on_erase_toggled(self, checked: bool) -> None:
        self.erase_mode = checked
        if checked:
            self.draw_btn.setChecked(False)
            self.draw_mode = False
        self._push_mode_to_editor()

    def _push_mode_to_editor(self) -> None:
        mode = "draw" if self.draw_mode else ("erase" if self.erase_mode else "default")
        self.editor_view.set_mode(mode)

    # Kept under the original name for any leftover callers; just forwards.
    def _update_cursor(self) -> None:
        self._push_mode_to_editor()

    # ----- step 1: isolate the orange line --------------------------------

    def isolate_orange(self) -> bool:
        """Step 1: HSV-threshold the solid #FF6600 line pixels, close 1-px
        gaps, then VECTORISE the resulting mask into polylines with control
        points. The right pane shows the vector line drawing.

        Returns True if at least one polyline was extracted; False otherwise.
        """
        if self.source_rgb is None:
            return False
        rgb = self.source_rgb

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        orange_mask = cv2.inRange(hsv, self.HSV_LOW, self.HSV_HIGH)
        n_orange = int((orange_mask > 0).sum())
        if n_orange == 0:
            QMessageBox.warning(
                self, "No orange lines",
                "Couldn't find any solid #FF6600 pixels in the image. "
                "This tool expects a Voronoi panel generated with the "
                "'Orange dividing lines' option in image_to_voronoi.py.",
            )
            self.orange_mask = None
            self.polylines = None
            return False

        # Bridge 1-px gaps so the line network is fully connected.
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kern)
        self.orange_mask = orange_mask

        # Vectorise: skeletonise → walk into polylines, simplify with
        # Douglas-Peucker (1 px) so we end up with a manageable number of
        # control points per polyline rather than one per pixel.
        skel = skeletonize(orange_mask > 0).astype(np.uint8)
        self.polylines = _skeleton_to_polylines(skel, simplify_eps=1.0)

        # Push into the vector editor view and switch to it.
        self._show_editor_with_polylines()

        n_polys = len(self.polylines)
        n_pts = sum(len(p) for p in self.polylines)
        self.statusBar().showMessage(
            f"Step 1 done — {n_orange} orange pixels isolated → "
            f"{n_polys} polylines, {n_pts} control points. "
            f"Edit with Draw / Erase / drag the dots, then click Detect.",
        )
        self.tile_count_label.setText(f"Polylines: {n_polys}")
        # Tiles from any previous run are stale once the lines are rebuilt.
        self.tiles = []
        self._update_buttons()
        return True

    def _show_editor_with_polylines(self) -> None:
        """Populate the vector editor view with the source image and the
        current polyline list, then bring it to the front of the stack."""
        if self.source_rgb is None:
            return
        rgb = np.ascontiguousarray(self.source_rgb)
        h, w = rgb.shape[:2]
        qimg = QImage(
            rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888,
        ).copy()
        self.editor_view.set_line_width(int(self.target_width_spin.value()))
        self.editor_view.set_background(qimg)
        self.editor_view.set_polylines(self.polylines)
        self.editor_view.set_mode(
            "draw" if self.draw_mode
            else ("erase" if self.erase_mode else "default")
        )

    # ----- step 1b: thin the line to a uniform width ----------------------

    def thin_line(self) -> None:
        """Skeletonise self.orange_mask down to a 1-pixel centerline, then
        re-inflate to a uniform width given by ``target_width_spin``. Useful
        when the AI's orange line has variable thickness — produces a clean
        uniform line that Detect can extract more reliably.

        Implementation: skeletonise (skimage), then use the distance
        transform of the skeleton's inverse to pick all pixels within
        ``(width - 1) / 2`` of the centerline. That gives exactly `width`
        pixels of uniform thickness regardless of how thick the original
        line was.
        """
        if self.orange_mask is None:
            QMessageBox.warning(
                self, "No orange mask",
                "Run 'Isolate orange' first so there's a mask to thin.",
            )
            return

        target = int(self.target_width_spin.value())
        skel = skeletonize(self.orange_mask > 0).astype(np.uint8) * 255
        if int((skel > 0).sum()) == 0:
            QMessageBox.warning(
                self, "Empty skeleton",
                "Skeletonising the orange mask produced no pixels. The "
                "input mask may already be empty.",
            )
            return

        if target <= 1:
            new_mask = skel
        else:
            non_skel = (skel == 0).astype(np.uint8) * 255
            dist = cv2.distanceTransform(non_skel, cv2.DIST_L2, 3)
            new_mask = (dist <= (target - 1) / 2.0).astype(np.uint8) * 255

        self.orange_mask = new_mask

        # Re-vectorise from the cleaned-up skeleton (always use skel here, not
        # new_mask — skeleton walking expects 1-px-wide input).
        self.polylines = _skeleton_to_polylines(skel, simplify_eps=1.0)

        n_polys = len(self.polylines)
        n_pts = sum(len(p) for p in self.polylines)
        self.statusBar().showMessage(
            f"Thinned to {target}-px uniform width → "
            f"{n_polys} polylines, {n_pts} control points.",
        )
        self.tile_count_label.setText(f"Polylines: {n_polys}")
        # Polygons from a previous Detect are stale.
        self.tiles = []
        self._show_editor_with_polylines()
        self._update_buttons()

    # ----- step 2: detect polygons from the isolated line -----------------

    def detect(self) -> None:
        """Extract cell polygons from the current vector line drawing. If
        Isolate orange hasn't been run yet, run it first as a convenience.
        After any Draw/Erase edits, the polyline list takes precedence over
        the raw orange mask — we rasterise the polylines fresh at the
        chosen line width and run connected components on that."""
        if self.source_rgb is None:
            return
        if self.polylines is None:
            if not self.isolate_orange():
                return

        # The vector editor is the source of truth — pull the user's latest
        # edits (CP drags / draws / erases) back into self.polylines.
        self.polylines = [pl.copy() for pl in self.editor_view.polylines]
        if not self.polylines:
            QMessageBox.warning(
                self, "No polylines",
                "The vector line drawing is empty. Run Isolate orange again "
                "or draw some lines with the Draw tool.",
            )
            return

        rgb = self.source_rgb
        h, w = rgb.shape[:2]

        # Rasterise the (possibly edited) polylines back into a clean mask.
        line_w = max(1, int(self.target_width_spin.value()))
        raster = np.zeros((h, w), dtype=np.uint8)
        for pl in self.polylines:
            if len(pl) >= 2:
                cv2.polylines(
                    raster, [pl.astype(np.int32).reshape(-1, 1, 2)],
                    isClosed=False, color=255,
                    thickness=line_w, lineType=cv2.LINE_8,
                )
        # Close any 1-px pinholes so cells don't leak through.
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.orange_mask = cv2.morphologyEx(raster, cv2.MORPH_CLOSE, kern)
        orange_mask = self.orange_mask

        tile_mask = (255 - orange_mask).astype(np.uint8)

        # Pad so cells that touch the image edge become bounded by an outer
        # orange ring instead of merging with the "outside" component.
        PAD = 5
        tile_padded = cv2.copyMakeBorder(
            tile_mask, PAD, PAD, PAD, PAD,
            cv2.BORDER_CONSTANT, value=0,
        )
        rgb_padded = cv2.copyMakeBorder(
            rgb, PAD, PAD, PAD, PAD,
            cv2.BORDER_CONSTANT, value=(255, 102, 0),
        )

        self.statusBar().showMessage("Detecting polygons from orange lines...")
        QApplication.processEvents()

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            tile_padded, connectivity=4,
        )

        min_area = int(self.min_area_spin.value())
        eps_ratio = float(self.eps_spin.value())
        ph, pw = tile_padded.shape[:2]

        tiles = []
        for lid in range(1, num_labels):
            area = int(stats[lid, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x0 = int(stats[lid, cv2.CC_STAT_LEFT])
            y0 = int(stats[lid, cv2.CC_STAT_TOP])
            ww = int(stats[lid, cv2.CC_STAT_WIDTH])
            hh = int(stats[lid, cv2.CC_STAT_HEIGHT])
            # Drop the outer ring (any component reaching the padded edge).
            if x0 == 0 or y0 == 0 or x0 + ww == pw or y0 + hh == ph:
                continue
            sub_mask = (labels[y0:y0 + hh, x0:x0 + ww] == lid).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 3:
                continue
            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = max(0.5, eps_ratio * perimeter)
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            if len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2).astype(np.float64)
            pts[:, 0] += x0 - PAD
            pts[:, 1] += y0 - PAD
            cell_pixels = rgb_padded[labels == lid]
            mean_rgb = (cell_pixels.mean(axis=0) / 255.0).tolist()
            tiles.append((pts, tuple(mean_rgb)))

        self.tiles = tiles
        if not tiles:
            QMessageBox.warning(
                self, "No polygons",
                f"Found {int((orange_mask > 0).sum())} orange pixels but "
                f"extracted zero polygons. Try lowering Min tile area or "
                f"the simplify ε.",
            )
            self.statusBar().showMessage("No polygons detected.")
            self._update_buttons()
            return

        preview = render_polygons(tiles, w, h)
        self.result_pane.set_pil_image(
            preview, f"{len(tiles)} cells  |  {w} × {h} px",
        )
        self.tile_count_label.setText(f"Cells: {len(tiles)}")
        self.statusBar().showMessage(
            f"Detected {len(tiles)} polygons — ready to save as CSV.",
        )
        self._update_buttons()


def main() -> int:
    app = QApplication(sys.argv)
    win = VoronoiToCsv()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
