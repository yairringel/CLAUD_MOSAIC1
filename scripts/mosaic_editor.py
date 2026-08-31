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
import os
import csv
import io
import json
import pickle
import zipfile
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QFrame, QLabel, QPushButton, QFileDialog, QCheckBox,
    QSpinBox, QDoubleSpinBox, QMessageBox, QScrollArea, QColorDialog,
    QInputDialog, QDialog, QDialogButtonBox,
)
from PyQt5.QtCore import Qt, QRectF, QPointF, QBuffer
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
    (None, None) if the polygon has no valid bbox.

    Bbox is the polygon's REAL (unclamped) bbox — pixels outside the
    background image simply stay transparent so Save Tile Image can
    render the polygon in its full extent instead of a clipped strip.

    Mask is built with PIL's ImageDraw.polygon — reliable across Qt/PIL
    versions and doesn't depend on any painter-to-Grayscale8 support."""
    if bg_rgba is None or bg_rgba.size == 0:
        return None, None
    H, W = bg_rgba.shape[:2]

    xs = [p[0] for p in polygon_points]; ys = [p[1] for p in polygon_points]
    x0 = int(np.floor(min(xs))); y0 = int(np.floor(min(ys)))
    x1 = int(np.ceil (max(xs))); y1 = int(np.ceil (max(ys)))
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None, None

    # Allocate the full polygon-bbox RGBA buffer up front — every pixel
    # starts as fully transparent black. Any pixel outside the bg image
    # STAYS transparent; pixels inside get filled from the background.
    crop = np.zeros((h, w, 4), dtype=np.uint8)

    # Rectangle of overlap between polygon bbox and bg image (in bg
    # image coords).
    ix0 = max(0, x0); iy0 = max(0, y0)
    ix1 = min(W, x1); iy1 = min(H, y1)
    if ix1 > ix0 and iy1 > iy0:
        # Where the overlap lands inside our local (0..w, 0..h) buffer.
        lx0 = ix0 - x0; ly0 = iy0 - y0
        lx1 = lx0 + (ix1 - ix0)
        ly1 = ly0 + (iy1 - iy0)
        crop[ly0:ly1, lx0:lx1] = bg_rgba[iy0:iy1, ix0:ix1]
    # (else: polygon entirely off-image → buffer stays fully
    # transparent; only the shape's mask survives after the next step)

    # Build a single-channel alpha mask via PIL. Polygon coords are
    # remapped into the crop's local (0..w, 0..h) space. ImageDraw
    # accepts floats and rasterises the whole polygon in one call.
    from PIL import Image as _PILImage, ImageDraw as _ImageDraw
    mask_img = _PILImage.new('L', (w, h), 0)
    poly_local = [(float(px - x0), float(py - y0))
                  for (px, py) in polygon_points]
    _ImageDraw.Draw(mask_img).polygon(poly_local, fill=255)
    mask_arr = np.array(mask_img, dtype=np.uint8)   # (h, w) uint8

    # Multiply the crop's existing alpha channel by the mask. Pixels
    # OUTSIDE the polygon shape become fully transparent. Pixels INSIDE
    # the polygon but outside the bg image keep alpha 0 too (the
    # `crop` buffer starts at zero everywhere), which prints as
    # transparent-white when the tile is composited on the paper
    # background later — same as never having a fill there.
    crop[:, :, 3] = (
        (crop[:, :, 3].astype(np.uint16) * mask_arr.astype(np.uint16)) // 255
    ).astype(np.uint8)

    # Convert to a QImage that owns its data (copy() detaches the
    # numpy buffer so it survives after `crop` goes out of scope).
    fill_qimg = QImage(crop.data, w, h, w * 4,
                       QImage.Format_RGBA8888).copy()
    return fill_qimg, (x0, y0, w, h)


def _qimage_to_numpy_rgba(qimg: QImage) -> np.ndarray | None:
    """Convert a QImage to an (H, W, 4) uint8 RGBA numpy array. Robust
    against padded rows (bytesPerLine > w*4) and against numpy versions
    that refuse to reshape non-contiguous slices."""
    if qimg is None or qimg.isNull():
        return None
    qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    stride = qimg.bytesPerLine()
    ptr = qimg.constBits(); ptr.setsize(h * stride)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, stride)
    if stride == w * 4:
        return arr.reshape(h, w, 4).copy()
    # Padded rows — copy each row's first w*4 bytes into a fresh
    # (h, w, 4) contiguous buffer.
    out = np.empty((h, w, 4), dtype=np.uint8)
    for i in range(h):
        out[i] = arr[i, : w * 4].reshape(w, 4)
    return out


def _resample_polygon_boundary(pts, n):
    """Resample a polygon boundary to n points, evenly spaced by arc
    length. Returns an (n, 2) float32 array. Ported from
    image_strech._resample_polygon_boundary."""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 3:
        return pts
    loop = np.vstack([pts, pts[:1]])
    seg_diffs = np.diff(loop, axis=0)
    seg_lens  = np.linalg.norm(seg_diffs, axis=1)
    cum_lens  = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_lens[-1]
    if total <= 0:
        return pts
    target_lens = np.linspace(0.0, total, n, endpoint=False)
    out = np.zeros((n, 2), dtype=np.float32)
    for i, t in enumerate(target_lens):
        seg_i = int(np.searchsorted(cum_lens, t, side="right") - 1)
        seg_i = max(0, min(len(seg_lens) - 1, seg_i))
        seg_len = max(float(seg_lens[seg_i]), 1e-6)
        frac = (t - cum_lens[seg_i]) / seg_len
        p0 = pts[seg_i]
        p1 = pts[(seg_i + 1) % len(pts)]
        out[i] = p0 + frac * (p1 - p0)
    return out


def warp_fill_piecewise_affine(src_qimg: QImage, src_bbox, src_points,
                               tgt_points, n_resample: int = 24):
    """Warp `src_qimg` (a polygon-shaped RGBA crop with top-left at
    `src_bbox[0], src_bbox[1]` on the canvas, covering `src_points`
    in world coords) so it fits `tgt_points` (new polygon shape in
    world coords). Uses fan triangulation from the centroid and
    per-triangle cv2.getAffineTransform + cv2.warpAffine.

    Returns (new_qimg, new_bbox) where new_bbox is
    (min_x, min_y, tw, th) in world coords. Returns (None, None) if
    the inputs are degenerate."""
    if src_qimg is None or src_qimg.isNull():
        return None, None
    if len(src_points) < 3 or len(tgt_points) < 3:
        return None, None

    src_rgba = _qimage_to_numpy_rgba(src_qimg)
    if src_rgba is None:
        return None, None

    # Local (src_qimg) coords of the source polygon.
    src_pts_local = np.array(
        [(p[0] - src_bbox[0], p[1] - src_bbox[1]) for p in src_points],
        dtype=np.float32,
    )
    # Target polygon's bbox on canvas + local coords.
    tgt_arr = np.array(tgt_points, dtype=np.float32)
    tx1 = float(tgt_arr[:, 0].min()); ty1 = float(tgt_arr[:, 1].min())
    tx2 = float(tgt_arr[:, 0].max()); ty2 = float(tgt_arr[:, 1].max())
    tw = max(1, int(round(tx2 - tx1)))
    th = max(1, int(round(ty2 - ty1)))
    tgt_pts_local = tgt_arr - np.array([tx1, ty1], dtype=np.float32)

    # Boundary-length resampling for one-to-one triangle correspondence.
    N = n_resample
    src_boundary = _resample_polygon_boundary(src_pts_local, N)
    tgt_boundary = _resample_polygon_boundary(tgt_pts_local, N)
    src_center = src_pts_local.mean(axis=0)
    tgt_center = tgt_pts_local.mean(axis=0)

    # NB: slicing [:, :, :3] returns a non-contiguous view of the RGBA
    # buffer, which cv2.warpAffine handles incorrectly on some builds
    # (producing an all-zero or shifted output). Force contiguity so
    # every warp reads valid RGB bytes.
    src_rgb = np.ascontiguousarray(src_rgba[:, :, :3])
    warped_rgb   = np.zeros((th, tw, 3), dtype=np.uint8)
    warped_alpha = np.zeros((th, tw),    dtype=np.uint8)

    for i in range(N):
        j = (i + 1) % N
        src_tri = np.array(
            [src_center, src_boundary[i], src_boundary[j]],
            dtype=np.float32,
        )
        tgt_tri = np.array(
            [tgt_center, tgt_boundary[i], tgt_boundary[j]],
            dtype=np.float32,
        )
        v1 = tgt_tri[1] - tgt_tri[0]
        v2 = tgt_tri[2] - tgt_tri[0]
        if abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1e-3:
            continue

        M = cv2.getAffineTransform(src_tri, tgt_tri)
        warped_full = cv2.warpAffine(
            src_rgb, M, (tw, th),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        tri_mask = np.zeros((th, tw), dtype=np.uint8)
        cv2.fillConvexPoly(
            tri_mask,
            np.round(tgt_tri).astype(np.int32),
            255,
        )
        sel = tri_mask > 0
        warped_rgb  [sel] = warped_full[sel]
        warped_alpha[sel] = 255

    warped_rgba = np.ascontiguousarray(
        np.dstack([warped_rgb, warped_alpha])
    )
    # Some PyQt5 builds don't accept a numpy memoryview via `.data` —
    # they treat it as an opaque pointer whose lifetime doesn't extend
    # past the numpy array. tobytes() forces an explicit bytes copy
    # QImage can safely wrap; then .copy() detaches into a QImage that
    # owns its own buffer. Robust across builds.
    new_qimg = QImage(warped_rgba.tobytes(), tw, th, tw * 4,
                      QImage.Format_RGBA8888).copy()
    return new_qimg, (tx1, ty1, tw, th)


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
        # Absolute path of the file `canvas_background` was loaded from
        # (or None if none set). Save Project stores just this path;
        # Load Project reloads the file so the .mep never carries the
        # background bitmap.
        self.canvas_background_source: str | None = None
        # World-space position of the canvas background's top-left
        # corner. Moved by the sidebar's arrow buttons when
        # bg_affected is True.
        self.bg_offset_x = 0.0
        self.bg_offset_y = 0.0
        # Cumulative scale factor applied to the canvas background when
        # bg_affected is True. Displayed size = (orig × bg_scale), so
        # values >1 grow the background, <1 shrink it. Kept as float so
        # repeated Apply calls compound cleanly.
        self.bg_scale    = 1.0
        self.bg_affected = True

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

        # Show polygon outline strokes. When False, only the filled
        # interiors render — useful when you want to preview the mosaic
        # as-printed. The SELECTED polygon still gets its red outline
        # so selection is always visible regardless of this flag.
        self.show_polygon_lines = True

        # Interaction state.
        self.selected_index = -1
        self._dragging_polygon = False
        self._drag_start_world = (0.0, 0.0)
        self._drag_start_points: list[tuple[float, float]] = []
        self._drag_start_bbox: tuple | None = None
        self._drag_start_orig_points: list[tuple[float, float]] | None = None
        self._drag_start_orig_bbox: tuple | None = None
        # Vertex-drag state: LMB grab of a single control point on the
        # selected polygon. On mouseMove we update just that vertex and
        # re-warp the fill (if any).
        self._dragging_vertex = False
        self._vertex_drag_idx = -1
        # Screen-space hit radius for control-point handles.
        self._vertex_hit_radius = 8

        # Eraser tool — when active, LMB drag sweeps a circular
        # cursor over the canvas and every polygon in an affected
        # layer that intersects the circle gets deleted.
        self.eraser_mode           = False
        self.eraser_radius         = 20.0        # world units
        self._eraser_pressed       = False
        self._eraser_mouse_screen  = (0, 0)      # last known cursor pos

        # Origin-picker tool — when active, LMB click snaps to the
        # nearest gridline intersection and hands (x, y) in world
        # coords to a stored callback. Used by Save Array so the
        # exported CSV's (0,0) is a user-chosen grid intersection.
        self._picking_origin        = False
        self._picking_hover_screen  = (0, 0)
        self._origin_pick_callback  = None

        # Undo — bounded stack of state snapshots. Each snapshot is a
        # dict with deep-copies of polygons + groups + canvas-bg fields.
        # Callers push a snapshot BEFORE mutating; undo() pops and
        # restores. 30 is more than enough for realistic editing.
        self._undo_stack: list[dict] = []
        self._undo_limit = 30
        self._panning = False
        self._pan_start_screen = (0, 0)
        self._pan_start_offset = (0.0, 0.0)

    # ── layers management ─────────────────────────────────────────────
    def add_group(self, name: str, kind: str,
                  source_path: str | None = None) -> int:
        gid = self._next_group_id
        self._next_group_id += 1
        # applied_scale tracks the group's current size as a factor of
        # its original (as-loaded) size. Every scale_affected(pct)
        # brings the group to `pct/100` * original — the ratio to the
        # current applied_scale is what actually gets multiplied into
        # the coordinates, so scaling is always absolute-from-original.
        # source_path (optional, absolute) is where this group's data
        # originally came from — a .mosaic bundle for image-fill layers,
        # a .csv for polygon-only layers. Save Project stores just this
        # path (not the images), and Load Project reloads the file to
        # rebuild every polygon's fill.
        self.groups.append({'id': gid, 'name': name, 'kind': kind,
                            'visible': True, 'affected': True,
                            'applied_scale': 1.0,
                            'source_path': source_path})
        return gid

    def set_group_affected(self, gid: int, affected: bool) -> None:
        for g in self.groups:
            if g['id'] == gid:
                g['affected'] = affected
                break

    def move_affected(self, dx: float, dy: float) -> None:
        """Translate every polygon in `affected` groups (and the canvas
        background if it too is `affected`) by (dx, dy) in world units.
        `dx / dy` are in the same units as polygon coords, so callers
        typically pass ±step from the step-size spinbox."""
        if dx == 0.0 and dy == 0.0:
            return
        self._push_undo()
        affected_gids = {g['id'] for g in self.groups if g.get('affected', True)}
        for poly in self.polygons:
            if poly['group_id'] not in affected_gids:
                continue
            poly['points'] = [(x + dx, y + dy) for x, y in poly['points']]
            if poly.get('fill_bbox') is not None:
                x0, y0, w, h = poly['fill_bbox']
                poly['fill_bbox'] = (x0 + dx, y0 + dy, w, h)
            # Keep the warp reference aligned with the current shape so
            # a subsequent vertex-drag warps from the polygon's actual
            # current position, not from the pre-translation position.
            if poly.get('orig_points') is not None:
                poly['orig_points'] = [
                    (x + dx, y + dy) for x, y in poly['orig_points']
                ]
            if poly.get('orig_fill_bbox') is not None:
                x0, y0, w, h = poly['orig_fill_bbox']
                poly['orig_fill_bbox'] = (x0 + dx, y0 + dy, w, h)
        if self.bg_affected and self.canvas_background is not None:
            self.bg_offset_x += dx
            self.bg_offset_y += dy
        self.update()

    def scale_all_polygons_individual(self, pct: float) -> int:
        """Scale EVERY polygon (regardless of layer / affected flag) by
        `pct` percent around ITS OWN centroid — 110 = each polygon
        grows 10 % about its own centre, 90 = each shrinks 10 %.

        Unlike `scale_affected` which is ABSOLUTE-FROM-ORIGINAL and
        uses per-group centroids, this is a RELATIVE MULTIPLIER applied
        per polygon: applying 110 twice grows every polygon by 21 %,
        applying 100 is a no-op. Returns the number of polygons
        actually scaled.

        source_points is left untouched — it lives in the source
        .mosaic bundle's local coord system, so Load Project's
        `warp_fill_piecewise_affine(source_points → orig_points)`
        naturally captures the new scale."""
        if pct <= 0 or abs(pct - 100.0) < 1e-9:
            return 0
        ratio = pct / 100.0
        if not self.polygons:
            return 0

        self._push_undo()
        n = 0
        for poly in self.polygons:
            pts = poly.get('points') or []
            if len(pts) < 3:
                continue
            cx = sum(x for (x, _) in pts) / len(pts)
            cy = sum(y for (_, y) in pts) / len(pts)
            poly['points'] = [
                (cx + (x - cx) * ratio, cy + (y - cy) * ratio)
                for x, y in pts
            ]
            if poly.get('fill_bbox') is not None:
                x0, y0, w, h = poly['fill_bbox']
                poly['fill_bbox'] = (
                    cx + (x0 - cx) * ratio,
                    cy + (y0 - cy) * ratio,
                    w * ratio,
                    h * ratio,
                )
            if poly.get('orig_points') is not None:
                poly['orig_points'] = [
                    (cx + (x - cx) * ratio, cy + (y - cy) * ratio)
                    for x, y in poly['orig_points']
                ]
            if poly.get('orig_fill_bbox') is not None:
                x0, y0, w, h = poly['orig_fill_bbox']
                poly['orig_fill_bbox'] = (
                    cx + (x0 - cx) * ratio,
                    cy + (y0 - cy) * ratio,
                    w * ratio,
                    h * ratio,
                )
            n += 1
        self.update()
        return n

    def scale_affected(self, pct: float) -> None:
        """Scale every affected element ABSOLUTELY relative to its
        original (as-loaded) size — 100 = original, 200 = 2× original,
        50 = half original, regardless of prior scales. Internally the
        ratio applied is (pct/100) / current applied_scale, so:
          * Apply 200 twice → still 200 % (second apply is a no-op).
          * Apply 100 after any scale → restore original size.
        Each affected polygon group is scaled around its own current
        centroid so moved layers stay put. The canvas background scales
        around its visible centre and its bg_scale is set absolutely."""
        if pct <= 0:
            return
        target_factor = pct / 100.0
        self._push_undo()

        # Bucket affected polygons by group.
        affected_gids = {g['id'] for g in self.groups if g.get('affected', True)}
        buckets: dict[int, list[dict]] = {}
        for poly in self.polygons:
            if poly['group_id'] in affected_gids:
                buckets.setdefault(poly['group_id'], []).append(poly)

        gid_to_group = {g['id']: g for g in self.groups}
        for gid, polys in buckets.items():
            group        = gid_to_group.get(gid)
            current_fac  = group.get('applied_scale', 1.0) if group else 1.0
            if abs(target_factor - current_fac) < 1e-9:
                continue
            ratio = target_factor / current_fac

            all_pts = [pt for p in polys for pt in p['points']]
            if not all_pts:
                continue
            cx = sum(pt[0] for pt in all_pts) / len(all_pts)
            cy = sum(pt[1] for pt in all_pts) / len(all_pts)
            for poly in polys:
                poly['points'] = [
                    (cx + (x - cx) * ratio, cy + (y - cy) * ratio)
                    for x, y in poly['points']
                ]
                if poly.get('fill_bbox') is not None:
                    x0, y0, w, h = poly['fill_bbox']
                    poly['fill_bbox'] = (
                        cx + (x0 - cx) * ratio,
                        cy + (y0 - cy) * ratio,
                        w * ratio,
                        h * ratio,
                    )
                # Same ratio, same centroid → keep the warp reference
                # in lockstep with the visible shape.
                if poly.get('orig_points') is not None:
                    poly['orig_points'] = [
                        (cx + (x - cx) * ratio, cy + (y - cy) * ratio)
                        for x, y in poly['orig_points']
                    ]
                if poly.get('orig_fill_bbox') is not None:
                    x0, y0, w, h = poly['orig_fill_bbox']
                    poly['orig_fill_bbox'] = (
                        cx + (x0 - cx) * ratio,
                        cy + (y0 - cy) * ratio,
                        w * ratio,
                        h * ratio,
                    )
            if group is not None:
                group['applied_scale'] = target_factor

        # Canvas background — set bg_scale absolutely and re-derive
        # bg_offset so the visible CENTRE stays where it was.
        if self.bg_affected and self.canvas_background is not None:
            bg = self.canvas_background
            orig_w, orig_h = bg.width(), bg.height()
            if abs(self.bg_scale - target_factor) > 1e-9:
                center_x = self.bg_offset_x + orig_w * self.bg_scale / 2.0
                center_y = self.bg_offset_y + orig_h * self.bg_scale / 2.0
                self.bg_scale   = target_factor
                self.bg_offset_x = center_x - orig_w * target_factor / 2.0
                self.bg_offset_y = center_y - orig_h * target_factor / 2.0

        self.update()

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
        self._push_undo()
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
        self._push_undo()
        gid = self.add_group(Path(path).name, 'csv')
        for p in polys:
            self.polygons.append({
                'points':    p['points'],
                'fill_type': ('solid' if p['color'].alpha() > 0 else 'none'),
                'color':     p['color'],
                'fill_qimg': None,
                'fill_bbox': None,
                # No image fill on CSV-only polygons, but keep the
                # orig_* keys so save/load / warp code paths don't
                # need to special-case their absence.
                'orig_points':    list(p['points']),
                'orig_fill_qimg': None,
                'orig_fill_bbox': None,
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
        self._push_undo()
        gid = self.add_group(Path(path).name, 'mosaic',
                             source_path=str(Path(path).resolve()))
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
                # Stable warp reference: original polygon shape + fill
                # image at load time. Kept in sync with translations
                # and scales so it always represents the polygon's
                # "pre vertex-drag" shape. Vertex-drag warp uses
                # this as the source so the fill doesn't accumulate
                # quality loss over many small vertex adjustments.
                'orig_points':    list(p['points']),
                'orig_fill_qimg': (fill_qimg.copy()
                                   if fill_qimg is not None else None),
                'orig_fill_bbox': fill_bbox,
                # Snapshot of the polygon's coords in the SOURCE .mosaic's
                # local coordinate system (== `points` at load time, before
                # any translation / scale / vertex-drag). Save Project
                # stores this and Load Project uses it to re-sample the
                # bundle's background.png so the fill can be rebuilt
                # without embedding image bytes in the .mep file.
                'source_points':  [(float(x), float(y))
                                   for (x, y) in p['points']],
                'image_zoom':     1.0,
                'group_id':  gid,
            })
        self.selected_index = -1
        self.update()
        return len(polys)

    def load_canvas_background(self, path: str) -> None:
        pm = QPixmap(path)
        if pm.isNull():
            raise ValueError(f"Could not load image: {path}")
        self._push_undo()
        self.canvas_background = pm
        self.canvas_background_source = str(Path(path).resolve())
        # Reset background offset + scale — a freshly-loaded image
        # starts at world (0, 0) and 1:1, same as a freshly-loaded
        # CSV / mosaic.
        self.bg_offset_x = 0.0
        self.bg_offset_y = 0.0
        self.bg_scale    = 1.0
        self.update()

    def clear_canvas_background(self) -> None:
        if self.canvas_background is None:
            return
        self._push_undo()
        self.canvas_background = None
        self.canvas_background_source = None
        self.bg_offset_x = 0.0
        self.bg_offset_y = 0.0
        self.bg_scale    = 1.0
        self.update()

    # ── project save / load ──────────────────────────────────────────
    #
    # Uses pickle over a dict whose values are all primitives (or
    # PNG-encoded bytes for Qt image types), so the pickle file survives
    # Qt/PyQt version changes and doesn't rely on QColor/QImage having
    # working __reduce__ implementations.
    #
    # V2 (2026): the .mep no longer embeds any high-resolution image
    # bytes. Groups store their source_path, polygons store their
    # source_points (in source-local coords), and the canvas background
    # stores its source path. Load Project reloads the source files and
    # rebuilds every fill via build_image_fill (+ optional
    # warp_fill_piecewise_affine to reach the current shape).
    # V1 files (with fill_png / orig_fill_png / canvas_background_png)
    # still open — the loader falls back to the embedded bytes when a
    # source_path is absent.
    PROJECT_VERSION = 2

    @staticmethod
    def _qcolor_to_tuple(c: QColor) -> tuple:
        return (c.red(), c.green(), c.blue(), c.alpha())

    @staticmethod
    def _tuple_to_qcolor(t) -> QColor:
        r, g, b, a = t
        return QColor(int(r), int(g), int(b), int(a))

    @staticmethod
    def _qimage_to_png_bytes(qimg: QImage | None) -> bytes | None:
        if qimg is None or qimg.isNull():
            return None
        buf = QBuffer(); buf.open(QBuffer.WriteOnly)
        qimg.save(buf, "PNG")
        return bytes(buf.data())

    @staticmethod
    def _png_bytes_to_qimage(data: bytes | None) -> QImage | None:
        if not data:
            return None
        qimg = QImage.fromData(data)
        return qimg if not qimg.isNull() else None

    def save_project(self, path: str) -> None:
        """Serialise the canvas state to a .mep pickle file — INFORMATION
        ONLY. No high-resolution image bytes are stored: instead each
        group carries its `source_path` (e.g. the .mosaic bundle it was
        loaded from) and each polygon carries its `source_points` (its
        outline in that source's local coord system). Load Project
        reloads the source files and rebuilds every polygon's fill via
        build_image_fill + warp_fill_piecewise_affine.

        Result: a typical .mep drops from tens of MB (embedded PNG
        crops) to a few dozen KB (points + metadata), while still
        round-tripping the full scene provided the source files remain
        reachable at load time."""
        data = {
            'version': self.PROJECT_VERSION,
            'groups':  [dict(g) for g in self.groups],
            'next_group_id':          self._next_group_id,
            'polygons': [
                {
                    'points':    list(p['points']),
                    'fill_type': p['fill_type'],
                    'color':     self._qcolor_to_tuple(p['color']),
                    'fill_bbox': p.get('fill_bbox'),
                    # Warp-reference snapshot — needed so vertex-drag
                    # stretches from the stable original geometry
                    # after a project round-trip.
                    'orig_points':    (list(p['orig_points'])
                                       if p.get('orig_points') is not None
                                       else None),
                    'orig_fill_bbox': p.get('orig_fill_bbox'),
                    'image_zoom':     float(p.get('image_zoom', 1.0)),
                    # source_points: the polygon's outline in its source
                    # .mosaic bundle's local coord system (== `points`
                    # at load time). Load Project uses this + the
                    # group's source_path to re-sample the source
                    # background and rebuild the fill without any PNG
                    # bytes in the .mep file. None → polygon has no
                    # image-fill source (e.g. loaded from CSV).
                    'source_points':  (list(p['source_points'])
                                       if p.get('source_points') is not None
                                       else None),
                    'group_id':  p['group_id'],
                } for p in self.polygons
            ],
            # NB: no canvas_background_png — the bitmap is reloaded from
            # canvas_background_source at Load Project time.
            'canvas_background_source': self.canvas_background_source,
            'bg_offset_x': self.bg_offset_x,
            'bg_offset_y': self.bg_offset_y,
            'bg_scale':    self.bg_scale,
            'bg_affected': self.bg_affected,
            'zoom_factor': self.zoom_factor,
            'pan_x':       self.pan_x,
            'pan_y':       self.pan_y,
            'grid_enabled':      self.grid_enabled,
            'grid_size_percent': self.grid_size_percent,
            'grid_size_world':   self.grid_size_world,
            'grid_color':        self._qcolor_to_tuple(self.grid_color),
            'grid_thickness':    self.grid_thickness,
            'grid_offset_x':     self.grid_offset_x,
            'grid_offset_y':     self.grid_offset_y,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def _polygon_mean_color(self, poly: dict) -> QColor:
        """Best guess at a polygon's "dominant" color for CSV writers
        that expect one solid color per polygon. Rules:
          * solid → polygon['color'] unchanged.
          * image → mean RGB of fill_qimg over alpha>0 pixels.
          * none  → opaque white."""
        ft = poly.get('fill_type', 'none')
        if ft == 'solid':
            c = poly.get('color')
            if c is not None:
                return QColor(c)
        if ft == 'image' and poly.get('fill_qimg') is not None:
            arr = _qimage_to_numpy_rgba(poly['fill_qimg'])
            if arr is not None:
                mask = arr[:, :, 3] > 0
                if mask.any():
                    r = int(round(arr[mask, 0].mean()))
                    g = int(round(arr[mask, 1].mean()))
                    b = int(round(arr[mask, 2].mean()))
                    return QColor(r, g, b, 255)
        return QColor(255, 255, 255, 255)

    def save_array_csv(self, path: str, scale: float = 1.0,
                       origin: tuple = (0.0, 0.0)) -> int:
        """Write every visible polygon to a CSV using the shared
        save_array schema (same 11 columns image_strech.py writes).
        Each vertex is transformed as:
              new = (world - origin) * scale
        so `origin` becomes (0, 0) in the exported coordinate system
        and one output unit equals one target grid box. Color columns
        are written as 0,0,0,0 (no color fill — just polygon points).
        Returns the number of polygons written."""
        if not self.polygons:
            return 0
        ox, oy = float(origin[0]), float(origin[1])
        gid_visible = {g['id']: g['visible'] for g in self.groups}
        with open(path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'polygon_id', 'coordinates',
                'color_r', 'color_g', 'color_b', 'color_a',
                'frame_r', 'frame_g', 'frame_b', 'frame_a',
                'group_id',
            ])
            written = 0
            for poly in self.polygons:
                # Skip hidden layers so the CSV matches "what I see".
                if not gid_visible.get(poly['group_id'], True):
                    continue
                pts = poly.get('points') or []
                if len(pts) < 3:
                    continue
                coords_json = json.dumps([
                    [(float(x) - ox) * scale,
                     (float(y) - oy) * scale]
                    for (x, y) in pts
                ])
                writer.writerow([
                    written, coords_json,
                    0.0, 0.0, 0.0, 0.0,           # no color fill
                    0.0, 0.0, 0.0, 0.0,           # no frame fill
                    int(poly.get('group_id', 0)),
                ])
                written += 1
        return written

    def save_mosaic(self, path: str) -> tuple[int, int, int]:
        """Flatten the current canvas into a portable .mosaic bundle:
        one PNG that is the composite of the shared canvas background
        (if any) + every visible polygon's fill, and a polygons.csv
        listing each polygon's outline in that PNG's coordinate space.
        Returns (polygons_written, bg_width, bg_height).

        Semantics vs save_project():
          * Save Project (.mep) preserves per-layer geometry, image
            fills, warp references, image_zoom, etc. — for re-editing.
          * Save Mosaic (.mosaic) is FLATTENED — receivers only see
            the composite PNG + one flat polygon list, exactly the
            format image_strech.py's Save Mosaic writes."""
        if not self.polygons and self.canvas_background is None:
            raise RuntimeError("Canvas is empty — nothing to save.")

        # ── 1. Union bounding box of everything visible on canvas ──
        xs: list[float] = []
        ys: list[float] = []
        if self.canvas_background is not None:
            bg   = self.canvas_background
            bg_w = bg.width()  * self.bg_scale
            bg_h = bg.height() * self.bg_scale
            xs += [self.bg_offset_x, self.bg_offset_x + bg_w]
            ys += [self.bg_offset_y, self.bg_offset_y + bg_h]
        gid_visible = {g['id']: g['visible'] for g in self.groups}
        visible_polys = [
            p for p in self.polygons
            if gid_visible.get(p['group_id'], True)
            and p.get('points') and len(p['points']) >= 3
        ]
        for p in visible_polys:
            for (x, y) in p['points']:
                xs.append(float(x)); ys.append(float(y))
        if not xs:
            raise RuntimeError("Nothing visible to save.")
        x0 = min(xs); y0 = min(ys)
        x1 = max(xs); y1 = max(ys)
        w  = max(1, int(round(x1 - x0)))
        h  = max(1, int(round(y1 - y0)))

        # ── 2. Composite everything into one RGBA image ──
        composite = QImage(w, h, QImage.Format_ARGB32)
        composite.fill(Qt.transparent)
        painter = QPainter(composite)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # 2a. Canvas background (behind polygons).
        if self.canvas_background is not None:
            bg = self.canvas_background
            sx = self.bg_offset_x - x0
            sy = self.bg_offset_y - y0
            sw = bg.width()  * self.bg_scale
            sh = bg.height() * self.bg_scale
            painter.drawPixmap(
                QRectF(sx, sy, sw, sh), bg,
                QRectF(0, 0, bg.width(), bg.height()),
            )

        # 2b. Every visible polygon with its fill.
        for p in visible_polys:
            pts = p['points']
            qpoly = QPolygonF([QPointF(px - x0, py - y0)
                               for (px, py) in pts])
            if p['fill_type'] == 'image' and p.get('fill_qimg') is not None:
                bx, by, bw, bh = p['fill_bbox']
                painter.drawImage(
                    QRectF(bx - x0, by - y0, bw, bh),
                    p['fill_qimg'],
                )
            elif p['fill_type'] == 'solid':
                painter.setBrush(QBrush(p['color']))
                painter.setPen(Qt.NoPen)
                painter.drawPolygon(qpoly)
        painter.end()

        # ── 3. Encode composite as PNG bytes ──
        buf = QBuffer(); buf.open(QBuffer.WriteOnly)
        composite.save(buf, "PNG")
        png_bytes = bytes(buf.data())

        # ── 4. Build polygons.csv (save_array-compatible schema) ──
        # Coordinates are relative to the composite PNG's top-left,
        # so opening the bundle in any receiver aligns polygon
        # outlines to their baked-in pixel content 1:1.
        csv_buf = io.StringIO()
        writer  = csv.writer(csv_buf)
        writer.writerow([
            'polygon_id', 'coordinates',
            'color_r', 'color_g', 'color_b', 'color_a',
        ])
        written = 0
        for p in visible_polys:
            coords_json = json.dumps(
                [[float(px - x0), float(py - y0)] for (px, py) in p['points']]
            )
            # For image-filled polygons write 0,0,0,0 — the receiver
            # should sample from the composite PNG, exactly like the
            # convention image_strech.py's Save Mosaic uses.
            if p['fill_type'] == 'image':
                r = g = b = a = 0.0
            else:
                c = p.get('color') or QColor(0, 0, 0, 0)
                r = c.red()   / 255.0
                g = c.green() / 255.0
                b = c.blue()  / 255.0
                a = c.alpha() / 255.0
            writer.writerow([written, coords_json, r, g, b, a])
            written += 1

        # ── 5. Manifest ──
        manifest = {
            "version": 1,
            "background": "background.png",
            "background_size": [int(w), int(h)],
            "polygons_csv": "polygons.csv",
            "polygon_count": written,
            "notes": (
                "Flattened mosaic saved by mosaic_editor.py. "
                "background.png is the composite of the shared canvas "
                "background plus every visible polygon's image fill "
                "(image-filled polygons drawn from their orig_fill "
                "or fill_qimg, solid-color polygons filled with their "
                "own color). polygons.csv holds each polygon's outline "
                "in the PNG's coordinate space (0,0 = top-left)."
            ),
        }

        # ── 6. Write the ZIP ──
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('polygons.csv',   csv_buf.getvalue())
            zf.writestr('background.png', png_bytes)
            zf.writestr('manifest.json',  json.dumps(manifest, indent=2))

        return (written, w, h)

    def load_project(self, path: str) -> list[str]:
        """Restore canvas state from a .mep pickle file. Replaces every
        current layer / background / transform / grid setting.

        For V2 files (info-only), each polygon's fill is rebuilt by
        re-reading the source .mosaic bundle referenced by its group
        (`build_image_fill` at the polygon's `source_points`, then
        `warp_fill_piecewise_affine` to the current shape). V1 files
        with embedded PNG bytes still load — the bytes are used
        directly and no source-reload is attempted.

        Returns a list of human-readable warnings (missing source files,
        polygons that couldn't be reconstructed, etc.). Empty list on a
        fully-clean load."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or 'polygons' not in data:
            raise ValueError("Not a valid mosaic_editor project file.")

        # Reset transient state.
        self.selected_index      = -1
        self._dragging_polygon   = False
        self._drag_start_bbox    = None
        self._panning            = False

        self.groups         = [dict(g) for g in data.get('groups', [])]
        # Older files predate group['source_path']; default it in so
        # downstream code can always .get without a KeyError.
        for g in self.groups:
            g.setdefault('source_path', None)
        self._next_group_id = int(
            data.get('next_group_id',
                     max((g['id'] for g in self.groups), default=0) + 1)
        )

        polys = []
        for p in data.get('polygons', []):
            polys.append({
                'points':    [(float(x), float(y)) for x, y in p['points']],
                'fill_type': p.get('fill_type', 'none'),
                'color':     self._tuple_to_qcolor(p.get('color',
                                                        (0, 0, 0, 0))),
                # V1 files carry fill_png / orig_fill_png bytes. V2
                # files omit them; we set None here and let the
                # source-reload pass below fill them in.
                'fill_qimg': self._png_bytes_to_qimage(p.get('fill_png')),
                'fill_bbox': (tuple(p['fill_bbox'])
                              if p.get('fill_bbox') is not None else None),
                'orig_points': ([(float(x), float(y))
                                 for x, y in p['orig_points']]
                                if p.get('orig_points') is not None
                                else None),
                'orig_fill_qimg': self._png_bytes_to_qimage(
                    p.get('orig_fill_png')
                ),
                'orig_fill_bbox': (tuple(p['orig_fill_bbox'])
                                   if p.get('orig_fill_bbox') is not None
                                   else None),
                'image_zoom': float(p.get('image_zoom', 1.0)),
                'source_points': ([(float(x), float(y))
                                   for x, y in p['source_points']]
                                  if p.get('source_points') is not None
                                  else None),
                'group_id':  int(p['group_id']),
            })
        self.polygons = polys

        # Canvas background.
        warnings: list[str] = []
        self.canvas_background_source = data.get(
            'canvas_background_source'
        )
        # V2 path: reload the source file. V1 fallback: use the embedded
        # PNG bytes if they're there.
        bg_qimg = None
        if self.canvas_background_source:
            src = self.canvas_background_source
            if Path(src).exists():
                pm = QPixmap(src)
                self.canvas_background = pm if not pm.isNull() else None
                if pm.isNull():
                    warnings.append(
                        f"Canvas background: could not decode {src}"
                    )
            else:
                self.canvas_background = None
                warnings.append(
                    f"Canvas background source missing: {src}"
                )
        else:
            bg_qimg = self._png_bytes_to_qimage(
                data.get('canvas_background_png')
            )
            self.canvas_background = (QPixmap.fromImage(bg_qimg)
                                      if bg_qimg is not None else None)
        self.bg_offset_x = float(data.get('bg_offset_x', 0.0))
        self.bg_offset_y = float(data.get('bg_offset_y', 0.0))
        self.bg_scale    = float(data.get('bg_scale',    1.0))
        self.bg_affected = bool (data.get('bg_affected', True))

        # Reconstruct polygon fills from source .mosaic bundles for
        # every polygon that has (a) a source_points snapshot and (b)
        # a group whose source_path resolves to an existing file. One
        # bundle load per unique source path (cached), so a mosaic with
        # thousands of polygons doesn't re-open the zip per polygon.
        self._reload_fills_from_sources(warnings)

        # View.
        self.zoom_factor = float(data.get('zoom_factor', 1.0))
        self.pan_x       = float(data.get('pan_x',       0.0))
        self.pan_y       = float(data.get('pan_y',       0.0))

        # Grid.
        self.grid_enabled      = bool (data.get('grid_enabled',      False))
        self.grid_size_percent = float(data.get('grid_size_percent', 10.0))
        self.grid_size_world   = float(data.get('grid_size_world',   100.0))
        self.grid_color        = self._tuple_to_qcolor(
            data.get('grid_color', (255, 105, 180, 255))
        )
        self.grid_thickness    = int  (data.get('grid_thickness', 2))
        self.grid_offset_x     = float(data.get('grid_offset_x',  0.0))
        self.grid_offset_y     = float(data.get('grid_offset_y',  0.0))

        # Clear undo history — a project load is a fresh session.
        self._undo_stack.clear()

        self.update()
        return warnings

    def _reload_fills_from_sources(self, warnings: list[str]) -> None:
        """For every polygon whose fill isn't already populated (V2
        info-only projects), rebuild `fill_qimg` / `orig_fill_qimg`
        by re-reading the group's source .mosaic bundle. The polygon's
        `source_points` snapshot (in source-local coords) picks the
        crop; `warp_fill_piecewise_affine` maps it back to the world
        `orig_points` and then to the current `points` shape.

        One bundle per unique `source_path` — cached in a local dict
        so a mosaic with thousands of polygons re-reads the zip once.
        Missing / unreadable sources are appended to `warnings`."""
        groups_by_id = {g['id']: g for g in self.groups}
        cache: dict = {}       # source_path -> bg_np (numpy) or False
        missing_reported: set = set()

        for poly in self.polygons:
            # Already has an image fill (V1 embedded bytes, or built
            # elsewhere) → nothing to do.
            if poly.get('fill_qimg') is not None:
                continue
            if poly.get('fill_type') != 'image':
                continue
            src_pts = poly.get('source_points')
            if not src_pts or len(src_pts) < 3:
                continue

            g = groups_by_id.get(poly.get('group_id'))
            if g is None:
                continue
            src_path = g.get('source_path')
            if not src_path:
                continue

            bg_np = cache.get(src_path)
            if bg_np is None:
                if not Path(src_path).exists():
                    if src_path not in missing_reported:
                        warnings.append(
                            f"Source missing: {src_path} — polygons "
                            f"from group '{g.get('name', '?')}' will "
                            f"load without fills."
                        )
                        missing_reported.add(src_path)
                    cache[src_path] = False
                    continue
                try:
                    _polys, bg_np = load_mosaic_bundle(src_path)
                except Exception as e:
                    warnings.append(
                        f"Source unreadable: {src_path} ({e})"
                    )
                    cache[src_path] = False
                    continue
                if bg_np is None:
                    warnings.append(
                        f"Source has no background.png: {src_path}"
                    )
                    cache[src_path] = False
                    continue
                cache[src_path] = bg_np

            if bg_np is False:
                continue

            # Sample the source at source_points → orig-quality fill in
            # source-local coords.
            src_fill_qimg, src_fill_bbox = build_image_fill(src_pts, bg_np)
            if src_fill_qimg is None:
                continue

            # Warp source → orig_points (current world-space "resting"
            # shape). If the group has never been transformed, orig_points
            # equals source_points and the warp is essentially a copy.
            orig_pts = poly.get('orig_points') or list(poly['points'])
            orig_fill_qimg, orig_fill_bbox = warp_fill_piecewise_affine(
                src_fill_qimg, src_fill_bbox, src_pts, orig_pts,
            )
            if orig_fill_qimg is None:
                # Warp degenerate — fall back to the source crop as-is;
                # it'll sit at the source coord and at least the pixels
                # exist.
                orig_fill_qimg, orig_fill_bbox = src_fill_qimg, src_fill_bbox

            poly['orig_fill_qimg'] = orig_fill_qimg
            poly['orig_fill_bbox'] = orig_fill_bbox

            # Warp orig → current shape. If they match, no warp needed.
            cur_pts = poly['points']
            if list(cur_pts) == list(orig_pts):
                poly['fill_qimg'] = orig_fill_qimg.copy()
                poly['fill_bbox'] = orig_fill_bbox
            else:
                new_qimg, new_bbox = warp_fill_piecewise_affine(
                    orig_fill_qimg, orig_fill_bbox, orig_pts, cur_pts,
                )
                if new_qimg is None:
                    poly['fill_qimg'] = orig_fill_qimg.copy()
                    poly['fill_bbox'] = orig_fill_bbox
                else:
                    poly['fill_qimg'] = new_qimg
                    poly['fill_bbox'] = new_bbox

    # ── undo ─────────────────────────────────────────────────────────
    def _snapshot_state(self) -> dict:
        """Deep-copy just enough state to restore polygons + groups +
        canvas-background positioning. View/grid state is deliberately
        excluded so undo doesn't fight zoom/pan and grid toggles.
        QImage / QPixmap are Qt-owned buffers we replace wholesale on
        mutation — safe to share by reference here."""
        return {
            'polygons': [
                {
                    'points':    list(p['points']),
                    'fill_type': p['fill_type'],
                    'color':     QColor(p['color']),
                    'fill_qimg': (p['fill_qimg'].copy()
                                  if p.get('fill_qimg') is not None else None),
                    'fill_bbox': p.get('fill_bbox'),
                    'orig_points':    (list(p['orig_points'])
                                       if p.get('orig_points') is not None
                                       else None),
                    'orig_fill_qimg': (p['orig_fill_qimg'].copy()
                                       if p.get('orig_fill_qimg') is not None
                                       else None),
                    'orig_fill_bbox': p.get('orig_fill_bbox'),
                    'image_zoom':     float(p.get('image_zoom', 1.0)),
                    'group_id':  p['group_id'],
                }
                for p in self.polygons
            ],
            'groups':               [dict(g) for g in self.groups],
            'next_group_id':        self._next_group_id,
            'canvas_background':    self.canvas_background,
            'bg_offset_x':          self.bg_offset_x,
            'bg_offset_y':          self.bg_offset_y,
            'bg_scale':             self.bg_scale,
            'selected_index':       self.selected_index,
        }

    def _push_undo(self) -> None:
        """Called BEFORE any state mutation. Bounded at self._undo_limit
        entries — oldest are discarded to cap memory."""
        self._undo_stack.append(self._snapshot_state())
        if len(self._undo_stack) > self._undo_limit:
            del self._undo_stack[0]

    def undo(self) -> bool:
        """Restore the top of the undo stack. Returns True if anything
        was popped. Emits nothing on empty stack."""
        if not self._undo_stack:
            return False
        s = self._undo_stack.pop()
        self.polygons          = s['polygons']
        self.groups            = s['groups']
        self._next_group_id    = s['next_group_id']
        self.canvas_background = s['canvas_background']
        self.bg_offset_x       = s['bg_offset_x']
        self.bg_offset_y       = s['bg_offset_y']
        self.bg_scale          = s['bg_scale']
        idx = s.get('selected_index', -1)
        self.selected_index    = (idx if 0 <= idx < len(self.polygons) else -1)
        # Reset any transient drag state so the restored polygon
        # geometry isn't overwritten by a stale in-progress drag.
        self._dragging_polygon = False
        self._dragging_vertex  = False
        self._vertex_drag_idx  = -1
        self._panning          = False
        self.update()
        return True

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
            sx, sy = self.world_to_screen(self.bg_offset_x, self.bg_offset_y)
            sw = bg.width()  * self.bg_scale * self.zoom_factor
            sh = bg.height() * self.bg_scale * self.zoom_factor
            painter.drawPixmap(QRectF(sx, sy, sw, sh), bg,
                               QRectF(0, 0, bg.width(), bg.height()))

        # 2. Polygons.
        gid_visible = {g['id']: g['visible'] for g in self.groups}
        for i, poly in enumerate(self.polygons):
            if not gid_visible.get(poly['group_id'], True):
                continue
            self._draw_polygon(painter, i, poly)

        # 3. Grid overlay — painted LAST so it always sits on top of
        # every element (background + polygons + image fills). Ensures
        # the grid is visible for alignment regardless of what else is
        # loaded on the canvas.
        if self.grid_enabled:
            self._draw_grid(painter)

        # 4. Eraser cursor — drawn only while eraser mode is on. Sits
        # on top of everything so the user always sees exactly how big
        # the erase footprint is at the current zoom.
        if self.eraser_mode:
            cx, cy = self._eraser_mouse_screen
            r_screen = self.eraser_radius * self.zoom_factor
            painter.setBrush(QBrush(QColor(255, 60, 60, 60)))
            painter.setPen(QPen(QColor(255, 60, 60), 2))
            painter.drawEllipse(
                QRectF(cx - r_screen, cy - r_screen,
                       2 * r_screen, 2 * r_screen)
            )

        # 5. Origin-pick snap marker — bright-green cross at the
        # gridline intersection nearest the cursor while picking.
        if self._picking_origin:
            hx, hy = self._picking_hover_screen
            wx, wy = self.screen_to_world(hx, hy)
            ox, oy = self._snap_to_grid_intersection(wx, wy)
            sx, sy = self.world_to_screen(ox, oy)
            painter.setPen(QPen(QColor(0, 200, 0), 3))
            painter.setBrush(QBrush(QColor(0, 200, 0, 90)))
            r = 10
            painter.drawEllipse(QRectF(sx - r, sy - r, 2 * r, 2 * r))
            painter.drawLine(int(sx - r * 1.6), int(sy),
                             int(sx + r * 1.6), int(sy))
            painter.drawLine(int(sx), int(sy - r * 1.6),
                             int(sx), int(sy + r * 1.6))

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

        # Outline (thicker + red when selected). Skipped entirely when
        # `show_polygon_lines` is off — unless the polygon is the
        # selected one, whose red outline is always drawn so the user
        # can still see what they're editing.
        if i == self.selected_index:
            painter.setPen(QPen(QColor(255, 30, 30), 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(qpoly)
        elif self.show_polygon_lines:
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(qpoly)

        # Control-point handles for the SELECTED polygon: small yellow
        # squares at each vertex so the user can grab and drag them
        # individually. Painted after the outline so they sit on top.
        if i == self.selected_index:
            r = self._vertex_hit_radius
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.setBrush(QBrush(QColor(255, 220, 0)))
            for x, y in pts:
                sx, sy = self.world_to_screen(x, y)
                painter.drawRect(QRectF(sx - r, sy - r, 2 * r, 2 * r))

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
        # Origin-picker has the HIGHEST priority — while it's active,
        # LMB is exclusively for choosing a gridline intersection.
        if ev.button() == Qt.LeftButton and self._picking_origin:
            wx, wy = self.screen_to_world(ev.x(), ev.y())
            ox, oy = self._snap_to_grid_intersection(wx, wy)
            cb = self._origin_pick_callback
            self.cancel_origin_pick()
            if cb is not None:
                cb(ox, oy)
            return
        # Eraser tool wins over everything else while active — LMB
        # starts an erase pass, drag continues it.
        if ev.button() == Qt.LeftButton and self.eraser_mode:
            self._push_undo()
            self._eraser_pressed = True
            wx, wy = self.screen_to_world(ev.x(), ev.y())
            self._erase_polygons_under(wx, wy)
            return
        if ev.button() == Qt.LeftButton:
            # Priority: if a polygon is currently selected and the click
            # landed near one of its vertices, enter vertex-drag mode
            # BEFORE running the polygon-pick logic (so users can grab
            # a vertex handle that sits outside the polygon body).
            if self.selected_index >= 0:
                vidx = self._hit_vertex_handle(self.selected_index,
                                               ev.x(), ev.y())
                if vidx >= 0:
                    self._begin_vertex_drag(self.selected_index, vidx)
                    return

            wx, wy = self.screen_to_world(ev.x(), ev.y())
            idx = self._pick_polygon_at(wx, wy)
            self.selected_index = idx
            if idx >= 0:
                # Snapshot BEFORE the drag mutates points so undo goes
                # back to pre-drag geometry in one step.
                self._push_undo()
                self._dragging_polygon = True
                self._drag_start_world = (wx, wy)
                self._drag_start_points = list(self.polygons[idx]['points'])
                # Snapshot the fill bbox + orig_* state at drag start so
                # mouseMove applies the total delta since drag-start,
                # NOT accumulating onto already-shifted values.
                fb = self.polygons[idx].get('fill_bbox')
                self._drag_start_bbox = tuple(fb) if fb is not None else None
                op = self.polygons[idx].get('orig_points')
                ofb = self.polygons[idx].get('orig_fill_bbox')
                self._drag_start_orig_points = list(op) if op else None
                self._drag_start_orig_bbox = tuple(ofb) if ofb else None
                self.setCursor(Qt.ClosedHandCursor)
            self.update()

    def mouseMoveEvent(self, ev):
        # Origin-picker hover — updates the green snap marker.
        if self._picking_origin:
            self._picking_hover_screen = (ev.x(), ev.y())
            self.update()
            return
        # Track cursor for the eraser cursor circle even without any
        # button held down — moving the mouse should update where the
        # circle is drawn.
        if self.eraser_mode:
            self._eraser_mouse_screen = (ev.x(), ev.y())
            if self._eraser_pressed:
                wx, wy = self.screen_to_world(ev.x(), ev.y())
                self._erase_polygons_under(wx, wy)
            self.update()
            return
        if self._panning:
            dx = ev.x() - self._pan_start_screen[0]
            dy = ev.y() - self._pan_start_screen[1]
            self.pan_x = self._pan_start_offset[0] + dx
            self.pan_y = self._pan_start_offset[1] + dy
            self.update()
            return
        if self._dragging_vertex and self.selected_index >= 0:
            wx, wy = self.screen_to_world(ev.x(), ev.y())
            poly = self.polygons[self.selected_index]
            pts = list(poly['points'])
            if 0 <= self._vertex_drag_idx < len(pts):
                pts[self._vertex_drag_idx] = (wx, wy)
                poly['points'] = pts
            # Stretch the image fill to the polygon's NEW bounding rect
            # AND re-mask it to the polygon's new shape.
            try:
                changed = self._restretch_fill_to_current_shape(poly)
                if not changed:
                    print(
                        f"[mosaic_editor] restretch skipped — "
                        f"orig_qimg={poly.get('orig_fill_qimg') is not None}, "
                        f"orig_bbox={poly.get('orig_fill_bbox') is not None}, "
                        f"orig_pts={poly.get('orig_points') is not None}",
                        file=sys.stderr,
                    )
            except Exception as e:
                import traceback
                print(
                    f"[mosaic_editor] restretch error: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
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
            # Keep the warp reference (orig_*) in sync with the visible
            # translation. Same total-delta-from-snapshot pattern.
            if self._drag_start_orig_points is not None:
                poly['orig_points'] = [
                    (px + dx, py + dy)
                    for (px, py) in self._drag_start_orig_points
                ]
            if self._drag_start_orig_bbox is not None:
                x0, y0, w, h = self._drag_start_orig_bbox
                poly['orig_fill_bbox'] = (x0 + dx, y0 + dy, w, h)
            self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        if ev.button() == Qt.LeftButton:
            if self.eraser_mode and self._eraser_pressed:
                self._eraser_pressed = False
                return
            if self._dragging_vertex:
                self._dragging_vertex = False
                self._vertex_drag_idx = -1
                self.setCursor(Qt.ArrowCursor)
                return
            if self._dragging_polygon:
                self._dragging_polygon = False
                self.setCursor(Qt.ArrowCursor)

    def _grid_cell_world(self) -> float:
        """Current grid cell size in world units — matches how the
        grid is actually drawn (see _draw_grid)."""
        if self.canvas_background is not None:
            return max(
                1.0,
                self.canvas_background.width()
                * (self.grid_size_percent / 100.0),
            )
        return max(1.0, self.grid_size_world)

    def _snap_to_grid_intersection(self, wx: float, wy: float) -> tuple:
        """Return the nearest gridline intersection to (wx, wy) in
        world coords."""
        cell = self._grid_cell_world()
        ox = self.grid_offset_x
        oy = self.grid_offset_y
        i = round((wx - ox) / cell)
        j = round((wy - oy) / cell)
        return (ox + i * cell, oy + j * cell)

    def start_origin_pick(self, callback) -> None:
        """Enter origin-picking mode. The next LMB click on the canvas
        snaps to the nearest gridline intersection and invokes
        `callback(world_x, world_y)`. Escape cancels."""
        self.cancel_origin_pick()   # in case one is already in flight
        self._picking_origin       = True
        self._origin_pick_callback = callback
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self.setFocus()             # so Escape reaches keyPressEvent
        self.update()

    def cancel_origin_pick(self) -> None:
        """Exit origin-picking mode without firing the callback."""
        if self._picking_origin:
            self._picking_origin       = False
            self._origin_pick_callback = None
            self.setCursor(Qt.ArrowCursor)
            self.update()

    def set_eraser_mode(self, on: bool) -> None:
        """Toggle the eraser tool. When on, the mouse cursor becomes a
        blank pointer (we draw our own eraser-radius circle on top of
        the canvas) and every LMB drag deletes any polygon in an
        affected layer that intersects the circle."""
        self.eraser_mode = bool(on)
        if self.eraser_mode:
            self.setCursor(Qt.CrossCursor)
            self.setMouseTracking(True)
        else:
            self.setCursor(Qt.ArrowCursor)
            self._eraser_pressed = False
        self.update()

    def _erase_polygons_under(self, wx: float, wy: float) -> int:
        """Delete every polygon in an *affected* layer whose centroid
        OR any vertex is within `eraser_radius` world units of the
        cursor. Returns how many were deleted. Undo is snapshotted by
        the caller (mousePressEvent) so a full drag counts as ONE
        undo step, not one per polygon."""
        if not self.polygons:
            return 0
        affected_gids = {
            g['id'] for g in self.groups if g.get('affected', True)
        }
        if not affected_gids:
            return 0
        r  = float(self.eraser_radius)
        r2 = r * r
        keep = []
        removed = 0
        for poly in self.polygons:
            if poly['group_id'] not in affected_gids:
                keep.append(poly)
                continue
            pts = poly.get('points') or []
            if not pts:
                keep.append(poly)
                continue
            # Centroid distance.
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            if (cx - wx) ** 2 + (cy - wy) ** 2 <= r2:
                removed += 1
                continue
            # Any-vertex distance — catches polygons much larger than
            # the eraser radius.
            hit = False
            for (px, py) in pts:
                if (px - wx) ** 2 + (py - wy) ** 2 <= r2:
                    hit = True
                    break
            if hit:
                removed += 1
            else:
                keep.append(poly)
        if removed:
            self.polygons = keep
            self.selected_index = -1
            self.update()
        return removed

    def keyPressEvent(self, ev):
        # Escape cancels an origin-pick in progress. Highest priority
        # so nothing else swallows the key while picking.
        if ev.key() == Qt.Key_Escape and self._picking_origin:
            self.cancel_origin_pick()
            return
        # Delete / Backspace removes the single currently-selected
        # polygon. The sidebar's "Erase Affected" button erases whole
        # layers instead — that's a heavier operation, so it's not
        # bound to a stray key press.
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace) \
                and 0 <= self.selected_index < len(self.polygons):
            self._push_undo()
            del self.polygons[self.selected_index]
            self.selected_index = -1
            self.update()
            return
        # P → zoom the image content of the selected polygon (shape
        # stays put, pixels inside become a magnified centre-crop of
        # the source fill). Shift+P zooms out — clamped so we never
        # go below the original 1:1 sampling.
        if ev.key() == Qt.Key_P:
            factor = (1.0 / 1.25) if (ev.modifiers() & Qt.ShiftModifier) \
                                   else 1.25
            self.zoom_polygon_image(factor)
            return

    def erase_affected(self) -> tuple[int, int]:
        """Delete every polygon that belongs to a layer whose ⇢
        (affected) checkbox is on, then remove those layer groups
        themselves (so the layers panel doesn't show empty rows).
        Returns (polygons_removed, groups_removed). Undoable."""
        affected_gids = {
            g['id'] for g in self.groups if g.get('affected', True)
        }
        if not affected_gids:
            return (0, 0)
        n_polys_before  = len(self.polygons)
        n_groups_before = len(self.groups)

        self._push_undo()
        self.polygons = [
            p for p in self.polygons if p['group_id'] not in affected_gids
        ]
        self.groups = [
            g for g in self.groups if g['id'] not in affected_gids
        ]
        self.selected_index = -1
        self._dragging_polygon = False
        self._dragging_vertex  = False
        self.update()
        return (n_polys_before  - len(self.polygons),
                n_groups_before - len(self.groups))

    def _hit_vertex_handle(self, poly_idx: int, sx: int, sy: int) -> int:
        """Return the index of the vertex (in poly.points) whose on-
        screen position is within self._vertex_hit_radius of the mouse
        (sx, sy), or -1 if none is close enough."""
        if not (0 <= poly_idx < len(self.polygons)):
            return -1
        poly = self.polygons[poly_idx]
        r2 = self._vertex_hit_radius ** 2
        best_i, best_d = -1, r2 + 1
        for i, (x, y) in enumerate(poly['points']):
            vx, vy = self.world_to_screen(x, y)
            d = (vx - sx) ** 2 + (vy - sy) ** 2
            if d <= r2 and d < best_d:
                best_i, best_d = i, d
        return best_i

    def _restretch_fill_to_current_shape(self, poly: dict) -> bool:
        """Rebuild `poly['fill_qimg']` + `poly['fill_bbox']` by
        BILINEAR-STRETCHING the *original* fill image (before any
        vertex drags) into the current polygon's bounding rectangle,
        then RE-MASKING with the current polygon's shape so the
        silhouette matches the deformed outline exactly. Returns True
        if it actually rebuilt the fill, False if it was skipped."""
        orig_qimg = poly.get('orig_fill_qimg')
        orig_bbox = poly.get('orig_fill_bbox')
        orig_pts  = poly.get('orig_points')
        cur_pts   = poly.get('points')
        if (orig_qimg is None or orig_bbox is None
                or orig_pts is None or not cur_pts or len(cur_pts) < 3):
            return False

        # New polygon bounding rect.
        xs = [p[0] for p in cur_pts]
        ys = [p[1] for p in cur_pts]
        nx0 = min(xs); ny0 = min(ys)
        nx1 = max(xs); ny1 = max(ys)
        nw = max(1, int(round(nx1 - nx0)))
        nh = max(1, int(round(ny1 - ny0)))

        # 1a. If the polygon has an image_zoom > 1.0, crop a smaller
        #     centred subregion of the ORIGINAL fill first. Cropping
        #     less area then stretching to (nw × nh) is exactly what
        #     "zoom in" means — the polygon shape doesn't move but the
        #     visible pixels are a magnified view of the source.
        img_zoom = float(poly.get('image_zoom', 1.0))
        if img_zoom < 1.0:
            img_zoom = 1.0
        if img_zoom > 1.0:
            ow, oh = orig_qimg.width(), orig_qimg.height()
            crop_w = max(1, int(round(ow / img_zoom)))
            crop_h = max(1, int(round(oh / img_zoom)))
            crop_x = max(0, (ow - crop_w) // 2)
            crop_y = max(0, (oh - crop_h) // 2)
            src_img = orig_qimg.copy(crop_x, crop_y, crop_w, crop_h)
        else:
            src_img = orig_qimg

        # 1b. Bilinear-scale that source (either the whole orig or a
        #     centred crop) into a fresh (nw × nh) QImage.
        stretched = src_img.scaled(
            nw, nh,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        )
        # Ensure RGBA so we can rewrite the alpha channel.
        if stretched.format() != QImage.Format_RGBA8888:
            stretched = stretched.convertToFormat(QImage.Format_RGBA8888)

        # 2. Build a fresh alpha mask matching the polygon's CURRENT
        #    silhouette (in the new bbox's local coords). Use PIL —
        #    identical technique to build_image_fill so we know it
        #    works on this environment.
        from PIL import Image as _PILImage, ImageDraw as _ImageDraw
        mask_img = _PILImage.new('L', (nw, nh), 0)
        poly_local = [(float(px - nx0), float(py - ny0))
                      for (px, py) in cur_pts]
        _ImageDraw.Draw(mask_img).polygon(poly_local, fill=255)
        mask_arr = np.array(mask_img, dtype=np.uint8)          # (nh, nw)

        # 3. Rewrite the stretched image's alpha channel with the mask.
        stretched_rgba = _qimage_to_numpy_rgba(stretched)
        if stretched_rgba is None:
            return False
        stretched_rgba[:, :, 3] = mask_arr
        stretched_rgba = np.ascontiguousarray(stretched_rgba)
        new_qimg = QImage(
            stretched_rgba.tobytes(), nw, nh, nw * 4,
            QImage.Format_RGBA8888,
        ).copy()

        poly['fill_qimg'] = new_qimg
        poly['fill_bbox'] = (nx0, ny0, nw, nh)
        return True

    def _begin_vertex_drag(self, poly_idx: int, vertex_idx: int) -> None:
        """Enter vertex-drag mode for polygon poly_idx / vertex_idx.
        Also backfills missing orig_* fields from the current fill so
        the restretch path works on polygons loaded from older files
        (or any case where the warp reference wasn't saved)."""
        # Snapshot BEFORE the drag so undo returns to pre-drag geometry
        # (and pre-drag image fill) in one step.
        self._push_undo()
        self._dragging_vertex = True
        self._vertex_drag_idx = int(vertex_idx)
        self.selected_index = int(poly_idx)
        self.setCursor(Qt.ClosedHandCursor)
        poly = self.polygons[poly_idx]
        self._ensure_orig_fill_snapshot(poly)
        has_img = poly.get('orig_fill_qimg') is not None
        print(
            f"[mosaic_editor] vertex-drag start: polygon #{poly_idx}, "
            f"vertex #{vertex_idx}, image-fill={has_img}",
            file=sys.stderr,
        )

    @staticmethod
    def _ensure_orig_fill_snapshot(poly: dict) -> None:
        """Promote whatever current fill state exists into the warp
        reference slots. No-op when orig_* is already populated (fresh
        .mosaic load) — used by both vertex-drag and image-zoom paths
        so polygons loaded from older projects still stretch/zoom."""
        if poly.get('orig_points') is None:
            poly['orig_points'] = list(poly.get('points') or [])
        if (poly.get('orig_fill_qimg') is None
                and poly.get('fill_qimg') is not None):
            poly['orig_fill_qimg'] = poly['fill_qimg'].copy()
        if poly.get('orig_fill_bbox') is None:
            poly['orig_fill_bbox'] = poly.get('fill_bbox')

    def zoom_polygon_image(self, factor: float) -> bool:
        """Zoom the image content of the currently selected polygon by
        `factor` (>1 zooms IN, <1 zooms out; clamped at min 1.0 so we
        never sample outside the source polygon). The polygon shape
        stays exactly where it is; only the pixels visible inside
        change to show a smaller cropped-and-stretched region of the
        original fill. Returns True on success."""
        if not (0 <= self.selected_index < len(self.polygons)):
            return False
        poly = self.polygons[self.selected_index]
        self._ensure_orig_fill_snapshot(poly)
        if poly.get('orig_fill_qimg') is None:
            return False
        old = float(poly.get('image_zoom', 1.0))
        new = max(1.0, old * float(factor))
        if abs(new - old) < 1e-9:
            return False
        self._push_undo()
        poly['image_zoom'] = new
        self._restretch_fill_to_current_shape(poly)
        self.update()
        return True

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

        # Header row explains the two-checkbox layout so users can tell
        # them apart at a glance.
        header = QFrame()
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(4, 2, 4, 2); h_lay.setSpacing(4)
        h_lay.addWidget(self._mini_label("👁"))   # visibility
        h_lay.addWidget(self._mini_label("⇢"))    # action / affected
        h_lay.addWidget(QLabel("Layer"), 1)
        h_lay.addWidget(self._mini_label(" "))    # delete slot
        self._layout.insertWidget(self._layout.count() - 1, header)

        # Canvas background — its own row when one is loaded.
        if self.canvas.canvas_background is not None:
            row = QFrame()
            row.setFrameStyle(QFrame.StyledPanel)
            row.setStyleSheet("background-color: #eef2ff;")
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(4, 2, 4, 2); row_lay.setSpacing(4)

            vis_cb = self._mini_checkbox(True, enabled=False,
                                         tip="Canvas background is always visible while loaded.")
            row_lay.addWidget(vis_cb)

            act_cb = self._mini_checkbox(
                self.canvas.bg_affected,
                tip="If checked, the movement arrows move the canvas background."
            )
            act_cb.toggled.connect(self._on_bg_affected_toggled)
            row_lay.addWidget(act_cb)

            row_lay.addWidget(QLabel("[bg] canvas background"), 1)
            row_lay.addWidget(self._mini_label(" "))

            self._layout.insertWidget(self._layout.count() - 1, row)

        for g in self.canvas.groups:
            row = QFrame()
            row.setFrameStyle(QFrame.StyledPanel)
            row.setStyleSheet("background-color: #f5f5f5;")
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(4, 2, 4, 2)
            row_lay.setSpacing(4)

            vis_cb = self._mini_checkbox(
                g['visible'],
                tip="Show / hide this layer."
            )
            vis_cb.toggled.connect(
                lambda checked, gid=g['id']:
                    self._on_visibility_toggled(gid, checked)
            )
            row_lay.addWidget(vis_cb)

            act_cb = self._mini_checkbox(
                g.get('affected', True),
                tip="If checked, the movement arrows move this layer."
            )
            act_cb.toggled.connect(
                lambda checked, gid=g['id']:
                    self._on_affected_toggled(gid, checked)
            )
            row_lay.addWidget(act_cb)

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

    @staticmethod
    def _mini_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFixedWidth(16)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size: 11px; color: #555;")
        return lbl

    @staticmethod
    def _mini_checkbox(checked: bool, tip: str = "", enabled: bool = True):
        cb = QCheckBox()
        cb.setChecked(checked)
        cb.setEnabled(enabled)
        cb.setFixedWidth(16)
        cb.setToolTip(tip)
        return cb

    def _on_visibility_toggled(self, gid: int, checked: bool) -> None:
        self.canvas.set_group_visible(gid, checked)

    def _on_affected_toggled(self, gid: int, checked: bool) -> None:
        self.canvas.set_group_affected(gid, checked)

    def _on_bg_affected_toggled(self, checked: bool) -> None:
        self.canvas.bg_affected = checked

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
        btn_img_clear.clicked.connect(self.on_clear_canvas_background)
        sb.addWidget(btn_img_clear)

        sb.addSpacing(6)
        # Project save / load — snapshots every layer, canvas
        # background, view transform, and grid state into a single
        # .mep pickle file. Load restores everything.
        btn_save_project = QPushButton("Save Project")
        btn_save_project.clicked.connect(self.on_save_project)
        sb.addWidget(btn_save_project)

        btn_load_project = QPushButton("Load Project")
        btn_load_project.clicked.connect(self.on_load_project)
        sb.addWidget(btn_load_project)

        # Save Mosaic — flatten everything on the canvas into one
        # portable .mosaic bundle (composite PNG + polygon CSV).
        # Same format image_strech.py's Save Mosaic writes, so the
        # result can be reopened here or in any other tool that
        # reads the shared schema.
        btn_save_mosaic = QPushButton("Save Mosaic")
        btn_save_mosaic.setToolTip(
            "Combine every array on the canvas into a single "
            ".mosaic bundle: background.png is the flattened "
            "composite of all polygon fills + canvas background, "
            "polygons.csv holds every polygon's outline in that "
            "PNG's coordinate space."
        )
        btn_save_mosaic.clicked.connect(self.on_save_mosaic)
        sb.addWidget(btn_save_mosaic)

        btn_save_array = QPushButton("Save Array")
        btn_save_array.setToolTip(
            "Write every visible polygon to a CSV in the shared "
            "save_array schema (11 columns matching image_strech.py). "
            "Asks for a target grid-box size and scales every "
            "polygon coordinate by (target / current)."
        )
        btn_save_array.clicked.connect(self.on_save_array)
        sb.addWidget(btn_save_array)

        sb.addSpacing(8)
        sb.addWidget(self._section_label("View"))
        btn_reset_view = QPushButton("Reset Zoom / Pan")
        btn_reset_view.clicked.connect(self.canvas.reset_view)
        sb.addWidget(btn_reset_view)

        sb.addSpacing(8)
        sb.addWidget(self._section_label("Move"))
        # Step size — world units per arrow click.
        row = QHBoxLayout()
        row.addWidget(QLabel("Step:"))
        self.move_step_spin = QDoubleSpinBox()
        self.move_step_spin.setRange(0.1, 100000.0)
        self.move_step_spin.setValue(10.0)
        self.move_step_spin.setDecimals(1)
        self.move_step_spin.setSuffix(" px")
        self.move_step_spin.setToolTip(
            "Distance in world units each arrow click moves the "
            "affected elements."
        )
        row.addWidget(self.move_step_spin)
        sb.addLayout(row)

        # 3×3 arrow grid: ↑ ← → ↓ around an empty centre.
        arrows = QGridLayout()
        arrows.setSpacing(2)
        up_btn    = QPushButton("↑")
        left_btn  = QPushButton("←")
        right_btn = QPushButton("→")
        down_btn  = QPushButton("↓")
        for b in (up_btn, left_btn, right_btn, down_btn):
            b.setFixedSize(36, 28)
        up_btn.clicked.connect   (lambda: self.on_move_arrow( 0, -1))
        left_btn.clicked.connect (lambda: self.on_move_arrow(-1,  0))
        right_btn.clicked.connect(lambda: self.on_move_arrow( 1,  0))
        down_btn.clicked.connect (lambda: self.on_move_arrow( 0,  1))
        arrows.addWidget(up_btn,    0, 1)
        arrows.addWidget(left_btn,  1, 0)
        arrows.addWidget(right_btn, 1, 2)
        arrows.addWidget(down_btn,  2, 1)
        sb.addLayout(arrows)

        # Scale (%) — same-styled spinbox as image_strech's Scale
        # Polygon Array. Applied only to elements whose ⇢ checkbox is
        # ticked. Each Apply multiplies the current size by (pct/100)
        # cumulatively — 100 is a no-op, 200 doubles, 50 halves.
        row = QHBoxLayout()
        row.addWidget(QLabel("Scale:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 10000.0)
        self.scale_spin.setDecimals(1)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setValue(100.0)
        self.scale_spin.setSuffix(" %")
        self.scale_spin.setToolTip(
            "Absolute scale — every affected element becomes this "
            "percentage of ITS ORIGINAL (as-loaded) size, regardless "
            "of prior scales. 100 = original. Apply 200 twice → still "
            "200 %. Apply 100 to restore original size."
        )
        row.addWidget(self.scale_spin)
        self.apply_scale_btn = QPushButton("Apply")
        self.apply_scale_btn.setFixedWidth(60)
        self.apply_scale_btn.clicked.connect(self.on_apply_scale)
        row.addWidget(self.apply_scale_btn)
        sb.addLayout(row)

        # Scale All Polygons (%) — RELATIVE multiplier applied to every
        # polygon around ITS OWN centroid. Ignores the ⇢ affected
        # checkboxes and doesn't touch the per-group applied_scale.
        # 110 = each polygon grows 10 % about its own centre.
        row = QHBoxLayout()
        row.addWidget(QLabel("Scale All:"))
        self.scale_all_spin = QDoubleSpinBox()
        self.scale_all_spin.setRange(0.1, 10000.0)
        self.scale_all_spin.setDecimals(1)
        self.scale_all_spin.setSingleStep(1.0)
        self.scale_all_spin.setValue(100.0)
        self.scale_all_spin.setSuffix(" %")
        self.scale_all_spin.setToolTip(
            "Scale EVERY polygon around its own centroid by this "
            "percentage. Relative multiplier — 110 grows by 10 %, "
            "90 shrinks by 10 %, 100 is a no-op. Ignores the ⇢ "
            "affected checkboxes."
        )
        row.addWidget(self.scale_all_spin)
        self.apply_scale_all_btn = QPushButton("Apply")
        self.apply_scale_all_btn.setFixedWidth(60)
        self.apply_scale_all_btn.clicked.connect(self.on_apply_scale_all)
        row.addWidget(self.apply_scale_all_btn)
        sb.addLayout(row)

        # Edit — erase affected layers + undo.
        # "Erase Affected" wipes every polygon whose layer has ⇢
        # (affected) ticked, then removes those layer entries too.
        # Single-polygon deletion is still available via the
        # Delete / Backspace key on the canvas.
        row = QHBoxLayout()
        self.erase_btn = QPushButton("Erase Affected")
        self.erase_btn.setToolTip(
            "Delete every polygon in every ⇢-checked layer, then "
            "remove those layers. Undoable with Ctrl+Z. "
            "(To delete a single polygon: select it and press Delete.)"
        )
        self.erase_btn.clicked.connect(self.on_erase_affected)
        row.addWidget(self.erase_btn)

        self.undo_btn = QPushButton("Undo (Ctrl+Z)")
        self.undo_btn.setToolTip(
            "Undo the last canvas edit (load, move, scale, drag, "
            "erase). Up to 30 steps."
        )
        self.undo_btn.clicked.connect(self.on_undo)
        row.addWidget(self.undo_btn)
        sb.addLayout(row)

        sb.addSpacing(8)
        sb.addWidget(self._section_label("View"))
        self.polygon_lines_cb = QCheckBox("Show polygon lines")
        self.polygon_lines_cb.setChecked(self.canvas.show_polygon_lines)
        self.polygon_lines_cb.setToolTip(
            "Toggle the black polygon outlines. Off = filled shapes only "
            "(as it would look printed). The selected polygon's red "
            "outline stays visible either way."
        )
        self.polygon_lines_cb.toggled.connect(self.on_polygon_lines_toggled)
        sb.addWidget(self.polygon_lines_cb)

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

        # ── right-side canvas toolbar ─────────────────────────────
        # Tool buttons for direct-manipulation modes on the canvas.
        # Currently just Delete (eraser) but scales with more tools.
        right_bar = QFrame()
        right_bar.setFrameStyle(QFrame.StyledPanel)
        right_bar.setFixedWidth(140)
        rb = QVBoxLayout(right_bar)
        rb.setContentsMargins(6, 6, 6, 6)
        rb.setSpacing(6)

        self.delete_tool_btn = QPushButton("Delete")
        self.delete_tool_btn.setCheckable(True)
        self.delete_tool_btn.setFixedHeight(40)
        self.delete_tool_btn.setToolTip(
            "Eraser tool. When active, the cursor becomes a red circle;\n"
            "click / drag over polygons to delete them. Only polygons\n"
            "in ⇢-checked layers are erased. Ctrl+Z to undo."
        )
        self.delete_tool_btn.toggled.connect(self.on_delete_tool_toggled)
        rb.addWidget(self.delete_tool_btn)

        rb.addWidget(QLabel("Radius:"))
        self.eraser_radius_spin = QDoubleSpinBox()
        self.eraser_radius_spin.setRange(1.0, 10000.0)
        self.eraser_radius_spin.setDecimals(0)
        self.eraser_radius_spin.setValue(self.canvas.eraser_radius)
        self.eraser_radius_spin.setSuffix(" px")
        self.eraser_radius_spin.setToolTip(
            "Eraser radius in world units."
        )
        self.eraser_radius_spin.valueChanged.connect(
            self.on_eraser_radius_changed
        )
        rb.addWidget(self.eraser_radius_spin)

        rb.addSpacing(10)

        # Save Tile Image — cut each selected tile from the composite
        # canvas using the OFFSET boundary in its box_<label>.dxf
        # (same DXF-driven workflow as image_strech.py).
        self.save_tile_btn = QPushButton("Save Tile\nImage")
        self.save_tile_btn.setFixedHeight(48)
        self.save_tile_btn.setToolTip(
            "Save selected grid cells as print-ready PNGs cut with "
            "their box_<label>.dxf offset boundary. Steps: 1) click "
            "a gridline intersection for origin, 2) pick cells from "
            "the 6×6 dialog, 3) choose a DXF root folder, 4) enter "
            "the target grid box size you used at Save Array time. "
            "Tile size (mm) and DPI come from the fields below."
        )
        self.save_tile_btn.clicked.connect(self.on_save_tile_image)
        rb.addWidget(self.save_tile_btn)

        rb.addWidget(QLabel("Grid box mm:"))
        self.tile_size_spin = QDoubleSpinBox()
        self.tile_size_spin.setRange(1.0, 10000.0)
        self.tile_size_spin.setDecimals(1)
        self.tile_size_spin.setValue(200.0)
        self.tile_size_spin.setSuffix(" mm")
        self.tile_size_spin.setToolTip(
            "Physical size of ONE GRID BOX in the output print.\n"
            "This is NOT the final PNG size — the PNG scales with the "
            "DXF polygon's world-coord width relative to one grid box, "
            "so an offset polygon slightly bigger than a grid box comes "
            "out slightly bigger than this value at the given DPI. Same "
            "convention as image_strech.py's Save Tile Image."
        )
        rb.addWidget(self.tile_size_spin)

        rb.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 2400)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setToolTip(
            "Print DPI baked into each PNG's pHYs metadata."
        )
        rb.addWidget(self.dpi_spin)

        # Debug mode — when checked, Save Tile Image writes an extra
        # `<label>_debug.png` (render + overlays of DXF polygon, grid
        # cell, and every canvas polygon in the region) plus a
        # `<label>_debug.txt` log next to every tile PNG.
        self.debug_tile_chk = QCheckBox("Debug Tile")
        self.debug_tile_chk.setToolTip(
            "When checked, Save Tile Image writes an extra "
            "<label>_debug.png (render + DXF polygon overlay + "
            "polygon outlines) and <label>_debug.txt log next to "
            "each PNG. Useful for diagnosing why a tile is missing "
            "content."
        )
        rb.addWidget(self.debug_tile_chk)

        rb.addStretch(1)

        # ── assemble ──────────────────────────────────────────────────
        main.addWidget(sidebar)
        main.addWidget(canvas_frame, 1)
        main.addWidget(right_bar)

        # Ctrl+Z → undo. Attached at the main-window scope so it fires
        # regardless of which widget has focus (sidebar spinbox,
        # layers panel, canvas, ...).
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._undo_shortcut.activated.connect(self.on_undo)

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
        self.layers_panel.rebuild()   # background row must now show
        self.statusBar().showMessage(
            f"Canvas background: {Path(path).name}"
        )

    def on_clear_canvas_background(self) -> None:
        self.canvas.clear_canvas_background()
        self.layers_panel.rebuild()

    def on_move_arrow(self, dx_sign: int, dy_sign: int) -> None:
        """Move every affected layer (and the affected canvas
        background) by ±step in x / y. dx_sign, dy_sign ∈ {-1, 0, +1}."""
        step = float(self.move_step_spin.value())
        self.canvas.move_affected(dx_sign * step, dy_sign * step)

    def on_apply_scale(self) -> None:
        """Scale every affected element by the sidebar's percentage."""
        pct = float(self.scale_spin.value())
        self.canvas.scale_affected(pct)

    def on_apply_scale_all(self) -> None:
        """Scale EVERY polygon around its own centroid by the Scale All
        spinbox's percentage. Ignores affected / group state."""
        pct = float(self.scale_all_spin.value())
        n = self.canvas.scale_all_polygons_individual(pct)
        if n:
            self.statusBar().showMessage(
                f"Scaled {n} polygon(s) to {pct:g} % around their own "
                f"centres."
            )

    # ── Save Tile Image chain ─────────────────────────────────────────
    # Multi-step flow triggered by the right-toolbar "Save Tile Image"
    # button:  origin-pick → 6×6 tile checkbox dialog → target folder →
    # tile size (mm) → DPI → render + save. Each step's callback runs
    # via a Qt event, so state (origin, selected cells, folder) is
    # threaded through method args rather than kept as instance state.

    def on_save_tile_image(self) -> None:
        """Step 1: guard + start origin pick."""
        if not self.canvas.polygons and self.canvas.canvas_background is None:
            QMessageBox.information(
                self, "Save Tile Image",
                "Canvas is empty — nothing to save."
            )
            return
        # Force the grid on so the user sees the intersections.
        if not self.canvas.grid_enabled:
            self.canvas.grid_enabled = True
            self.grid_cb.blockSignals(True)
            self.grid_cb.setChecked(True)
            self.grid_cb.blockSignals(False)
            self.canvas.update()
        self.statusBar().showMessage(
            "Save Tile Image — click a gridline intersection to set the "
            "top-left corner (A1) of the tile grid. Press Esc to cancel."
        )
        self.canvas.start_origin_pick(
            lambda ox, oy: self._save_tile_pick_cells(ox, oy)
        )

    def _save_tile_pick_cells(self, ox: float, oy: float) -> None:
        """Step 2: show a 6×6 checkbox dialog matching image_strech's UX
        so the user picks which cells to export. On Ok, chain to folder
        + tile-size prompts."""
        ROWS = 6; COLS = 6

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Tiles to Save")
        dlg_layout = QVBoxLayout(dlg)

        cell_world = self.canvas._grid_cell_world()
        info = QLabel(
            f"Origin: ({ox:.1f}, {oy:.1f})  ·  "
            f"Cell size: {cell_world:.1f} world-units  ·  "
            f"Grid: {ROWS} rows × {COLS} columns"
        )
        dlg_layout.addWidget(info)

        # Select-all / deselect-all
        sel_row = QHBoxLayout()
        sel_all   = QPushButton("Select All")
        desel_all = QPushButton("Deselect All")
        sel_row.addWidget(sel_all); sel_row.addWidget(desel_all)
        dlg_layout.addLayout(sel_row)

        # Checkbox grid: rows A-F top→bottom, cols 1-6 left→right.
        checkboxes: dict[tuple[int, int], QCheckBox] = {}
        grid_holder = QWidget()
        grid = QGridLayout(grid_holder)
        grid.setSpacing(2)
        # Column header row (1, 2, 3, ...) at grid row 0
        for c in range(COLS):
            hdr = QLabel(str(c + 1))
            hdr.setAlignment(Qt.AlignCenter)
            hdr.setStyleSheet("font-weight: bold;")
            grid.addWidget(hdr, 0, c + 1)
        for r in range(ROWS):
            row_letter = chr(ord('A') + r)
            row_hdr = QLabel(row_letter)
            row_hdr.setAlignment(Qt.AlignCenter)
            row_hdr.setStyleSheet("font-weight: bold;")
            grid.addWidget(row_hdr, r + 1, 0)
            for c in range(COLS):
                cb = QCheckBox(f"{row_letter}{c + 1}")
                grid.addWidget(cb, r + 1, c + 1)
                checkboxes[(r, c)] = cb
        dlg_layout.addWidget(grid_holder)

        sel_all.clicked.connect(
            lambda: [cb.setChecked(True) for cb in checkboxes.values()]
        )
        desel_all.clicked.connect(
            lambda: [cb.setChecked(False) for cb in checkboxes.values()]
        )

        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        dlg_layout.addWidget(btn_box)

        if dlg.exec_() != QDialog.Accepted:
            self.statusBar().showMessage("Save Tile Image cancelled.")
            return
        selected = [(r, c) for (r, c), cb in checkboxes.items()
                    if cb.isChecked()]
        if not selected:
            QMessageBox.information(
                self, "Save Tile Image",
                "No tiles selected."
            )
            return
        self._save_tile_ask_dxf(ox, oy, selected)

    def _save_tile_ask_dxf(self, ox: float, oy: float,
                           selected: list) -> None:
        """Step 3: pick the DXF root directory. It should contain a
        subdirectory per box, each holding a `box_<label>.dxf` written
        by mosaic_studio.py's save_boxes."""
        dxf_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose DXF Directory (per-box subdirs with box_*.dxf inside)",
        )
        if not dxf_dir:
            self.statusBar().showMessage("Save Tile Image cancelled.")
            return

        # Step 3b: ask the user to name the OUTPUT subfolder. Every
        # tile PNG (and per-tile debug artifact when enabled) is
        # written into this single folder as `<label>.png` instead of
        # being scattered into each box's own sub-DXF folder.
        name, ok = QInputDialog.getText(
            self,
            "Output Folder Name",
            f"Name of the new folder to hold all tile PNGs.\n\n"
            f"It will be created inside:\n{dxf_dir}",
            text="tiles",
        )
        if not ok:
            self.statusBar().showMessage("Save Tile Image cancelled.")
            return
        name = name.strip()
        # Strip characters Windows/Unix filesystems reject in folder
        # names — keeps ASCII letters/digits/-/_/space and turns
        # anything else into `_` so a paste of an odd string doesn't
        # explode inside makedirs.
        safe = "".join(ch if (ch.isalnum() or ch in "-_ .") else "_"
                       for ch in name) or "tiles"
        out_dir = os.path.join(dxf_dir, safe)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(
                self, "Save Tile Image",
                f"Could not create output folder:\n{out_dir}\n\n{e}"
            )
            return

        self._save_tile_ask_scale(ox, oy, selected, dxf_dir, out_dir)

    def _save_tile_ask_scale(self, ox: float, oy: float,
                             selected: list, dxf_dir: str,
                             out_dir: str) -> None:
        """Step 4 (single dialog, matching image_strech.py): ask for
        the target grid box size the user entered at Save Array time.
        Tile size (mm) + DPI come from the persistent right-toolbar
        spinboxes, so we don't prompt for them here. `out_dir` is the
        folder we already resolved in `_save_tile_ask_dxf` — every PNG
        lands there as `<label>.png`."""
        current_cell_px = self.canvas._grid_cell_world()
        target_cell_px, ok = QInputDialog.getDouble(
            self,
            "Un-scale DXF Coordinates",
            "What target grid box size did you enter when you ran "
            "'Save Array' to make the CSV that produced these DXFs?\n\n"
            f"Current canvas grid box is {current_cell_px:.2f} px.\n"
            "Enter the SAME value you entered at Save Array time.\n\n"
            "DXF coords will be divided by (target / current) so they "
            "land back on the canvas.",
            value=round(current_cell_px, 2),
            min=0.01, max=1_000_000.0, decimals=2,
        )
        if not ok:
            self.statusBar().showMessage("Save Tile Image cancelled.")
            return
        inverse_scale = current_cell_px / max(1e-9, target_cell_px)

        mm  = float(self.tile_size_spin.value())
        dpi = int  (self.dpi_spin.value())
        try:
            saved, no_dxf, other = self._save_tiles_via_dxf(
                ox, oy, selected, dxf_dir, out_dir,
                inverse_scale, current_cell_px, mm, dpi,
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Save Tile Image",
                f"Failed to save tiles: {type(e).__name__}: {e}"
            )
            return
        msg = (
            f"Saved {len(saved)} tile(s) into:\n{out_dir}\n\n"
            f"Each PNG is named after its box (A1.png, B2.png, …), "
            f"was cut with that box's OFFSET boundary (color 5) from "
            f"the DXF, and carries DPI ({dpi}) in its pHYs metadata."
        )
        if no_dxf:
            msg += (f"\n\nSkipped (no box_<label>.dxf found or unreadable): "
                    f"{', '.join(no_dxf)}")
        if other:
            msg += (f"\n\nSkipped (out of bounds or write error): "
                    f"{', '.join(other)}")
        QMessageBox.information(self, "Save Tile Image", msg)

    @staticmethod
    def _read_tile_polygon_from_dxf(dxf_path: str):
        """Read a `box_<label>.dxf` and return the OFFSET-BOUNDARY
        polygon as an (N, 2) float32 numpy array in DXF-unit coords.
        Returns None on failure. Same rule as image_strech.py: prefer
        the color-5 offset polyline; fall back to largest non-frame."""
        try:
            import ezdxf
        except Exception:
            return None
        try:
            doc = ezdxf.readfile(str(dxf_path))
        except Exception:
            return None
        candidates = []
        for e in doc.modelspace().query("LWPOLYLINE"):
            try:
                colour = int(getattr(e.dxf, "color", 0))
            except Exception:
                colour = 0
            pts = []
            try:
                for v in e.get_points():
                    pts.append((float(v[0]), float(v[1])))
            except Exception:
                continue
            if len(pts) < 3:
                continue
            candidates.append((np.asarray(pts, dtype=np.float32), colour))
        if not candidates:
            return None
        offset_only = [(p, c) for (p, c) in candidates if c == 5]
        if offset_only:
            return offset_only[0][0]
        non_frame = [(p, c) for (p, c) in candidates if c != 8]
        if not non_frame:
            non_frame = candidates
        non_frame.sort(
            key=lambda c: (
                (c[0][:, 0].max() - c[0][:, 0].min())
                * (c[0][:, 1].max() - c[0][:, 1].min())
            ),
            reverse=True,
        )
        return non_frame[0][0]

    # Two distinct fill colors used by Save Tile Image:
    #   _CANVAS_BACKDROP  — INSIDE the DXF offset polygon, wherever the
    #                       canvas has no image content (matches the
    #                       canvas backdrop the user sees while editing).
    #   _BACKGROUND_FILL  — OUTSIDE the DXF offset polygon (white paper
    #                       around the cut tile).
    _CANVAS_BACKDROP = QColor(48, 48, 48)
    _BACKGROUND_FILL = QColor(255, 255, 255)

    def _render_canvas_region_rgb(self, wx0: float, wy0: float,
                                  w: int, h: int,
                                  out_w: int | None = None,
                                  out_h: int | None = None) -> np.ndarray:
        """Render the composite of canvas background + all visible
        polygon fills within the world-coord rectangle (wx0, wy0,
        w × h). If `out_w`/`out_h` are given, the QImage is created
        at that pixel resolution and painter.scale() maps world coords
        → output pixels so source content (canvas_bg pixmap, polygon
        fill_qimg) is bilinear-sampled from its native resolution
        directly into the output — much sharper than rendering at
        world resolution and then upscaling. When both are None the
        output size defaults to the world size (1 world-unit per
        pixel), matching the previous behaviour."""
        canvas = self.canvas
        if out_w is None: out_w = max(1, w)
        if out_h is None: out_h = max(1, h)
        img = QImage(out_w, out_h, QImage.Format_ARGB32)
        img.fill(self._CANVAS_BACKDROP)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Map "world units" → "output pixels" via a painter scale so
        # every drawImage/drawPixmap below can use world-coord offsets
        # (existing `- wx0` / `- wy0` pattern) and Qt bilinear-samples
        # source content directly from its native resolution into the
        # output. Sharper than rendering at world resolution + upscaling.
        painter.scale(out_w / max(1.0, float(w)),
                      out_h / max(1.0, float(h)))

        # Canvas background.
        if canvas.canvas_background is not None:
            bg = canvas.canvas_background
            bg_w = bg.width()  * canvas.bg_scale
            bg_h = bg.height() * canvas.bg_scale
            sx = canvas.bg_offset_x - wx0
            sy = canvas.bg_offset_y - wy0
            painter.drawPixmap(
                QRectF(sx, sy, bg_w, bg_h), bg,
                QRectF(0, 0, bg.width(), bg.height()),
            )

        gid_visible = {g['id']: g['visible'] for g in canvas.groups}
        # NB: no vertex-bbox culling here. It used to skip polygons
        # whose vertex bbox fell outside (wx0, wy0)–(wx1, wy1), but that
        # missed content when the DXF's offset polygon (which drives
        # the render region) was smaller than the actual mosaic
        # coverage — e.g. polygons added after mosaic_studio generated
        # the DXFs. QPainter clips out-of-range draws anyway, so the
        # cost of iterating everything is negligible next to the
        # correctness win.
        for poly in canvas.polygons:
            if not gid_visible.get(poly['group_id'], True):
                continue
            pts = poly.get('points') or []
            if len(pts) < 3:
                continue
            qpoly = QPolygonF([QPointF(px - wx0, py - wy0)
                               for (px, py) in pts])
            if (poly['fill_type'] == 'image'
                    and poly.get('fill_qimg') is not None):
                bx, by, bw, bh = poly['fill_bbox']
                painter.drawImage(
                    QRectF(bx - wx0, by - wy0, bw, bh),
                    poly['fill_qimg'],
                )
            elif poly['fill_type'] == 'solid':
                painter.setBrush(QBrush(poly['color']))
                painter.setPen(Qt.NoPen)
                painter.drawPolygon(qpoly)
        painter.end()

        # QImage → numpy RGB.
        arr = _qimage_to_numpy_rgba(img)
        return np.ascontiguousarray(arr[:, :, :3])

    def _save_tiles_via_dxf(self, ox: float, oy: float,
                            selected: list, dxf_dir: str, out_dir: str,
                            inverse_scale: float,
                            current_cell_px: float,
                            tile_size_mm: float, dpi: int) -> tuple:
        """Ports image_strech.py's Save Tile Image cutting logic to
        this editor: for each (r, c) in `selected`, read
        `<dxf_dir>/<label>/box_<label>.dxf`, un-scale + translate the
        offset boundary into canvas world coords, render the composite
        of canvas + polygons inside that polygon's bbox, mask outside
        to white, resize proportionally to the requested mm + DPI, and
        save the PNG as `<out_dir>/<label>.png`. Every tile lands in a
        SINGLE flat folder (`out_dir`) named after its box — A1.png,
        A2.png, … — instead of being scattered into each box's own
        sub-DXF folder. Each PNG carries a PngInfo tEXt chunk named
        `mosaic_box_corners` with a JSON blob giving the four
        containing-box corners in PNG-pixel coords so mosaic_aranger
        can place / scale the tile onto a target grid box later.

        Returns (saved, no_dxf, other) lists of tile labels."""
        from PIL import Image as _PILImage
        from PIL.PngImagePlugin import PngInfo as _PngInfo

        # Same scale convention as image_strech.py:
        #   target_w = tile_size_mm * dpi / 25.4  (output px per cell)
        #   scale    = target_w / cell_w          (out_px per src_px)
        target_w = int((tile_size_mm / 25.4) * dpi)
        scale = target_w / max(1.0, current_cell_px)

        saved: list = []
        no_dxf: list = []
        other: list = []

        for (r, c) in selected:
            row_letter = chr(ord('A') + r)
            tile_name  = f"{row_letter}{c + 1}"

            dxf_candidates = [
                os.path.join(dxf_dir, tile_name, f"box_{tile_name}.dxf"),
                os.path.join(dxf_dir, tile_name, f"Box_{tile_name}.dxf"),
            ]
            dxf_path = next(
                (p for p in dxf_candidates if os.path.isfile(p)), None,
            )
            if dxf_path is None:
                no_dxf.append(tile_name)
                continue

            polygon = self._read_tile_polygon_from_dxf(dxf_path)
            if polygon is None or len(polygon) < 3:
                no_dxf.append(tile_name)
                continue

            # DXF coord → canvas world coord:
            #   world = dxf * inverse_scale + origin
            polygon = polygon * inverse_scale
            polygon = polygon + np.array([ox, oy], dtype=np.float32)

            # Crop region = the DXF polygon's own bbox (so the whole
            # offset boundary lives in the PNG — nothing gets clipped
            # by the grid cell rectangle). `Grid box mm` is used only
            # as a SCALING REFERENCE below: it defines the pixel size
            # of one grid box at print, and the polygon is scaled by
            # the same ratio, so a polygon 10 % larger than a grid
            # box comes out 10 % larger than `Grid box mm` in the PNG.
            x_min = float(np.floor(polygon[:, 0].min()))
            y_min = float(np.floor(polygon[:, 1].min()))
            x_max = float(np.ceil (polygon[:, 0].max()))
            y_max = float(np.ceil (polygon[:, 1].max()))
            w_crop = max(1, int(round(x_max - x_min)))
            h_crop = max(1, int(round(y_max - y_min)))
            if w_crop < 1 or h_crop < 1:
                other.append(tile_name)
                continue

            # Output pixel dimensions — one grid box maps to target_w,
            # so this polygon's world width × scale gives its PNG width.
            out_w = max(1, int(round(w_crop * scale)))
            out_h = max(1, int(round(h_crop * scale)))

            # Composite canvas → RGB numpy DIRECTLY at output resolution.
            # Rendering at output size (via QPainter transform inside
            # _render_canvas_region_rgb) preserves source detail —
            # canvas_bg pixmap and polygon fill_qimg get bilinear-sampled
            # once from their native resolution to the output, instead
            # of first being drawn at low world-coord resolution and
            # then upscaled. This matches image_strech.py's effective
            # quality: it crops from cv_image at native res and does
            # one cv2.resize at the end.
            rgb = self._render_canvas_region_rgb(
                x_min, y_min, w_crop, h_crop,
                out_w=out_w, out_h=out_h,
            )

            # DXF polygon → OUTPUT-pixel coords for masking (multiply
            # local coords by scale). Outside the polygon → background
            # fill; inside keeps the canvas content already rendered.
            mask = np.zeros((out_h, out_w), dtype=np.uint8)
            local_pts = polygon.copy()
            local_pts[:, 0] = (local_pts[:, 0] - x_min) * scale
            local_pts[:, 1] = (local_pts[:, 1] - y_min) * scale
            cv2.fillPoly(mask, [local_pts.astype(np.int32)], 255)
            outside = (
                self._BACKGROUND_FILL.red(),
                self._BACKGROUND_FILL.green(),
                self._BACKGROUND_FILL.blue(),
            )
            rgb[mask == 0] = outside

            # rgb is already at output resolution — no cv2.resize step.
            tile_resized = rgb

            # ── Compute the containing-box corner metadata ────────
            # Cell (r, c) occupies world rect anchored at the user-
            # picked origin. Its four corners are:
            #   TL: (ox + c*cell,       oy + r*cell)
            #   TR: (ox + (c+1)*cell,   oy + r*cell)
            #   BR: (ox + (c+1)*cell,   oy + (r+1)*cell)
            #   BL: (ox + c*cell,       oy + (r+1)*cell)
            # Convert each world corner to PNG-pixel coords:
            #   png_px = (world - polygon_bbox_min) * scale
            cell = current_cell_px
            world_corners = [
                (ox + c * cell,       oy + r * cell),         # TL
                (ox + (c + 1) * cell, oy + r * cell),         # TR
                (ox + (c + 1) * cell, oy + (r + 1) * cell),   # BR
                (ox + c * cell,       oy + (r + 1) * cell),   # BL
            ]
            corners_px = [
                [float(round((wx - x_min) * scale, 3)),
                 float(round((wy - y_min) * scale, 3))]
                for (wx, wy) in world_corners
            ]

            meta = {
                "label":         tile_name,
                "grid_box_mm":   float(tile_size_mm),
                "dpi":           int(dpi),
                "png_size_px":   [int(out_w), int(out_h)],
                # Clockwise from top-left; each entry is [x_px, y_px]
                # measured from the PNG's top-left corner (0, 0).
                "corners_px":    corners_px,
                "corner_order":  ["TL", "TR", "BR", "BL"],
                "coord_system": (
                    "PNG-pixel space, (0,0) = PNG top-left. Sub-pixel "
                    "precision. Load the PNG, read the "
                    "`mosaic_box_corners` PngInfo tEXt chunk, and warp "
                    "these four points onto the target grid box's four "
                    "corners to align the tile."
                ),
            }
            pnginfo = _PngInfo()
            pnginfo.add_text("mosaic_box_corners", json.dumps(meta))

            # PNG saved into the user-named flat output folder as
            # `<label>.png`, with DPI in pHYs and the box-corner
            # metadata in tEXt.
            out_path = os.path.join(out_dir, f"{tile_name}.png")
            try:
                _PILImage.fromarray(tile_resized).save(
                    out_path, format="PNG",
                    dpi=(float(dpi), float(dpi)),
                    pnginfo=pnginfo,
                )
                saved.append(tile_name)
            except Exception:
                other.append(tile_name)

            # ── Debug outputs (per-tile diagnostic) ────────────────
            # Sidebar "Debug Tile" checkbox controls this. Writes:
            #   <label>_debug.txt  — per-polygon log
            #   <label>_debug.png  — render + overlays (DXF polygon,
            #                        grid cell, canvas polygons)
            # Both go into the same flat output folder as the tile PNG.
            if self.debug_tile_chk.isChecked():
                try:
                    self._write_tile_debug(
                        out_dir, tile_name,
                        dxf_path, polygon,
                        origin_x=ox, origin_y=oy,
                        cell_world=current_cell_px,
                        cell_r=r, cell_c=c,
                        x_min=x_min, y_min=y_min,
                        w_crop=w_crop, h_crop=h_crop,
                        out_w=out_w, out_h=out_h,
                        scale=scale, rgb=rgb,
                    )
                except Exception as e:
                    print(
                        f"[mosaic_editor] debug write failed for "
                        f"{tile_name}: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
        return saved, no_dxf, other

    def _write_tile_debug(self, out_dir: str, label: str,
                          dxf_path: str, polygon_world,
                          origin_x: float, origin_y: float,
                          cell_world: float,
                          cell_r: int, cell_c: int,
                          x_min: float, y_min: float,
                          w_crop: int, h_crop: int,
                          out_w: int, out_h: int, scale: float,
                          rgb) -> None:
        """Emit `<label>_debug.txt` + `<label>_debug.png` next to the
        saved tile PNG. The txt lists every canvas polygon plus its
        state; the png draws the render (unmasked) with the DXF
        polygon (red), grid cell (blue), and every polygon that
        intersected the render region (green) overlaid on top. Only
        called when the sidebar's `Debug Tile` checkbox is on."""
        import os
        from PIL import Image as _PILImage, ImageDraw as _ImageDraw

        wx0, wy0 = float(x_min), float(y_min)
        wx1, wy1 = wx0 + float(w_crop), wy0 + float(h_crop)

        # Grid cell world corners.
        cell_x0 = origin_x + cell_c * cell_world
        cell_y0 = origin_y + cell_r * cell_world
        cell_x1 = cell_x0 + cell_world
        cell_y1 = cell_y0 + cell_world

        # ── Text log ──────────────────────────────────────────────
        lines: list[str] = []
        lines.append(f"=== Debug: tile {label} (r={cell_r}, c={cell_c}) ===")
        lines.append(f"DXF path:              {dxf_path}")
        lines.append(f"Chosen origin (world): ({origin_x:.3f}, {origin_y:.3f})")
        lines.append(f"Grid cell size (world): {cell_world:.3f}")
        lines.append(f"Grid cell bbox (world): "
                     f"({cell_x0:.2f}, {cell_y0:.2f}) → "
                     f"({cell_x1:.2f}, {cell_y1:.2f})")
        lines.append(f"DXF polygon (un-scaled + translated), "
                     f"{len(polygon_world)} pts.")
        px_min = float(polygon_world[:, 0].min())
        py_min = float(polygon_world[:, 1].min())
        px_max = float(polygon_world[:, 0].max())
        py_max = float(polygon_world[:, 1].max())
        lines.append(f"  bbox (world): ({px_min:.2f}, {py_min:.2f}) → "
                     f"({px_max:.2f}, {py_max:.2f})")
        lines.append(f"Render region (world): ({wx0:.2f}, {wy0:.2f}) → "
                     f"({wx1:.2f}, {wy1:.2f})  "
                     f"[{w_crop}×{h_crop} world units]")
        lines.append(f"Output PNG size:       {out_w}×{out_h} px, "
                     f"scale = {scale:.4f} px/world")
        lines.append(f"Cell relative to DXF polygon bbox: "
                     f"cell_inside_bbox="
                     f"{px_min <= cell_x0 and py_min <= cell_y0 and px_max >= cell_x1 and py_max >= cell_y1}")

        canvas = self.canvas
        gid_visible = {g['id']: g['visible'] for g in canvas.groups}
        gid_name = {g['id']: g.get('name', '?') for g in canvas.groups}

        # Per-polygon breakdown, restricted to polygons whose bbox
        # intersects either the render region or the grid cell.
        lines.append(f"\n=== Polygons (visible + intersecting either "
                     f"render region or grid cell) ===")
        drawn_count = 0
        in_cell_only = 0
        in_region_only = 0
        for i, poly in enumerate(canvas.polygons):
            group_visible = gid_visible.get(poly['group_id'], True)
            pts = poly.get('points') or []
            if len(pts) < 3:
                continue
            pxs = [p[0] for p in pts]; pys = [p[1] for p in pts]
            pbx0 = min(pxs); pby0 = min(pys)
            pbx1 = max(pxs); pby1 = max(pys)
            hits_region = (pbx1 >= wx0 and pbx0 <= wx1
                           and pby1 >= wy0 and pby0 <= wy1)
            hits_cell   = (pbx1 >= cell_x0 and pbx0 <= cell_x1
                           and pby1 >= cell_y0 and pby0 <= cell_y1)
            if not (hits_region or hits_cell):
                continue
            if group_visible and hits_region:
                drawn_count += 1
            if hits_cell and not hits_region:
                in_cell_only += 1
            if hits_region and not hits_cell:
                in_region_only += 1
            ft = poly.get('fill_type', '?')
            fill_img = poly.get('fill_qimg')
            fill_bbox = poly.get('fill_bbox')
            fq_w = fill_img.size().width()  if fill_img is not None else 0
            fq_h = fill_img.size().height() if fill_img is not None else 0
            lines.append(
                f"[{i}] group={poly['group_id']} "
                f"({gid_name.get(poly['group_id'], '?')}) "
                f"vis={group_visible} fill={ft} "
                f"pts_bbox=({pbx0:.1f}, {pby0:.1f}, {pbx1:.1f}, {pby1:.1f}) "
                f"fill_bbox={fill_bbox} fill_qimg={fq_w}×{fq_h} "
                f"in_region={hits_region} in_cell={hits_cell}"
            )

        lines.append(f"\n=== Counts ===")
        lines.append(f"Polygons drawn (visible + in render region): {drawn_count}")
        lines.append(f"Polygons in cell but OUTSIDE render region:  {in_cell_only}")
        lines.append(f"Polygons in region but OUTSIDE cell:         {in_region_only}")

        txt_path = os.path.join(out_dir, f"{label}_debug.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        # ── Debug PNG (overlays on the pre-mask render) ───────────
        # Take the numpy render (rgb, size out_h × out_w × 3, no mask
        # applied) as the base, then draw:
        #   • red thick outline: DXF polygon (in output pixel coords)
        #   • blue thick outline: grid cell rectangle
        #   • green thin outlines: polygons that intersect the region
        base = _PILImage.fromarray(rgb).convert("RGBA")
        draw = _ImageDraw.Draw(base)

        def w2p(x, y):
            return ((x - wx0) * scale, (y - wy0) * scale)

        # DXF polygon (red).
        px = [w2p(px_i, py_i) for (px_i, py_i) in polygon_world]
        draw.line(px + [px[0]], fill=(255, 40, 40, 255), width=4)

        # Grid cell rectangle (blue).
        c_tl = w2p(cell_x0, cell_y0)
        c_tr = w2p(cell_x1, cell_y0)
        c_br = w2p(cell_x1, cell_y1)
        c_bl = w2p(cell_x0, cell_y1)
        draw.line([c_tl, c_tr, c_br, c_bl, c_tl],
                  fill=(40, 100, 255, 255), width=4)

        # Every visible polygon that hits the region — green thin.
        for poly in canvas.polygons:
            if not gid_visible.get(poly['group_id'], True):
                continue
            pts = poly.get('points') or []
            if len(pts) < 3:
                continue
            pxs = [p[0] for p in pts]; pys = [p[1] for p in pts]
            if max(pxs) < wx0 or min(pxs) > wx1: continue
            if max(pys) < wy0 or min(pys) > wy1: continue
            pcoords = [w2p(px_, py_) for (px_, py_) in pts]
            draw.line(pcoords + [pcoords[0]],
                      fill=(50, 200, 60, 220), width=1)

        png_path = os.path.join(out_dir, f"{label}_debug.png")
        base.save(png_path, format="PNG")

    def on_delete_tool_toggled(self, checked: bool) -> None:
        """Toggle the on-canvas eraser tool from the right toolbar."""
        self.canvas.set_eraser_mode(checked)
        if checked:
            self.statusBar().showMessage(
                "Eraser tool ON — drag over polygons in ⇢-checked "
                "layers to delete. Ctrl+Z to undo."
            )
        else:
            self.statusBar().showMessage("Eraser tool off.")

    def on_eraser_radius_changed(self, v: float) -> None:
        self.canvas.eraser_radius = float(v)
        self.canvas.update()

    def on_erase_affected(self) -> None:
        """Confirm, then delete every polygon in every ⇢-checked
        layer (and remove those layers). Destructive but undoable."""
        # Preview count so the user knows what they're about to erase.
        affected_gids = {
            g['id'] for g in self.canvas.groups if g.get('affected', True)
        }
        n_polys = sum(1 for p in self.canvas.polygons
                      if p['group_id'] in affected_gids)
        n_groups = len(affected_gids)
        if n_polys == 0 and n_groups == 0:
            QMessageBox.information(
                self, "Erase Affected",
                "No affected layers — tick the ⇢ column on the "
                "layers you want to erase."
            )
            return
        reply = QMessageBox.question(
            self, "Erase Affected",
            f"Delete {n_polys} polygon(s) across "
            f"{n_groups} layer(s)?\nUndoable with Ctrl+Z.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        n_p, n_g = self.canvas.erase_affected()
        self.layers_panel.rebuild()
        self.statusBar().showMessage(
            f"Erased {n_p} polygon(s) and {n_g} layer(s). "
            f"Ctrl+Z to undo."
        )

    def on_undo(self) -> None:
        """Undo the most recent mutation. Rebuilds the layers panel
        because groups may have changed (loads / deletes)."""
        if self.canvas.undo():
            self.layers_panel.rebuild()
            self.statusBar().showMessage("Undo")
        else:
            self.statusBar().showMessage("Nothing to undo.")

    def on_save_array(self) -> None:
        """Save all visible polygons to a CSV in the shared save_array
        schema. 3 steps:
          1. Ask for the target grid-box size (scale = target /
             current).
          2. Ask the user to click a gridline intersection — that
             point becomes (0, 0) in the exported coordinate system.
          3. Prompt for a save path and write the CSV with no color
             fill (just polygon points)."""
        if not self.canvas.polygons:
            QMessageBox.information(
                self, "Save Array",
                "No polygons on the canvas — nothing to save."
            )
            return
        # Step 1: current gridbox size in world/pixel units.
        if self.canvas.canvas_background is not None:
            current_cell_px = max(
                1.0,
                self.canvas.canvas_background.width()
                * (self.canvas.grid_size_percent / 100.0),
            )
        else:
            current_cell_px = max(1.0, self.canvas.grid_size_world)

        new_cell_px, ok = QInputDialog.getDouble(
            self,
            "Calibrate Grid Box Size",
            f"Current grid box is {current_cell_px:.2f} px.\n"
            "Enter the target grid box size in pixels.\n"
            "All coordinates will be scaled by (target / current):",
            value=round(current_cell_px, 2),
            min=0.01,
            max=100000.0,
            decimals=2,
        )
        if not ok:
            return
        scale = new_cell_px / current_cell_px

        # Turn the grid on if it isn't already so the user can see
        # the intersections they're picking from.
        if not self.canvas.grid_enabled:
            self.canvas.grid_enabled = True
            self.grid_cb.blockSignals(True)
            self.grid_cb.setChecked(True)
            self.grid_cb.blockSignals(False)
            self.canvas.update()

        # Step 2: enter origin-pick mode. Step 3 fires from the
        # callback once the user clicks a gridline intersection.
        self.statusBar().showMessage(
            "Save Array — click a gridline intersection to set it as "
            "(0, 0) for the exported array. Press Esc to cancel."
        )
        self.canvas.start_origin_pick(
            lambda ox, oy: self._save_array_finish(scale, ox, oy)
        )

    def _save_array_finish(self, scale: float,
                           origin_x: float, origin_y: float) -> None:
        """Step 3 of on_save_array — prompt for a save path and
        write the CSV using the scale + origin the user chose."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Array as CSV", "polygons.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            self.statusBar().showMessage("Save Array cancelled.")
            return
        try:
            n = self.canvas.save_array_csv(
                path, scale=scale, origin=(origin_x, origin_y),
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Save Array",
                f"Failed to save array: {type(e).__name__}: {e}"
            )
            return
        self.statusBar().showMessage(
            f"Array saved: {Path(path).name}  "
            f"({n} polygon(s), origin=({origin_x:.1f}, {origin_y:.1f}), "
            f"scale × {scale:.3f})"
        )

    def on_save_mosaic(self) -> None:
        """Flatten the whole canvas into one portable .mosaic bundle
        (composite PNG + polygon CSV). Same format image_strech.py's
        Save Mosaic writes; loadable in this app via Load .mosaic."""
        if not self.canvas.polygons and self.canvas.canvas_background is None:
            QMessageBox.information(
                self, "Save Mosaic",
                "Canvas is empty — nothing to save."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mosaic", "canvas.mosaic",
            "Mosaic bundle (*.mosaic *.zip);;All files (*.*)"
        )
        if not path:
            return
        try:
            n, w, h = self.canvas.save_mosaic(path)
        except Exception as e:
            QMessageBox.critical(
                self, "Save Mosaic",
                f"Failed to save mosaic: {type(e).__name__}: {e}"
            )
            return
        self.statusBar().showMessage(
            f"Mosaic saved: {Path(path).name}  "
            f"({n} polygon(s), {w}×{h} composite)"
        )

    def on_save_project(self) -> None:
        if not self.canvas.polygons and self.canvas.canvas_background is None:
            QMessageBox.information(
                self, "Save Project",
                "Canvas is empty — nothing to save."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mosaic Editor Project",
            "project.mep",
            "Mosaic Editor Project (*.mep);;All files (*.*)"
        )
        if not path:
            return
        try:
            self.canvas.save_project(path)
        except Exception as e:
            QMessageBox.critical(
                self, "Save Project",
                f"Failed to save project: {type(e).__name__}: {e}"
            )
            return
        self.statusBar().showMessage(
            f"Project saved: {Path(path).name} "
            f"({len(self.canvas.polygons)} polygon(s), "
            f"{len(self.canvas.groups)} layer(s))"
        )

    def on_load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Mosaic Editor Project", "",
            "Mosaic Editor Project (*.mep);;All files (*.*)"
        )
        if not path:
            return
        try:
            warnings = self.canvas.load_project(path) or []
        except Exception as e:
            QMessageBox.critical(
                self, "Load Project",
                f"Failed to load project: {type(e).__name__}: {e}"
            )
            return
        if warnings:
            QMessageBox.warning(
                self, "Load Project — partial",
                "Loaded, but with warnings:\n\n" + "\n".join(warnings),
            )
        # Sync sidebar widgets to the newly-loaded canvas state so the
        # spinboxes / checkbox positions don't disagree with the canvas.
        self.grid_cb.blockSignals(True)
        self.grid_cb.setChecked(self.canvas.grid_enabled)
        self.grid_cb.blockSignals(False)

        self.grid_size_spin.blockSignals(True)
        # Pick whichever field matches the current mode.
        if self.canvas.canvas_background is not None:
            self.grid_size_spin.setValue(self.canvas.grid_size_percent)
        else:
            self.grid_size_spin.setValue(self.canvas.grid_size_world)
        self.grid_size_spin.blockSignals(False)

        self.grid_thickness_spin.blockSignals(True)
        self.grid_thickness_spin.setValue(self.canvas.grid_thickness)
        self.grid_thickness_spin.blockSignals(False)
        self._refresh_grid_color_btn()

        self.layers_panel.rebuild()
        self.statusBar().showMessage(
            f"Project loaded: {Path(path).name} "
            f"({len(self.canvas.polygons)} polygon(s), "
            f"{len(self.canvas.groups)} layer(s))"
        )

    def on_grid_toggled(self, checked: bool) -> None:
        self.canvas.grid_enabled = checked
        self.canvas.update()

    def on_polygon_lines_toggled(self, checked: bool) -> None:
        self.canvas.show_polygon_lines = bool(checked)
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
    # Open maximised — "full screen" in the practical sense (window
    # frame + title bar retained; whole screen used).
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
