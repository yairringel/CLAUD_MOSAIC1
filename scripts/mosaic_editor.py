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
import pickle
import zipfile
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QFrame, QLabel, QPushButton, QFileDialog, QCheckBox,
    QSpinBox, QDoubleSpinBox, QMessageBox, QScrollArea, QColorDialog,
)
from PyQt5.QtCore import Qt, QRectF, QBuffer
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
    def add_group(self, name: str, kind: str) -> int:
        gid = self._next_group_id
        self._next_group_id += 1
        # applied_scale tracks the group's current size as a factor of
        # its original (as-loaded) size. Every scale_affected(pct)
        # brings the group to `pct/100` * original — the ratio to the
        # current applied_scale is what actually gets multiplied into
        # the coordinates, so scaling is always absolute-from-original.
        self.groups.append({'id': gid, 'name': name, 'kind': kind,
                            'visible': True, 'affected': True,
                            'applied_scale': 1.0})
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
    PROJECT_VERSION = 1

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
        """Serialise the full canvas state to a .mep pickle file:
        every polygon (points + fill colour + fill image + bbox +
        group_id), every group's metadata, canvas background, view
        transform, and grid settings."""
        data = {
            'version': self.PROJECT_VERSION,
            'groups':  [dict(g) for g in self.groups],
            'next_group_id':          self._next_group_id,
            'polygons': [
                {
                    'points':    list(p['points']),
                    'fill_type': p['fill_type'],
                    'color':     self._qcolor_to_tuple(p['color']),
                    'fill_png':  self._qimage_to_png_bytes(p.get('fill_qimg')),
                    'fill_bbox': p.get('fill_bbox'),
                    # Warp-reference snapshot — needed so vertex-drag
                    # stretches from the stable original geometry
                    # after a project round-trip.
                    'orig_points':    (list(p['orig_points'])
                                       if p.get('orig_points') is not None
                                       else None),
                    'orig_fill_png':  self._qimage_to_png_bytes(
                        p.get('orig_fill_qimg')
                    ),
                    'orig_fill_bbox': p.get('orig_fill_bbox'),
                    'group_id':  p['group_id'],
                } for p in self.polygons
            ],
            'canvas_background_png':
                self._qimage_to_png_bytes(
                    self.canvas_background.toImage()
                    if self.canvas_background is not None else None
                ),
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

    def load_project(self, path: str) -> None:
        """Restore canvas state from a .mep pickle file. Replaces every
        current layer / background / transform / grid setting."""
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
                'group_id':  int(p['group_id']),
            })
        self.polygons = polys

        # Canvas background.
        bg_qimg = self._png_bytes_to_qimage(
            data.get('canvas_background_png')
        )
        self.canvas_background = (QPixmap.fromImage(bg_qimg)
                                  if bg_qimg is not None else None)
        self.bg_offset_x = float(data.get('bg_offset_x', 0.0))
        self.bg_offset_y = float(data.get('bg_offset_y', 0.0))
        self.bg_scale    = float(data.get('bg_scale',    1.0))
        self.bg_affected = bool (data.get('bg_affected', True))

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
            if self._dragging_vertex:
                self._dragging_vertex = False
                self._vertex_drag_idx = -1
                self.setCursor(Qt.ArrowCursor)
                return
            if self._dragging_polygon:
                self._dragging_polygon = False
                self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, ev):
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

        # 1. Bilinear-scale the original (already polygon-masked) fill
        #    image into a fresh (nw × nh) QImage. `QImage.scaled` with
        #    SmoothTransformation does the bilinear resample.
        stretched = orig_qimg.scaled(
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

        # Promote whatever fill state exists into the warp-reference
        # slots so the stretch code has something to work from. This
        # is a no-op when orig_* is already populated (from a fresh
        # .mosaic load).
        if poly.get('orig_points') is None:
            poly['orig_points'] = list(poly.get('points') or [])
        if (poly.get('orig_fill_qimg') is None
                and poly.get('fill_qimg') is not None):
            poly['orig_fill_qimg'] = poly['fill_qimg'].copy()
        if poly.get('orig_fill_bbox') is None:
            poly['orig_fill_bbox'] = poly.get('fill_bbox')

        has_img = poly.get('orig_fill_qimg') is not None
        print(
            f"[mosaic_editor] vertex-drag start: polygon #{poly_idx}, "
            f"vertex #{vertex_idx}, image-fill={has_img}",
            file=sys.stderr,
        )

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
            self.canvas.load_project(path)
        except Exception as e:
            QMessageBox.critical(
                self, "Load Project",
                f"Failed to load project: {type(e).__name__}: {e}"
            )
            return
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
