"""sam.py — Mosaic image → polygon CSV via Meta's Segment Anything.

Pipeline is a direct port of mosaic-vectorizer-py's engine (which the
user confirmed works well on their mosaics), re-wrapped around the
PyQt5 UI shell:

  1. Resize the image to a workable size (default 2048 px long edge)
     and remember the scale factor so output polygons map back to
     ORIGINAL pixel coords.
  2. Split the work image into N × N overlapping tiles. For each tile:
       a. Compute the SAM image embedding ONCE (expensive step).
       b. Scatter a grid of point prompts across the tile core.
       c. Skip prompts that already sit inside a committed stone.
       d. Batch the remaining prompts through SamPredictor.predict_torch
          with multimask_output → 3 candidate masks per point.
       e. For each prompt pick the candidate that passes:
            * predicted IoU ≥ threshold
            * manual stability (logit > 1 area / logit > -1 area)
              ≥ threshold
            * area in [min_area, max_area_frac × image]
            * does not touch an INNER tile edge (border masks are
              left for the neighbouring tile)
            * contains its own prompt point
          Ties broken on higher IoU.
       f. Commit the chosen mask into a per-pixel StoneMap (label
          image). The commit rejects masks that overlap existing
          stones too much; otherwise it claims only the unclaimed
          pixels.
  3. After every tile is processed, convert each surviving stone's
     mask to a contour, run cv2.approxPolyDP at the user's tolerance,
     optionally Chaikin-smooth, scale back to original pixels, and
     sample the mean RGB from the SOURCE image under the stone.

CSV schema matches mosaic_to_csv.py so polygon_viewer.py loads this
file unchanged:
    coordinates, color_r, color_g, color_b, color_a, color_hex

Setup (one-time):
  1. pip install torch torchvision segment-anything opencv-python Pillow
  2. Either click "Choose Checkpoint..." to point at a pre-downloaded
     .pth, or just pick a variant (vit_b/l/h) and click "Load Model" —
     the checkpoint is auto-downloaded to IMAGE_TO_MOSAIC/checkpoints/
     on first use.
"""
from __future__ import annotations

# IMPORTANT — on Windows, torch MUST be imported before PyQt5. PyQt5
# loads a Qt-bundled C++ runtime that shadows the one torch's bundled
# CUDA DLLs link against, causing `OSError: [WinError 1114] c10.dll` on
# the next `import torch`. Loading torch first lets it bind its own
# DLLs into the process cleanly; later Qt imports don't disturb what's
# already resolved. If this import fails, the UI never comes up — which
# is correct: without torch, sam.py has nothing to offer.
try:
    import torch                                            # noqa: E402, F401
except Exception as _torch_import_err:                       # pragma: no cover
    import sys as _sys
    print(
        "sam.py: failed to import torch BEFORE PyQt5.\n"
        "Install torch for your GPU first:\n"
        "    # CUDA 12.8 (RTX 50 series Blackwell):\n"
        "    pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu128\n"
        "    # CPU-only (slower):\n"
        "    pip install torch torchvision\n\n"
        f"Original error: {type(_torch_import_err).__name__}: "
        f"{_torch_import_err}",
        file=_sys.stderr,
    )
    raise

import csv
import os
import sys
import traceback
import urllib.request
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from PyQt5.QtCore import QEvent, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

ROOT         = Path(__file__).resolve().parent.parent    # IMAGE_TO_MOSAIC/
PROJECT_ROOT = ROOT.parent
INPUT_DIR    = ROOT / "input"
OUTPUT_DIR   = ROOT / "output"
SAM_PATH_MEMO = ROOT / ".sam_path"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

CHECKPOINTS = {
    'vit_b': ('sam_vit_b_01ec64.pth',
              'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
              '375 MB'),
    'vit_l': ('sam_vit_l_0b3195.pth',
              'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
              '1.2 GB'),
    'vit_h': ('sam_vit_h_4b8939.pth',
              'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
              '2.5 GB'),
}


# ---------------------------------------------------------------------------
# Checkpoint path persistence + download
# ---------------------------------------------------------------------------

def read_memo_path() -> str | None:
    """Return the remembered manual checkpoint path, or None."""
    if not SAM_PATH_MEMO.is_file():
        return None
    try:
        p = SAM_PATH_MEMO.read_text(encoding="utf-8").strip()
        if p and Path(p).is_file():
            return p
    except Exception:
        return None
    return None


def write_memo_path(path: str) -> None:
    try:
        SAM_PATH_MEMO.write_text(str(Path(path).resolve()), encoding="utf-8")
    except Exception:
        pass


def detect_variant(path: str) -> str:
    """Infer the SAM registry key from the checkpoint filename."""
    name = Path(path).name.lower()
    if "vit_h" in name or "4b8939" in name:
        return "vit_h"
    if "vit_l" in name or "0b3195" in name:
        return "vit_l"
    if "vit_b" in name or "01ec64" in name:
        return "vit_b"
    return "vit_h"


def ensure_checkpoint(variant: str,
                      progress: Callable[[float], None] | None = None
                      ) -> str:
    """Return the local path to the variant's checkpoint, downloading
    into `IMAGE_TO_MOSAIC/scripts/checkpoints/` if it isn't there yet."""
    name, url, _ = CHECKPOINTS[variant]
    path = CHECKPOINT_DIR / name
    if path.exists():
        return str(path)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")

    def hook(blocks, block_size, total):
        if progress is not None and total > 0:
            progress(min(1.0, blocks * block_size / total))

    urllib.request.urlretrieve(url, str(tmp), hook)
    os.replace(tmp, path)
    return str(path)


# ---------------------------------------------------------------------------
# StoneMap — direct port of mosaic-vectorizer-py's sam_engine.StoneMap.
# Per-pixel labels, per-stone bbox + area + mean colour. Commits reject
# masks that overlap existing stones too much; the first stone there
# wins the pixels.
# ---------------------------------------------------------------------------

class StoneMap:
    def __init__(self, image_rgb: np.ndarray):
        self.image = image_rgb
        self.H, self.W = image_rgb.shape[:2]
        self.labels = np.zeros((self.H, self.W), np.int32)
        self.stones: dict = {}
        self.next_id = 1

    def commit(self, crop: np.ndarray, x0: int, y0: int,
               max_overlap: float = 0.3, min_area: int = 0) -> int:
        """Add a stone from a boolean mask crop placed at (x0, y0).

        Rejected (returns 0) when the mask overlaps already-committed
        stones by more than `max_overlap` × its own area. Otherwise
        only the unclaimed pixels are taken; the stone is cleaned up
        (largest connected piece, holes filled) and dropped if the
        remaining area falls below `min_area`."""
        n = int(crop.sum())
        if not n:
            return 0
        h, w = crop.shape
        region = self.labels[y0:y0 + h, x0:x0 + w]
        if int((region[crop] != 0).sum()) > max_overlap * n:
            return 0
        sid = self.next_id
        self.next_id += 1
        take = crop & (region == 0)
        region[take] = sid
        self.stones[sid] = {'bbox': (x0, y0, x0 + w, y0 + h)}
        self._cleanup(sid)
        s = self.stones.get(sid)
        if s is None or s['area'] < min_area:
            if s is not None:
                self.remove(sid)
            return 0
        return sid

    def _cleanup(self, sid: int) -> None:
        """Keep the largest 4-connected piece, fill enclosed holes,
        recompute bbox / area / mean colour."""
        s = self.stones.get(sid)
        if s is None:
            return
        x0, y0, x1, y1 = s['bbox']
        region = self.labels[y0:y1, x0:x1]
        m = (region == sid).astype(np.uint8)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        if n <= 1:
            del self.stones[sid]
            return
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        region[(m == 1) & (cc != best)] = 0
        keep = (cc == best).astype(np.uint8)
        contours, _ = cv2.findContours(
            keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(keep)
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
        region[(filled == 1) & (keep == 0) & (region == 0)] = sid
        mask = region == sid
        ys, xs = np.nonzero(mask)
        color = self.image[y0:y1, x0:x1][mask].mean(axis=0)
        s.update(
            bbox=(x0 + int(xs.min()),
                  y0 + int(ys.min()),
                  x0 + int(xs.max()) + 1,
                  y0 + int(ys.max()) + 1),
            area=int(mask.sum()),
            color=tuple(int(round(c)) for c in color),
        )

    def remove(self, sid: int) -> None:
        s = self.stones.pop(sid, None)
        if s is None:
            return
        x0, y0, x1, y1 = s['bbox']
        region = self.labels[y0:y1, x0:x1]
        region[region == sid] = 0


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def make_tiles(W: int, H: int, n: int, overlap: float = 0.15) -> list:
    """n × n grid of tiles, each with a `core` (where its seed points
    live) and a larger `crop` (what SAM sees). Overlap keeps a stone
    straddling two cores catchable from the owning tile."""
    tw, th = W / n, H / n
    ox, oy = round(tw * overlap), round(th * overlap)
    tiles = []
    for j in range(n):
        for i in range(n):
            kx0, kx1 = round(i * tw),       round((i + 1) * tw)
            ky0, ky1 = round(j * th),       round((j + 1) * th)
            tiles.append({
                'core': (kx0, ky0, kx1, ky1),
                'crop': (max(0, kx0 - ox),
                         max(0, ky0 - oy),
                         min(W, kx1 + ox),
                         min(H, ky1 + oy)),
            })
    return tiles


# ---------------------------------------------------------------------------
# Polygon extraction — port of vectorize.stone_polygon, Chaikin corner
# smoothing optional.
# ---------------------------------------------------------------------------

def chaikin(pts: np.ndarray, iterations: int) -> np.ndarray:
    for _ in range(iterations):
        nxt = np.roll(pts, -1, axis=0)
        pts = np.stack([0.75 * pts + 0.25 * nxt,
                        0.25 * pts + 0.75 * nxt], axis=1).reshape(-1, 2)
    return pts


def stone_polygon(smap: StoneMap, sid: int,
                  tolerance: float, smoothing: int) -> np.ndarray | None:
    """Outer outline of one stone in WORK-image pixel coords. Padded by
    1 px before findContours so the edge-of-bbox pixels get a complete
    contour; the padding is subtracted back out at the end."""
    s = smap.stones.get(sid)
    if s is None:
        return None
    x0, y0, x1, y1 = s['bbox']
    m = np.pad((smap.labels[y0:y1, x0:x1] == sid).astype(np.uint8), 1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=len)
    if tolerance > 0:
        c = cv2.approxPolyDP(c, tolerance, True)
    pts = c.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return None
    pts = chaikin(pts, smoothing)
    # Undo the 1-px pad and shift by 0.5 so coordinates represent pixel
    # centres (the same convention mosaic-vectorizer-py uses for its
    # SVG / DXF exports).
    pts += (x0 - 1 + 0.5, y0 - 1 + 0.5)
    return pts


# ---------------------------------------------------------------------------
# Engine — SamPredictor-driven, batched point prompts per tile.
# ---------------------------------------------------------------------------

def pick_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# Module-level cache so re-running Detect on the same checkpoint
# doesn't reload the .pth every click.
_CACHED_SAM = None
_CACHED_PREDICTOR = None
_CACHED_CHECKPOINT: str | None = None
_CACHED_VARIANT: str | None = None
_CACHED_DEVICE: str | None = None


def load_sam(checkpoint_path: str, variant: str, device: str):
    """Load (or reuse) a SAM model + predictor."""
    global _CACHED_SAM, _CACHED_PREDICTOR
    global _CACHED_CHECKPOINT, _CACHED_VARIANT, _CACHED_DEVICE
    if (_CACHED_SAM is not None
            and _CACHED_CHECKPOINT == checkpoint_path
            and _CACHED_VARIANT == variant
            and _CACHED_DEVICE == device):
        return _CACHED_SAM, _CACHED_PREDICTOR
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry[variant](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    predictor = SamPredictor(sam)
    _CACHED_SAM = sam
    _CACHED_PREDICTOR = predictor
    _CACHED_CHECKPOINT = checkpoint_path
    _CACHED_VARIANT = variant
    _CACHED_DEVICE = device
    return sam, predictor


def auto_segment(predictor, device: str, smap: StoneMap,
                 points_per_side: int, tiles: int,
                 pred_iou_thresh: float, stability_thresh: float,
                 min_area: int, max_area_frac: float,
                 max_overlap: float, batch: int,
                 progress_cb: Callable[[str, float], None] | None,
                 should_stop: Callable[[], bool]) -> int:
    """Run the tiled grid-of-prompts SAM pipeline into `smap`. Returns
    the number of stones added."""
    max_area = max_area_frac * smap.W * smap.H
    tile_list = make_tiles(smap.W, smap.H, tiles)
    total = len(tile_list) * points_per_side ** 2
    done = 0
    added = 0
    emb_cache: dict[tuple, tuple] = {}

    for ti, tile in enumerate(tile_list):
        if should_stop():
            return added
        if progress_cb:
            progress_cb(
                f"Tile {ti + 1}/{len(tile_list)}: image embedding…",
                done / total,
            )
        cx0, cy0, cx1, cy1 = tile['crop']
        kx0, ky0, kx1, ky1 = tile['core']
        with torch.inference_mode():
            crop_key = tile['crop']
            if crop_key in emb_cache:
                predictor.features, predictor.original_size, predictor.input_size = \
                    emb_cache[crop_key]
                predictor.is_image_set = True
            else:
                predictor.set_image(np.ascontiguousarray(
                    smap.image[cy0:cy1, cx0:cx1]))
                emb_cache[crop_key] = (
                    predictor.features,
                    predictor.original_size,
                    predictor.input_size,
                )

        n = points_per_side
        seeds = [
            (kx0 + (a + 0.5) * (kx1 - kx0) / n,
             ky0 + (b + 0.5) * (ky1 - ky0) / n)
            for b in range(n) for a in range(n)
        ]
        # Which crop edges are INNER tile edges (not the real image
        # border): a mask touching one of those is left for the
        # neighbour tile.
        inner = (
            cx0 > 0, cy0 > 0,
            cx1 < smap.W, cy1 < smap.H,
        )

        idx = 0
        while idx < len(seeds):
            if should_stop():
                return added
            pts = []
            while idx < len(seeds) and len(pts) < batch:
                sx, sy = seeds[idx]
                idx += 1
                done += 1
                # Skip prompts already inside a committed stone.
                if not smap.labels[int(sy), int(sx)]:
                    pts.append((sx, sy))
            if not pts:
                continue
            local = np.array([(x - cx0, y - cy0) for x, y in pts])
            with torch.inference_mode():
                coords = torch.as_tensor(
                    predictor.transform.apply_coords(
                        local, predictor.original_size),
                    dtype=torch.float, device=device,
                )[:, None, :]
                labels = torch.ones(
                    (len(pts), 1), dtype=torch.int, device=device)
                logits, iou, _ = predictor.predict_torch(
                    coords, labels,
                    multimask_output=True, return_logits=True,
                )
                pos = logits > 0
                areas = pos.flatten(2).sum(-1)
                stab = ((logits > 1).flatten(2).sum(-1)
                        / (logits > -1).flatten(2).sum(-1).clamp(min=1))
                edges = torch.stack([
                    pos[..., :, 0].any(-1),
                    pos[..., 0, :].any(-1),
                    pos[..., :, -1].any(-1),
                    pos[..., -1, :].any(-1),
                ], -1)

            areas = areas.cpu().numpy()
            stab = stab.cpu().numpy()
            iou_np = iou.cpu().numpy()
            edges = edges.cpu().numpy()

            for i, (sx, sy) in enumerate(pts):
                if smap.labels[int(sy), int(sx)]:
                    continue   # covered by a stone found earlier in this batch
                lx, ly = int(sx - cx0), int(sy - cy0)
                best = None
                for k in range(iou_np.shape[1]):
                    if iou_np[i, k] < pred_iou_thresh:
                        continue
                    if stab[i, k] < stability_thresh:
                        continue
                    if not (min_area <= areas[i, k] <= max_area):
                        continue
                    if any(e and inner[j]
                           for j, e in enumerate(edges[i, k])):
                        continue
                    if not pos[i, k, ly, lx]:
                        continue    # mask must contain its prompt
                    if best is None or iou_np[i, k] > iou_np[i, best]:
                        best = k
                if best is None:
                    continue
                m = pos[i, best].cpu().numpy()
                ys, xs = np.nonzero(m)
                x0 = int(xs.min()); y0 = int(ys.min())
                x1 = int(xs.max()) + 1; y1 = int(ys.max()) + 1
                if smap.commit(m[y0:y1, x0:x1],
                               int(cx0 + x0), int(cy0 + y0),
                               max_overlap=max_overlap,
                               min_area=int(min_area)):
                    added += 1
            if progress_cb:
                progress_cb(
                    f"Tile {ti + 1}/{len(tile_list)}: "
                    f"{added} stones found",
                    done / total,
                )
    return added


# ---------------------------------------------------------------------------
# Palette for raw-mask visualisation (same hash-step palette the
# vectorizer uses — stable across runs).
# ---------------------------------------------------------------------------

def _make_palette(n: int = 4096) -> np.ndarray:
    pal = np.zeros((n, 3), np.uint8)
    for i in range(n):
        h = (i * 137.508) % 360 / 60
        x = 1 - abs(h % 2 - 1)
        r, g, b = [(1, x, 0), (x, 1, 0), (0, 1, x),
                   (0, x, 1), (x, 0, 1), (1, 0, x)][int(h) % 6]
        pal[i] = [int(40 + 200 * v) for v in (r, g, b)]
    return pal


_PALETTE = _make_palette()


def render_mask_overlay(smap: StoneMap) -> np.ndarray:
    """Source image blended with a translucent per-stone palette fill
    plus a stronger-alpha border at mask edges."""
    H, W = smap.H, smap.W
    lab = smap.labels
    edge = np.zeros(lab.shape, bool)
    edge[:, 1:]  |= lab[:, 1:]  != lab[:, :-1]
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    edge[1:]     |= lab[1:]     != lab[:-1]
    edge[:-1]    |= lab[:-1]    != lab[1:]
    overlay = np.zeros((H, W, 3), np.uint8)
    overlay[:] = _PALETTE[lab % len(_PALETTE)]
    overlay[lab == 0] = 0
    # Compose: 55 % source, 45 % overlay; borders bias toward overlay.
    base = smap.image.astype(np.float32)
    ov = overlay.astype(np.float32)
    out = (base * 0.55 + ov * 0.45)
    out[edge & (lab != 0)] = ov[edge & (lab != 0)]
    return np.clip(out, 0, 255).astype(np.uint8)


def render_polygon_fills(polygons: list, w: int, h: int) -> np.ndarray:
    """One mean-coloured fill per stone + 1 px black outline on a WHITE
    canvas — matches mosaic-vectorizer's 'Vector' view and shows the
    polygons on their own, with no background image bleeding through.
    This is what the saved CSV encodes."""
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for _sid, color, pts in polygons:
        poly_int = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [poly_int], color=tuple(int(c) for c in color))
        cv2.polylines(canvas, [poly_int], isClosed=True,
                      color=(0, 0, 0), thickness=1)
    return canvas


def render_overlay_on_source(image: np.ndarray,
                             polygons: list) -> np.ndarray:
    """Source image with each polygon's outline drawn in yellow and a
    very light translucent fill on top. Good for sanity-checking
    detection against the real mosaic."""
    out = image.copy()
    for _sid, color, pts in polygons:
        poly_int = pts.astype(np.int32).reshape(-1, 1, 2)
        over = out.copy()
        cv2.fillPoly(over, [poly_int], color=tuple(int(c) for c in color))
        out = cv2.addWeighted(over, 0.25, out, 0.75, 0)
        cv2.polylines(out, [poly_int], isClosed=True,
                      color=(255, 255, 0), thickness=1)
    return out


# ---------------------------------------------------------------------------
# CSV output — same schema as mosaic_to_csv.py.
# ---------------------------------------------------------------------------

def write_csv(polygons: list, path: Path) -> None:
    """`polygons` = [(sid, color_tuple_0_255, Nx2 points in ORIGINAL
    pixels)] — colors in 0..255 integer tuples (StoneMap convention)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["coordinates", "color_r", "color_g",
                    "color_b", "color_a", "color_hex"])
        for _sid, color, pts in polygons:
            r, g, b = (float(color[0]) / 255.0,
                       float(color[1]) / 255.0,
                       float(color[2]) / 255.0)
            coords = "[" + ", ".join(
                f"({x}, {y})" for x, y in pts) + "]"
            hex_str = "#{:02x}{:02x}{:02x}".format(
                int(color[0]), int(color[1]), int(color[2]))
            w.writerow([coords, r, g, b, 1.0, hex_str])


def extract_polygons(smap: StoneMap, scale: float,
                     tolerance: float, smoothing: int) -> list:
    """Walk the StoneMap, build each stone's polygon, scale back to
    ORIGINAL image pixels. Returns [(sid, (r,g,b), pts)]."""
    out = []
    for sid, s in list(smap.stones.items()):
        pts = stone_polygon(smap, sid, tolerance, smoothing)
        if pts is None:
            continue
        out.append((sid, s['color'], pts * scale))
    return out


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class SamWorker(QThread):
    stage_image = pyqtSignal(str, object)         # (name, HxWx3 RGB)
    progress    = pyqtSignal(str, float)          # (message, 0..1)
    finished_ok = pyqtSignal(object, object)      # (polygons, stages)
    failed      = pyqtSignal(str)

    def __init__(self, source_rgb: np.ndarray,
                 checkpoint_path: str, variant: str, device: str,
                 params: dict):
        super().__init__()
        self.source_rgb = source_rgb
        self.checkpoint_path = checkpoint_path
        self.variant = variant
        self.device = device
        self.params = params
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def _should_stop(self) -> bool:
        return self._stop

    def _p(self, msg: str, frac: float = -1.0) -> None:
        self.progress.emit(msg, frac if frac >= 0.0 else 0.0)

    def run(self) -> None:
        try:
            p = self.params
            # Resize to the user-selected process size (long edge).
            self._p(f"Preparing image "
                    f"(process size={p['process_size']})...")
            src_h, src_w = self.source_rgb.shape[:2]
            long_edge = max(src_w, src_h)
            if p['process_size'] <= 0 or p['process_size'] >= long_edge:
                work = self.source_rgb
                scale = 1.0
            else:
                k = p['process_size'] / long_edge
                tw = max(1, int(round(src_w * k)))
                th = max(1, int(round(src_h * k)))
                pil = Image.fromarray(self.source_rgb)
                work = np.asarray(
                    pil.resize((tw, th), Image.LANCZOS)).copy()
                scale = src_w / tw

            smap = StoneMap(work)

            self._p(f"Loading SAM checkpoint ({self.variant}, "
                    f"device={self.device})...")
            try:
                _sam, predictor = load_sam(
                    self.checkpoint_path, self.variant, self.device)
            except Exception as e:
                self.failed.emit(
                    "Failed to load the SAM checkpoint. Common causes:\n"
                    " • Wrong variant for this filename.\n"
                    " • Corrupted .pth file (re-download).\n"
                    " • torch wasn't built with CUDA but you picked "
                    "the CUDA device.\n\n"
                    f"{type(e).__name__}: {e}")
                return

            self._p(f"Detecting stones on {smap.W}×{smap.H} work image "
                    "(tiled)...")
            added = auto_segment(
                predictor, self.device, smap,
                points_per_side=int(p['points_per_side']),
                tiles=int(p['tiles']),
                pred_iou_thresh=float(p['pred_iou_thresh']),
                stability_thresh=float(p['stability_thresh']),
                min_area=int(round(p['min_area_src'] / scale ** 2)),
                max_area_frac=float(p['max_area_frac']),
                max_overlap=float(p['max_overlap']),
                batch=16,
                progress_cb=lambda m, f: self.progress.emit(m, f),
                should_stop=self._should_stop,
            )
            if self._stop:
                self._p("Stopped by user — rendering what we have...")
            self._p(f"{added} stones found. Building polygons...", 1.0)

            # Stages.
            stages: dict[str, np.ndarray] = {}
            overlay = render_mask_overlay(smap)
            stages['1. SAM mask overlay (work resolution)'] = overlay
            self.stage_image.emit(
                '1. SAM mask overlay (work resolution)', overlay)

            polygons = extract_polygons(
                smap, scale,
                tolerance=float(p['tolerance']),
                smoothing=int(p['smoothing']),
            )
            final = render_polygon_fills(polygons, src_w, src_h)
            stages['2. Final polygons (mean colour)'] = final
            self.stage_image.emit(
                '2. Final polygons (mean colour)', final)

            final_overlay = render_overlay_on_source(
                self.source_rgb, polygons)
            stages['3. Overlay on original image'] = final_overlay
            self.stage_image.emit(
                '3. Overlay on original image', final_overlay)

            self._p(f"Done — {len(polygons)} polygons.", 1.0)
            self.finished_ok.emit(polygons, stages)
        except Exception as e:
            self.failed.emit(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Image pane — image_strech-style zoom (1.1× per notch, cursor-anchored).
# ---------------------------------------------------------------------------

class ImagePane(QFrame):
    ZOOM_STEP = 1.1
    ZOOM_MIN  = 0.05
    ZOOM_MAX  = 20.0

    def __init__(self, title: str, placeholder: str):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.placeholder_text = placeholder
        self._base_qimg: QImage | None = None
        self._info_text = "—"
        self.zoom = 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight:bold; padding:2px;")
        layout.addWidget(self.title_label)

        self.image_label = QLabel(placeholder)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background:#222; color:#bbb;")
        self.image_label.setMinimumSize(300, 300)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.image_label)
        self.scroll.setWidgetResizable(True)
        self.scroll.viewport().installEventFilter(self)
        layout.addWidget(self.scroll, 1)

        self.info_label = QLabel("—")
        self.info_label.setStyleSheet("color:#666;")
        layout.addWidget(self.info_label)

    def set_rgb(self, rgb: np.ndarray, info: str = "") -> None:
        h, w = rgb.shape[:2]
        buf = np.ascontiguousarray(rgb).data
        self._base_qimg = QImage(
            buf, w, h, w * 3, QImage.Format_RGB888,
        ).copy()
        self._info_text = info or f"{w} × {h} px"
        self.zoom = 1.0
        self._apply_zoom()

    def clear(self) -> None:
        self._base_qimg = None
        self.zoom = 1.0
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(self.placeholder_text)
        self.info_label.setText("—")

    def _apply_zoom(self) -> None:
        if self._base_qimg is None:
            return
        nw = max(1, int(round(self._base_qimg.width()  * self.zoom)))
        nh = max(1, int(round(self._base_qimg.height() * self.zoom)))
        tf = (Qt.SmoothTransformation
              if self.zoom < 2.0 else Qt.FastTransformation)
        scaled = self._base_qimg.scaled(nw, nh, Qt.KeepAspectRatio, tf)
        self.image_label.setPixmap(QPixmap.fromImage(scaled))
        self.image_label.resize(scaled.size())
        self.info_label.setText(
            f"{self._info_text}  |  zoom {int(self.zoom * 100)}%")

    def eventFilter(self, obj, event):
        if obj is self.scroll.viewport() and event.type() == QEvent.Wheel:
            if self._base_qimg is None:
                return True
            self._wheel_zoom(event)
            return True
        return super().eventFilter(obj, event)

    def _wheel_zoom(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        old = self.zoom
        new = old * (self.ZOOM_STEP if delta > 0
                     else 1.0 / self.ZOOM_STEP)
        new = max(self.ZOOM_MIN, min(self.ZOOM_MAX, new))
        if abs(new - old) < 1e-9:
            return
        pos = event.position() if hasattr(event, "position") else event.posF()
        vx, vy = pos.x(), pos.y()
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        old_sx = hbar.value()
        old_sy = vbar.value()
        self.zoom = new
        self._apply_zoom()
        ratio = new / old
        hbar.setValue(max(hbar.minimum(),
                          min(hbar.maximum(),
                              int(round((vx + old_sx) * ratio - vx)))))
        vbar.setValue(max(vbar.minimum(),
                          min(vbar.maximum(),
                              int(round((vy + old_sy) * ratio - vy)))))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

PROCESS_SIZES = [1024, 1536, 2048, 3072, 0]      # 0 = Full


class SamWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SAM → CSV Polygons")
        self.resize(1700, 1000)

        self.source_path: Path | None = None
        self.source_rgb:  np.ndarray | None = None
        self.checkpoint_path: str | None = read_memo_path()
        self.worker: SamWorker | None = None
        self.polygons: list = []
        self.stages: dict[str, np.ndarray] = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Toolbar
        bar = QHBoxLayout()
        self.load_btn       = QPushButton("Load Image...")
        self.checkpoint_btn = QPushButton("Choose Checkpoint...")
        self.detect_btn     = QPushButton("Detect")
        self.save_btn       = QPushButton("Save CSV...")
        self.cancel_btn     = QPushButton("Cancel")
        for b in (self.load_btn, self.checkpoint_btn,
                  self.detect_btn, self.save_btn, self.cancel_btn):
            bar.addWidget(b)
        bar.addStretch(1)
        self.tile_count_label = QLabel("Stones: —")
        bar.addWidget(self.tile_count_label)
        root.addLayout(bar)

        # Model + checkpoint info line
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(CHECKPOINTS.keys()))
        self.model_combo.setCurrentText('vit_b')
        self.model_combo.setToolTip(
            "vit_b (375 MB) = fastest; vit_l (1.2 GB) = balanced; "
            "vit_h (2.5 GB) = best quality. Variant is auto-detected "
            "when you point at a specific .pth file.")
        model_row.addWidget(self.model_combo)
        model_row.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CUDA", "CPU"])
        self.device_combo.setToolTip(
            "Auto picks CUDA when torch sees your GPU, else CPU.")
        model_row.addWidget(self.device_combo)
        self.checkpoint_label = QLabel("—")
        self.checkpoint_label.setStyleSheet("color:#444; padding:2px;")
        model_row.addWidget(self.checkpoint_label, 1)
        root.addLayout(model_row)

        # Parameters row 1 — process size + simplification knobs.
        row1 = QHBoxLayout()
        self.process_combo = QComboBox()
        for s in PROCESS_SIZES:
            self.process_combo.addItem("Full" if s == 0 else str(s), s)
        self.process_combo.setCurrentIndex(2)   # 2048
        self.process_combo.setToolTip(
            "Resize the long edge to this many pixels before running "
            "SAM (polygons are mapped back to the ORIGINAL image "
            "coordinates in the saved CSV). 2048 is a good balance; "
            "Full is slow and uses more VRAM.")
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 100_000_000)
        self.min_area_spin.setValue(300)
        self.min_area_spin.setSuffix(" px²")
        self.min_area_spin.setToolTip(
            "Minimum polygon area in ORIGINAL image pixels (converted "
            "to work-image pixels internally).")
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.0, 10.0)
        self.tolerance_spin.setDecimals(2)
        self.tolerance_spin.setSingleStep(0.25)
        self.tolerance_spin.setValue(1.0)
        self.tolerance_spin.setToolTip(
            "Douglas-Peucker tolerance in work-image pixels. Higher = "
            "fewer vertices per polygon. 1 px is a good start.")
        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(0, 6)
        self.smoothing_spin.setValue(0)
        self.smoothing_spin.setToolTip(
            "Chaikin corner-smoothing iterations. 0 = no smoothing. "
            "Each iteration roughly doubles vertex count.")
        for label, widget in (
            ("Process size:",   self.process_combo),
            ("Min polygon:",    self.min_area_spin),
            ("Simplify ε:",     self.tolerance_spin),
            ("Corner smooth:",  self.smoothing_spin),
        ):
            row1.addWidget(QLabel(label))
            row1.addWidget(widget)
        row1.addStretch(1)
        root.addLayout(row1)

        # Parameters row 2 — SAM / tiling knobs.
        row2 = QHBoxLayout()
        self.tiles_spin = QSpinBox()
        self.tiles_spin.setRange(1, 5)
        self.tiles_spin.setValue(2)
        self.tiles_spin.setToolTip(
            "n × n tiles. More = SAM sees finer detail. 1 = single "
            "pass (fast but misses small stones); 2 is the default "
            "that works well for most mosaics.")
        self.points_per_side_spin = QSpinBox()
        self.points_per_side_spin.setRange(8, 96)
        self.points_per_side_spin.setSingleStep(4)
        self.points_per_side_spin.setValue(32)
        self.points_per_side_spin.setToolTip(
            "Prompt grid density per tile. Higher = more stones found "
            "and more time.")
        self.pred_iou_spin = QDoubleSpinBox()
        self.pred_iou_spin.setRange(0.5, 0.99)
        self.pred_iou_spin.setDecimals(2)
        self.pred_iou_spin.setSingleStep(0.01)
        self.pred_iou_spin.setValue(0.80)
        self.pred_iou_spin.setToolTip(
            "Minimum SAM-predicted IoU. Lower = more candidate masks "
            "pass through.")
        self.stability_spin = QDoubleSpinBox()
        self.stability_spin.setRange(0.5, 0.99)
        self.stability_spin.setDecimals(2)
        self.stability_spin.setSingleStep(0.01)
        self.stability_spin.setValue(0.85)
        self.stability_spin.setToolTip(
            "Manual stability threshold (logit > 1 area / logit > -1 "
            "area). Mosaic-vectorizer's chosen default.")
        self.max_area_spin = QDoubleSpinBox()
        self.max_area_spin.setRange(0.1, 100.0)
        self.max_area_spin.setDecimals(1)
        self.max_area_spin.setValue(3.0)
        self.max_area_spin.setSuffix(" %")
        self.max_area_spin.setToolTip(
            "Reject masks larger than this fraction of the image — "
            "catches SAM picking up a whole floor tile instead of a "
            "stone.")
        self.max_overlap_spin = QDoubleSpinBox()
        self.max_overlap_spin.setRange(0.0, 1.0)
        self.max_overlap_spin.setDecimals(2)
        self.max_overlap_spin.setSingleStep(0.05)
        self.max_overlap_spin.setValue(0.3)
        self.max_overlap_spin.setToolTip(
            "Reject a new mask when it overlaps already-committed "
            "stones by more than this fraction of its own area.")

        for label, widget in (
            ("Tiles:",         self.tiles_spin),
            ("Points/side:",   self.points_per_side_spin),
            ("IoU ≥",          self.pred_iou_spin),
            ("Stability ≥",    self.stability_spin),
            ("Max area:",      self.max_area_spin),
            ("Max overlap:",   self.max_overlap_spin),
        ):
            row2.addWidget(QLabel(label))
            row2.addWidget(widget)
        row2.addStretch(1)
        root.addLayout(row2)

        # Stage dropdown + progress bar.
        stage_row = QHBoxLayout()
        stage_row.addWidget(QLabel("Show stage:"))
        self.stage_combo = QComboBox()
        self.stage_combo.setMinimumWidth(320)
        stage_row.addWidget(self.stage_combo)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        stage_row.addWidget(self.progress_bar, 1)
        root.addLayout(stage_row)

        # Panes.
        splitter = QSplitter(Qt.Horizontal)
        self.source_pane = ImagePane(
            "Source mosaic image",
            "Load a mosaic image to begin.")
        self.result_pane = ImagePane(
            "SAM output stages",
            "Click Detect once you've loaded an image.")
        splitter.addWidget(self.source_pane)
        splitter.addWidget(self.result_pane)
        splitter.setSizes([850, 850])
        root.addWidget(splitter, 1)

        self.statusBar().showMessage("Ready.")

        self.load_btn.clicked.connect(self.load_image)
        self.checkpoint_btn.clicked.connect(self.choose_checkpoint)
        self.detect_btn.clicked.connect(self.detect)
        self.save_btn.clicked.connect(self.save_csv)
        self.cancel_btn.clicked.connect(self.cancel_detect)
        self.stage_combo.currentIndexChanged.connect(self._on_stage_changed)
        self._refresh_checkpoint_label()
        self._update_buttons()

    # ----- buttons / state -------------------------------------------------

    def _update_buttons(self) -> None:
        running = self.worker is not None and self.worker.isRunning()
        self.detect_btn.setEnabled(
            self.source_rgb is not None and not running)
        self.save_btn.setEnabled(bool(self.polygons) and not running)
        self.load_btn.setEnabled(not running)
        self.checkpoint_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.model_combo.setEnabled(not running)
        self.device_combo.setEnabled(not running)

    def _refresh_checkpoint_label(self) -> None:
        if self.checkpoint_path:
            v = detect_variant(self.checkpoint_path)
            self.checkpoint_label.setText(
                f"Manual checkpoint: {self.checkpoint_path}  "
                f"[auto variant: {v}]")
            # Sync the model combo so the UI reflects what Detect will
            # actually use.
            if self.model_combo.findText(v) >= 0:
                self.model_combo.setCurrentText(v)
        else:
            self.checkpoint_label.setText(
                "No manual checkpoint set — will auto-download "
                "'{}' from Meta on first Detect ({})".format(
                    self.model_combo.currentText(),
                    CHECKPOINTS[self.model_combo.currentText()][2]))

    def choose_checkpoint(self) -> None:
        start = (str(Path(self.checkpoint_path).parent)
                 if self.checkpoint_path else str(ROOT))
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose SAM checkpoint", start,
            "SAM checkpoint (*.pth);;All files (*.*)",
        )
        if not path:
            return
        self.checkpoint_path = str(Path(path).resolve())
        write_memo_path(self.checkpoint_path)
        self._refresh_checkpoint_label()
        self.statusBar().showMessage(
            f"Checkpoint set: {Path(path).name}")
        self._update_buttons()

    def load_image(self) -> None:
        start_dir = str(INPUT_DIR) if INPUT_DIR.exists() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load mosaic image", start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;"
            "All files (*.*)",
        )
        if not path:
            return
        try:
            pil = Image.open(path); pil.load()
            rgb = np.array(pil.convert("RGB"))
        except Exception as e:
            QMessageBox.critical(
                self, "Load failed", f"{type(e).__name__}: {e}")
            return
        self.source_rgb = rgb
        self.source_path = Path(path)
        self.polygons = []
        self.stages = {}
        self.stage_combo.clear()
        self.result_pane.clear()
        self.tile_count_label.setText("Stones: —")
        self.progress_bar.setValue(0)
        h, w = rgb.shape[:2]
        self.source_pane.set_rgb(
            rgb, f"{Path(path).name}  |  {w} × {h} px")
        self.statusBar().showMessage(f"Loaded {Path(path).name}")
        self._update_buttons()

    # ----- detect / cancel -------------------------------------------------

    def _resolve_device(self) -> str:
        choice = self.device_combo.currentText()
        if choice == "CUDA":
            return "cuda"
        if choice == "CPU":
            return "cpu"
        return pick_device()

    def _resolve_checkpoint(self) -> tuple[str, str] | None:
        """Pick (checkpoint_path, variant). If the user set a manual
        path, use it + inferred variant. Otherwise auto-download the
        combo-selected variant. Returns None on download failure."""
        variant = self.model_combo.currentText()
        if self.checkpoint_path:
            return self.checkpoint_path, detect_variant(self.checkpoint_path)
        # Auto-download — do it synchronously with status bar updates.
        try:
            def _p(frac: float) -> None:
                pct = int(frac * 100)
                self.statusBar().showMessage(
                    f"Downloading {variant} checkpoint... {pct}%")
                self.progress_bar.setValue(pct)
                QApplication.processEvents()
            path = ensure_checkpoint(variant, _p)
            return path, variant
        except Exception as e:
            QMessageBox.critical(
                self, "Checkpoint download failed",
                f"{type(e).__name__}: {e}")
            return None

    def detect(self) -> None:
        if self.source_rgb is None:
            return
        try:
            import segment_anything  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self, "segment-anything not installed",
                "sam.py needs Meta's segment-anything package.\n\n"
                "Install with:\n"
                "    pip install segment-anything torch torchvision")
            return

        resolved = self._resolve_checkpoint()
        if resolved is None:
            return
        checkpoint_path, variant = resolved
        device = self._resolve_device()

        params = {
            'process_size':      self.process_combo.currentData(),
            'tiles':             self.tiles_spin.value(),
            'points_per_side':   self.points_per_side_spin.value(),
            'pred_iou_thresh':   self.pred_iou_spin.value(),
            'stability_thresh':  self.stability_spin.value(),
            'min_area_src':      self.min_area_spin.value(),
            'max_area_frac':     self.max_area_spin.value() / 100.0,
            'max_overlap':       self.max_overlap_spin.value(),
            'tolerance':         self.tolerance_spin.value(),
            'smoothing':         self.smoothing_spin.value(),
        }

        self.polygons = []
        self.stages = {}
        self.stage_combo.clear()
        self.progress_bar.setValue(0)
        self.worker = SamWorker(
            self.source_rgb, checkpoint_path, variant, device, params)
        self.worker.progress.connect(self._on_progress)
        self.worker.stage_image.connect(self._on_stage_ready)
        self.worker.finished_ok.connect(self._on_detect_ok)
        self.worker.failed.connect(self._on_detect_failed)
        self.worker.finished.connect(self._on_worker_done)
        self.detect_btn.setText("Detecting...")
        self.worker.start()
        self._update_buttons()

    def cancel_detect(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        # Cooperative cancel: the engine checks a flag per-batch and
        # per-tile. If it ignores us for too long we fall back to
        # terminate(). Cached model is preserved.
        self.worker.request_stop()
        if not self.worker.wait(5000):
            self.worker.terminate()
            self.worker.wait(2000)
        self.statusBar().showMessage("Detection cancelled.")

    def _on_progress(self, msg: str, frac: float) -> None:
        self.statusBar().showMessage(msg)
        if frac > 0:
            self.progress_bar.setValue(int(frac * 100))

    def _on_stage_ready(self, name: str, img: np.ndarray) -> None:
        self.stages[name] = img
        if self.stage_combo.findText(name) < 0:
            self.stage_combo.addItem(name)
        self.stage_combo.setCurrentText(name)

    def _on_detect_ok(self, polygons, stages) -> None:
        self.polygons = polygons
        self.stages = stages
        self.tile_count_label.setText(f"Stones: {len(polygons)}")
        # Default view when detection ends: the clean polygon fills on
        # white background — no source image behind, so the user sees
        # exactly what the CSV encodes. They can switch to the mask or
        # overlay stages from the dropdown.
        preferred = '2. Final polygons (mean colour)'
        if preferred in self.stages:
            self.stage_combo.setCurrentText(preferred)

    def _on_detect_failed(self, msg: str) -> None:
        QMessageBox.critical(self, "Detect failed", msg)
        self.statusBar().showMessage("Detect failed.")

    def _on_worker_done(self) -> None:
        self.worker = None
        self.detect_btn.setText("Detect")
        self._update_buttons()

    def _on_stage_changed(self, _idx: int) -> None:
        name = self.stage_combo.currentText()
        if not name or name not in self.stages:
            return
        img = self.stages[name]
        h, w = img.shape[:2]
        self.result_pane.set_rgb(img, f"{name}  |  {w} × {h} px")

    # ----- CSV save --------------------------------------------------------

    def save_csv(self) -> None:
        if not self.polygons:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        default_name = (
            (self.source_path.stem if self.source_path else "mosaic")
            + f"_sam_{len(self.polygons)}polys.csv"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save polygons CSV",
            str(OUTPUT_DIR / default_name),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        out = Path(path)
        if out.suffix.lower() != ".csv":
            out = out.with_suffix(".csv")
        try:
            write_csv(self.polygons, out)
        except Exception as e:
            QMessageBox.critical(
                self, "Save failed", f"{type(e).__name__}: {e}")
            return
        self.statusBar().showMessage(
            f"Saved {len(self.polygons)} polygons → {out.name}")
        QMessageBox.information(
            self, "Saved",
            f"Saved {len(self.polygons)} polygons to:\n{out}")


# ---------------------------------------------------------------------------

def main() -> int:
    app = QApplication(sys.argv)
    win = SamWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
