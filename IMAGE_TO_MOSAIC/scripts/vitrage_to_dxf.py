"""Vitrage → DXF — extract puzzle cut lines from a stained-glass image.

The lines between glass pieces are the CUTS that separate them. For a puzzle
that tessellates (no waste between pieces), the DXF must contain ONE LINE per
shared edge, not one closed polygon per piece. This script extracts the
LEAD-CAME SKELETON (1-pixel-wide centerline of the dark lines in the image)
and writes each strand to DXF as an open polyline.

Algorithm:
  1. Adaptive threshold on the source → tile_mask (white=glass, black=lead).
  2. skimage.morphology.skeletonize on the inverted mask → 1-px-wide
     centerline network.
  3. Walk the skeleton (junctions broken out into chains) → polylines.
  4. Douglas-Peucker simplification per polyline.
  5. Write each polyline + an outer image-border rectangle to DXF.

Builds on mosaic_to_csv.MosaicToCsv for UI / image loading / API key / solid-
white-transform machinery; overrides the detection and save paths.

Usage:
  python IMAGE_TO_MOSAIC/scripts/vitrage_to_dxf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from skimage.morphology import skeletonize

from mosaic_to_csv import MosaicToCsv, OUTPUT_DIR, ai_bytes_to_binary_mask


# ---------------------------------------------------------------------------
# Detection pipeline (skeleton-based)
# ---------------------------------------------------------------------------

def detect_lead_came_polylines(
    detection_rgb: np.ndarray,
    block_size: int,
    C: int,
    simplify_eps_px: float,
) -> list[np.ndarray]:
    """Extract puzzle cut lines from a stained-glass image.

    Returns a list of polylines, each an (N, 2) float array of (x, y) pixel
    coordinates. Adjacent pieces share their cut line, so the network is
    deduplicated by construction.
    """
    gray = cv2.cvtColor(detection_rgb, cv2.COLOR_RGB2GRAY)
    bs = max(3, int(block_size))
    if bs % 2 == 0:
        bs += 1
    tile_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        bs, int(C),
    )
    # Lead came (dark pixels) inverted → True where the cut line should run.
    lead_binary = (tile_mask == 0)
    skel = skeletonize(lead_binary).astype(np.uint8)
    return _skeleton_to_polylines(skel, simplify_eps_px)


def _skeleton_to_polylines(sk: np.ndarray, simplify_eps: float) -> list[np.ndarray]:
    """Decompose a 1-pixel-wide skeleton into individual polylines.

    Junctions (3+ neighbours) split the skeleton into chains; each chain is
    walked pixel-by-pixel from one endpoint to the other, then bridged back
    to its adjacent junctions so the network stays connected.
    """
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

        # Endpoint: pixel with only one in-chain neighbour
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

        # Stitch back to adjacent junctions so the network closes up
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


# ---------------------------------------------------------------------------
# Preview + DXF writers
# ---------------------------------------------------------------------------

def render_polylines(polylines, w: int, h: int, line_width: int = 1) -> Image.Image:
    """Black 1-px polylines on white — preview that mirrors what's in the DXF."""
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for pl in polylines:
        if len(pl) < 2:
            continue
        pts_int = pl.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            canvas, [pts_int], isClosed=False,
            color=(0, 0, 0), thickness=line_width,
        )
    # Outer frame in the preview, matching the DXF's outer rectangle.
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color=(0, 0, 0), thickness=1)
    return Image.fromarray(canvas, mode="RGB")


def write_dxf_polylines(polylines, out_path: Path,
                        image_w: int, image_h: int) -> None:
    """Each polyline → open LWPOLYLINE. Plus a closed outer rectangle.

    Y-axis is flipped (image y grows down, DXF y grows up) so the drawing is
    right-side-up in CAD viewers. Layer 0, color 7."""
    import ezdxf
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for pl in polylines:
        if len(pl) < 2:
            continue
        coords = [(float(x), float(image_h - y)) for x, y in pl]
        msp.add_lwpolyline(coords, close=False, dxfattribs={"color": 7})
    frame = [
        (0.0, 0.0),
        (float(image_w), 0.0),
        (float(image_w), float(image_h)),
        (0.0, float(image_h)),
    ]
    msp.add_lwpolyline(frame, close=True, dxfattribs={"color": 7})
    doc.saveas(str(out_path))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class VitrageToDxf(MosaicToCsv):
    """Stained-glass image → DXF of puzzle cut lines (one line per shared edge)."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vitrage → DXF (puzzle cut lines)")
        self.source_pane.title_label.setText("Source stained-glass image")
        self.result_pane.title_label.setText("Cut lines (DXF preview)")
        self.tile_count_label.setText("Lines: —")

        # Simplify ε is interpreted in PIXELS here (parent uses a ratio of
        # perimeter). Override the spinner range and default.
        self.eps_spin.setRange(0.0, 10.0)
        self.eps_spin.setDecimals(2)
        self.eps_spin.setSingleStep(0.1)
        self.eps_spin.setValue(1.0)
        # min_area is not used by the skeleton pipeline; keep visible but quiet.
        self.min_area_spin.setRange(0, 1_000_000)
        self.min_area_spin.setValue(0)

        # Repurpose the save button.
        self.save_btn.setText("Save DXF...")
        try:
            self.save_btn.clicked.disconnect()
        except TypeError:
            pass
        self.save_btn.clicked.connect(self.save_dxf)

    # ----- detection ------------------------------------------------------

    def detect(self) -> None:
        if self.source_rgb is None:
            return
        if not self.solid_white_chk.isChecked():
            self._run_skeleton_detect(self.source_rgb)
            return
        if self.solid_white_rgb is not None:
            self._run_skeleton_detect(self.solid_white_rgb)
            return
        # Cache miss — delegate to the parent's API worker dispatch. When
        # the worker finishes it'll call _on_solid_white_ready (overridden
        # below) which runs OUR skeleton pipeline.
        super().detect()

    def _on_solid_white_ready(self, png_bytes: bytes) -> None:
        if self.source_rgb is None:
            return
        h, w = self.source_rgb.shape[:2]
        try:
            self.solid_white_rgb = ai_bytes_to_binary_mask(png_bytes, w, h)
        except Exception as e:
            QMessageBox.critical(self, "Decode failed", f"{type(e).__name__}: {e}")
            return
        self.statusBar().showMessage(
            "AI transform ready — running skeleton pipeline...",
        )
        QApplication.processEvents()
        self._run_skeleton_detect(self.solid_white_rgb)

    def _run_skeleton_detect(self, detection_rgb: np.ndarray) -> None:
        self.statusBar().showMessage("Skeletonizing lead came...")
        QApplication.processEvents()
        try:
            polylines = detect_lead_came_polylines(
                detection_rgb,
                block_size=self.block_size_spin.value(),
                C=self.C_spin.value(),
                simplify_eps_px=self.eps_spin.value(),
            )
        except Exception as e:
            QMessageBox.critical(self, "Detect failed", f"{type(e).__name__}: {e}")
            self.statusBar().showMessage("Detect failed.")
            return

        # Stash as self.tiles so the inherited button-state logic enables Save.
        # Second element of each tuple is unused for skeleton output.
        self.tiles = [(pl, (0.0, 0.0, 0.0)) for pl in polylines]
        h, w = detection_rgb.shape[:2]
        preview = render_polylines(polylines, w, h)
        self.result_pane.set_pil_image(
            preview,
            f"{len(polylines)} cut lines  |  {w} × {h} px  |  DXF preview",
        )
        self.tile_count_label.setText(f"Lines: {len(polylines)}")
        self.statusBar().showMessage(
            f"Skeleton: {len(polylines)} cut lines — ready to save as DXF.",
        )
        self._update_buttons()

    # ----- save -----------------------------------------------------------

    def save_dxf(self) -> None:
        if not self.tiles or self.source_rgb is None:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = self.source_path.stem if self.source_path else "vitrage"
        default_name = f"{stem}_{len(self.tiles)}lines.dxf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save vitrage DXF", str(OUTPUT_DIR / default_name),
            "DXF files (*.dxf);;All files (*.*)",
        )
        if not path:
            return
        out_path = Path(path)
        if out_path.suffix.lower() != ".dxf":
            out_path = out_path.with_suffix(".dxf")
        h, w = self.source_rgb.shape[:2]
        polylines = [pts for pts, _color in self.tiles]
        try:
            write_dxf_polylines(polylines, out_path, image_w=w, image_h=h)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        self.statusBar().showMessage(
            f"Saved {len(polylines)} cut lines + outer frame → {out_path.name}",
        )
        QMessageBox.information(
            self, "Saved",
            f"Saved {len(polylines)} cut lines + outer frame to:\n{out_path}",
        )


def main() -> int:
    app = QApplication(sys.argv)
    win = VitrageToDxf()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
