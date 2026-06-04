"""venus_tile_by_tile.py — convert venus1.png into a polygons CSV by splitting
the 4096x4096 image into a 4x4 grid of 1024x1024 sub-images, running the
attempt-07 adaptive-threshold pipeline independently on each, and merging.

Why split? Each sub-image has more uniform local contrast; adaptive-threshold
parameters that work well on a 1024-px crop produce more consistent tiles than
running on the full 4096 image. Tiles touching SUB-CELL seams are kept;
only tiles touching the FULL-IMAGE outer border are dropped.

Output:
  IMAGE_TO_MOSAIC/output/venus1_4x4_<N>polys.csv
  IMAGE_TO_MOSAIC/output/venus1_4x4_<N>polys.png   (rendered preview)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent   # IMAGE_TO_MOSAIC/ (file lives in scripts/helpers/)
SOURCE = ROOT / "output" / "venus1.png"
OUT_DIR = ROOT / "output"

GRID = 4                # 4 x 4 -> 16 sub-images
MARGIN = 100            # px overlap around each cell — must exceed half the max tile width
BLOCK_SIZE = 51
C = 5
MIN_AREA = 200
EPSILON_RATIO = 0.02


def detect_tiles_subimage(rgb: np.ndarray,
                           drop_left: bool, drop_top: bool,
                           drop_right: bool, drop_bottom: bool):
    """Detect tiles in a single sub-image; drop tiles only on the specified outer sides."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    bs = max(3, int(BLOCK_SIZE))
    if bs % 2 == 0:
        bs += 1
    tile_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        bs, int(C),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tile_mask, connectivity=4)

    tiles = []
    for lid in range(1, num_labels):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            continue
        x0 = stats[lid, cv2.CC_STAT_LEFT]
        y0 = stats[lid, cv2.CC_STAT_TOP]
        ww = stats[lid, cv2.CC_STAT_WIDTH]
        hh = stats[lid, cv2.CC_STAT_HEIGHT]
        # Drop only on EXTERNAL borders (the ones that are also the full-image edge)
        if drop_left   and x0 == 0:          continue
        if drop_top    and y0 == 0:          continue
        if drop_right  and x0 + ww == w:     continue
        if drop_bottom and y0 + hh == h:     continue

        sub_mask = (labels[y0:y0 + hh, x0:x0 + ww] == lid).astype(np.uint8) * 255
        contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = max(0.5, EPSILON_RATIO * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2).astype(np.float64)
        pts[:, 0] += x0
        pts[:, 1] += y0

        mean_rgb = (rgb[labels == lid].mean(axis=0) / 255.0).tolist()
        tiles.append((pts, tuple(mean_rgb)))

    return tiles


def render_polygons(tiles, w: int, h: int) -> Image.Image:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for pts, mean_rgb in tiles:
        color = (int(mean_rgb[0] * 255), int(mean_rgb[1] * 255), int(mean_rgb[2] * 255))
        poly_int = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [poly_int], color=color)
        cv2.polylines(canvas, [poly_int], isClosed=True, color=(0, 0, 0), thickness=1)
    return Image.fromarray(canvas, mode="RGB")


def write_csv(tiles, out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["coordinates", "color_r", "color_g", "color_b", "color_a", "color_hex"])
        for pts, mean_rgb in tiles:
            coords_str = "[" + ", ".join(f"({x}, {y})" for x, y in pts) + "]"
            r, g, b = mean_rgb
            hex_str = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            writer.writerow([coords_str, r, g, b, 1.0, hex_str])


def main() -> int:
    if not SOURCE.is_file():
        print(f"ERROR: source not found: {SOURCE}")
        return 1
    print(f"Loading {SOURCE}...")
    rgb_full = cv2.cvtColor(cv2.imread(str(SOURCE)), cv2.COLOR_BGR2RGB)
    H, W = rgb_full.shape[:2]
    print(f"  {W} x {H}")
    cell_w, cell_h = W // GRID, H // GRID
    print(f"Grid {GRID}x{GRID} -> cells of {cell_w} x {cell_h}")

    all_tiles = []
    per_cell_counts = {}

    for gy in range(GRID):
        for gx in range(GRID):
            # Cell's "owned" region — a tile belongs to this cell iff its centroid is here.
            owned_x0 = gx * cell_w
            owned_y0 = gy * cell_h
            owned_x1 = W if gx == GRID - 1 else (gx + 1) * cell_w
            owned_y1 = H if gy == GRID - 1 else (gy + 1) * cell_h

            # Extended crop — owned region plus MARGIN overlap into neighbors. This lets
            # tiles that straddle a cell seam be detected as one whole polygon by the cell
            # whose owned region contains the centroid (instead of being cut in two).
            crop_x0 = max(0, owned_x0 - MARGIN)
            crop_y0 = max(0, owned_y0 - MARGIN)
            crop_x1 = min(W, owned_x1 + MARGIN)
            crop_y1 = min(H, owned_y1 + MARGIN)
            sub = rgb_full[crop_y0:crop_y1, crop_x0:crop_x1]

            # Drop tiles touching the FULL-IMAGE outer border. In sub-image coords, a
            # sub-image edge is also a full-image edge only if the corresponding crop
            # coordinate equals the image boundary.
            sub_tiles = detect_tiles_subimage(
                sub,
                drop_left   = (crop_x0 == 0),
                drop_top    = (crop_y0 == 0),
                drop_right  = (crop_x1 == W),
                drop_bottom = (crop_y1 == H),
            )

            kept = 0
            for pts, color in sub_tiles:
                # Shift to global image coordinates.
                pts_g = pts.copy()
                pts_g[:, 0] += crop_x0
                pts_g[:, 1] += crop_y0
                # Centroid-based ownership — each tile belongs to exactly one cell.
                cx = float(pts_g[:, 0].mean())
                cy = float(pts_g[:, 1].mean())
                if owned_x0 <= cx < owned_x1 and owned_y0 <= cy < owned_y1:
                    all_tiles.append((pts_g, color))
                    kept += 1
            per_cell_counts[(gy, gx)] = kept
            print(f"  cell ({gy+1},{gx+1})  owned x={owned_x0}-{owned_x1}, "
                  f"y={owned_y0}-{owned_y1}  kept {kept} (margin={MARGIN})")

    print(f"\nTotal tiles: {len(all_tiles)}")
    print("Per-cell tile counts:")
    for gy in range(GRID):
        row = "  " + " | ".join(f"{per_cell_counts[(gy, gx)]:>4}" for gx in range(GRID))
        print(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"venus1_4x4_{len(all_tiles)}polys.csv"
    png_path = OUT_DIR / f"venus1_4x4_{len(all_tiles)}polys.png"
    write_csv(all_tiles, csv_path)
    render_polygons(all_tiles, W, H).save(png_path)
    print(f"\nWrote CSV    -> {csv_path}")
    print(f"Wrote preview -> {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
