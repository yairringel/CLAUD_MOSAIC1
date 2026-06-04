"""Experiment harness — convert a mosaic image into a CSV of polygons, trying
multiple detection strategies and reporting per-attempt metrics. Results land
in output/csv_attempts_<source-stem>/.

Run:
  python IMAGE_TO_MOSAIC/scripts/csv_experiments.py                     # all on T1.png
  python IMAGE_TO_MOSAIC/scripts/csv_experiments.py 03 04               # specific
  python IMAGE_TO_MOSAIC/scripts/csv_experiments.py --source T1_SIMPLE.png 14 19 25
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage import segmentation, color as skcolor
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parent.parent.parent   # IMAGE_TO_MOSAIC/ (file lives in scripts/helpers/)
DEFAULT_SOURCE = ROOT / "output" / "T1.png"
TARGET_TILES = 5000

# These get set per-run inside main() based on the chosen source file.
SOURCE: Path = DEFAULT_SOURCE
ATTEMPTS_DIR: Path = ROOT / "output" / "csv_attempts"
REPORT_PATH: Path = ATTEMPTS_DIR / "report.md"
JSON_REPORT: Path = ATTEMPTS_DIR / "metrics.json"


# ---------------------------------------------------------------------------
# Shared helpers (mirroring mosaic_to_csv.py so this script is self-contained)
# ---------------------------------------------------------------------------

def write_csv(tiles, out_path: Path) -> None:
    """coordinates,color_r,color_g,color_b,color_a,color_hex (project format)."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["coordinates", "color_r", "color_g", "color_b", "color_a", "color_hex"])
        for pts, mean_rgb in tiles:
            coords_str = "[" + ", ".join(f"({x}, {y})" for x, y in pts) + "]"
            r, g, b = mean_rgb
            hex_str = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            writer.writerow([coords_str, r, g, b, 1.0, hex_str])


def render_polygons(tiles, w: int, h: int, grout_color=(0, 0, 0)) -> np.ndarray:
    """Render each polygon filled with its mean color, with 1-px black edges. Returns HxWx3 uint8 RGB."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = grout_color
    for pts, mean_rgb in tiles:
        color = (int(mean_rgb[0] * 255), int(mean_rgb[1] * 255), int(mean_rgb[2] * 255))
        poly_int = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [poly_int], color=color)
        cv2.polylines(canvas, [poly_int], isClosed=True, color=(0, 0, 0), thickness=1)
    return canvas


def contour_from_mask_label(labels, label_id, x0, y0, w, h, epsilon_ratio,
                             max_vertices=None):
    """Extract simplified polygon from one CC label. Returns Nx2 float or None.

    If max_vertices is given and the simplified polygon has more than that many
    points, this function returns None (the caller will skip that label). Real
    mosaic tiles are simple shapes — high-vertex polygons are almost always
    noise blobs or merged tile clusters.
    """
    sub = (labels[y0:y0 + h, x0:x0 + w] == label_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = max(0.5, epsilon_ratio * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    if len(approx) < 3:
        return None
    if max_vertices is not None and len(approx) > max_vertices:
        return None
    pts = approx.reshape(-1, 2).astype(np.float64)
    pts[:, 0] += x0
    pts[:, 1] += y0
    return pts


def tiles_from_labeled_mask(rgb, labels, num_labels, stats, min_area, epsilon_ratio,
                            drop_border=True, max_vertices=None,
                            max_area=None, max_aspect_ratio=None):
    """Given a CC-labeled mask, build (polygon, mean_rgb_0_1) list.

    Filters applied to each component, in order:
      - min_area: skip components below the area floor
      - max_area: skip components above the area ceiling (clearly merged tiles)
      - max_aspect_ratio: skip elongated components (W/H or H/W above this ratio)
      - drop_border: skip components that touch the image edge
      - max_vertices: skip simplified polygons with more vertices than this
    """
    h, w = rgb.shape[:2]
    tiles = []
    for lid in range(1, num_labels):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        x0 = stats[lid, cv2.CC_STAT_LEFT]; y0 = stats[lid, cv2.CC_STAT_TOP]
        ww = stats[lid, cv2.CC_STAT_WIDTH]; hh = stats[lid, cv2.CC_STAT_HEIGHT]
        if max_aspect_ratio is not None and ww > 0 and hh > 0:
            ar = max(ww / hh, hh / ww)
            if ar > max_aspect_ratio:
                continue
        if drop_border and (x0 == 0 or y0 == 0 or x0 + ww == w or y0 + hh == h):
            continue
        pts = contour_from_mask_label(labels, lid, x0, y0, ww, hh, epsilon_ratio,
                                       max_vertices=max_vertices)
        if pts is None:
            continue
        full_mask = labels == lid
        mean_rgb = (rgb[full_mask].mean(axis=0) / 255.0).tolist()
        tiles.append((pts, tuple(mean_rgb)))
    return tiles


def split_large_components(rgb, tile_mask, max_unsplit_area, dist_factor=0.35):
    """Run a per-component watershed split: any CC larger than max_unsplit_area gets
    distance-transform → peaks → watershed inside that component. Smaller components
    are left alone. Returns (labels, num_labels, stats) compatible with tiles_from_labeled_mask.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(tile_mask, connectivity=4)
    h, w = tile_mask.shape
    new_labels = labels.copy()
    new_stats = stats.tolist()
    next_id = n

    for lid in range(1, n):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area <= max_unsplit_area:
            continue
        x0 = stats[lid, cv2.CC_STAT_LEFT]; y0 = stats[lid, cv2.CC_STAT_TOP]
        ww = stats[lid, cv2.CC_STAT_WIDTH]; hh = stats[lid, cv2.CC_STAT_HEIGHT]
        # Work on a tight crop of the component for speed.
        sub_mask = (labels[y0:y0 + hh, x0:x0 + ww] == lid).astype(np.uint8) * 255
        dist = cv2.distanceTransform(sub_mask, cv2.DIST_L2, 3)
        _, peaks = cv2.threshold(dist, dist_factor * dist.max(), 255, 0)
        peaks = peaks.astype(np.uint8)
        n_peaks, peak_labels = cv2.connectedComponents(peaks)
        if n_peaks <= 2:
            continue   # only one peak → not really splittable, leave as is
        markers = peak_labels + 1
        markers[sub_mask == 0] = 0
        bgr_crop = cv2.cvtColor(rgb[y0:y0 + hh, x0:x0 + ww], cv2.COLOR_RGB2BGR)
        ws = cv2.watershed(bgr_crop, markers.astype(np.int32))
        # Reassign labels in the full-image label map.
        new_labels[y0:y0 + hh, x0:x0 + ww][labels[y0:y0 + hh, x0:x0 + ww] == lid] = 0
        new_stats[lid] = [0, 0, 0, 0, 0]
        for sub_id in range(2, n_peaks + 1):
            sub_pix = (ws == sub_id)
            count = int(sub_pix.sum())
            if count == 0:
                continue
            new_labels[y0:y0 + hh, x0:x0 + ww][sub_pix] = next_id
            ys, xs = np.where(sub_pix)
            new_stats.append([int(xs.min() + x0), int(ys.min() + y0),
                              int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1),
                              count])
            next_id += 1

    return new_labels, next_id, np.array(new_stats, dtype=np.int32)


# ---------------------------------------------------------------------------
# Approaches
# ---------------------------------------------------------------------------

def approach_classical(rgb, blur_sigma=1.0, threshold=80, min_area=200, epsilon_ratio=0.02,
                       invert=False):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if blur_sigma > 0:
        k = max(3, int(blur_sigma * 4) | 1)
        gray = cv2.GaussianBlur(gray, (k, k), blur_sigma)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, tile_mask = cv2.threshold(gray, threshold, 255, mode)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(tile_mask, connectivity=4)
    return tiles_from_labeled_mask(rgb, lab, n, stats, min_area, epsilon_ratio)


def approach_adaptive_threshold(rgb, block_size=51, C=5, min_area=200, epsilon_ratio=0.02,
                                max_vertices=None, max_area=None, max_aspect_ratio=None,
                                split_large_above=None):
    """Adaptive Gaussian threshold — handles uneven lighting better than a flat threshold.

    Optional post-processing:
      - split_large_above: any CC bigger than this area gets watershed-split (recovers
        merged tiles where the grout line was lost). Set to ~ 3× median tile area.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if block_size % 2 == 0:
        block_size += 1
    tile_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block_size, C,
    )
    if split_large_above is not None:
        lab, n, stats = split_large_components(rgb, tile_mask, split_large_above)
    else:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(tile_mask, connectivity=4)
    return tiles_from_labeled_mask(rgb, lab, n, stats, min_area, epsilon_ratio,
                                   max_vertices=max_vertices, max_area=max_area,
                                   max_aspect_ratio=max_aspect_ratio)


def approach_classical_eroded(rgb, blur_sigma=1.0, threshold=80, erode_iter=1,
                              min_area=200, epsilon_ratio=0.02, invert=False):
    """Same as classical, but erode the tile_mask before CC to separate touching tiles."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if blur_sigma > 0:
        k = max(3, int(blur_sigma * 4) | 1)
        gray = cv2.GaussianBlur(gray, (k, k), blur_sigma)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, tile_mask = cv2.threshold(gray, threshold, 255, mode)
    if erode_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        tile_mask = cv2.erode(tile_mask, kernel, iterations=erode_iter)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(tile_mask, connectivity=4)
    return tiles_from_labeled_mask(rgb, lab, n, stats, min_area, epsilon_ratio)


def approach_adaptive_plus_erode(rgb, block_size=31, C=3, erode_iter=1,
                                 min_area=60, epsilon_ratio=0.02, max_vertices=None):
    """Adaptive threshold + morphological erosion to separate touching tiles."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if block_size % 2 == 0:
        block_size += 1
    tile_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block_size, C,
    )
    if erode_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        tile_mask = cv2.erode(tile_mask, kernel, iterations=erode_iter)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(tile_mask, connectivity=4)
    return tiles_from_labeled_mask(rgb, lab, n, stats, min_area, epsilon_ratio,
                                   max_vertices=max_vertices)


def approach_adaptive_watershed(rgb, block_size=31, C=3, dist_factor=0.35,
                                min_area=60, epsilon_ratio=0.02):
    """Adaptive threshold for the tile/grout mask, then watershed-split touching tiles
    using the distance transform."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if block_size % 2 == 0:
        block_size += 1
    tile_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block_size, C,
    )
    dist = cv2.distanceTransform(tile_mask, cv2.DIST_L2, 3)
    _, peaks = cv2.threshold(dist, dist_factor * dist.max(), 255, 0)
    peaks = peaks.astype(np.uint8)
    n_markers, markers = cv2.connectedComponents(peaks)
    markers = markers + 1
    markers[tile_mask == 0] = 0
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    markers_ws = cv2.watershed(bgr, markers.astype(np.int32))
    h, w = rgb.shape[:2]
    final = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    stats_list = [[0, 0, w, h, 0]]
    for u in np.unique(markers_ws):
        if u <= 0:
            continue
        mask = (markers_ws == u)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        final[mask] = next_id
        stats_list.append([xs.min(), ys.min(), xs.max() - xs.min() + 1,
                           ys.max() - ys.min() + 1, mask.sum()])
        next_id += 1
    stats_arr = np.array(stats_list, dtype=np.int32)
    return tiles_from_labeled_mask(rgb, final, next_id, stats_arr, min_area, epsilon_ratio)


def approach_watershed(rgb, blur_sigma=1.5, threshold=80, dist_factor=0.4,
                       min_area=200, epsilon_ratio=0.02):
    """Watershed: distance transform on tile mask, take peaks as markers, watershed labels."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if blur_sigma > 0:
        k = max(3, int(blur_sigma * 4) | 1)
        gray = cv2.GaussianBlur(gray, (k, k), blur_sigma)
    _, tile_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(tile_mask, cv2.DIST_L2, 3)
    _, peaks = cv2.threshold(dist, dist_factor * dist.max(), 255, 0)
    peaks = peaks.astype(np.uint8)
    n_markers, markers = cv2.connectedComponents(peaks)
    markers = markers + 1
    markers[tile_mask == 0] = 0
    # cv2.watershed requires BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    markers_ws = cv2.watershed(bgr, markers.astype(np.int32))
    # Each marker label (>1) is now a tile. Build a labeled mask compatible with our extractor.
    h, w = rgb.shape[:2]
    final = np.zeros((h, w), dtype=np.int32)
    unique = np.unique(markers_ws)
    next_id = 1
    stats_list = []
    for u in unique:
        if u <= 0:
            continue  # background or boundary
        mask = (markers_ws == u)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        final[mask] = next_id
        stats_list.append([xs.min(), ys.min(), xs.max() - xs.min() + 1,
                           ys.max() - ys.min() + 1, mask.sum()])
        next_id += 1
    stats_arr = np.zeros((next_id, 5), dtype=np.int32)
    stats_arr[0] = [0, 0, w, h, 0]
    for i, s in enumerate(stats_list, start=1):
        stats_arr[i] = s
    return tiles_from_labeled_mask(rgb, final, next_id, stats_arr, min_area, epsilon_ratio)


def approach_slic(rgb, n_segments=6000, compactness=10.0, min_area=200, epsilon_ratio=0.02):
    """SLIC superpixels: cluster by Lab-color + spatial distance into N regions."""
    labels = segmentation.slic(
        rgb, n_segments=n_segments, compactness=compactness,
        start_label=1, channel_axis=-1,
    )
    # Build stats array (compatible with tiles_from_labeled_mask)
    h, w = rgb.shape[:2]
    n_labels = labels.max() + 1
    stats_list = [[0, 0, w, h, 0]]
    for lid in range(1, n_labels):
        ys, xs = np.where(labels == lid)
        if len(xs) == 0:
            stats_list.append([0, 0, 0, 0, 0])
            continue
        stats_list.append(
            [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, len(xs)],
        )
    stats_arr = np.array(stats_list, dtype=np.int32)
    return tiles_from_labeled_mask(rgb, labels.astype(np.int32), n_labels, stats_arr,
                                   min_area, epsilon_ratio, drop_border=False)


# ---------------------------------------------------------------------------
# Metrics & report
# ---------------------------------------------------------------------------

def metrics(source_rgb: np.ndarray, render_rgb: np.ndarray, tiles, runtime_s: float):
    h, w = source_rgb.shape[:2]
    n = len(tiles)
    if n:
        areas = [cv2.contourArea(t[0].astype(np.float32).reshape(-1, 1, 2)) for t in tiles]
        median_area = float(np.median(areas))
        mean_area = float(np.mean(areas))
        max_area = float(np.max(areas))
    else:
        median_area = mean_area = max_area = 0.0
    coverage = float(np.count_nonzero(np.any(render_rgb > 0, axis=2))) / (h * w)
    # Downscale to 512 before SSIM (full-res is slow and noisy)
    s_small = cv2.resize(source_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    r_small = cv2.resize(render_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    s_gray = cv2.cvtColor(s_small, cv2.COLOR_RGB2GRAY)
    r_gray = cv2.cvtColor(r_small, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(s_gray, r_gray, full=True)
    # Mean color error (Lab space) in the regions covered by polygons
    s_lab = skcolor.rgb2lab(s_small / 255.0)
    r_lab = skcolor.rgb2lab(r_small / 255.0)
    color_diff = float(np.mean(np.sqrt(np.sum((s_lab - r_lab) ** 2, axis=2))))
    return {
        "tile_count": n,
        "median_tile_area_px": round(median_area, 1),
        "mean_tile_area_px":   round(mean_area, 1),
        "max_tile_area_px":    round(max_area, 1),
        "coverage_pct":        round(100 * coverage, 1),
        "ssim_512":            round(float(score), 4),
        "lab_color_diff":      round(color_diff, 2),
        "runtime_s":           round(runtime_s, 2),
    }


# ---------------------------------------------------------------------------
# Attempts catalog
# ---------------------------------------------------------------------------

ATTEMPTS = [
    # name, callable returning tiles
    ("01_classical_default",
     lambda rgb: approach_classical(rgb, blur_sigma=1.0, threshold=80, min_area=200,
                                    epsilon_ratio=0.02)),
    ("02_classical_low_threshold",
     lambda rgb: approach_classical(rgb, blur_sigma=1.0, threshold=40, min_area=200,
                                    epsilon_ratio=0.02)),
    ("03_classical_high_threshold",
     lambda rgb: approach_classical(rgb, blur_sigma=1.0, threshold=120, min_area=200,
                                    epsilon_ratio=0.02)),
    ("04_classical_tight_simplify",
     lambda rgb: approach_classical(rgb, blur_sigma=1.0, threshold=80, min_area=200,
                                    epsilon_ratio=0.005)),
    ("05_classical_eroded",
     lambda rgb: approach_classical_eroded(rgb, blur_sigma=1.0, threshold=80, erode_iter=1,
                                           min_area=200, epsilon_ratio=0.02)),
    ("06_classical_eroded_strong",
     lambda rgb: approach_classical_eroded(rgb, blur_sigma=1.0, threshold=80, erode_iter=2,
                                           min_area=200, epsilon_ratio=0.02)),
    ("07_adaptive_threshold",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=51, C=5, min_area=200,
                                             epsilon_ratio=0.02)),
    ("08_adaptive_threshold_tight",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=31, C=3, min_area=150,
                                             epsilon_ratio=0.02)),
    ("09_watershed",
     lambda rgb: approach_watershed(rgb, blur_sigma=1.5, threshold=80,
                                    dist_factor=0.4, min_area=200, epsilon_ratio=0.02)),
    ("10_slic_6000",
     lambda rgb: approach_slic(rgb, n_segments=6000, compactness=10.0,
                               min_area=200, epsilon_ratio=0.02)),
    ("11_slic_8000_compact",
     lambda rgb: approach_slic(rgb, n_segments=8000, compactness=20.0,
                               min_area=150, epsilon_ratio=0.02)),
    # Round 2 — push adaptive_threshold to ≥5000 tiles while keeping rectangular shapes
    ("12_adapt_smaller_block",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=21, C=2, min_area=80,
                                             epsilon_ratio=0.02)),
    ("13_adapt_small_minarea",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=31, C=3, min_area=60,
                                             epsilon_ratio=0.02)),
    ("14_adapt_tighter_minarea",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=25, C=3, min_area=40,
                                             epsilon_ratio=0.02)),
    ("15_adapt_plus_erode",
     lambda rgb: approach_adaptive_plus_erode(rgb, block_size=31, C=3, erode_iter=1,
                                              min_area=60, epsilon_ratio=0.02)),
    ("16_adapt_plus_erode_strong",
     lambda rgb: approach_adaptive_plus_erode(rgb, block_size=31, C=3, erode_iter=2,
                                              min_area=60, epsilon_ratio=0.02)),
    ("17_adapt_watershed",
     lambda rgb: approach_adaptive_watershed(rgb, block_size=31, C=3,
                                             dist_factor=0.35,
                                             min_area=60, epsilon_ratio=0.02)),
    # Round 3 — apply the user's filters on top of the BEST round-2 config: #14
    # (block_size=25, C=3, min_area=40, epsilon_ratio=0.02).
    # These attempts only add the new filters; they don't change the threshold params.
    ("18_14_plus_maxv8",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=25, C=3, min_area=40,
                                             epsilon_ratio=0.02, max_vertices=8)),
    ("19_14_plus_maxv8_aspect3",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=25, C=3, min_area=40,
                                             epsilon_ratio=0.02, max_vertices=8,
                                             max_aspect_ratio=3.0)),
    ("20_14_plus_maxv8_maxarea15000",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=25, C=3, min_area=40,
                                             epsilon_ratio=0.02, max_vertices=8,
                                             max_area=15000, max_aspect_ratio=3.0)),
    ("21_14_plus_split_then_maxv8",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=25, C=3, min_area=40,
                                             epsilon_ratio=0.02, max_vertices=8,
                                             split_large_above=8000)),
    ("22_14_plus_split_full_filter",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=25, C=3, min_area=40,
                                             epsilon_ratio=0.02, max_vertices=8,
                                             max_aspect_ratio=3.0,
                                             split_large_above=8000)),
    # Round 4 — keep all tiles but simplify harder so small/jagged ones survive max_vertices=8.
    ("23_14_eps04_maxv8",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=25, C=3, min_area=40,
                                             epsilon_ratio=0.04, max_vertices=8)),
    ("24_14_eps03_maxv10",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=25, C=3, min_area=40,
                                             epsilon_ratio=0.03, max_vertices=10)),
    ("25_14_eps04_maxv10_split",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=25, C=3, min_area=40,
                                             epsilon_ratio=0.04, max_vertices=10,
                                             split_large_above=8000)),
    ("26_14_eps05_maxv8_split_aspect",
     lambda rgb: approach_adaptive_threshold(rgb, block_size=25, C=3, min_area=40,
                                             epsilon_ratio=0.05, max_vertices=8,
                                             max_aspect_ratio=3.0,
                                             split_large_above=8000)),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv):
    global SOURCE, ATTEMPTS_DIR, REPORT_PATH, JSON_REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None,
                        help="path to source mosaic image (default: output/T1.png). "
                             "Relative paths resolve under IMAGE_TO_MOSAIC/output/.")
    parser.add_argument("filters", nargs="*",
                        help="substrings — only run attempts whose name contains one")
    args = parser.parse_args(argv[1:])

    if args.source:
        candidate = Path(args.source)
        SOURCE = candidate if candidate.is_absolute() else (ROOT / "output" / args.source)
    else:
        SOURCE = DEFAULT_SOURCE

    # Default folder shares the name when source is T1.png (for backward compat),
    # otherwise gets a per-source subfolder so results don't collide.
    if SOURCE == DEFAULT_SOURCE:
        ATTEMPTS_DIR = ROOT / "output" / "csv_attempts"
    else:
        ATTEMPTS_DIR = ROOT / "output" / f"csv_attempts_{SOURCE.stem}"
    REPORT_PATH = ATTEMPTS_DIR / "report.md"
    JSON_REPORT = ATTEMPTS_DIR / "metrics.json"

    if not SOURCE.is_file():
        print(f"ERROR: source not found: {SOURCE}")
        return 1
    ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {SOURCE}...")
    rgb = cv2.cvtColor(cv2.imread(str(SOURCE)), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    print(f"  {w} × {h}")
    print(f"Output folder: {ATTEMPTS_DIR}")

    only = set(args.filters)
    selected = [(name, fn) for name, fn in ATTEMPTS if not only or any(o in name for o in only)]
    print(f"Running {len(selected)} attempts...")

    all_metrics = {}
    for name, fn in selected:
        print(f"\n=== {name} ===")
        try:
            t0 = time.perf_counter()
            tiles = fn(rgb)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            all_metrics[name] = {"error": f"{type(e).__name__}: {e}"}
            continue

        csv_path    = ATTEMPTS_DIR / f"{name}.csv"
        render_path = ATTEMPTS_DIR / f"{name}.png"
        write_csv(tiles, csv_path)
        render = render_polygons(tiles, w, h)
        Image.fromarray(render).save(render_path)
        m = metrics(rgb, render, tiles, elapsed)
        all_metrics[name] = m
        print(f"  tiles={m['tile_count']:>6}  median_area={m['median_tile_area_px']:>7.1f}  "
              f"max={m['max_tile_area_px']:>9.1f}  ssim={m['ssim_512']:.3f}  "
              f"lab_diff={m['lab_color_diff']:.2f}  runtime={m['runtime_s']}s")

    # Merge with any prior metrics so we keep history across runs
    if JSON_REPORT.is_file():
        try:
            prior = json.loads(JSON_REPORT.read_text(encoding="utf-8"))
            for k, v in prior.items():
                all_metrics.setdefault(k, v)
        except Exception:
            pass
    JSON_REPORT.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    write_report(all_metrics)
    print(f"\nReport: {REPORT_PATH}")
    return 0


def write_report(all_metrics: dict) -> None:
    lines = ["# T1.png polygon-extraction attempts", ""]
    lines.append(f"Source: `{SOURCE.relative_to(ROOT)}` (4096 × 4096)")
    lines.append(f"Target: ≥ {TARGET_TILES} tiles")
    lines.append("")
    lines.append("| attempt | tiles | median area | max area | SSIM | Lab Δ | runtime |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in sorted(all_metrics):
        m = all_metrics[name]
        if "error" in m:
            lines.append(f"| {name} | ERROR: {m['error']} |")
            continue
        meets = "✓" if m["tile_count"] >= TARGET_TILES else "·"
        lines.append(
            f"| {name} | {meets} {m['tile_count']} | {m['median_tile_area_px']} | "
            f"{m['max_tile_area_px']} | {m['ssim_512']} | {m['lab_color_diff']} | "
            f"{m['runtime_s']}s |",
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
