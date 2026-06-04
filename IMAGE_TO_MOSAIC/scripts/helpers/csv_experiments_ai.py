"""AI-augmented attempts for the T1.png → polygons experiment.

Run AFTER csv_experiments.py finishes — this script costs API calls.

Approaches:
  ai_12_mask_2k       — single Nano-Banana Pro call, 2K mask, current prompt
  ai_13_mask_4k       — single call, 4K mask  (better detail, slower)
  ai_14_mask_per_tile_4x4 — split source into 4×4 grid, mask each sub-image at 2K, stitch
  ai_15_mask_per_tile_2x2 — 2×2 grid (cheaper but coarser)
  ai_16_mask_4k_explicit_count — 4K mask, prompt explicitly asks for ~5000 tiles

Each call to Nano Banana Pro = 1 API unit. Budget is set by --budget; default 20.

Run:
  python IMAGE_TO_MOSAIC/scripts/csv_experiments_ai.py
  python IMAGE_TO_MOSAIC/scripts/csv_experiments_ai.py --only 14
  python IMAGE_TO_MOSAIC/scripts/csv_experiments_ai.py --budget 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent   # IMAGE_TO_MOSAIC/ (file lives in scripts/helpers/)
PROJECT_ROOT = ROOT.parent
SOURCE = ROOT / "output" / "T1.png"
ATTEMPTS_DIR = ROOT / "output" / "csv_attempts"
JSON_REPORT = ATTEMPTS_DIR / "metrics.json"
KEY_PATH_MEMO = ROOT / ".key_path"

# Reuse the existing harness for write_csv/render_polygons/metrics.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_experiments import (
    write_csv, render_polygons, metrics, tiles_from_labeled_mask, write_report,
    TARGET_TILES,
)

GEMINI_MODEL = "gemini-3-pro-image-preview"
api_calls_made = 0  # module-level counter


# ---------------------------------------------------------------------------
# API key (same scheme as photo_editor.py / mosaic_to_csv.py)
# ---------------------------------------------------------------------------

def read_key_from_file(path: Path):
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("GEMINI_API_KEY"):
            _, _, value = line.partition("=")
            return value.strip().strip('"').strip("'") or None
        if "=" not in line:
            return line.strip('"').strip("'")
    return None


def load_api_key():
    if KEY_PATH_MEMO.is_file():
        memo = Path(KEY_PATH_MEMO.read_text(encoding="utf-8").strip())
        key = read_key_from_file(memo)
        if key:
            return key
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()
    for p in (ROOT / ".env", PROJECT_ROOT / ".env", ROOT / "gemini.key"):
        key = read_key_from_file(p)
        if key:
            return key
    return None


# ---------------------------------------------------------------------------
# Gemini call wrapper
# ---------------------------------------------------------------------------

def call_gemini_for_mask(image_rgb: np.ndarray, prompt_text: str,
                          aspect_ratio: str = "1:1", image_size: str = "2K"):
    """Send image + prompt, return PNG bytes of the binary mask."""
    global api_calls_made
    api_calls_made += 1
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY available.")

    from google import genai
    from google.genai import types

    buf = BytesIO()
    Image.fromarray(image_rgb, mode="RGB").save(buf, "PNG")
    img_bytes = buf.getvalue()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            prompt_text,
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio, image_size=image_size,
            ),
        ),
    )
    for candidate in response.candidates or []:
        for part in (candidate.content.parts if candidate.content else []) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data
    raise RuntimeError("No image in Gemini response (safety filter or empty).")


MASK_PROMPT_BASIC = (
    "Identify every individual mosaic tile in the supplied image and output a "
    "binary image where the BACKGROUND is solid PURE BLACK (#000000) and every "
    "tile is drawn as ONE SOLID PURE WHITE (#FFFFFF) shape, matching the tile's "
    "exact position and outline.\n"
    "STRICT requirements:\n"
    "- Every pair of adjacent tiles must be separated by at least 3 pixels of "
    "BLACK between their white shapes — adjacent shapes must NEVER touch.\n"
    "- Fill every tile shape solid white edge-to-edge. No gray, no patterns.\n"
    "- Only #FFFFFF and #000000 — no other colors, no anti-aliasing.\n"
    "- Do not skip tiles. Every visible tile becomes a separate white shape.\n"
    "Output ONLY the binary image."
)


def MASK_PROMPT_EXPLICIT(target_count: int) -> str:
    return (
        f"Identify every individual mosaic tile in the supplied image. There "
        f"are approximately {target_count} tiles in this image — count them "
        f"carefully and represent ALL of them in the output. Output a binary "
        f"image where the BACKGROUND is solid PURE BLACK (#000000) and every "
        f"tile is drawn as ONE SOLID PURE WHITE (#FFFFFF) shape.\n"
        "STRICT requirements:\n"
        "- Approximately {n} separate white shapes total.\n".format(n=target_count) +
        "- Every pair of adjacent tiles must be separated by at least 3 pixels "
        "of BLACK between their white shapes — adjacent shapes must NEVER touch.\n"
        "- Fill every tile shape solid white edge-to-edge.\n"
        "- Only #FFFFFF and #000000.\n"
        "- Do not skip tiles.\n"
        "Output ONLY the binary image."
    )


# ---------------------------------------------------------------------------
# Mask post-processing → CC → polygons
# ---------------------------------------------------------------------------

def mask_to_tile_mask(png_bytes: bytes, target_w: int, target_h: int) -> np.ndarray:
    pil = Image.open(BytesIO(png_bytes)).convert("L")
    if pil.size != (target_w, target_h):
        pil = pil.resize((target_w, target_h), Image.NEAREST)
    arr = np.array(pil)
    return (arr >= 128).astype(np.uint8) * 255


def tiles_from_tile_mask(rgb, tile_mask, min_area=200, epsilon_ratio=0.02,
                        erode_iter=0):
    if erode_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        tile_mask = cv2.erode(tile_mask, kernel, iterations=erode_iter)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(tile_mask, connectivity=4)
    return tiles_from_labeled_mask(rgb, lab, n, stats, min_area, epsilon_ratio)


# ---------------------------------------------------------------------------
# Attempts
# ---------------------------------------------------------------------------

def attempt_mask_single(rgb, image_size: str, prompt_text: str, erode_iter: int = 0):
    h, w = rgb.shape[:2]
    png_bytes = call_gemini_for_mask(rgb, prompt_text, "1:1", image_size)
    tile_mask = mask_to_tile_mask(png_bytes, w, h)
    return tiles_from_tile_mask(rgb, tile_mask, min_area=200, epsilon_ratio=0.02,
                                erode_iter=erode_iter), tile_mask


def attempt_mask_per_tile_grid(rgb, grid: int, image_size: str, prompt_text: str,
                               erode_iter: int = 0):
    """Slice rgb into grid×grid sub-images, mask each, paste back, then CC."""
    h, w = rgb.shape[:2]
    th, tw = h // grid, w // grid
    full_mask = np.zeros((h, w), dtype=np.uint8)
    for gy in range(grid):
        for gx in range(grid):
            y0, x0 = gy * th, gx * tw
            y1 = h if gy == grid - 1 else y0 + th
            x1 = w if gx == grid - 1 else x0 + tw
            sub = rgb[y0:y1, x0:x1]
            print(f"    tile {gy+1},{gx+1} ({x1-x0} × {y1-y0})...", flush=True)
            png_bytes = call_gemini_for_mask(sub, prompt_text, "1:1", image_size)
            sub_mask = mask_to_tile_mask(png_bytes, x1 - x0, y1 - y0)
            # Carve a 2-pixel black border between grid cells so CC doesn't merge
            # tiles that span the seam.
            if gx > 0: sub_mask[:, :2] = 0
            if gy > 0: sub_mask[:2, :] = 0
            full_mask[y0:y1, x0:x1] = sub_mask
    return tiles_from_tile_mask(rgb, full_mask, min_area=200, epsilon_ratio=0.02,
                                erode_iter=erode_iter), full_mask


ATTEMPTS = [
    ("ai_12_mask_2k",
     lambda rgb: attempt_mask_single(rgb, "2K", MASK_PROMPT_BASIC)),
    ("ai_13_mask_4k",
     lambda rgb: attempt_mask_single(rgb, "4K", MASK_PROMPT_BASIC)),
    ("ai_14_mask_per_tile_4x4",
     lambda rgb: attempt_mask_per_tile_grid(rgb, 4, "2K", MASK_PROMPT_BASIC)),
    ("ai_15_mask_per_tile_2x2",
     lambda rgb: attempt_mask_per_tile_grid(rgb, 2, "2K", MASK_PROMPT_BASIC)),
    ("ai_16_mask_4k_explicit_5000",
     lambda rgb: attempt_mask_single(rgb, "4K", MASK_PROMPT_EXPLICIT(5000))),
]
ATTEMPT_CALL_COST = {
    "ai_12_mask_2k": 1, "ai_13_mask_4k": 1, "ai_16_mask_4k_explicit_5000": 1,
    "ai_15_mask_per_tile_2x2": 4, "ai_14_mask_per_tile_4x4": 16,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=[],
                        help="run only attempts whose name contains any of these tokens")
    parser.add_argument("--budget", type=int, default=20,
                        help="max Gemini API calls allowed total (default 20)")
    args = parser.parse_args()

    if not SOURCE.is_file():
        print(f"ERROR: source not found: {SOURCE}")
        return 1
    ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)

    rgb = cv2.cvtColor(cv2.imread(str(SOURCE)), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    print(f"Source: {w} × {h}")

    if load_api_key() is None:
        print("ERROR: no Gemini API key found.")
        return 1

    only = args.only
    selected = [(n, fn) for n, fn in ATTEMPTS if not only or any(o in n for o in only)]

    # Budget enforcement
    planned = sum(ATTEMPT_CALL_COST[n] for n, _ in selected)
    print(f"Selected attempts: {[n for n, _ in selected]}")
    print(f"Planned API calls: {planned}  |  budget: {args.budget}")
    if planned > args.budget:
        print("Refusing to start — planned calls exceed budget. Use --only to narrow, "
              "or --budget to raise.")
        return 1

    all_metrics = {}
    if JSON_REPORT.is_file():
        try:
            all_metrics = json.loads(JSON_REPORT.read_text(encoding="utf-8"))
        except Exception:
            pass

    for name, fn in selected:
        print(f"\n=== {name} ===  (API calls so far: {api_calls_made}/{args.budget})")
        try:
            t0 = time.perf_counter()
            tiles, mask = fn(rgb)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            all_metrics[name] = {"error": f"{type(e).__name__}: {e}"}
            JSON_REPORT.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
            continue

        csv_path    = ATTEMPTS_DIR / f"{name}.csv"
        render_path = ATTEMPTS_DIR / f"{name}.png"
        mask_path   = ATTEMPTS_DIR / f"{name}_mask.png"
        write_csv(tiles, csv_path)
        render = render_polygons(tiles, w, h)
        Image.fromarray(render).save(render_path)
        Image.fromarray(mask).save(mask_path)
        m = metrics(rgb, render, tiles, elapsed)
        m["api_calls"] = ATTEMPT_CALL_COST[name]
        all_metrics[name] = m
        JSON_REPORT.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
        print(f"  tiles={m['tile_count']:>6}  median={m['median_tile_area_px']:>7.1f}  "
              f"max={m['max_tile_area_px']:>9.1f}  ssim={m['ssim_512']:.3f}  "
              f"lab_diff={m['lab_color_diff']:.2f}  runtime={m['runtime_s']}s")

    write_report(all_metrics)
    print(f"\nTotal API calls used: {api_calls_made}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
