"""Divide each stone plate into a 4x4 grid of squares, saving each one named
by its perceptual-average color (Lab-space mean).

Process per source image (2000x2000):
  1. Crop into 16 squares of 500x500 (no resize, no resolution loss)
  2. For each square: compute Lab-space mean color, convert back to sRGB hex
  3. Save the 500x500 square as `output/roman colors/{HEX}.png`

Usage:
  python scripts/divide_into_tesserae.py                        # all 50 sources
  python scripts/divide_into_tesserae.py --only rosso_antico    # one source
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "output" / "stone_plates"
OUT_DIR = ROOT / "output" / "roman colors"
GRID = 4  # 4x4 = 16 squares per source image
MAX_PERTURB_ATTEMPTS = 20  # try up to N tiny color shifts to resolve hex collisions
# Each square gets a deliberate tint in Lab space based on its (row, col) position,
# giving 16 distinct subtle color directions for variety.
TINT_LEVELS_LAB = (-3.0, -1.0, 1.0, 3.0)  # 4 evenly-spaced shifts in Lab units


# ---------------------------------------------------------------------------
# sRGB <-> Lab conversions (D65 illuminant, vectorized over numpy arrays)
# ---------------------------------------------------------------------------

# sRGB -> Linear-light RGB -> XYZ (D65) -> Lab -> mean -> reverse

RGB_TO_XYZ_D65 = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
XYZ_TO_RGB_D65 = np.linalg.inv(RGB_TO_XYZ_D65)
WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])
DELTA = 6.0 / 29.0


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> Linear-light RGB [0,1]."""
    a = 0.055
    return np.where(srgb <= 0.04045,
                    srgb / 12.92,
                    ((srgb + a) / (1.0 + a)) ** 2.4)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Linear-light RGB [0,1] -> sRGB [0,1]."""
    a = 0.055
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308,
                    linear * 12.92,
                    (1.0 + a) * (linear ** (1.0 / 2.4)) - a)


def _f_lab(t: np.ndarray) -> np.ndarray:
    return np.where(t > DELTA ** 3, np.cbrt(t), t / (3.0 * DELTA ** 2) + 4.0 / 29.0)


def _f_lab_inv(t: np.ndarray) -> np.ndarray:
    return np.where(t > DELTA, t ** 3, 3.0 * DELTA ** 2 * (t - 4.0 / 29.0))


def rgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 sRGB -> (..., 3) float64 Lab."""
    srgb = rgb_uint8.astype(np.float64) / 255.0
    linear = srgb_to_linear(srgb)
    xyz = linear @ RGB_TO_XYZ_D65.T
    xyz_n = xyz / WHITE_D65
    f = _f_lab(xyz_n)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb_uint8(lab: np.ndarray) -> np.ndarray:
    """(..., 3) Lab -> (..., 3) uint8 sRGB."""
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    xyz_n = np.stack([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)], axis=-1)
    xyz = xyz_n * WHITE_D65
    linear = xyz @ XYZ_TO_RGB_D65.T
    srgb = linear_to_srgb(linear)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


def average_color_lab(square_rgb: np.ndarray) -> np.ndarray:
    """Compute the perceptual (Lab-space mean) average of a 2D RGB array.

    Returns (3,) uint8 sRGB.
    """
    lab = rgb_to_lab(square_rgb)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    return lab_to_rgb_uint8(mean_lab.reshape(1, 1, 3))[0, 0]


# ---------------------------------------------------------------------------
# processing
# ---------------------------------------------------------------------------

def hex_str(rgb: np.ndarray) -> str:
    return "{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def position_tint_offset(r: int, c: int) -> tuple[float, float, float]:
    """Per grid-position Lab offset. Produces 16 distinct (dL, da, db) tint vectors:
      row controls L*  (lightness): row 0 darker -> row 3 lighter
      col controls a*  (green-red): col 0 greener -> col 3 redder
      (r+c) controls b* (blue-yellow) for an additional diagonal pattern
    """
    dL = TINT_LEVELS_LAB[r]
    da = TINT_LEVELS_LAB[c]
    db = TINT_LEVELS_LAB[(r + c) % 4]
    return dL, da, db


def apply_lab_tint(square: np.ndarray, dL: float, da: float, db: float) -> np.ndarray:
    """Shift every pixel in `square` by (dL, da, db) in Lab space, return sRGB uint8."""
    lab = rgb_to_lab(square)
    lab[..., 0] += dL
    lab[..., 1] += da
    lab[..., 2] += db
    return lab_to_rgb_uint8(lab)


def perturb_square(square: np.ndarray, magnitude: int, rng: np.random.Generator) -> np.ndarray:
    """Apply an imperceptible color shift: pick a random RGB channel,
    add a signed offset of `magnitude` to every pixel in that channel,
    clipped to [0, 255]. magnitude=1 shifts the channel mean by ~1 unit.
    """
    ch = int(rng.integers(0, 3))
    sign = int(rng.choice([-1, 1]))
    delta = np.zeros(3, dtype=np.int16)
    delta[ch] = sign * magnitude
    shifted = square.astype(np.int16) + delta[None, None, :]
    return np.clip(shifted, 0, 255).astype(np.uint8)


def find_unique_average(square: np.ndarray, seen: set[str],
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute Lab-mean of `square`. If the resulting hex is already in `seen`,
    perturb the square by tiny color shifts until it produces a unique hex
    (or until MAX_PERTURB_ATTEMPTS is reached).

    Returns (final_square, final_avg_rgb, n_attempts_used).
    """
    current = square
    avg = average_color_lab(current)
    if hex_str(avg) not in seen:
        return current, avg, 0

    for attempt in range(1, MAX_PERTURB_ATTEMPTS + 1):
        # perturb from the ORIGINAL each time so we don't drift cumulatively
        candidate = perturb_square(square, attempt, rng)
        avg = average_color_lab(candidate)
        if hex_str(avg) not in seen:
            return candidate, avg, attempt

    # gave up — return last candidate even though it still collides
    return candidate, avg, MAX_PERTURB_ATTEMPTS


def process_source(src_path: Path, out_dir: Path, seen: set[str],
                   rng: np.random.Generator) -> list[dict]:
    img = Image.open(src_path).convert("RGB")
    arr = np.asarray(img)
    H, W = arr.shape[:2]
    cell_h = H // GRID
    cell_w = W // GRID
    print(f"  source: {W}x{H} -> {GRID}x{GRID} grid of {cell_w}x{cell_h} squares")

    results: list[dict] = []
    for r in range(GRID):
        for c in range(GRID):
            y0, x0 = r * cell_h, c * cell_w
            square = arr[y0:y0 + cell_h, x0:x0 + cell_w]
            # Always apply a deliberate per-position Lab tint so each of the 16
            # squares has its own distinct color direction.
            dL, da, db = position_tint_offset(r, c)
            tinted = apply_lab_tint(square, dL, da, db)
            # Collision-perturbation stays as a safety net (rarely fires after tinting).
            final_square, avg, attempts = find_unique_average(tinted, seen, rng)
            h = hex_str(avg)
            seen.add(h)
            fname = f"{h}.png"
            out_path = out_dir / fname
            Image.fromarray(final_square).save(out_path, "PNG", optimize=True)
            results.append({
                "row": r, "col": c, "hex": h,
                "rgb": tuple(int(x) for x in avg), "file": fname,
                "tint": (dL, da, db),
                "perturb_attempts": attempts,
            })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="process only one source image (id substring)")
    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        print(f"ERROR: source directory not found: {SOURCE_DIR}")
        return 1

    sources = sorted(SOURCE_DIR.glob("*.png"))
    if args.only:
        sources = [s for s in sources if args.only.lower() in s.stem.lower()]
        if not sources:
            print(f"no source matches --only {args.only!r}")
            return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output: {OUT_DIR}")
    print(f"sources: {len(sources)}\n")

    # global set of already-used hex strings — drives collision-perturbation
    seen: set[str] = set()
    rng = np.random.default_rng(seed=42)  # deterministic for reproducibility

    total = 0
    total_perturbed = 0
    for src in sources:
        print(f"[{src.stem}]")
        results = process_source(src, OUT_DIR, seen, rng)
        total += len(results)
        total_perturbed += sum(1 for r in results if r["perturb_attempts"] > 0)
        if len(sources) == 1:
            for r in results:
                tag = f" (perturbed x{r['perturb_attempts']})" if r["perturb_attempts"] > 0 else ""
                dL, da, db = r["tint"]
                print(f"  r{r['row']}c{r['col']}  tint=(L{dL:+.0f},a{da:+.0f},b{db:+.0f})  avg={r['hex']}  rgb={r['rgb']}  -> {r['file']}{tag}")

    print(f"\ndone: {total} squares written to {OUT_DIR}  ({total_perturbed} needed perturbation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
