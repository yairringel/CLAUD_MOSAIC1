"""Generate 50 mosaic stone plates from output/mosaic_palette_50.csv using Imagen 4.

Reads:
  output/mosaic_palette_50.csv              - color list (name + hex)
  workflows/stone_plate_from_color.md       - the prompt template (blockquote section)

Writes:
  output/stone_plates/{color_id}.png        - 2000x2000 PNG per color

Usage:
  python scripts/generate_stone_plates.py                       # generate all missing plates
  python scripts/generate_stone_plates.py --only rosso_antico   # one plate (id or prefix match)
  python scripts/generate_stone_plates.py --force               # overwrite existing files
  python scripts/generate_stone_plates.py --list                # show catalog
  python scripts/generate_stone_plates.py --dry-run             # print prompts, don't call API
  python scripts/generate_stone_plates.py --model fast          # use Imagen 4 Fast (default: standard)
  python scripts/generate_stone_plates.py --model ultra         # use Imagen 4 Ultra

Requires: GEMINI_API_KEY env var (NEVER paste the key into source or chat).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent   # CLAUDE_MOSAIC1.0/ (file lives in scripts/helpers/)
PALETTE_CSV = ROOT / "output" / "mosaic_palette_50.csv"
TEMPLATE_MD = ROOT / "workflows" / "stone_plate_from_color.md"
OUTPUT_DIR = ROOT / "output" / "stone_plates"
TARGET_SIZE = 2000
SLEEP_BETWEEN_CALLS = 1.5
MAX_RETRIES = 3

MODEL_IDS = {
    # Gemini image-gen models (free tier available)
    "nano-banana":       "gemini-2.5-flash-image",            # original Nano Banana, free tier
    "nano-banana-2":     "gemini-3.1-flash-image-preview",    # newer flash, fast + free tier
    "nano-banana-pro":   "gemini-3-pro-image-preview",        # best Gemini quality, free tier
    # Imagen models (require PAID Google AI Studio plan)
    "imagen-fast":       "imagen-4.0-fast-generate-001",
    "imagen-standard":   "imagen-4.0-generate-001",
    "imagen-ultra":      "imagen-4.0-ultra-generate-001",
}
DEFAULT_MODEL = "nano-banana-pro"


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_palette() -> list[dict]:
    """Return list of {index, name, family, hex, id} dicts from the palette CSV."""
    rows: list[dict] = []
    with open(PALETTE_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "index": int(row["index"]),
                "name": row["name"],
                "family": row["family"],
                "hex": row["hex"],
                "id": slugify(row["name"]),
            })
    return rows


def slugify(name: str) -> str:
    """Convert 'Rosso antico' -> 'rosso_antico'."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def load_prompt_template() -> str:
    """Return the entire prompt template file as the prompt string.

    The file is plain text (no markdown wrapping) with {COLOR_NAME} and
    {HEX_CODE} placeholders, so users can ctrl+A and paste directly.
    """
    return TEMPLATE_MD.read_text(encoding="utf-8").strip()


def build_prompt(template: str, color_name: str, hex_code: str) -> str:
    return template.replace("{COLOR_NAME}", color_name).replace("{HEX_CODE}", hex_code)


# ---------------------------------------------------------------------------
# image saving
# ---------------------------------------------------------------------------

def save_resized(img_bytes: bytes, out_path: Path) -> tuple[int, int]:
    img = Image.open(BytesIO(img_bytes))
    orig_size = img.size
    if img.mode != "RGB":
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")
    if img.size != (TARGET_SIZE, TARGET_SIZE):
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    img.save(out_path, "PNG", optimize=True)
    return orig_size


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_imagen(client, model_id: str, prompt: str):
    """Imagen models use the generate_images() endpoint."""
    from google.genai import types
    response = client.models.generate_images(
        model=model_id,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1",
            image_size="2K",
        ),
    )
    if not response.generated_images:
        return None
    return response.generated_images[0].image.image_bytes


def _call_gemini_image(client, model_id: str, prompt: str):
    """Gemini image-gen models use generate_content() with image modality."""
    from google.genai import types
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="1:1", image_size="2K"),
        ),
    )
    for candidate in response.candidates or []:
        for part in (candidate.content.parts if candidate.content else []) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data
    return None


def generate_one(client, model_id: str, prompt: str, out_path: Path) -> bool:
    print(f"  prompt: {prompt[:120]}...")
    is_imagen = model_id.startswith("imagen-")
    caller = _call_imagen if is_imagen else _call_gemini_image
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            img_bytes = caller(client, model_id, prompt)
            if img_bytes is None:
                print(f"  attempt {attempt}: response had no images (safety filter or empty response)")
            else:
                orig = save_resized(img_bytes, out_path)
                print(f"  ok  ({orig[0]}x{orig[1]} -> {TARGET_SIZE}x{TARGET_SIZE}) -> {out_path.name}")
                return True
        except Exception as e:
            print(f"  attempt {attempt} failed: {type(e).__name__}: {e}")
        if attempt < MAX_RETRIES:
            time.sleep(5 * attempt)
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only", help="generate just one color (id or prefix or name substring)")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("--list", action="store_true", help="list palette entries and exit")
    parser.add_argument("--dry-run", action="store_true", help="print prompts, don't call API")
    parser.add_argument("--model", choices=list(MODEL_IDS.keys()), default=DEFAULT_MODEL,
                        help=f"Imagen 4 variant (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    palette = load_palette()

    if args.list:
        for c in palette:
            print(f"  {c['index']:02d}  {c['id']:<24} {c['hex']:<8} ({c['family']:<6}) {c['name']}")
        return 0

    if args.only:
        needle = args.only.lower()
        palette = [c for c in palette if needle in c["id"].lower() or needle in c["name"].lower()]
        if not palette:
            print(f"no color matches --only {args.only!r}")
            return 1

    template = load_prompt_template()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        if not os.environ.get("GEMINI_API_KEY"):
            print("ERROR: GEMINI_API_KEY env var not set.")
            print("       In PowerShell:  $env:GEMINI_API_KEY = \"AIzaSy...\"")
            print("       Get a free key at https://aistudio.google.com/apikey")
            return 1
        from google import genai
        client = genai.Client()
    else:
        client = None

    model_id = MODEL_IDS[args.model]
    print(f"model: {model_id}\n")

    successes, skipped, failures = [], [], []
    for i, color in enumerate(palette, 1):
        out_path = OUTPUT_DIR / f"{color['id']}.png"
        print(f"[{i}/{len(palette)}] {color['name']} ({color['hex']})")

        if out_path.exists() and not args.force:
            print(f"  skip (already exists): {out_path.name}")
            skipped.append(color["id"])
            continue

        prompt = build_prompt(template, color["name"], color["hex"])

        if args.dry_run:
            print(f"  [dry-run] would generate -> {out_path.name}")
            print(f"  prompt: {prompt}\n")
            continue

        if generate_one(client, model_id, prompt, out_path):
            successes.append(color["id"])
        else:
            failures.append(color["id"])

        if i < len(palette):
            time.sleep(SLEEP_BETWEEN_CALLS)

    print("\n" + "=" * 60)
    print(f"done: {len(successes)} generated, {len(skipped)} skipped, {len(failures)} failed")
    if failures:
        print("failed:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
