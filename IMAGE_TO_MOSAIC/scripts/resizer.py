"""Resizer — reshape an image to a specific print size at 300 DPI.

Workflow:
  1. Load Image — pick any image (JPG / PNG / TIFF …).
  2. Enter tile size in centimetres (e.g. 20 × 45 cm).
  3. Generate — the source + a "reframe" prompt are sent to Gemini's
     image model. The prompt tells the model to keep the original
     content intact and EXPAND (never crop) into the new aspect
     ratio, filling the added strips with content that matches the
     source's style, lighting and edges.
  4. The returned image is fitted to the exact target pixel size
     (TARGET_W_CM × 300 / 2.54 by TARGET_H_CM × 300 / 2.54) and
     tagged with 300 DPI in its PNG pHYs metadata.
  5. Save Result — writes the PNG to disk (default: output/).

The Gemini API key is loaded the same way photo_editor.py loads it —
env var, .env, .key file, or a saved key-file path.

Requires: google-genai, Pillow, PyQt5.
Usage:    python IMAGE_TO_MOSAIC/scripts/resizer.py
"""
from __future__ import annotations

import os
import sys
import traceback
from io import BytesIO
from pathlib import Path

from PIL import Image
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSplitter, QVBoxLayout, QWidget,
)

ROOT         = Path(__file__).resolve().parent.parent       # IMAGE_TO_MOSAIC/
PROJECT_ROOT = ROOT.parent                                  # CLAUDE_MOSAIC1.0/
OUTPUT_DIR   = ROOT / "output"
INPUT_DIR    = ROOT / "input"
PROMPTS_DIR  = ROOT / "prompts"
PROMPT_FILE  = PROMPTS_DIR / "resize_to_ratio.txt"

DEFAULT_MODEL = "gemini-3-pro-image-preview"    # Nano Banana Pro
DPI           = 300


# ---------------------------------------------------------------------------
# Key loading — same conventions as photo_editor.py so a single key file
# serves the whole IMAGE_TO_MOSAIC toolset.
# ---------------------------------------------------------------------------

KEY_PATH_MEMO = ROOT / ".key_path"


def read_key_from_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("GEMINI_API_KEY"):
            _, _, value = line.partition("=")
            return value.strip().strip('"').strip("'") or None
        if "=" not in line:
            return line.strip('"').strip("'")
    return None


def load_api_key() -> str | None:
    """Same source order as photo_editor.load_api_key."""
    if KEY_PATH_MEMO.is_file():
        memo = Path(KEY_PATH_MEMO.read_text(encoding="utf-8").strip())
        key = read_key_from_file(memo)
        if key:
            return key
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()
    for env_path in (ROOT / ".env", PROJECT_ROOT / ".env", ROOT / "gemini.key"):
        key = read_key_from_file(env_path)
        if key:
            return key
    return None


# ---------------------------------------------------------------------------
# Aspect ratio helpers
# ---------------------------------------------------------------------------

# The set of aspect ratios Gemini's image model accepts (verified against
# the API's INVALID_ARGUMENT rejection message). NOTE: 1:2 and 2:1 are
# NOT accepted — the closest tall / wide options are 9:16 and 16:9, or
# the extreme 1:4 / 4:1 if the source is very elongated. We pick the
# closest supported value to the user's tile ratio; the returned image
# is then fitted to the exact requested pixels via `fit_to_exact_size`.
_GEMINI_ASPECTS = {
    "1:1":  1 / 1,
    "1:4":  1 / 4,
    "1:8":  1 / 8,
    "2:3":  2 / 3,
    "3:2":  3 / 2,
    "3:4":  3 / 4,
    "4:1":  4 / 1,
    "4:3":  4 / 3,
    "4:5":  4 / 5,
    "5:4":  5 / 4,
    "8:1":  8 / 1,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
}


def closest_gemini_aspect(target_ratio: float) -> str:
    return min(_GEMINI_ASPECTS.items(),
               key=lambda kv: abs(kv[1] - target_ratio))[0]


def cm_to_px_at_300dpi(cm: float) -> int:
    """cm × (300 dpi / 2.54 cm-per-inch), rounded to nearest int."""
    return max(1, int(round(cm * DPI / 2.54)))


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def integer_aspect(w_px: int, h_px: int) -> tuple[int, int]:
    """Reduce w:h to smallest integer ratio (only for readability in the
    prompt — not used for math)."""
    g = gcd(max(1, w_px), max(1, h_px)) or 1
    return w_px // g, h_px // g


def readable_cm_aspect(w_cm: float, h_cm: float) -> tuple[int, int]:
    """Reduce w_cm:h_cm to a small integer ratio for the prompt/hint.
    Multiplies by 10 to catch tenths, then GCD-reduces. E.g. 20×45 → 4:9,
    30×30 → 1:1, 20×30.5 → 40:61."""
    a = max(1, int(round(w_cm * 10)))
    b = max(1, int(round(h_cm * 10)))
    g = gcd(a, b) or 1
    return a // g, b // g


# ---------------------------------------------------------------------------
# Fit / pad the API result to the EXACT target pixel size + aspect.
# Gemini returns an image close to the requested aspect ratio but not
# necessarily the exact pixel dimensions we need, so we finish the job
# in PIL. Aspect mismatch (rare — target isn't in Gemini's 7-value set)
# is resolved by extending edges rather than cropping.
# ---------------------------------------------------------------------------

def _detect_letterbox_crop(img: Image.Image) -> Image.Image:
    """Trim uniform BLACK letterbox / pillarbox bars from the outside
    of an image. Gemini sometimes returns an aspect-fitted canvas with
    a solid black frame around the actual scene; if we don't strip it
    the downstream fit-to-target just scales the black up. Threshold
    ≤ 8/255 per channel, so near-black bars are caught too.
    Returns the unchanged image if no letterbox is detected."""
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    import numpy as _np
    arr = _np.asarray(img)
    if arr.ndim != 3:
        return img
    rgb = arr[..., :3]
    # A row / column is "black" when every channel of every pixel in
    # it is ≤ threshold. `.all(axis=(1,2))` collapses to a per-row bool.
    thr = 8
    row_black = (rgb <= thr).all(axis=(1, 2))
    col_black = (rgb <= thr).all(axis=(0, 2))
    if not row_black.any() and not col_black.any():
        return img
    # First / last non-black row.
    non_black_rows = _np.where(~row_black)[0]
    non_black_cols = _np.where(~col_black)[0]
    if non_black_rows.size == 0 or non_black_cols.size == 0:
        return img          # entirely black (unlikely) — leave alone
    y0, y1 = int(non_black_rows[0]), int(non_black_rows[-1]) + 1
    x0, x1 = int(non_black_cols[0]), int(non_black_cols[-1]) + 1
    if (x0, y0, x1, y1) == (0, 0, img.width, img.height):
        return img
    return img.crop((x0, y0, x1, y1))


def fit_to_exact_size(img: Image.Image, target_w: int,
                       target_h: int) -> Image.Image:
    """Return a copy of `img` whose canvas is exactly (target_w × target_h)
    with the source scaled to fill one axis and the OTHER axis padded
    with WHITE. Never crops the subject; never extends by mirroring
    (which produced striped-black bars when Gemini's return already
    had a black letterbox). If Gemini letterboxed its own output,
    those bars are stripped first via `_detect_letterbox_crop` so the
    scaled subject fills as much of the requested canvas as possible."""
    # 1. Trim any solid-black letterbox / pillarbox Gemini added.
    img = _detect_letterbox_crop(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    src_w, src_h = img.size
    tgt_ratio = target_w / max(1, target_h)
    src_ratio = src_w / max(1, src_h)

    if abs(src_ratio - tgt_ratio) < 1e-3:
        # Aspect already matches — just resize.
        return img.resize((target_w, target_h), Image.LANCZOS)

    fill = ((255, 255, 255) if img.mode == "RGB"
            else (255, 255, 255, 255))
    canvas = Image.new(img.mode, (target_w, target_h), fill)

    if src_ratio > tgt_ratio:
        # Source WIDER than target → fit width, WHITE bars on top / bottom.
        new_w = target_w
        new_h = max(1, int(round(new_w / src_ratio)))
        scaled = img.resize((new_w, new_h), Image.LANCZOS)
        canvas.paste(scaled, (0, (target_h - new_h) // 2))
    else:
        # Source TALLER than target → fit height, WHITE bars on left / right.
        new_h = target_h
        new_w = max(1, int(round(new_h * src_ratio)))
        scaled = img.resize((new_w, new_h), Image.LANCZOS)
        canvas.paste(scaled, ((target_w - new_w) // 2, 0))
    return canvas


# ---------------------------------------------------------------------------
# Gemini worker — image + reframe prompt → new PNG bytes.
# ---------------------------------------------------------------------------

class ResizeWorker(QThread):
    finished_ok = pyqtSignal(bytes)
    failed      = pyqtSignal(str)
    progress    = pyqtSignal(str)

    def __init__(self, source_image: Image.Image, prompt_text: str,
                 aspect_str: str, image_size: str):
        super().__init__()
        self.source_image = source_image
        self.prompt_text  = prompt_text
        self.aspect_str   = aspect_str
        self.image_size   = image_size

    def run(self) -> None:
        try:
            api_key = load_api_key()
            if not api_key:
                self.failed.emit(
                    "No Gemini API key found.\n\n"
                    "Set GEMINI_API_KEY env var, or save the key in "
                    "IMAGE_TO_MOSAIC/.env  (line: GEMINI_API_KEY=...)  "
                    "or in IMAGE_TO_MOSAIC/gemini.key (plain key)."
                )
                return

            self.progress.emit("Preparing image...")
            buf = BytesIO()
            img = self.source_image
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            img.save(buf, "PNG")
            img_bytes = buf.getvalue()

            self.progress.emit(f"Calling {DEFAULT_MODEL} "
                               f"(aspect {self.aspect_str})...")
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[
                    types.Part.from_bytes(data=img_bytes,
                                          mime_type="image/png"),
                    self.prompt_text,
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=self.aspect_str,
                        image_size=self.image_size,
                    ),
                ),
            )

            result_bytes = None
            for candidate in response.candidates or []:
                for part in (candidate.content.parts
                             if candidate.content else []) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline is not None and getattr(inline, "data", None):
                        result_bytes = inline.data
                        break
                if result_bytes is not None:
                    break

            if result_bytes is None:
                self.failed.emit(
                    "No image returned (safety filter or empty response)."
                )
                return
            self.finished_ok.emit(result_bytes)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n\n"
                             f"{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Simple image pane — a QLabel inside a scroll area, that fits-to-window.
# ---------------------------------------------------------------------------

class ImagePane(QFrame):
    def __init__(self, placeholder: str):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.caption = QLabel(placeholder)
        self.caption.setStyleSheet(
            "color: #555; font-size: 11px; padding: 2px;"
        )
        layout.addWidget(self.caption)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.label = QLabel("(no image)")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #888;")
        self.scroll.setWidget(self.label)
        layout.addWidget(self.scroll, 1)
        self._pil_image: Image.Image | None = None

    def set_image(self, img: Image.Image | None, caption: str = "") -> None:
        self._pil_image = img
        if img is None:
            self.label.setText("(no image)")
            self.label.setPixmap(QPixmap())
            self.caption.setText(caption or "")
            return
        buf = BytesIO()
        (img if img.mode in ("RGB", "RGBA") else img.convert("RGB")
         ).save(buf, "PNG")
        qimg = QImage.fromData(buf.getvalue(), "PNG")
        pm = QPixmap.fromImage(qimg)
        # Fit to the current scroll-viewport size while keeping aspect,
        # so the whole result is visible without manual scrolling.
        vp = self.scroll.viewport().size()
        if pm.width() > vp.width() or pm.height() > vp.height():
            pm = pm.scaled(vp.width(), vp.height(),
                           Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(pm)
        self.caption.setText(caption or f"{img.width} × {img.height} px")

    def image(self) -> Image.Image | None:
        return self._pil_image


# ---------------------------------------------------------------------------
# Main window.
# ---------------------------------------------------------------------------

class ResizerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resizer — image → 300 DPI tile size")
        self.setGeometry(100, 100, 1200, 720)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Toolbar row ────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self.load_btn = QPushButton("Load Image...")
        self.load_btn.clicked.connect(self.on_load)
        bar.addWidget(self.load_btn)

        bar.addSpacing(20)
        bar.addWidget(QLabel("Tile size:"))
        self.w_cm = QDoubleSpinBox()
        self.w_cm.setRange(0.1, 500.0)
        self.w_cm.setDecimals(1)
        self.w_cm.setSingleStep(0.5)
        self.w_cm.setValue(20.0)
        self.w_cm.setSuffix(" cm")
        self.w_cm.setToolTip("Target print WIDTH in cm at 300 DPI.")
        bar.addWidget(self.w_cm)

        bar.addWidget(QLabel("×"))
        self.h_cm = QDoubleSpinBox()
        self.h_cm.setRange(0.1, 500.0)
        self.h_cm.setDecimals(1)
        self.h_cm.setSingleStep(0.5)
        self.h_cm.setValue(45.0)
        self.h_cm.setSuffix(" cm")
        self.h_cm.setToolTip("Target print HEIGHT in cm at 300 DPI.")
        bar.addWidget(self.h_cm)

        self.px_hint = QLabel("")
        self.px_hint.setStyleSheet(
            "color: #555; font-style: italic; padding-left: 10px;"
        )
        bar.addWidget(self.px_hint)
        self.w_cm.valueChanged.connect(self._refresh_px_hint)
        self.h_cm.valueChanged.connect(self._refresh_px_hint)
        self._refresh_px_hint()

        bar.addStretch(1)

        self.gen_btn = QPushButton("Generate")
        self.gen_btn.setToolTip(
            "Send image + prompt to Gemini and reframe to the "
            "tile size at 300 DPI. Preserves the original as-is; "
            "only extends the frame outward where the aspect changes."
        )
        self.gen_btn.clicked.connect(self.on_generate)
        self.gen_btn.setEnabled(False)
        bar.addWidget(self.gen_btn)

        self.save_btn = QPushButton("Save Result...")
        self.save_btn.clicked.connect(self.on_save)
        self.save_btn.setEnabled(False)
        bar.addWidget(self.save_btn)

        root.addLayout(bar)

        # ── Status line ────────────────────────────────────────────
        self.status = QLabel("Load an image to start.")
        self.status.setStyleSheet(
            "color: #333; padding: 2px 4px; background: #eef; border: 1px solid #ccd;"
        )
        root.addWidget(self.status)

        # ── Panes: Source | Result ─────────────────────────────────
        split = QSplitter(Qt.Horizontal)
        self.source_pane = ImagePane("Source")
        self.result_pane = ImagePane("Result (300 DPI)")
        split.addWidget(self.source_pane)
        split.addWidget(self.result_pane)
        split.setSizes([1, 1])
        root.addWidget(split, 1)

        self._worker: ResizeWorker | None = None
        self._prompt_template = self._load_prompt_template()
        self._gen_ctx: dict | None = None

    # ── UI helpers ────────────────────────────────────────────────
    def _load_prompt_template(self) -> str:
        try:
            return PROMPT_FILE.read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.warning(
                self, "Prompt missing",
                f"Could not read {PROMPT_FILE}:\n{e}\n\n"
                "A generic reframe fallback will be used.",
            )
            return (
                "Reframe this image to a {ASPECT_W}:{ASPECT_H} aspect "
                "ratio ({TARGET_W_CM}×{TARGET_H_CM} cm, "
                "{TARGET_W_PX}×{TARGET_H_PX} px). Preserve the original "
                "content exactly; extend only the edges where the new "
                "aspect requires it."
            )

    def _refresh_px_hint(self) -> None:
        w_cm = float(self.w_cm.value())
        h_cm = float(self.h_cm.value())
        w_px = cm_to_px_at_300dpi(w_cm)
        h_px = cm_to_px_at_300dpi(h_cm)
        ar_w, ar_h = readable_cm_aspect(w_cm, h_cm)
        self.px_hint.setText(
            f"= {w_px} × {h_px} px @ 300 DPI (aspect {ar_w}:{ar_h})"
        )

    # ── Slots ─────────────────────────────────────────────────────
    def on_load(self) -> None:
        start_dir = str(INPUT_DIR if INPUT_DIR.is_dir() else Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;"
            "All Files (*)",
        )
        if not path:
            return
        try:
            img = Image.open(path)
            img.load()          # detach from file handle
        except Exception as e:
            QMessageBox.critical(self, "Load Image",
                                 f"Failed to open {path}:\n{e}")
            return
        self.source_pane.set_image(
            img, caption=f"{Path(path).name} — {img.width}×{img.height} px",
        )
        self.result_pane.set_image(None, caption="Result (300 DPI)")
        self.save_btn.setEnabled(False)
        self.gen_btn.setEnabled(True)
        self.status.setText(
            f"Loaded {Path(path).name}. Set tile size and press Generate."
        )

    def on_generate(self) -> None:
        src = self.source_pane.image()
        if src is None:
            return
        w_cm = float(self.w_cm.value())
        h_cm = float(self.h_cm.value())
        w_px = cm_to_px_at_300dpi(w_cm)
        h_px = cm_to_px_at_300dpi(h_cm)
        ar_w, ar_h = readable_cm_aspect(w_cm, h_cm)
        aspect_str = closest_gemini_aspect(w_cm / max(1e-6, h_cm))
        # Pick the smallest Gemini output size that comfortably covers
        # the target (cheaper and faster than always requesting 4K).
        longest = max(w_px, h_px)
        image_size = ("1K" if longest <= 1100 else
                      "2K" if longest <= 2100 else "4K")

        prompt = (self._prompt_template
                  .replace("{TARGET_W_CM}", f"{w_cm:g}")
                  .replace("{TARGET_H_CM}", f"{h_cm:g}")
                  .replace("{TARGET_W_PX}", str(w_px))
                  .replace("{TARGET_H_PX}", str(h_px))
                  .replace("{ASPECT_W}",    str(ar_w))
                  .replace("{ASPECT_H}",    str(ar_h)))

        self.gen_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.status.setText(
            f"Generating {w_px}×{h_px} @ 300 DPI "
            f"(Gemini aspect {aspect_str}, {image_size}) — stage 1/2..."
        )

        # Remember what target + parameters are in flight so the
        # follow-up outpaint stage (if needed) can request the same
        # size / substitute the same physical dimensions.
        self._gen_ctx = {
            "w_cm":       w_cm,
            "h_cm":       h_cm,
            "w_px":       w_px,
            "h_px":       h_px,
            "ar_w":       ar_w,
            "ar_h":       ar_h,
            "aspect_str": aspect_str,
            "image_size": image_size,
        }

        self._worker = ResizeWorker(src, prompt, aspect_str, image_size)
        self._worker.progress.connect(
            lambda s: self.status.setText(s)
        )
        self._worker.finished_ok.connect(
            lambda data: self._on_generation_ok(data, w_px, h_px)
        )
        self._worker.failed.connect(self._on_generation_failed)
        self._worker.start()

    def _on_generation_ok(self, data: bytes, target_w: int,
                          target_h: int) -> None:
        try:
            api_img = Image.open(BytesIO(data))
            api_img.load()
        except Exception as e:
            self._on_generation_failed(f"Could not decode result: {e}")
            return
        try:
            fitted = fit_to_exact_size(api_img, target_w, target_h)
        except Exception as e:
            self._on_generation_failed(f"Fit failed: {e}")
            return
        self.result_pane.set_image(
            fitted,
            caption=(f"{fitted.width}×{fitted.height} px "
                     f"({self.w_cm.value():g}×{self.h_cm.value():g} cm @ 300 DPI)"),
        )
        self.save_btn.setEnabled(True)
        self.gen_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.status.setText(
            f"Done. Result: {fitted.width}×{fitted.height} px @ 300 DPI. "
            "Press Save Result to write PNG."
        )
        self._gen_ctx = None

    def _on_generation_failed(self, msg: str) -> None:
        self.gen_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.status.setText("Generation failed — see dialog.")
        QMessageBox.critical(self, "Generation failed", msg)

    def on_save(self) -> None:
        img = self.result_pane.image()
        if img is None:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        default_name = (
            f"resized_{int(self.w_cm.value())}x"
            f"{int(self.h_cm.value())}cm_300dpi.png"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result",
            str(OUTPUT_DIR / default_name),
            "PNG image (*.png);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            # PIL writes the pHYs chunk (physical pixel size) when
            # `dpi=(x, y)` is passed — receivers that respect the tag
            # (Photoshop, Illustrator, Inkscape) then open the file at
            # the correct physical print size instead of defaulting to
            # 72 DPI.
            (img if img.mode in ("RGB", "RGBA") else img.convert("RGB")
             ).save(path, "PNG", dpi=(DPI, DPI))
        except Exception as e:
            QMessageBox.critical(self, "Save failed",
                                 f"{type(e).__name__}: {e}")
            return
        self.status.setText(f"Saved: {path}")
        QMessageBox.information(
            self, "Saved",
            f"Wrote {img.width}×{img.height} PNG @ 300 DPI:\n{path}",
        )


def main() -> None:
    app = QApplication(sys.argv)
    win = ResizerWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
