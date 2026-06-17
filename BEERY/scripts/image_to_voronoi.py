"""Image → Voronoi Puzzle — generate a Voronoi-style tessellation of an
image via the Gemini API.

The output image is a flat-color, thin-pure-black-bordered Voronoi panel,
designed so it can be fed straight into the existing polygon-detection
pipelines:
  - IMAGE_TO_MOSAIC/scripts/vitrage_to_dxf.py  → DXF cut lines (skeleton)
  - IMAGE_TO_MOSAIC/scripts/mosaic_to_csv.py   → CSV polygons (per cell)

Subclasses PhotoEditor from IMAGE_TO_MOSAIC/scripts/photo_editor.py to
inherit the Gemini API integration, key handling, source/result panes,
preview, generate, and save flow.

Differences from PhotoEditor:
  - Window title: "Image → Voronoi Puzzle (BEERY)"
  - Default prompts folder: BEERY/prompts/
  - Auto-loads `voronoi_puzzle.txt` on startup
  - Hides the red-square, max-tiles-across, and background-fill controls
    (none apply to a pure Voronoi tessellation)
  - Drops the SUBJECT RENDERING RULES teeth clause (not relevant)

Usage:
  python BEERY/scripts/image_to_voronoi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `from photo_editor import …` even though we live in BEERY/scripts.
ROOT = Path(__file__).resolve().parent.parent       # BEERY/
PROJECT_ROOT = ROOT.parent                          # CLAUDE_MOSAIC1.0/
IMAGE_TO_MOSAIC_SCRIPTS = PROJECT_ROOT / "IMAGE_TO_MOSAIC" / "scripts"
if str(IMAGE_TO_MOSAIC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(IMAGE_TO_MOSAIC_SCRIPTS))

import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QLabel, QMessageBox, QPushButton,
)

from photo_editor import (
    PhotoEditor as _PhotoEditor,
    GenerationWorker,
    DEFAULT_MODEL,
    closest_aspect_ratio,
    load_api_key,
)
from mosaic_to_csv import (
    render_polygons as _render_polygons,
    write_csv as _write_csv,
    OUTPUT_DIR as _OUTPUT_DIR,
)

VORONOI_PROMPTS_DIR = ROOT / "prompts"
DEFAULT_PROMPT_FILE = VORONOI_PROMPTS_DIR / "voronoi_puzzle.txt"

# Toolbar QLabels that we hide alongside their hidden widgets. The labels
# weren't stored as attributes on the parent so we look them up by text.
_HIDDEN_LABELS = {"Red square (px):", "Max tiles across:"}


class VoronoiPuzzleEditor(_PhotoEditor):
    """PhotoEditor restricted to the Voronoi-puzzle workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image → Voronoi Puzzle (BEERY)")

        # Hide and zero out the controls that don't apply.
        self.square_size_input.setValue(0)
        self.square_size_input.setVisible(False)
        self.use_red_square_chk.setChecked(False)
        self.use_red_square_chk.setVisible(False)
        self.max_tiles_input.setValue(0)
        self.max_tiles_input.setVisible(False)
        self.fill_bg_chk.setChecked(False)
        self.fill_bg_chk.setVisible(False)
        # Always keep the source aspect ratio — there's no other behavior we
        # want here. Force the inherited checkbox on and hide it so the user
        # can't accidentally disable it.
        self.keep_aspect_chk.setChecked(True)
        self.keep_aspect_chk.setVisible(False)
        for label in self.findChildren(QLabel):
            if label.text() in _HIDDEN_LABELS:
                label.setVisible(False)

        # Add the Stone-palette checkbox next to the inherited Keep-aspect one.
        self.stone_palette_chk = QCheckBox("Stone palette")
        self.stone_palette_chk.setToolTip(
            "When checked, ask the model to shift the cell colours toward "
            "muted earthy mineral / stone tones (limestone, travertine, "
            "ochre, sienna, terracotta, umber, sage, slate, charcoal) — the "
            "image subject stays recognisable but the palette feels like "
            "natural stone rather than vivid photographic colour."
        )
        toolbar = self._toolbar_containing(self.keep_aspect_chk)
        if toolbar is not None:
            toolbar.addWidget(self.stone_palette_chk)

        # Main toolbar additions: Orange line (2nd-pass recolour) + Convert to CSV.
        self.orange_btn = QPushButton("Orange line")
        self.orange_btn.setToolTip(
            "Step 2 (after Generate): send the current black-lined result "
            "back to Gemini asking it to RECOLOUR every black dividing line "
            "to pure orange #FF6600, keeping everything else identical. "
            "Two-pass approach: black is easier for the AI to draw cleanly, "
            "then a focused recolour pass gives us the detection-friendly "
            "orange line for Convert to CSV."
        )
        self.orange_btn.clicked.connect(self.orange_line)

        self.no_line_btn = QPushButton("No line")
        self.no_line_btn.setToolTip(
            "Optional step: send the CURRENT RESULT image (not the source!) "
            "back to Gemini asking it to remove every dividing line and "
            "inpaint over the gaps. The model keeps everything else "
            "identical — same cells, same colours, same composition — just "
            "without the line network."
        )
        self.no_line_btn.clicked.connect(self.no_line)

        self.convert_csv_btn = QPushButton("Convert to CSV")
        self.convert_csv_btn.setToolTip(
            "Use the ORANGE dividing lines in the result image to detect "
            "each cell as a polygon (HSV threshold + connected components "
            "+ Douglas-Peucker), sample mean colour per cell, render the "
            "polygons on the LEFT pane, and save them in the project's "
            "CSV polygon format. Click Orange line first if the result "
            "still has black lines."
        )
        self.convert_csv_btn.clicked.connect(self.convert_to_csv)
        main_toolbar = self._toolbar_containing(self.save_res_btn)
        if main_toolbar is not None:
            main_toolbar.addWidget(self.orange_btn)
            main_toolbar.addWidget(self.no_line_btn)
            main_toolbar.addWidget(self.convert_csv_btn)

        self._auto_load_default_prompt()

    def _toolbar_containing(self, widget):
        """Return the QHBoxLayout (toolbar row) that holds ``widget``, by
        walking the central widget's main layout. Used to slot new controls
        into the inherited PhotoEditor toolbar."""
        central = self.centralWidget()
        if central is None or central.layout() is None:
            return None
        root_layout = central.layout()
        for i in range(root_layout.count()):
            sub = root_layout.itemAt(i).layout()
            if sub is None:
                continue
            for j in range(sub.count()):
                if sub.itemAt(j).widget() is widget:
                    return sub
        return None

    def _auto_load_default_prompt(self) -> None:
        """Pre-select BEERY/prompts/voronoi_puzzle.txt so Generate is ready."""
        if not DEFAULT_PROMPT_FILE.is_file():
            return
        try:
            text = DEFAULT_PROMPT_FILE.read_text(encoding="utf-8")
        except Exception:
            return
        if not text.strip():
            return
        self.current_prompt_path = DEFAULT_PROMPT_FILE
        self.current_prompt_text = text
        self.prompt_label.setText(
            f"Prompt: {DEFAULT_PROMPT_FILE.name}  ({len(text)} chars)"
        )
        self.statusBar().showMessage(f"Prompt loaded: {DEFAULT_PROMPT_FILE.name}")
        self._update_button_states()

    def choose_prompt(self) -> None:
        """Like the parent, but the dialog opens in BEERY/prompts/."""
        if VORONOI_PROMPTS_DIR.exists():
            start_dir = str(VORONOI_PROMPTS_DIR)
        else:
            start_dir = ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Voronoi prompt", start_dir,
            "Text files (*.txt);;All files (*.*)",
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Read failed", f"{type(e).__name__}: {e}")
            return
        if not text.strip():
            QMessageBox.warning(self, "Empty prompt", "The selected file is empty.")
            return
        self.current_prompt_path = Path(path)
        self.current_prompt_text = text
        self.prompt_label.setText(f"Prompt: {Path(path).name}  ({len(text)} chars)")
        self.statusBar().showMessage(f"Prompt loaded: {Path(path).name}")
        self._update_button_states()

    def _inject_size_overrides(self, prompt_text: str) -> str:
        """Voronoi version: aspect-rewrite + optional stone-palette add-on.
        The first generation pass always uses the prompt's pure-black thin
        lines; recolouring to orange is a separate second pass via the
        Orange line button. Skip the background-fill block and the SUBJECT
        RENDERING RULES teeth clause (irrelevant for Voronoi)."""
        text = prompt_text
        if self.keep_aspect_chk.isChecked():
            text = self._rewrite_square_output_clause(text)
        if getattr(self, "stone_palette_chk", None) is not None and \
                self.stone_palette_chk.isChecked():
            text = text + self._stone_palette_addendum()
        return text

    @staticmethod
    def _stone_palette_addendum() -> str:
        """Prompt clause appended when the Stone palette checkbox is on.
        Shifts the ceramic glaze colours toward muted natural mineral tones
        while keeping the subject recognisable."""
        return (
            "\n\n"
            "==== STONE-COLOUR PALETTE ====\n"
            "Shift the colour palette of every ceramic cell toward muted, "
            "earthy MINERAL / STONE tones: cream limestone, bone white, "
            "warm travertine, ochre, sienna, oxblood, brick red, terracotta, "
            "umber, olive, sage, slate blue-grey, charcoal, jet black. The "
            "image subject and the layout MUST remain recognisable — keep "
            "relative brightness, shading, and the positions of facial / "
            "subject features intact; just remap the hues into the stone "
            "palette as if the ceramic glaze were tinted with natural "
            "mineral pigments.\n"
            "Hard constraints:\n"
            "- NO neon, NO saturated jewel tones (no cobalt, no ruby, no "
            "emerald, no royal purple), NO synthetic-looking hues.\n"
            "- NO black borders or grout — those stay as already specified "
            "(pure black, thin, 2-4 px). The stone palette applies ONLY to "
            "the cell interiors.\n"
            "- The picture must still read as the same photograph, just "
            "rendered in stone tones."
        )


    def _on_worker_done(self) -> None:
        """Reset our extra-button labels after the API worker finishes (parent
        already resets generate_btn / background_btn / expand_btn)."""
        super()._on_worker_done()
        if hasattr(self, "orange_btn"):
            self.orange_btn.setText("Orange line")
        if hasattr(self, "no_line_btn"):
            self.no_line_btn.setText("No line")

    def _on_generated(self, img_bytes: bytes) -> None:
        """Override: Gemini can only emit images at its fixed set of
        aspect ratios and resolutions. After it returns, resize the result
        to EXACTLY match the source image's pixel dimensions so every pass
        (Generate / Orange line / No line) lands at the loaded image's
        resolution regardless of what the model produced."""
        from io import BytesIO
        from PIL import Image as _PILImage
        try:
            img = _PILImage.open(BytesIO(img_bytes))
            img.load()
        except Exception as e:
            QMessageBox.critical(self, "Bad response",
                                 f"Could not decode image: {e}")
            return
        # Resize to the source image's exact dimensions (LANCZOS for quality).
        if self.source_pane.pil_image is not None:
            target_w = self.source_pane.pil_image.width
            target_h = self.source_pane.pil_image.height
            if (img.width, img.height) != (target_w, target_h):
                img = img.resize((target_w, target_h), _PILImage.LANCZOS)
        self.result_pane.set_pil_image(
            img, f"Generated  |  {img.width} × {img.height} px (matched source)",
        )
        self.statusBar().showMessage(
            f"Generated and resized to {img.width} × {img.height} px "
            f"(source dimensions).",
        )

    # ----- second-pass recolour: black lines → orange lines ---------------

    def orange_line(self) -> None:
        """Send the current black-lined result back to Gemini, asking it to
        recolour every black dividing line to pure orange (#FF6600) and
        leave everything else unchanged. The new image lands in the result
        pane (replaces the black-lined version) so Convert to CSV can then
        run the HSV detection."""
        if self.result_pane.pil_image is None:
            QMessageBox.information(
                self, "No result image",
                "Run Generate first to produce a Voronoi panel with black "
                "dividing lines. Then click Orange line to recolour them.",
            )
            return
        if load_api_key() is None:
            QMessageBox.critical(
                self, "API key missing",
                "No Gemini API key found.\n\n"
                "Set the GEMINI_API_KEY env var, or click 'Choose Key "
                "File...' to point at a key file.",
            )
            return

        src = self.result_pane.pil_image.convert("RGB")
        prompt = (
            "The supplied image is a Voronoi panel with thin pure-BLACK "
            "lines dividing the cells.\n\n"
            "YOUR ONLY TASK: replace every black dividing line with PURE "
            "ORANGE (#FF6600). Do nothing else.\n\n"
            "Rules:\n"
            "- EVERY pixel that is currently black (#000000), and is part "
            "of a dividing line, must become PURE ORANGE (#FF6600).\n"
            "- EVERY OTHER PIXEL must remain IDENTICAL to the input — "
            "same cell content, same colours, same composition, same "
            "framing, same dimensions, same lighting. Do not redraw, "
            "restyle, recolour, or reshape any cell.\n"
            "- Line WIDTH, SHARPNESS, and CONTINUITY stay the same — "
            "only the colour swaps from black to orange.\n"
            "- The orange must be SOLID, UNIFORM, SHARP-EDGED #FF6600. No "
            "anti-aliasing, no gradient, no transparency, no glow.\n"
            "- DO NOT introduce orange anywhere except where there was "
            "black line before. Existing orange-ish content in the cells "
            "(skin tones, warm photo areas) stays exactly as in the input.\n"
            "Output: a 4K (or matching aspect) image, the same picture as "
            "the input but with the black lines replaced by orange lines."
        )

        aspect = closest_aspect_ratio(src.width, src.height)
        self.worker = GenerationWorker(
            src, prompt, DEFAULT_MODEL, aspect_ratio=aspect,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_generated)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()

        self.statusBar().showMessage("Orange-line pass: recolouring black → orange...")
        self.orange_btn.setText("Orange line (running)")
        self.generate_btn.setText("Generating...")
        self._update_button_states()

    # ----- second-pass: remove the dividing lines -------------------------

    def no_line(self) -> None:
        """Send the CURRENT RESULT image (which already has dividing lines)
        back to Gemini, asking it to remove every dividing line and inpaint
        seamlessly over the gaps. Input is the result pane, NOT the original
        source — so the picture's framing / cell colours / ceramic feel from
        the generation step are preserved exactly, only the line network is
        removed."""
        if self.result_pane.pil_image is None:
            QMessageBox.information(
                self, "No result image",
                "Run Generate first to produce a Voronoi panel. Then click "
                "No line to ask the model to inpaint over the dividing lines.",
            )
            return
        if load_api_key() is None:
            QMessageBox.critical(
                self, "API key missing",
                "No Gemini API key found.\n\n"
                "Set the GEMINI_API_KEY env var, or click 'Choose Key "
                "File...' to point at a key file.",
            )
            return

        src = self.result_pane.pil_image.convert("RGB")
        prompt = (
            "The supplied image is a Voronoi panel with a network of thin "
            "dividing lines (black, orange, or red) marking the shape "
            "boundaries of the picture.\n\n"
            "YOUR ONLY TASK: REMOVE EVERY DIVIDING LINE from the image. "
            "Inpaint seamlessly across each gap so the picture looks "
            "continuous, as if the lines were never there. Do nothing else.\n"
            "\n"
            "Rules:\n"
            "- EVERY pixel that is currently part of a dividing line (any "
            "line colour: black, orange, red, or otherwise) must be REPLACED "
            "by the colour and content of the surrounding cell, blending "
            "smoothly with adjacent cells so the join is invisible.\n"
            "- EVERY OTHER PIXEL must remain IDENTICAL to the input — same "
            "cells, same colours, same composition, same framing, same "
            "lighting, same ceramic / stone surface character. Do not "
            "restyle, recolour, redraw, or reshape any cell.\n"
            "- The output is the input image with the line network erased "
            "and inpainted — nothing added, nothing else changed.\n"
            "- No new lines, no new edges, no new shapes, no new colours.\n"
            "Output: a 4K (or matching aspect) image showing the same "
            "picture as the input but with the dividing lines invisibly "
            "removed."
        )

        aspect = closest_aspect_ratio(src.width, src.height)
        self.worker = GenerationWorker(
            src, prompt, DEFAULT_MODEL, aspect_ratio=aspect,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_generated)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()

        self.statusBar().showMessage("No-line pass: erasing dividing lines...")
        self.no_line_btn.setText("No line (running)")
        self.generate_btn.setText("Generating...")
        self._update_button_states()

    # ----- orange-line → CSV polygons -------------------------------------

    def convert_to_csv(self) -> None:
        """Extract polygons from the orange dividing lines in the result image,
        render them on the LEFT pane, and save them as CSV.

        Pipeline:
          1. HSV-threshold orange pixels in the result image → grout mask.
          2. Invert to tile mask (white = cell interior, black = orange grout).
          3. mosaic_to_csv.detect_tiles → list of (polygon, mean_rgb), sampling
             mean colour from the original (non-binarised) result image.
          4. mosaic_to_csv.render_polygons → coloured polygon preview on left.
          5. Save dialog → write CSV via mosaic_to_csv.write_csv (per-cell colour).
        """
        if self.result_pane.pil_image is None:
            QMessageBox.information(
                self, "No result image",
                "Generate a Voronoi panel first (check 'Orange dividing "
                "lines' before Generate so the AI uses orange #FF6600 lines).",
            )
            return

        pil = self.result_pane.pil_image.convert("RGB")
        result_rgb = np.array(pil)
        h, w = result_rgb.shape[:2]

        # TIGHT HSV threshold around #FF6600 — only SOLID, saturated orange
        # pixels. In OpenCV HSV (H 0-179) #FF6600 is H≈12, S=255, V=255. The
        # high-S / high-V floor rejects anti-aliased line edges, orange-tinted
        # skin tones, warm photo content, etc.
        hsv = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2HSV)
        orange_mask = cv2.inRange(
            hsv, (8, 200, 180), (16, 255, 255),
        )
        if int((orange_mask > 0).sum()) == 0:
            QMessageBox.warning(
                self, "No orange lines",
                "Couldn't find any solid #FF6600 pixels in the result "
                "image. Make sure 'Orange dividing lines' was checked when "
                "you ran Generate.",
            )
            return

        # Bridge 1-px gaps in the orange line so the polygon boundary is closed,
        # but use a small kernel so we don't eat into adjacent cells.
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kern)

        # Tile mask: 255 inside cells, 0 on the orange line.
        tile_mask = (255 - orange_mask).astype(np.uint8)

        # Pad with 0 (orange grout in mask space) so cells touching the image
        # edge become bounded by the padded ring — otherwise the outermost
        # cell merges with the "outside" component and gets dropped.
        PAD = 5
        tile_padded = cv2.copyMakeBorder(
            tile_mask, PAD, PAD, PAD, PAD,
            cv2.BORDER_CONSTANT, value=0,
        )
        result_padded = cv2.copyMakeBorder(
            result_rgb, PAD, PAD, PAD, PAD,
            cv2.BORDER_CONSTANT, value=(255, 102, 0),
        )

        self.statusBar().showMessage("Detecting polygons from orange lines...")
        QApplication.processEvents()

        # Connected components directly on the clean binary mask — NO adaptive
        # threshold pass (which would smear / fragment an already-binary input).
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            tile_padded, connectivity=4,
        )

        MIN_AREA = 200
        EPS_RATIO = 0.005
        ph, pw = tile_padded.shape[:2]
        tiles = []
        for lid in range(1, num_labels):
            area = int(stats[lid, cv2.CC_STAT_AREA])
            if area < MIN_AREA:
                continue
            x0 = int(stats[lid, cv2.CC_STAT_LEFT])
            y0 = int(stats[lid, cv2.CC_STAT_TOP])
            ww = int(stats[lid, cv2.CC_STAT_WIDTH])
            hh = int(stats[lid, cv2.CC_STAT_HEIGHT])
            # Drop the outer ring (any component reaching the padded image border).
            if x0 == 0 or y0 == 0 or x0 + ww == pw or y0 + hh == ph:
                continue
            sub_mask = (labels[y0:y0 + hh, x0:x0 + ww] == lid).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 3:
                continue
            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = max(0.5, EPS_RATIO * perimeter)
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            if len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2).astype(np.float64)
            # Translate from padded → original-image coordinates.
            pts[:, 0] += x0 - PAD
            pts[:, 1] += y0 - PAD
            # Mean cell colour from the original (non-padded) result image.
            cell_pixels = result_padded[labels == lid]
            mean_rgb = (cell_pixels.mean(axis=0) / 255.0).tolist()
            tiles.append((pts, tuple(mean_rgb)))

        if not tiles:
            QMessageBox.warning(
                self, "No polygons",
                "Found orange pixels but extracted zero polygons. The orange "
                "lines may be broken or too thin — try regenerating with a "
                "cleaner orange border.",
            )
            return

        # Render the polygons (filled with mean colour + 1-px outline) and
        # show them on the LEFT pane, replacing the source preview.
        preview = _render_polygons(tiles, w, h)
        self.source_pane.set_pil_image(
            preview,
            f"Detected polygons: {len(tiles)} cells  |  {w} × {h} px",
        )
        self.statusBar().showMessage(
            f"Detected {len(tiles)} polygons — ready to save as CSV.",
        )

        # Save dialog → write CSV via mosaic_to_csv format (per-cell colour).
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = (
            self.current_source_path.stem if self.current_source_path
            else "voronoi"
        )
        default_name = f"{stem}_voronoi_{len(tiles)}polys.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV polygons", str(_OUTPUT_DIR / default_name),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        out_path = Path(path)
        if out_path.suffix.lower() != ".csv":
            out_path = out_path.with_suffix(".csv")
        try:
            _write_csv(tiles, out_path)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        self.statusBar().showMessage(
            f"Saved {len(tiles)} polygons → {out_path.name}",
        )
        QMessageBox.information(
            self, "Saved",
            f"Saved {len(tiles)} polygons (with per-cell mean colours) to:\n"
            f"{out_path}",
        )


def main() -> None:
    app = QApplication(sys.argv)
    w = VoronoiPuzzleEditor()
    w.resize(1500, 900)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
