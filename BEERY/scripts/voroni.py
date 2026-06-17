"""Voroni — two-prompt image transformer.

After loading an image:
- "Paint" sends the SOURCE image to Gemini asking for a realistic hand-
  painted rendering on a continuous ceramic surface — no plate edges
  visible, no lighting highlights. (BEERY/prompts/paint_ceramic.txt)
- "Add lines" sends the CURRENT RESULT (or the source if no result yet)
  to Gemini asking it to overlay a thin pure-orange Voronoi line network,
  keeping the image otherwise IDENTICAL.
  (BEERY/prompts/add_voronoi_lines.txt)

Both passes preserve the source's exact pixel dimensions via a post-API
LANCZOS resize (Gemini only emits a fixed set of aspect ratios at its own
resolutions; the resize is the source of truth).

Usage:
  python BEERY/scripts/voroni.py
"""
from __future__ import annotations

import pickle
import sys
from io import BytesIO
from pathlib import Path

# Reach IMAGE_TO_MOSAIC/scripts (sibling tree) for PhotoEditor.
ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
IMAGE_TO_MOSAIC_SCRIPTS = PROJECT_ROOT / "IMAGE_TO_MOSAIC" / "scripts"
if str(IMAGE_TO_MOSAIC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(IMAGE_TO_MOSAIC_SCRIPTS))

import cv2
import numpy as np
from PIL import Image as _PILImage
from skimage.morphology import skeletonize as _skeletonize
from PyQt5.QtCore import QEvent, QPoint, QRect, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QLabel, QMessageBox, QPushButton,
    QRubberBand, QSpinBox,
)

from photo_editor import (
    PhotoEditor as _PhotoEditor,
    GenerationWorker,
    DEFAULT_MODEL,
    OUTPUT_DIR as _PHOTO_OUTPUT_DIR,
    closest_aspect_ratio,
    load_api_key,
)

PROMPTS_DIR = ROOT / "prompts"
PAINT_PROMPT_FILE = PROMPTS_DIR / "paint_ceramic.txt"
# Add-lines prompt variants — user picks one from the toolbar combobox
# before clicking "Add lines". The first entry is the default.
LINES_PROMPT_OPTIONS: list[tuple[str, Path]] = [
    ("Open",       PROMPTS_DIR / "add_voronoi_lines.txt"),
    ("Closed",     PROMPTS_DIR / "add_voronoi_lines_closed.txt"),
    ("Even-area",  PROMPTS_DIR / "add_voronoi_lines_even_area.txt"),
]

# Inherited toolbar widgets we want gone — they don't apply to this workflow.
_HIDDEN_BUTTON_TEXTS = {
    "Background...", "Expand to Circle", "Choose Prompt...",
    "Preview Prompt", "Generate →",
}
_HIDDEN_LABEL_TEXTS = {"Red square (px):", "Max tiles across:"}


class VoroniEditor(_PhotoEditor):
    """PhotoEditor restricted to two operations: Paint and Add lines."""

    # Multiplier applied to the base image's pixel values in the displayed
    # composite, so the orange lines stand out. 0.35 = 35% of original
    # brightness. The orange lines and the 3-px frame are NOT dimmed.
    DARKEN_FACTOR = 0.35

    # Fixed eraser radius (in source-image pixels). The brush_spin only
    # affects the Draw stroke width; Erase always uses this circular brush.
    ERASE_BRUSH_RADIUS = 10

    def __init__(self) -> None:
        # Set subclass state BEFORE super().__init__(): the parent's
        # PhotoEditor.__init__ calls self._update_button_states() at the
        # end of its setup, and Python's MRO dispatches that to OUR
        # override, which reads self._raw_line_mask. If we set it after
        # super(), the first call sees no attribute → AttributeError and
        # the window never finishes constructing.
        self._current_pass: str | None = None
        self._add_lines_base: _PILImage.Image | None = None
        self._raw_line_mask: np.ndarray | None = None
        # Draw / erase editing state.
        self._editing_dragging: bool = False
        self._last_edit_pt: tuple[int, int] | None = None
        # Zoom-rectangle (drag-to-zoom) state.
        self._zoom_dragging: bool = False
        self._zoom_origin: QPoint | None = None
        self._rubber_band: QRubberBand | None = None

        super().__init__()
        self.setWindowTitle("Voroni — Paint + Voronoi lines")

        # Hide all the inherited inputs / checkboxes we don't need.
        self.square_size_input.setValue(0)
        self.square_size_input.setVisible(False)
        self.use_red_square_chk.setChecked(False)
        self.use_red_square_chk.setVisible(False)
        self.max_tiles_input.setValue(0)
        self.max_tiles_input.setVisible(False)
        self.fill_bg_chk.setChecked(False)
        self.fill_bg_chk.setVisible(False)
        # Aspect-ratio preservation is forced on — every pass uses the
        # closest-aspect-ratio API match, then we resize to source dims.
        self.keep_aspect_chk.setChecked(True)
        self.keep_aspect_chk.setVisible(False)

        for label in self.findChildren(QLabel):
            if label.text() in _HIDDEN_LABEL_TEXTS:
                label.setVisible(False)
            elif label is getattr(self, "prompt_label", None):
                label.setVisible(False)

        # Hide inherited buttons that don't fit this two-action workflow.
        # The new Paint / Add lines buttons replace Generate.
        for btn in self.findChildren(QPushButton):
            if btn.text() in _HIDDEN_BUTTON_TEXTS:
                btn.setVisible(False)

        # New buttons.
        self.paint_btn = QPushButton("Paint")
        self.paint_btn.setToolTip(
            "Send the SOURCE image to Gemini and ask for a realistic hand-"
            "painted rendering on a continuous ceramic surface. No plate "
            "edges, no lighting highlights, same aspect ratio. Replaces "
            "whatever is currently in the result pane."
        )
        self.paint_btn.clicked.connect(self.run_paint)

        self.add_lines_btn = QPushButton("Add lines")
        self.add_lines_btn.setToolTip(
            "Send the CURRENT result image (or the source image if no "
            "result yet) to Gemini and ask it to add thin orange Voronoi "
            "lines on top, keeping the rest of the image IDENTICAL. Pair "
            "with Paint to add lines on top of the painted version, or "
            "use alone to add lines directly to the source image. Pick "
            "the prompt variant in the dropdown to the right."
        )
        self.add_lines_btn.clicked.connect(self.run_add_lines)

        # Prompt-variant selector for Add lines. The default (index 0) is
        # the original "Open" prompt; "Closed" adds strict no-loose-ends
        # rules so every endpoint terminates at another line or the frame.
        self.lines_prompt_combo = QComboBox()
        for label, path in LINES_PROMPT_OPTIONS:
            self.lines_prompt_combo.addItem(label, str(path))
        self.lines_prompt_combo.setToolTip(
            "Which 'Add lines' prompt to send to Gemini:\n"
            "  Open      — the original prompt (allows loose ends).\n"
            "  Closed    — strict prompt that requires every line endpoint "
            "to terminate at another line (T/X junction) or the image "
            "frame; the network must be a fully-closed planar graph.\n"
            "  Even-area — asks for a Voronoi tessellation where every "
            "region has roughly the same area (max 2× ratio between "
            "largest and smallest). Eyes / eyebrows remain single "
            "regions; features are otherwise de-emphasised."
        )

        # Stretch input: after Add lines lands, lets the user horizontally
        # stretch the orange line layer (anchored at the LEFT edge) to align
        # it with the source's features. 100 % = no stretch. The QSpinBox
        # accepts direct keyboard entry of the percentage.
        self.stretch_label = QLabel("Stretch:")
        self.stretch_spin = QSpinBox()
        self.stretch_spin.setRange(50, 200)
        self.stretch_spin.setValue(100)
        self.stretch_spin.setSuffix(" %")
        self.stretch_spin.setSingleStep(1)
        self.stretch_spin.setToolTip(
            "Horizontally stretch the orange line layer to the right "
            "(anchored at the left edge of the image) so the lines align "
            "with the source's features. 100 % = no stretch. Re-composites "
            "live as you change the value."
        )
        self.stretch_spin.valueChanged.connect(self._on_stretch_changed)
        # Hidden until a line composite exists.
        self.stretch_label.setVisible(False)
        self.stretch_spin.setVisible(False)

        # Line width: re-skeletonises the detected orange mask and re-renders
        # it at the chosen uniform pixel width. Lets the user dial the line
        # weight up / down after Add lines.
        self.line_width_label = QLabel("Line:")
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 20)
        self.line_width_spin.setValue(2)
        self.line_width_spin.setSuffix(" px")
        self.line_width_spin.setToolTip(
            "Width of the detected orange line network, in source-image "
            "pixels. Changing this re-skeletonises the current mask and "
            "re-renders it at the chosen uniform width."
        )
        self.line_width_spin.valueChanged.connect(self._on_line_width_changed)
        self.line_width_label.setVisible(False)
        self.line_width_spin.setVisible(False)

        # Draw / Erase tools: click + drag on the result pane to paint or
        # erase orange line pixels. Operate on the cached line MASK only —
        # the underlying base image is never modified.
        self.draw_btn = QPushButton("Draw")
        self.draw_btn.setCheckable(True)
        self.draw_btn.setToolTip(
            "Toggle Draw mode. Click + drag on the result pane to add "
            "orange line pixels at the chosen brush width. The base image "
            "stays untouched (lines are painted onto the line mask only)."
        )
        self.draw_btn.clicked.connect(self._on_draw_toggled)

        self.erase_btn = QPushButton("Erase")
        self.erase_btn.setCheckable(True)
        self.erase_btn.setToolTip(
            "Toggle Erase mode. Click + drag on the result pane to remove "
            "orange line pixels with a CIRCULAR eraser of fixed radius "
            f"{self.ERASE_BRUSH_RADIUS} px. The base image stays untouched."
        )
        self.erase_btn.clicked.connect(self._on_erase_toggled)

        self.brush_label = QLabel("Brush:")
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 30)
        self.brush_spin.setValue(3)
        self.brush_spin.setSuffix(" px")
        self.brush_spin.setToolTip(
            "DRAW stroke thickness, in pixels. The eraser uses a fixed "
            f"{self.ERASE_BRUSH_RADIUS}-px-radius circular brush and "
            "ignores this value."
        )

        # Zoom-area toggle + Fit-to-viewport. Mouse wheel still zooms
        # centered on the cursor; these add explicit controls.
        self.zoom_area_btn = QPushButton("Zoom Area")
        self.zoom_area_btn.setCheckable(True)
        self.zoom_area_btn.setToolTip(
            "Toggle Zoom Area mode. Click + drag a rectangle on the result "
            "pane and release — the view will zoom to fit that rectangle. "
            "(The mouse wheel always zooms centered on the cursor too.)"
        )
        self.zoom_area_btn.clicked.connect(self._on_zoom_area_toggled)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setToolTip(
            "Reset the result pane's zoom so the whole image fits in view."
        )
        self.fit_btn.clicked.connect(self._fit_to_viewport)

        # Hidden until Add lines completes (need a line mask to edit /
        # zoom into).
        for w in (self.draw_btn, self.erase_btn,
                  self.brush_label, self.brush_spin,
                  self.zoom_area_btn, self.fit_btn):
            w.setVisible(False)

        # Slot Paint + Add lines + Stretch + Draw/Erase widgets into the
        # main toolbar right after Load Image.
        target_layout = self._toolbar_containing(self.load_btn)
        if target_layout is not None:
            insert_at = self._index_of_widget(target_layout, self.load_btn) + 1
            for widget in (
                self.paint_btn,
                self.add_lines_btn,
                self.lines_prompt_combo,
                self.stretch_label,
                self.stretch_spin,
                self.line_width_label,
                self.line_width_spin,
                self.draw_btn,
                self.erase_btn,
                self.brush_label,
                self.brush_spin,
                self.zoom_area_btn,
                self.fit_btn,
            ):
                target_layout.insertWidget(insert_at, widget)
                insert_at += 1

        # Mouse events on the result pane → Draw / Erase handlers. Filter
        # is always installed; it's a no-op when neither mode is checked.
        self.result_pane.image_label.installEventFilter(self)
        self.result_pane.scroll.viewport().installEventFilter(self)

        # Save Lines: exports just the orange line layer (lines + frame on
        # solid black) at the source image's exact resolution. Placed right
        # after the inherited Save Result button.
        self.save_lines_btn = QPushButton("Save Lines")
        self.save_lines_btn.setToolTip(
            "Save a Voroni project file (.voroni) containing the bright "
            "base image + the orange line mask + the current editing "
            "parameters (frame width, stretch %, line width). Future "
            "polygon-extraction GUI loads this file and rebuilds the "
            "scene. Available after Add lines."
        )
        self.save_lines_btn.clicked.connect(self.save_lines)
        save_res_toolbar = self._toolbar_containing(self.save_res_btn)
        if save_res_toolbar is not None:
            insert_after = self._index_of_widget(
                save_res_toolbar, self.save_res_btn,
            ) + 1
            save_res_toolbar.insertWidget(insert_after, self.save_lines_btn)

        self._update_button_states()

    # ----- helpers --------------------------------------------------------

    def _toolbar_containing(self, widget):
        central = self.centralWidget()
        if central is None or central.layout() is None:
            return None
        root_layout = central.layout()
        for i in range(root_layout.count()):
            sub = root_layout.itemAt(i).layout()
            if sub is None:
                continue
            if self._index_of_widget(sub, widget) >= 0:
                return sub
        return None

    @staticmethod
    def _index_of_widget(layout, widget) -> int:
        for j in range(layout.count()):
            if layout.itemAt(j).widget() is widget:
                return j
        return -1

    def _load_prompt(self, path: Path) -> str | None:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(
                self, "Prompt file missing",
                f"Could not read prompt file:\n{path}\n\n"
                f"{type(e).__name__}: {e}",
            )
            return None
        if not text.strip():
            QMessageBox.critical(
                self, "Prompt file empty", f"{path} is empty.",
            )
            return None
        return text

    def _start_pass(self, src_image, prompt: str, label_btn,
                    running_text: str, status_text: str) -> None:
        """Common helper: spin up the API worker with the chosen src + prompt."""
        if load_api_key() is None:
            QMessageBox.critical(
                self, "API key missing",
                "No Gemini API key found.\n\n"
                "Set the GEMINI_API_KEY env var, or click 'Choose Key "
                "File...' to point at a key file.",
            )
            return
        src = src_image.convert("RGB")
        aspect = closest_aspect_ratio(src.width, src.height)
        self.worker = GenerationWorker(
            src, prompt, DEFAULT_MODEL, aspect_ratio=aspect,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_generated)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()
        self.statusBar().showMessage(status_text)
        label_btn.setText(running_text)
        self._update_button_states()

    # ----- actions --------------------------------------------------------

    def run_paint(self) -> None:
        """Paint pass — always operates on the source image."""
        if self.source_pane.pil_image is None:
            QMessageBox.information(
                self, "No source image",
                "Load an image first (Load Image...) before running Paint.",
            )
            return
        prompt = self._load_prompt(PAINT_PROMPT_FILE)
        if prompt is None:
            return
        self._current_pass = "paint"
        self._add_lines_base = None
        # The new paint result invalidates any previous line composite, so
        # hide the stretch UI and drop the cached mask.
        self._reset_stretch_ui()
        self._start_pass(
            self.source_pane.pil_image, prompt, self.paint_btn,
            "Paint (running)", "Painting pass running...",
        )

    def run_add_lines(self) -> None:
        """Add-lines pass — the API output is treated as a LINE MASK (the
        prompt asks for orange-on-black), and we composite the orange pixels
        onto a copy of the local base image (current result if any, else the
        source). This guarantees the underlying picture is byte-for-byte
        preserved — only the orange pixels change."""
        base = (
            self.result_pane.pil_image
            if self.result_pane.pil_image is not None
            else self.source_pane.pil_image
        )
        if base is None:
            QMessageBox.information(
                self, "No image",
                "Load an image first (Load Image...) before running Add lines.",
            )
            return
        # Resolve the prompt path from the toolbar combo. Falls back to
        # the first option if the combo somehow has no current data.
        prompt_path_str = self.lines_prompt_combo.currentData()
        if not prompt_path_str:
            prompt_path_str = str(LINES_PROMPT_OPTIONS[0][1])
        prompt_path = Path(prompt_path_str)
        prompt = self._load_prompt(prompt_path)
        if prompt is None:
            return
        # Stash the base for compositing in _on_generated. The image we send
        # to the API is the SAME image — it acts as a reference so the model
        # knows where the features are, even though the prompt tells it to
        # produce only the line network on black.
        self._current_pass = "add_lines"
        self._add_lines_base = base.copy()
        variant_label = self.lines_prompt_combo.currentText()
        self._start_pass(
            base, prompt, self.add_lines_btn,
            "Add lines (running)",
            f"Add-lines pass running ({variant_label}: orange-on-black)...",
        )

    # ----- overrides for resize + button-label reset ----------------------

    def _on_generated(self, img_bytes: bytes) -> None:
        """Two paths depending on which pass started this worker:

        Paint pass — the API result IS the new picture; just resize to the
        source's pixel dimensions and show it.

        Add-lines pass — the API result is treated as an orange-on-black
        line MASK. We HSV-threshold its orange pixels and stamp those pixels
        onto a copy of the local base image (untouched apart from the line
        pixels). This guarantees byte-for-byte preservation of the
        underlying picture — only the orange line pixels change.
        """
        try:
            img = _PILImage.open(BytesIO(img_bytes))
            img.load()
        except Exception as e:
            QMessageBox.critical(
                self, "Bad response", f"Could not decode image: {e}",
            )
            return

        # Resize the API output to the source's exact pixel dimensions so the
        # line positions register one-to-one with the local base image.
        if self.source_pane.pil_image is not None:
            tw = self.source_pane.pil_image.width
            th = self.source_pane.pil_image.height
            if (img.width, img.height) != (tw, th):
                img = img.resize((tw, th), _PILImage.LANCZOS)

        if self._current_pass == "add_lines" and self._add_lines_base is not None:
            self._composite_lines_onto_base(img)
            return

        # Default (paint pass or anything else): show the API result directly.
        self.result_pane.set_pil_image(
            img,
            f"Generated  |  {img.width} × {img.height} px (matched source)",
        )
        self.statusBar().showMessage(
            f"Generated and resized to {img.width} × {img.height} px "
            f"(source dimensions).",
        )

    def _composite_lines_onto_base(self, api_img: _PILImage.Image) -> None:
        """Treat ``api_img`` as a line mask (orange on black) and paint its
        orange pixels onto a copy of ``self._add_lines_base`` at #FF6600.
        Every other pixel in the base is preserved EXACTLY.

        Stores the raw line mask in ``self._raw_line_mask`` so the stretch
        slider can re-composite on demand without re-running the API."""
        base = self._add_lines_base
        assert base is not None
        api_rgb = np.array(api_img.convert("RGB"), dtype=np.uint8)
        base_h, base_w = base.height, base.width
        if api_rgb.shape[:2] != (base_h, base_w):
            # Defensive: shouldn't happen because we resized above.
            api_pil = _PILImage.fromarray(api_rgb, "RGB").resize(
                (base_w, base_h), _PILImage.LANCZOS,
            )
            api_rgb = np.array(api_pil, dtype=np.uint8)

        # HSV threshold for the orange line. Use a loose floor so any pixel
        # that "reads orange" in the API output is treated as a line pixel,
        # even with a bit of anti-aliasing in the response.
        hsv = cv2.cvtColor(api_rgb, cv2.COLOR_RGB2HSV)
        orange_mask = cv2.inRange(hsv, (5, 120, 120), (22, 255, 255))
        n_line_px = int((orange_mask > 0).sum())
        if n_line_px == 0:
            QMessageBox.warning(
                self, "No lines detected",
                "The API didn't return any clearly-orange pixels. Try "
                "running 'Add lines' again — the model may have produced an "
                "image instead of the requested line mask.",
            )
            self._raw_line_mask = None
            return

        # Strip orange pixels in the outermost band (= the frame's width).
        # The AI tends to draw lines that reach the image border; those would
        # visibly shift when the user stretches and would conflict with our
        # fixed orange frame. Removing them leaves the frame as the SOLE
        # border-orange, painted at fixed positions every recomposite — so
        # the frame never moves regardless of the stretch percentage.
        fw = self.FRAME_WIDTH_PX
        h_m, w_m = orange_mask.shape[:2]
        if fw > 0 and h_m > 2 * fw and w_m > 2 * fw:
            orange_mask = orange_mask.copy()
            orange_mask[:fw, :] = 0
            orange_mask[-fw:, :] = 0
            orange_mask[:, :fw] = 0
            orange_mask[:, -fw:] = 0

        # Cache the (border-stripped) mask at base dims so the stretch
        # spin-box has something to re-stretch from on every change.
        self._raw_line_mask = orange_mask
        # Reset stretch to 100% for each new line mask without triggering a
        # re-composite (we composite once below at 100%).
        self.stretch_spin.blockSignals(True)
        self.stretch_spin.setValue(100)
        self.stretch_spin.blockSignals(False)
        # Show the stretch + line-width + edit + zoom controls now that we
        # have a mask to operate on.
        for w in (
            self.stretch_label, self.stretch_spin,
            self.line_width_label, self.line_width_spin,
            self.draw_btn, self.erase_btn,
            self.brush_label, self.brush_spin,
            self.zoom_area_btn, self.fit_btn,
        ):
            w.setVisible(True)
        # Apply the chosen line width to the freshly-detected mask so the
        # user sees a clean uniform-width line immediately.
        self._reskeletonize_and_render(int(self.line_width_spin.value()))

        # First composite at 100% (no stretch).
        self._recomposite_with_stretch(100)
        # Refresh button states so Save Lines becomes enabled.
        self._update_button_states()

    def _stretch_mask_horizontally(
            self, mask: np.ndarray, scale_pct: int) -> np.ndarray:
        """Return ``mask`` horizontally scaled by ``scale_pct`` percent,
        anchored at the LEFT edge (column 0 stays put; the right side
        moves outward / inward with the scale). The result is cropped or
        padded with black on the right to keep the original width — so it
        composites pixel-aligned with the base image."""
        if scale_pct == 100:
            return mask
        h, w = mask.shape[:2]
        new_w = max(1, int(round(w * scale_pct / 100.0)))
        # Nearest-neighbour resize keeps the line crisp (no AA blur on the
        # mask edges that would dilate or blur the line afterwards).
        scaled = cv2.resize(mask, (new_w, h), interpolation=cv2.INTER_NEAREST)
        if new_w == w:
            return scaled
        if new_w > w:
            # Crop the right side to fit base width.
            return scaled[:, :w]
        # new_w < w → pad black on the right.
        out = np.zeros((h, w), dtype=mask.dtype)
        out[:, :new_w] = scaled
        return out

    # Width (in pixels) of the pure-orange frame painted around every
    # add-lines composite so the output has a clean outer border.
    FRAME_WIDTH_PX = 3

    def _recomposite_with_stretch(self, scale_pct: int) -> None:
        """Stamp the (possibly stretched) line mask onto a fresh copy of the
        base image, paint a 3-px orange frame around the whole image, and
        display the result. Called by both the initial composite and every
        spin-box change."""
        base = self._add_lines_base
        mask = self._raw_line_mask
        if base is None or mask is None:
            return
        stretched = self._stretch_mask_horizontally(mask, scale_pct)
        base_rgb = np.array(base.convert("RGB"), dtype=np.uint8).copy()
        # Dim the base so the orange lines + frame visibly pop. Applied
        # BEFORE stamping the lines / frame so those stay at full brightness.
        if self.DARKEN_FACTOR < 1.0:
            base_rgb = (base_rgb.astype(np.float32) * self.DARKEN_FACTOR).clip(
                0, 255,
            ).astype(np.uint8)
        base_rgb[stretched > 0] = (255, 102, 0)
        # 3-px pure-orange frame around the entire image (top / bottom /
        # left / right). Applied AFTER the line composite so the frame is
        # always solid orange even where lines happen to touch the border.
        fw = self.FRAME_WIDTH_PX
        h, w = base_rgb.shape[:2]
        fw_h = min(fw, h)
        fw_w = min(fw, w)
        if fw_h > 0:
            base_rgb[:fw_h, :] = (255, 102, 0)
            base_rgb[-fw_h:, :] = (255, 102, 0)
        if fw_w > 0:
            base_rgb[:, :fw_w] = (255, 102, 0)
            base_rgb[:, -fw_w:] = (255, 102, 0)
        out_pil = _PILImage.fromarray(base_rgb)
        n = int((stretched > 0).sum())
        suffix = "" if scale_pct == 100 else f"  |  stretched to {scale_pct}%"
        # IMPORTANT: do NOT call set_pil_image — it resets pane.zoom to 1.0
        # and snaps the scrollbars back to (0, 0), undoing any zoom the user
        # has set. Use the zoom-preserving helper instead.
        self._update_result_keep_zoom(
            out_pil,
            f"Lines composited  |  {out_pil.width} × {out_pil.height} px  |  "
            f"{n} line pixels{suffix}",
        )
        if scale_pct == 100:
            self.statusBar().showMessage(
                f"Composited {n} orange-line pixels onto the base — "
                f"underlying image preserved exactly.",
            )
        else:
            self.statusBar().showMessage(
                f"Lines stretched to {scale_pct}% width — {n} line pixels."
            )

    def _on_stretch_changed(self, value: int) -> None:
        """Spin-box callback: re-composite the line mask at the new width
        percentage."""
        self._recomposite_with_stretch(value)

    def _on_line_width_changed(self, value: int) -> None:
        """Spin-box callback: re-skeletonise the current line mask and
        re-render it at the new uniform pixel width, then re-composite."""
        if self._raw_line_mask is None:
            return
        self._reskeletonize_and_render(int(value))
        self._recomposite_with_stretch(int(self.stretch_spin.value()))

    def _reskeletonize_and_render(self, width: int) -> None:
        """Replace ``self._raw_line_mask`` with a clean uniform-width version:
        skeletonise the current mask down to a 1-pixel centerline, then
        re-inflate to ``width`` pixels via a distance-transform threshold.
        This keeps every line edge a perfect ``width`` pixels thick
        regardless of how thick / variable the input mask was."""
        if self._raw_line_mask is None:
            return
        # Skeletonise (skimage gives the medial axis as a boolean array).
        skel_bool = _skeletonize(self._raw_line_mask > 0)
        skel = skel_bool.astype(np.uint8) * 255
        if width <= 1:
            self._raw_line_mask = skel
            return
        # Distance transform on the inverted skeleton: each non-skeleton
        # pixel knows its distance to the nearest skeleton pixel. Pixels
        # within (width - 1) / 2 of the centerline become the new line —
        # giving a line of exactly ``width`` pixels uniform thickness.
        inverted = (skel == 0).astype(np.uint8) * 255
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)
        self._raw_line_mask = (
            dist <= (width - 1) / 2.0
        ).astype(np.uint8) * 255

    # ----- Draw / Erase tools (operate on the line mask only) -------------

    def _on_draw_toggled(self, checked: bool) -> None:
        if checked:
            self.erase_btn.setChecked(False)
            self._ensure_baked_for_edit()
            self.result_pane.image_label.setCursor(Qt.CrossCursor)
        else:
            self.result_pane.image_label.setCursor(Qt.ArrowCursor)
        self._editing_dragging = False
        self._last_edit_pt = None

    def _on_erase_toggled(self, checked: bool) -> None:
        if checked:
            self.draw_btn.setChecked(False)
            self._ensure_baked_for_edit()
            self.result_pane.image_label.setCursor(Qt.CrossCursor)
        else:
            self.result_pane.image_label.setCursor(Qt.ArrowCursor)
        self._editing_dragging = False
        self._last_edit_pt = None

    def _ensure_baked_for_edit(self) -> None:
        """When the user enters Draw / Erase mode while the stretch is not
        100 %, bake the current stretched mask into the raw mask and reset
        the stretch to 100 %. From that point, mouse coordinates on the
        result pane map 1:1 to mask pixels (no stretch math needed)."""
        if self._raw_line_mask is None:
            return
        current = int(self.stretch_spin.value())
        if current == 100:
            return
        baked = self._stretch_mask_horizontally(self._raw_line_mask, current)
        self._raw_line_mask = baked
        self.stretch_spin.blockSignals(True)
        self.stretch_spin.setValue(100)
        self.stretch_spin.blockSignals(False)
        # Re-composite at 100% so what the user sees matches the new raw mask.
        self._recomposite_with_stretch(100)

    def eventFilter(self, obj, event):
        """Catch mouse events on the result pane for Draw / Erase / Zoom
        Area. Wheel events pass through to the inherited ImagePane filter
        (cursor-centered zoom)."""
        viewport = self.result_pane.scroll.viewport()
        is_result_widget = (
            obj is self.result_pane.image_label or obj is viewport
        )
        if not is_result_widget:
            return super().eventFilter(obj, event)

        t = event.type()

        # ---- Mouse-wheel cursor-centered zoom ----
        # Handled directly here (rather than relying on ImagePane's own
        # wheel filter). Catches wheel on EITHER image_label or viewport:
        # since image_label fills the viewport, wheel events normally fire
        # on image_label first — we need to catch them there too. Returning
        # True consumes the event so QScrollArea's default scrolling
        # behaviour doesn't kick in.
        if t == QEvent.Wheel:
            self._handle_wheel_zoom(event)
            return True

        # ---- Zoom Area drag (rubber band + zoom-to-rect on release) ----
        if hasattr(self, "zoom_area_btn") and self.zoom_area_btn.isChecked():
            if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                # Convert event pos to viewport-relative coords so the
                # rubber band geometry is in the same coord system.
                gp = self._event_global_pos(event)
                vp_pt = viewport.mapFromGlobal(gp)
                self._zoom_origin = vp_pt
                if self._rubber_band is None:
                    self._rubber_band = QRubberBand(
                        QRubberBand.Rectangle, viewport,
                    )
                self._rubber_band.setGeometry(
                    QRect(vp_pt, vp_pt).normalized(),
                )
                self._rubber_band.show()
                self._rubber_band.raise_()
                self._zoom_dragging = True
                # CRITICAL: the filter consumed the press, so the widget's
                # mousePressEvent never ran → no automatic mouse grab. We
                # grab manually so subsequent moves still come through our
                # filter even if the cursor wanders outside the widget.
                viewport.grabMouse()
                return True
            elif t == QEvent.MouseMove and self._zoom_dragging:
                gp = self._event_global_pos(event)
                vp_pt = viewport.mapFromGlobal(gp)
                if self._zoom_origin is not None and self._rubber_band is not None:
                    self._rubber_band.setGeometry(
                        QRect(self._zoom_origin, vp_pt).normalized(),
                    )
                return True
            elif t == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self._zoom_dragging:
                    self._zoom_dragging = False
                    # Release the grab we set on press.
                    if viewport.mouseGrabber() is viewport:
                        viewport.releaseMouse()
                    if self._rubber_band is not None:
                        rect = self._rubber_band.geometry()
                        self._rubber_band.hide()
                        self._zoom_to_viewport_rect(rect)
                    self._zoom_origin = None
                return True

        # ---- Draw / Erase (existing) ----
        in_edit_mode = (
            self.draw_btn.isChecked() or self.erase_btn.isChecked()
        ) if hasattr(self, "draw_btn") else False
        if in_edit_mode and self._raw_line_mask is not None:
            if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                pt = self._event_to_image_coords(event)
                if pt is not None:
                    self._editing_dragging = True
                    self._last_edit_pt = pt
                    self._apply_brush(pt, pt)
                    return True
            elif t == QEvent.MouseMove and self._editing_dragging:
                pt = self._event_to_image_coords(event)
                if pt is not None and self._last_edit_pt is not None:
                    self._apply_brush(self._last_edit_pt, pt)
                    self._last_edit_pt = pt
                    return True
            elif t == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self._editing_dragging:
                    self._editing_dragging = False
                    self._last_edit_pt = None
                    return True
        return super().eventFilter(obj, event)

    # ----- Zoom Area + Fit -------------------------------------------------

    def _on_zoom_area_toggled(self, checked: bool) -> None:
        if checked:
            # Mutually exclusive with Draw / Erase.
            if hasattr(self, "draw_btn"):
                self.draw_btn.setChecked(False)
                self.erase_btn.setChecked(False)
            self.result_pane.image_label.setCursor(Qt.CrossCursor)
        else:
            self.result_pane.image_label.setCursor(Qt.ArrowCursor)
            if self._rubber_band is not None:
                self._rubber_band.hide()
            self._zoom_dragging = False
            self._zoom_origin = None

    def _zoom_to_viewport_rect(self, vp_rect) -> None:
        """Scale + scroll the ImagePane so the given viewport-coords
        rectangle fills the viewport. Tiny rects (likely a click rather
        than a drag) are ignored."""
        pane = self.result_pane
        if pane._base_qimg is None or pane.pil_image is None:
            return
        if vp_rect.width() < 5 or vp_rect.height() < 5:
            return  # treat as a stray click
        hbar = pane.scroll.horizontalScrollBar()
        vbar = pane.scroll.verticalScrollBar()
        old_zoom = float(pane.zoom)
        # Rectangle in image-space coords.
        ix1 = (hbar.value() + vp_rect.left())  / old_zoom
        iy1 = (vbar.value() + vp_rect.top())   / old_zoom
        ix2 = (hbar.value() + vp_rect.right()) / old_zoom
        iy2 = (vbar.value() + vp_rect.bottom())/ old_zoom
        rect_w_img = max(1.0, ix2 - ix1)
        rect_h_img = max(1.0, iy2 - iy1)
        # New zoom = fit the rect into the viewport.
        vp_size = pane.scroll.viewport().size()
        new_zoom = min(
            vp_size.width()  / rect_w_img,
            vp_size.height() / rect_h_img,
        )
        new_zoom = max(pane.ZOOM_MIN, min(pane.ZOOM_MAX, new_zoom))
        pane.zoom = new_zoom
        pane._apply_zoom()
        # Scroll so the rect's centre lands at the viewport centre.
        cx_img = (ix1 + ix2) / 2.0
        cy_img = (iy1 + iy2) / 2.0
        target_h = int(round(cx_img * new_zoom - vp_size.width()  / 2.0))
        target_v = int(round(cy_img * new_zoom - vp_size.height() / 2.0))
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), target_h)))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), target_v)))

    def _update_result_keep_zoom(self, pil_img, info: str) -> None:
        """Push a new PIL image into the result pane WITHOUT touching the
        pane's zoom factor or scrollbar positions. ImagePane.set_pil_image
        always resets ``zoom = 1.0`` and re-applies, which snaps the view
        back to the top-left every time we re-composite (after a draw,
        erase, stretch change, line-width change, etc.). This helper
        updates pil_image + _base_qimg + the displayed pixmap while
        preserving the user's current zoom + scroll."""
        pane = self.result_pane
        hbar = pane.scroll.horizontalScrollBar()
        vbar = pane.scroll.verticalScrollBar()
        old_h = hbar.value()
        old_v = vbar.value()
        pane.pil_image = pil_img
        rgba = pil_img.convert("RGBA")
        pane._base_qimg = QImage(
            rgba.tobytes("raw", "RGBA"),
            pil_img.width, pil_img.height,
            QImage.Format_RGBA8888,
        ).copy()
        pane._apply_zoom()
        # Restore scrollbars (clamp to new max in case the widget grew).
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), old_h)))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), old_v)))
        if info:
            pane.info_label.setText(info)

    @staticmethod
    def _event_global_pos(event):
        """Cross-version helper: PyQt5 has event.globalPos(); newer PyQt
        replaced it with event.globalPosition().toPoint()."""
        try:
            return event.globalPos()
        except AttributeError:
            return event.globalPosition().toPoint()

    def _handle_wheel_zoom(self, event) -> None:
        """Cursor-centered wheel zoom on the result pane. Always uses the
        cursor's VIEWPORT-relative coords via mapFromGlobal — so the math
        is correct no matter which child widget the wheel event fired on
        (image_label, viewport, or anywhere)."""
        pane = self.result_pane
        if pane._base_qimg is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        old_zoom = float(pane.zoom)
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_zoom = max(
            float(pane.ZOOM_MIN),
            min(float(pane.ZOOM_MAX), old_zoom * factor),
        )
        if abs(new_zoom - old_zoom) < 1e-9:
            return
        # Cursor position in VIEWPORT-local coords. Using globalPos →
        # mapFromGlobal(viewport) means it doesn't matter which widget the
        # event fired on; we always get the right viewport-relative point.
        viewport = pane.scroll.viewport()
        gp = self._event_global_pos(event)
        vp_pt = viewport.mapFromGlobal(gp)
        vx, vy = float(vp_pt.x()), float(vp_pt.y())
        hbar = pane.scroll.horizontalScrollBar()
        vbar = pane.scroll.verticalScrollBar()
        # Image-space coords currently under the cursor.
        img_x = (hbar.value() + vx) / old_zoom
        img_y = (vbar.value() + vy) / old_zoom
        pane.zoom = new_zoom
        pane._apply_zoom()
        # Reposition scrollbars so the same image point stays under cursor.
        new_h = int(round(img_x * new_zoom - vx))
        new_v = int(round(img_y * new_zoom - vy))
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), new_h)))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), new_v)))

    def _fit_to_viewport(self) -> None:
        """Reset the result pane's zoom so the whole image fits in the
        viewport with the longest side touching."""
        pane = self.result_pane
        if pane.pil_image is None:
            return
        vp_size = pane.scroll.viewport().size()
        img_w = float(pane.pil_image.width)
        img_h = float(pane.pil_image.height)
        if img_w <= 0 or img_h <= 0:
            return
        new_zoom = min(vp_size.width() / img_w, vp_size.height() / img_h)
        new_zoom = max(pane.ZOOM_MIN, min(pane.ZOOM_MAX, new_zoom))
        pane.zoom = new_zoom
        pane._apply_zoom()
        # Centre the scroll on the image.
        hbar = pane.scroll.horizontalScrollBar()
        vbar = pane.scroll.verticalScrollBar()
        hbar.setValue(int((img_w * new_zoom - vp_size.width())  / 2.0))
        vbar.setValue(int((img_h * new_zoom - vp_size.height()) / 2.0))

    def _event_to_image_coords(self, event) -> tuple[int, int] | None:
        """Translate a mouse event into integer image-space coords on the
        result pane, regardless of which widget the event fired on."""
        if self._raw_line_mask is None:
            return None
        try:
            gp = event.globalPos()
        except AttributeError:
            gp = event.globalPosition().toPoint()
        label = self.result_pane.image_label
        local = label.mapFromGlobal(gp)
        z = max(float(self.result_pane.zoom), 1e-6)
        img_x = int(round(local.x() / z))
        img_y = int(round(local.y() / z))
        h, w = self._raw_line_mask.shape[:2]
        if 0 <= img_x < w and 0 <= img_y < h:
            return img_x, img_y
        return None

    def _apply_brush(self, pt1: tuple[int, int], pt2: tuple[int, int]) -> None:
        """Paint (draw mode) or erase (erase mode) a segment from pt1 to pt2
        on the line MASK. Then re-composite so the change is visible.
        The base image is never touched — only the mask changes.

        Draw uses a flat line at the user's chosen brush width. Erase uses
        a fixed circular brush of radius ``ERASE_BRUSH_RADIUS`` (filled
        circles stamped at both endpoints + a thick line between them →
        true round caps that wipe out neighbouring line pixels cleanly).
        """
        if self._raw_line_mask is None:
            return
        if self.erase_btn.isChecked():
            r = int(self.ERASE_BRUSH_RADIUS)
            # Filled circles at both endpoints give the eraser truly round
            # caps (cv2.line's caps are otherwise flat / square-ish). The
            # thick line between them fills the swept area on fast drags.
            cv2.circle(
                self._raw_line_mask, pt1, r,
                color=0, thickness=-1, lineType=cv2.LINE_8,
            )
            cv2.circle(
                self._raw_line_mask, pt2, r,
                color=0, thickness=-1, lineType=cv2.LINE_8,
            )
            if pt1 != pt2:
                cv2.line(
                    self._raw_line_mask, pt1, pt2,
                    color=0, thickness=2 * r, lineType=cv2.LINE_8,
                )
        else:
            width = max(1, int(self.brush_spin.value()))
            cv2.line(
                self._raw_line_mask, pt1, pt2,
                color=255, thickness=width, lineType=cv2.LINE_8,
            )
        # Re-composite at the current stretch percentage (always 100% after
        # _ensure_baked_for_edit, but read defensively).
        self._recomposite_with_stretch(int(self.stretch_spin.value()))

    # ----- save the line layer alone --------------------------------------

    PROJECT_SCHEMA_VERSION = 1
    PROJECT_FILE_EXT = ".voroni"

    def save_lines(self) -> None:
        """Save a Voroni project file containing:
          - base image (bright, undimmed) as PNG bytes
          - orange line mask as a uint8 numpy array at source dimensions
          - frame width, current stretch %, line width, darken factor
          - source dimensions + (optional) source file path
        The future polygon-extraction GUI loads this file and rebuilds the
        editing state (it composites lines + frame onto the base, then runs
        polygon detection on the inverted mask)."""
        if self._raw_line_mask is None:
            QMessageBox.information(
                self, "No line mask",
                "Run 'Add lines' first to generate the orange line mask, "
                "then this button will save the project file.",
            )
            return
        if self._add_lines_base is None:
            QMessageBox.critical(
                self, "No base image",
                "Internal: no base image cached. Re-run Add lines.",
            )
            return

        base = self._add_lines_base
        if base.mode != "RGB":
            base = base.convert("RGB")

        # PNG-encode the base. PIL versions change pickling — PNG bytes are
        # rock-solid and stay small.
        base_buf = BytesIO()
        base.save(base_buf, format="PNG", optimize=True)
        base_png_bytes = base_buf.getvalue()

        # BAKE the current horizontal stretch into the saved mask. The
        # downstream consumer (lines_to_vec.py) reads `line_mask` as-is
        # and does not re-apply `stretch_pct`, so we must save the
        # already-stretched mask to honour the user's adjustment. We
        # store stretch_pct = 100 in the file to signal "already baked".
        current_stretch_pct = int(self.stretch_spin.value())
        stretched_mask = self._stretch_mask_horizontally(
            self._raw_line_mask, current_stretch_pct,
        )

        project = {
            "schema_version": self.PROJECT_SCHEMA_VERSION,
            "source_path": (
                str(self.current_source_path)
                if getattr(self, "current_source_path", None) is not None
                else None
            ),
            "source_size_px": (base.width, base.height),
            "base_image_png": base_png_bytes,
            "line_mask": stretched_mask.copy(),
            "frame_width_px": int(self.FRAME_WIDTH_PX),
            # 100 because the stretch is now baked into line_mask. The
            # original user-applied value is preserved in
            # 'applied_stretch_pct' for reference / audit only.
            "stretch_pct": 100,
            "applied_stretch_pct": current_stretch_pct,
            "line_width_px": int(self.line_width_spin.value()),
            "darken_factor": float(self.DARKEN_FACTOR),
        }

        # File dialog. Default name uses the source path's stem when known.
        try:
            _PHOTO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        stem = "voroni_project"
        if self.current_source_path is not None:
            stem = f"{self.current_source_path.stem}_project"
        default_name = f"{stem}{self.PROJECT_FILE_EXT}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Voroni project",
            str(_PHOTO_OUTPUT_DIR / default_name),
            f"Voroni project (*{self.PROJECT_FILE_EXT});;All files (*.*)",
        )
        if not path:
            return
        out_path = Path(path)
        if out_path.suffix.lower() != self.PROJECT_FILE_EXT:
            out_path = out_path.with_suffix(self.PROJECT_FILE_EXT)

        try:
            with open(out_path, "wb") as f:
                pickle.dump(project, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            QMessageBox.critical(
                self, "Save failed", f"{type(e).__name__}: {e}",
            )
            return

        size_mb = out_path.stat().st_size / (1024 * 1024)
        self.statusBar().showMessage(
            f"Saved Voroni project ({size_mb:.1f} MB) → {out_path.name}",
        )
        QMessageBox.information(
            self, "Saved",
            f"Saved Voroni project to:\n{out_path}\n\n"
            f"Contents:\n"
            f"  - base image: {base.width} × {base.height} px (PNG-encoded)\n"
            f"  - line mask: {stretched_mask.shape[1]} × "
            f"{stretched_mask.shape[0]} px (stretch baked in)\n"
            f"  - frame width: {project['frame_width_px']} px\n"
            f"  - applied stretch: {current_stretch_pct} %\n"
            f"  - line width: {project['line_width_px']} px\n"
            f"  - darken factor: {project['darken_factor']:.2f}\n"
            f"\nFile size: {size_mb:.1f} MB",
        )

    def _reset_stretch_ui(self) -> None:
        """Drop the cached line mask + base and hide the stretch widgets.
        Called when starting a new pass that invalidates the existing
        composite (Paint, or loading a new image)."""
        self._raw_line_mask = None
        self._add_lines_base = None
        if hasattr(self, "stretch_spin"):
            self.stretch_spin.blockSignals(True)
            self.stretch_spin.setValue(100)
            self.stretch_spin.blockSignals(False)
            self.stretch_label.setVisible(False)
            self.stretch_spin.setVisible(False)
        # Hide the Line width spin too.
        if hasattr(self, "line_width_spin"):
            self.line_width_label.setVisible(False)
            self.line_width_spin.setVisible(False)
        # Hide Draw / Erase tools too.
        if hasattr(self, "draw_btn"):
            self.draw_btn.setChecked(False)
            self.erase_btn.setChecked(False)
            for w in (self.draw_btn, self.erase_btn,
                      self.brush_label, self.brush_spin):
                w.setVisible(False)
        self._editing_dragging = False
        self._last_edit_pt = None
        # Hide Zoom Area + Fit too.
        if hasattr(self, "zoom_area_btn"):
            self.zoom_area_btn.setChecked(False)
            for w in (self.zoom_area_btn, self.fit_btn):
                w.setVisible(False)
        self._zoom_dragging = False
        self._zoom_origin = None
        if self._rubber_band is not None:
            self._rubber_band.hide()
        # Refresh button states so Save Lines disables until next Add lines.
        self._update_button_states()

    def load_image(self) -> None:
        """Override the inherited load to also drop the stretch UI / cached
        line mask so a fresh image starts with no stale composite state."""
        super().load_image()
        self._reset_stretch_ui()

    def _on_worker_done(self) -> None:
        """Reset Paint / Add-lines button labels + pass state after a worker
        finishes. _add_lines_base and _raw_line_mask are deliberately KEPT
        so the stretch slider can keep re-compositing without re-running the
        API. They get cleared when a new pass starts (run_paint / run_add_lines)
        or when a new image is loaded."""
        super()._on_worker_done()
        if hasattr(self, "paint_btn"):
            self.paint_btn.setText("Paint")
        if hasattr(self, "add_lines_btn"):
            self.add_lines_btn.setText("Add lines")
        self._current_pass = None

    def _update_button_states(self) -> None:
        """Gate Paint / Add-lines / Save Lines on the same conditions
        Generate uses (source loaded, no worker running); Save Lines also
        requires a line mask from a successful Add lines pass."""
        super()._update_button_states()
        running = self.worker is not None and self.worker.isRunning()
        has_src = self.source_pane.pil_image is not None
        has_lines = self._raw_line_mask is not None
        if hasattr(self, "paint_btn"):
            self.paint_btn.setEnabled(has_src and not running)
        if hasattr(self, "add_lines_btn"):
            self.add_lines_btn.setEnabled(has_src and not running)
        if hasattr(self, "save_lines_btn"):
            self.save_lines_btn.setEnabled(has_lines and not running)


def main() -> None:
    app = QApplication(sys.argv)
    w = VoroniEditor()
    w.resize(1500, 900)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
