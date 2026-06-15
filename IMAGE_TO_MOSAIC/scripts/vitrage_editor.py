"""Vitrage Editor — image-to-stained-glass editor.

A stained-glass-only sibling of photo_editor.py. The square overlay,
red-square tile-size input, and max-tiles-across input are hidden (none
apply to stained glass, which uses large flowing pieces, not measured
tesserae). The prompt picker is scoped to prompts/vitrage/, and the
SUBJECT RENDERING RULES teeth-as-white-stone clause is dropped (a tooth
in stained glass becomes one solid piece of glass, not a small white
stone).

Usage:
  python IMAGE_TO_MOSAIC/scripts/vitrage_editor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QMessageBox

from photo_editor import (
    BG_FILL_PROMPT_FILE,
    PhotoEditor as _PhotoEditor,
    PROMPTS_DIR,
)

VITRAGE_PROMPTS_DIR = PROMPTS_DIR / "vitrage"
DEFAULT_PROMPT_FILE = VITRAGE_PROMPTS_DIR / "stained_glass.txt"

# Labels that share a toolbar row with the hidden widgets; their text is
# matched to hide them too (the labels weren't stored as attributes on the
# parent so we look them up by text).
_HIDDEN_LABELS = {"Red square (px):", "Max tiles across:"}


class VitrageEditor(_PhotoEditor):
    """Photo Editor restricted to the stained-glass workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vitrage Editor — Stained Glass")

        # Force the size-related controls off and hide them.
        self.square_size_input.setValue(0)
        self.square_size_input.setVisible(False)
        self.use_red_square_chk.setChecked(False)
        self.use_red_square_chk.setVisible(False)
        self.max_tiles_input.setValue(0)
        self.max_tiles_input.setVisible(False)
        for label in self.findChildren(QLabel):
            if label.text() in _HIDDEN_LABELS:
                label.setVisible(False)

        self._auto_load_default_prompt()

    def _auto_load_default_prompt(self) -> None:
        """Pre-select prompts/vitrage/stained_glass.txt so the user can hit
        Generate immediately after loading an image."""
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
        """Same as the parent, but the dialog opens in prompts/vitrage/."""
        if VITRAGE_PROMPTS_DIR.exists():
            start_dir = str(VITRAGE_PROMPTS_DIR)
        elif PROMPTS_DIR.exists():
            start_dir = str(PROMPTS_DIR)
        else:
            start_dir = ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose stained-glass prompt", start_dir,
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
        """Stained-glass version: keep the aspect-rewrite and background-fill
        blocks, but DROP the SUBJECT RENDERING RULES teeth clause (in stained
        glass a tooth becomes one solid glass piece, not a white stone).
        """
        text = prompt_text
        if self.keep_aspect_chk.isChecked():
            text = self._rewrite_square_output_clause(text)
        if self.fill_bg_chk.isChecked():
            try:
                file_text = BG_FILL_PROMPT_FILE.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                QMessageBox.warning(
                    self, "Background fill file missing",
                    f"Expected file not found:\n{BG_FILL_PROMPT_FILE}\n\n"
                    "Generating without the background-fill instructions.",
                )
                file_text = ""
            except Exception as e:
                QMessageBox.warning(
                    self, "Background fill file unreadable",
                    f"{type(e).__name__}: {e}\n\n"
                    "Generating without the background-fill instructions.",
                )
                file_text = ""
            if file_text:
                text = text + "\n\n" + file_text
        return text


def main() -> None:
    app = QApplication(sys.argv)
    w = VitrageEditor()
    w.resize(1500, 900)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
