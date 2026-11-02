"""Image Prompt — send a free-text prompt (plus an optional image) to the
Google Gemini image API and show the result.

Workflow:
  1. (Optional) Load Image... or drag an image onto the left pane.
  2. Type a prompt.
  3. Pick model / aspect / size, click Generate.
  4. The result appears on the right and is auto-saved to
     output/prompt_results/<timestamp>.png. Save Result... saves a copy
     anywhere. Any text the model returns is shown under the result.

API key lookup matches IMAGE_TO_MOSAIC/scripts/photo_editor.py (same
.key_path memo, GEMINI_API_KEY env var, .env files, gemini.key), so a key
that works there works here.

Usage:
  python scripts/image_prompt.py
  (or the packaged output/ImagePrompt.exe)
"""
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget,
)


def _find_project_root() -> Path:
    """Walk up from the script (or the frozen .exe) to the folder holding
    CLAUDE.md, so paths work both from source and from output/ImagePrompt.exe."""
    start = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve()
    for p in [start.parent, *start.parents]:
        if (p / "CLAUDE.md").is_file():
            return p
    return start.parent


PROJECT_ROOT = _find_project_root()
I2M_ROOT = PROJECT_ROOT / "IMAGE_TO_MOSAIC"
RESULTS_DIR = PROJECT_ROOT / "output" / "prompt_results"
KEY_PATH_MEMO = I2M_ROOT / ".key_path"      # shared with photo_editor.py

MODELS = [
    ("Nano Banana Pro (gemini-3-pro-image-preview)", "gemini-3-pro-image-preview"),
    ("Nano Banana (gemini-2.5-flash-image)",         "gemini-2.5-flash-image"),
]
ASPECTS = ["Match image", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
SIZES = ["2K", "1K", "4K"]


def closest_aspect_ratio(w: int, h: int) -> str:
    target = w / h
    candidates = {
        "1:1": 1.0, "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3,
        "3:4": 3 / 4, "3:2": 3 / 2, "2:3": 2 / 3,
    }
    return min(candidates.items(), key=lambda kv: abs(kv[1] - target))[0]


def read_key_from_file(path: Path) -> str | None:
    """Plain key, or a GEMINI_API_KEY=value line."""
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


def load_api_key() -> str | None:
    if KEY_PATH_MEMO.is_file():
        key = read_key_from_file(Path(KEY_PATH_MEMO.read_text(encoding="utf-8").strip()))
        if key:
            return key
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()
    for p in (I2M_ROOT / ".env", PROJECT_ROOT / ".env", I2M_ROOT / "gemini.key"):
        key = read_key_from_file(p)
        if key:
            return key
    return None


def pil_to_pixmap(img: Image.Image) -> QPixmap:
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


# ---------------------------------------------------------------------------
# Gemini worker thread
# ---------------------------------------------------------------------------

class GenerationWorker(QThread):
    finished_ok = pyqtSignal(object, str)   # image bytes (or None), text reply
    failed      = pyqtSignal(str)

    def __init__(self, image: Image.Image | None, prompt: str, model_id: str,
                 aspect: str, size: str):
        super().__init__()
        self.image = image
        self.prompt = prompt
        self.model_id = model_id
        self.aspect = aspect
        self.size = size

    def run(self) -> None:
        try:
            api_key = load_api_key()
            if not api_key:
                self.failed.emit(
                    "No Gemini API key found.\n\n"
                    "Click 'Choose Key File...' and pick the file holding your key, "
                    "or set the GEMINI_API_KEY environment variable."
                )
                return

            from google import genai
            from google.genai import types

            contents: list = []
            if self.image is not None:
                img = self.image
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                buf = BytesIO()
                img.save(buf, "PNG")
                contents.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
            contents.append(self.prompt)

            image_config = {"aspect_ratio": self.aspect}
            if self.model_id.startswith("gemini-3"):
                image_config["image_size"] = self.size   # 2.5 Flash has a fixed size

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(**image_config),
                ),
            )

            image_bytes = None
            texts: list[str] = []
            for cand in response.candidates or []:
                for part in (cand.content.parts if cand.content else []) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline is not None and getattr(inline, "data", None) and image_bytes is None:
                        image_bytes = inline.data
                    elif getattr(part, "text", None) and not getattr(part, "thought", False):
                        texts.append(part.text.strip())

            if image_bytes is None and not texts:
                reason = ""
                fb = getattr(response, "prompt_feedback", None)
                if fb is not None and getattr(fb, "block_reason", None):
                    reason = f"\nBlock reason: {fb.block_reason}"
                self.failed.emit("The model returned nothing (safety filter or empty response)." + reason)
                return
            self.finished_ok.emit(image_bytes, "\n\n".join(texts))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Image pane
# ---------------------------------------------------------------------------

class ImagePane(QLabel):
    """Shows a PIL image scaled to fit. The input pane also accepts drops."""
    dropped = pyqtSignal(str)

    def __init__(self, placeholder: str, accept_drops: bool = False):
        super().__init__(placeholder)
        self.placeholder = placeholder
        self.pil_image: Image.Image | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setStyleSheet("QLabel { background: #2b2b2b; color: #aaa; border: 1px solid #444; }")
        self.setAcceptDrops(accept_drops)

    def set_image(self, img: Image.Image | None) -> None:
        self.pil_image = img
        if img is None:
            self.setPixmap(QPixmap())
            self.setText(self.placeholder)
        else:
            self._refresh()

    def _refresh(self) -> None:
        if self.pil_image is None:
            return
        pm = pil_to_pixmap(self.pil_image)
        self.setPixmap(pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.dropped.emit(urls[0].toLocalFile())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Prompt — Gemini")
        self.resize(1300, 820)
        self.worker: GenerationWorker | None = None
        self.result_bytes: bytes | None = None

        central = QWidget()
        root = QVBoxLayout(central)

        # Top bar
        bar = QHBoxLayout()
        self.load_btn = QPushButton("Load Image...")
        self.clear_btn = QPushButton("Clear Image")
        self.key_btn = QPushButton("Choose Key File...")
        bar.addWidget(self.load_btn)
        bar.addWidget(self.clear_btn)
        bar.addSpacing(16)
        bar.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        for label, _ in MODELS:
            self.model_combo.addItem(label)
        bar.addWidget(self.model_combo)
        bar.addWidget(QLabel("Aspect:"))
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(ASPECTS)
        bar.addWidget(self.aspect_combo)
        bar.addWidget(QLabel("Size:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(SIZES)
        bar.addWidget(self.size_combo)
        bar.addStretch(1)
        bar.addWidget(self.key_btn)
        root.addLayout(bar)

        # Image panes
        split = QSplitter(Qt.Horizontal)
        self.input_pane = ImagePane("No image loaded (optional)\n\nDrop an image here or click Load Image...",
                                    accept_drops=True)
        self.result_pane = ImagePane("Result will appear here")
        split.addWidget(self.input_pane)
        split.addWidget(self.result_pane)
        root.addWidget(split, 1)

        # Prompt + model text reply
        prompt_row = QHBoxLayout()
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Write your prompt here...")
        self.prompt_edit.setFixedHeight(130)
        self.reply_view = QPlainTextEdit()
        self.reply_view.setReadOnly(True)
        self.reply_view.setPlaceholderText("Text reply from the model (if any)")
        self.reply_view.setFixedHeight(130)
        prompt_row.addWidget(self.prompt_edit, 3)
        prompt_row.addWidget(self.reply_view, 2)
        root.addLayout(prompt_row)

        # Bottom buttons
        bottom = QHBoxLayout()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setMinimumHeight(36)
        self.generate_btn.setStyleSheet("font-weight: bold;")
        self.save_btn = QPushButton("Save Result...")
        self.use_btn = QPushButton("Use Result as Input")
        bottom.addWidget(self.generate_btn, 2)
        bottom.addWidget(self.use_btn, 1)
        bottom.addWidget(self.save_btn, 1)
        root.addLayout(bottom)

        self.setCentralWidget(central)

        self.load_btn.clicked.connect(self.load_image)
        self.clear_btn.clicked.connect(lambda: self._set_input(None))
        self.key_btn.clicked.connect(self.choose_key_file)
        self.generate_btn.clicked.connect(self.generate)
        self.save_btn.clicked.connect(self.save_result)
        self.use_btn.clicked.connect(self.use_result_as_input)
        self.input_pane.dropped.connect(self._open_path)

        self._update_buttons()
        self.statusBar().showMessage(
            "API key found." if load_api_key() else
            "No API key found — click 'Choose Key File...'."
        )

    # ── actions ─────────────────────────────────────────────────────────
    def load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load image", str(PROJECT_ROOT / "resources"),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if path:
            self._open_path(path)

    def _open_path(self, path: str) -> None:
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            QMessageBox.warning(self, "Load image", f"Could not open the image:\n{e}")
            return
        self._set_input(img)
        self.statusBar().showMessage(f"Loaded {Path(path).name}  ({img.width} × {img.height})")

    def _set_input(self, img: Image.Image | None) -> None:
        self.input_pane.set_image(img)
        self._update_buttons()

    def choose_key_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose the file holding your Gemini API key", str(I2M_ROOT),
            "Key files (*.key *.txt *.env);;All files (*.*)",
        )
        if not path:
            return
        if not read_key_from_file(Path(path)):
            QMessageBox.warning(self, "Key file", "No key found in that file.")
            return
        KEY_PATH_MEMO.parent.mkdir(parents=True, exist_ok=True)
        KEY_PATH_MEMO.write_text(path, encoding="utf-8")
        self.statusBar().showMessage(f"Using key from {Path(path).name}")

    def generate(self) -> None:
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "Generate", "Write a prompt first.")
            return
        img = self.input_pane.pil_image
        aspect = self.aspect_combo.currentText()
        if aspect == "Match image":
            aspect = closest_aspect_ratio(img.width, img.height) if img is not None else "1:1"
        model_id = MODELS[self.model_combo.currentIndex()][1]

        self.worker = GenerationWorker(img, prompt, model_id, aspect, self.size_combo.currentText())
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._update_buttons)
        self.worker.start()
        self._update_buttons()
        what = "prompt + image" if img is not None else "prompt only"
        self.statusBar().showMessage(f"Sending {what} to {model_id} ({aspect}) — this can take a minute...")

    def _on_done(self, image_bytes, text: str) -> None:
        self.reply_view.setPlainText(text)
        if image_bytes is None:
            self.statusBar().showMessage("The model answered with text only (no image).")
            return
        self.result_bytes = image_bytes
        img = Image.open(BytesIO(image_bytes))
        img.load()
        self.result_pane.set_image(img)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.png"
        img.save(out, "PNG")
        self.statusBar().showMessage(f"Done — {img.width} × {img.height}, saved to output/prompt_results/{out.name}")
        self._update_buttons()

    def _on_failed(self, msg: str) -> None:
        self.statusBar().showMessage("Generation failed.")
        QMessageBox.critical(self, "Generation failed", msg)

    def save_result(self) -> None:
        img = self.result_pane.pil_image
        if img is None:
            return
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", str(RESULTS_DIR / "result.png"),
            "PNG (*.png);;JPEG (*.jpg *.jpeg)",
        )
        if not path:
            return
        if path.lower().endswith((".jpg", ".jpeg")):
            img.convert("RGB").save(path, "JPEG", quality=95)
        else:
            img.save(path, "PNG")
        self.statusBar().showMessage(f"Saved {path}")

    def use_result_as_input(self) -> None:
        if self.result_pane.pil_image is not None:
            self._set_input(self.result_pane.pil_image.copy())
            self.statusBar().showMessage("Result moved to the input — write the next prompt.")

    def _update_buttons(self) -> None:
        running = self.worker is not None and self.worker.isRunning()
        has_result = self.result_pane.pil_image is not None
        self.generate_btn.setEnabled(not running)
        self.generate_btn.setText("Generating..." if running else "Generate")
        self.load_btn.setEnabled(not running)
        self.clear_btn.setEnabled(not running and self.input_pane.pil_image is not None)
        self.save_btn.setEnabled(has_result)
        self.use_btn.setEnabled(has_result and not running)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
