"""Polygon Viewer — load a CSV of polygon definitions and render them on a canvas.

CSV format (one polygon per row, header required):
  coordinates,color_r,color_g,color_b,color_a,color_hex

Where:
  coordinates  = Python-syntax list of (x, y) tuples, e.g. "[(1.0, 2.0), (3.0, 4.0), ...]"
  color_r/g/b  = floats in [0, 1]
  color_a      = float in [0, 1]  (optional, defaults to 1.0)
  color_hex    = "#RRGGBB"        (optional, color_r/g/b take precedence)

Usage:
  python scripts/polygon_viewer.py            # opens empty viewer, File > Open
  python scripts/polygon_viewer.py A1.csv     # opens the file immediately

Controls (matplotlib toolbar):
  pan / zoom-rect / home / save-as-png — all built in
"""
from __future__ import annotations

import argparse
import ast
import csv
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, simpledialog

import numpy as np
from PIL import Image, ImageDraw

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import PolyCollection


# ---------------------------------------------------------------------------
# Lab color helpers (duplicated from divide_into_tesserae.py for self-containment)
# ---------------------------------------------------------------------------

_RGB_TO_XYZ_D65 = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])
_DELTA = 6.0 / 29.0


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1.0 + a)) ** 2.4)


def _f_lab(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA ** 3, np.cbrt(t), t / (3.0 * _DELTA ** 2) + 4.0 / 29.0)


def rgb_to_lab(rgb_float: np.ndarray) -> np.ndarray:
    """Convert sRGB float [0,1] (..., 3) -> Lab (..., 3). Numpy-vectorized."""
    linear = _srgb_to_linear(rgb_float)
    xyz = linear @ _RGB_TO_XYZ_D65.T
    xyz_n = xyz / _WHITE_D65
    f = _f_lab(xyz_n)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def hex_to_rgb_float(hex_str: str) -> tuple[float, float, float]:
    h = hex_str.lstrip("#")
    return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0


# ---------------------------------------------------------------------------
# Tile library — index over output/roman colors/
# ---------------------------------------------------------------------------

class TileLibrary:
    """Lazy-loaded index of color tiles in output/roman colors/.
    Each PNG filename is the hex code of its average color.
    """
    def __init__(self, tiles_dir: Path):
        self.tiles_dir = tiles_dir
        self.hexes: list[str] = []         # ["5E251A", ...]
        self.lab: np.ndarray | None = None  # (N, 3) Lab values
        self._cache: dict[str, Image.Image] = {}  # hex -> PIL.Image RGB 500x500

    def is_available(self) -> bool:
        return self.tiles_dir.exists() and any(self.tiles_dir.glob("*.png"))

    def load_index(self) -> None:
        if self.lab is not None:
            return
        rgbs_float: list[tuple[float, float, float]] = []
        for p in sorted(self.tiles_dir.glob("*.png")):
            stem = p.stem
            if len(stem) != 6:
                continue
            try:
                rgb = hex_to_rgb_float("#" + stem)
            except ValueError:
                continue
            self.hexes.append(stem)
            rgbs_float.append(rgb)
        if not self.hexes:
            raise RuntimeError(f"No tile PNGs found in {self.tiles_dir}")
        arr = np.asarray(rgbs_float, dtype=np.float64)
        self.lab = rgb_to_lab(arr)

    def nearest_hex(self, polygon_rgb_float: tuple[float, float, float]) -> str:
        """Return the hex of the tile whose Lab color is closest to the given RGB."""
        if self.lab is None:
            self.load_index()
        target_lab = rgb_to_lab(np.asarray([polygon_rgb_float], dtype=np.float64))[0]
        diffs = self.lab - target_lab  # type: ignore[operator]
        d2 = np.sum(diffs * diffs, axis=1)
        return self.hexes[int(np.argmin(d2))]

    def get_tile(self, hex_str: str) -> Image.Image:
        if hex_str not in self._cache:
            img = Image.open(self.tiles_dir / f"{hex_str}.png").convert("RGB")
            self._cache[hex_str] = img
        return self._cache[hex_str]


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def load_polygons(csv_path: Path) -> tuple[list, list, tuple[float, float, float, float]]:
    """Parse the polygon CSV.

    Returns (coords_list, colors_list, bbox) where:
      coords_list[i] = list of (x, y) points for polygon i
      colors_list[i] = (r, g, b, a) tuple in [0, 1]
      bbox = (xmin, ymin, xmax, ymax) — overall bounding box
    """
    coords_list: list = []
    colors_list: list = []
    xs_all: list = []
    ys_all: list = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"coordinates", "color_r", "color_g", "color_b"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV missing required columns. Expected at least {sorted(required)}, "
                f"got {reader.fieldnames}"
            )

        for line_no, row in enumerate(reader, start=2):
            try:
                pts = ast.literal_eval(row["coordinates"])
                if not isinstance(pts, list) or len(pts) < 3:
                    continue  # need at least 3 points
                r = float(row["color_r"])
                g = float(row["color_g"])
                b = float(row["color_b"])
                a = float(row.get("color_a") or 1.0)
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"Bad row at line {line_no}: {e}") from e

            coords_list.append(pts)
            colors_list.append((r, g, b, a))
            for x, y in pts:
                xs_all.append(x)
                ys_all.append(y)

    if not coords_list:
        raise ValueError("No valid polygons found in CSV")

    bbox = (min(xs_all), min(ys_all), max(xs_all), max(ys_all))
    return coords_list, colors_list, bbox


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TILES_DIR = ROOT_DIR / "output" / "roman colors"


def _count_hex_pngs(folder: Path) -> int:
    """How many *.png files in `folder` have a 6-hex-char stem (tile library format)."""
    if not folder.exists():
        return 0
    n = 0
    for p in folder.glob("*.png"):
        stem = p.stem
        if len(stem) == 6:
            try:
                int(stem, 16)
                n += 1
            except ValueError:
                pass
    return n


class PolygonViewer:
    def __init__(self, root: tk.Tk, initial_csv: Path | None = None):
        self.root = root
        self.root.title("Polygon Viewer")
        self.root.geometry("1100x850")
        self.tile_library: TileLibrary | None = None  # chosen on first Create Image
        self.bg_color: str = "#ffffff"  # applied to preview and exported PNG

        # menu
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open CSV...", command=self.open_csv, accelerator="Ctrl+O")
        filemenu.add_command(label="Reload", command=self.reload, accelerator="F5")
        filemenu.add_separator()
        filemenu.add_command(label="Create Image...", command=self.create_image,
                             accelerator="Ctrl+E")
        filemenu.add_command(label="Set Tile Library...", command=self.set_tile_library,
                             accelerator="Ctrl+L")
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=filemenu)

        viewmenu = tk.Menu(menubar, tearoff=0)
        viewmenu.add_command(label="Fit to data", command=self.fit_view, accelerator="F")
        self.show_outlines = tk.BooleanVar(value=False)
        viewmenu.add_checkbutton(label="Show outlines", variable=self.show_outlines,
                                 command=self._render)
        self.invert_y = tk.BooleanVar(value=True)
        viewmenu.add_checkbutton(label="Invert Y axis (image style)", variable=self.invert_y,
                                 command=self._render)
        viewmenu.add_separator()
        viewmenu.add_command(label="Background Color...", command=self.choose_bg_color)
        menubar.add_cascade(label="View", menu=viewmenu)
        root.config(menu=menubar)

        # bindings
        root.bind_all("<Control-o>", lambda e: self.open_csv())
        root.bind_all("<Control-q>", lambda e: root.quit())
        root.bind_all("<Control-e>", lambda e: self.create_image())
        root.bind_all("<Control-l>", lambda e: self.set_tile_library())
        root.bind_all("<F5>", lambda e: self.reload())
        root.bind_all("<f>", lambda e: self.fit_view())

        # matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8), dpi=100)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor(self.bg_color)
        self.fig.tight_layout()

        # canvas + toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_frame = tk.Frame(root)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        # status bar
        self.status = tk.Label(
            root,
            text="Open a CSV to view polygons   (File > Open CSV  or  Ctrl+O)",
            bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=8,
        )
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # state
        self.coords_list: list = []
        self.colors_list: list = []
        self.bbox: tuple | None = None
        self.current_path: Path | None = None

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if initial_csv:
            self.load_and_render(initial_csv)
        else:
            self._initial_message()

    def _on_close(self) -> None:
        """Cleanly shut down so closing via the window 'X' doesn't hang on the matplotlib figure."""
        plt.close(self.fig)
        self.root.quit()
        self.root.destroy()

    def _initial_message(self) -> None:
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_facecolor(self.bg_color)
        self.ax.text(0.5, 0.5, "Open a CSV  (File > Open CSV)",
                     transform=self.ax.transAxes, ha="center", va="center",
                     fontsize=14, color="#888")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    def open_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Open polygon CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.load_and_render(Path(path))

    def reload(self) -> None:
        if self.current_path is not None:
            self.load_and_render(self.current_path)

    def load_and_render(self, path: Path) -> None:
        try:
            coords, colors, bbox = load_polygons(path)
        except Exception as e:
            messagebox.showerror("Error loading CSV", f"{type(e).__name__}: {e}")
            return

        self.coords_list = coords
        self.colors_list = colors
        self.bbox = bbox
        self.current_path = path
        self._render()
        self.fit_view()

        xmin, ymin, xmax, ymax = bbox
        self.status.config(
            text=(f"{path.name}   |   {len(coords)} polygons   |   "
                  f"bbox: x [{xmin:.1f}, {xmax:.1f}], y [{ymin:.1f}, {ymax:.1f}]")
        )

    def _render(self) -> None:
        if not self.coords_list:
            return
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_facecolor(self.bg_color)

        edge = "#222" if self.show_outlines.get() else "none"
        lw = 0.5 if self.show_outlines.get() else 0
        coll = PolyCollection(
            self.coords_list,
            facecolors=self.colors_list,
            edgecolors=edge,
            linewidths=lw,
        )
        self.ax.add_collection(coll)

        if self.invert_y.get() and not self.ax.yaxis_inverted():
            self.ax.invert_yaxis()
        elif not self.invert_y.get() and self.ax.yaxis_inverted():
            self.ax.invert_yaxis()

        if self.bbox:
            xmin, ymin, xmax, ymax = self.bbox
            pad_x = (xmax - xmin) * 0.02
            pad_y = (ymax - ymin) * 0.02
            self.ax.set_xlim(xmin - pad_x, xmax + pad_x)
            if self.invert_y.get():
                self.ax.set_ylim(ymax + pad_y, ymin - pad_y)
            else:
                self.ax.set_ylim(ymin - pad_y, ymax + pad_y)
        self.canvas.draw()

    def fit_view(self) -> None:
        if not self.bbox:
            return
        xmin, ymin, xmax, ymax = self.bbox
        pad_x = (xmax - xmin) * 0.02
        pad_y = (ymax - ymin) * 0.02
        self.ax.set_xlim(xmin - pad_x, xmax + pad_x)
        if self.invert_y.get():
            self.ax.set_ylim(ymax + pad_y, ymin - pad_y)
        else:
            self.ax.set_ylim(ymin - pad_y, ymax + pad_y)
        self.canvas.draw()

    # -----------------------------------------------------------------------
    # Create Image: render polygons filled with their nearest-matching tiles
    # -----------------------------------------------------------------------

    def _pick_tile_library(self, title: str = "Choose tile library folder") -> bool:
        """Prompt for a tile-library folder, validate it, store on self.tile_library.

        Returns True if a valid library is now set, False if the user cancelled or
        the chosen folder has no hex-named PNGs.
        """
        if self.tile_library is not None:
            initial = str(self.tile_library.tiles_dir)
        elif DEFAULT_TILES_DIR.exists():
            initial = str(DEFAULT_TILES_DIR)
        else:
            initial = str(ROOT_DIR / "output") if (ROOT_DIR / "output").exists() else str(ROOT_DIR)

        chosen = filedialog.askdirectory(
            title=title,
            initialdir=initial,
            mustexist=True,
            parent=self.root,
        )
        if not chosen:
            return False

        folder = Path(chosen)
        n = _count_hex_pngs(folder)
        if n == 0:
            messagebox.showerror(
                "Not a tile library",
                f"No hex-named PNGs (e.g. 5E251A.png) found in:\n{folder}\n\n"
                "Pick a folder produced by scripts/divide_into_tesserae.py.",
            )
            return False

        self.tile_library = TileLibrary(folder)
        self.status.config(text=f"tile library: {folder}  ({n} tiles)")
        return True

    def set_tile_library(self) -> None:
        """Menu command: switch the active tile library."""
        self._pick_tile_library(title="Set tile library folder")

    def choose_bg_color(self) -> None:
        """Pick a background color applied to the preview and exported PNG."""
        _, hex_str = colorchooser.askcolor(color=self.bg_color, parent=self.root,
                                           title="Background color")
        if not hex_str:
            return
        self.bg_color = hex_str
        if self.coords_list:
            self._render()
        else:
            self._initial_message()

    def create_image(self) -> None:
        if not self.coords_list:
            messagebox.showinfo("No data", "Open a CSV first.")
            return
        if self.tile_library is None:
            if not self._pick_tile_library(title="Choose tile library for rendering"):
                return

        dpi = simpledialog.askinteger(
            "Output DPI",
            "Pixels per inch (300 = print quality):",
            initialvalue=300, minvalue=72, maxvalue=2400, parent=self.root,
        )
        if dpi is None:
            return

        # Suggest the bbox size as default original. User may override.
        xmin, ymin, xmax, ymax = self.bbox  # type: ignore[misc]
        default_orig = round(max(xmax - xmin, ymax - ymin), 1)
        orig_mm = simpledialog.askfloat(
            "Original tile size",
            "Original tile size in mm\n(the size the polygon design was made for):",
            initialvalue=default_orig, minvalue=1.0, maxvalue=10000.0, parent=self.root,
        )
        if orig_mm is None:
            return

        final_mm = simpledialog.askfloat(
            "Final tile size",
            f"Desired final tile size in mm\n"
            f"(same as original = no scaling.  Original was {orig_mm} mm):",
            initialvalue=orig_mm, minvalue=1.0, maxvalue=10000.0, parent=self.root,
        )
        if final_mm is None:
            return

        scale = final_mm / orig_mm
        scale_tag = "" if abs(scale - 1.0) < 1e-6 else f"_x{scale:.3f}"
        default_name = (
            (self.current_path.stem if self.current_path else "polygons")
            + f"_{dpi}dpi{scale_tag}.png"
        )
        out_path = filedialog.asksaveasfilename(
            title="Save rendered image as...",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=default_name,
            initialdir=str(ROOT_DIR / "output"),
        )
        if not out_path:
            return

        try:
            self._do_render(Path(out_path), dpi, orig_mm=orig_mm, final_mm=final_mm)
        except Exception as e:
            messagebox.showerror("Render failed", f"{type(e).__name__}: {e}")
            self.status.config(text=f"render failed: {e}")
            return

    def _do_render(self, out_path: Path, dpi: int,
                    orig_mm: float = 1.0, final_mm: float = 1.0) -> None:
        # Lazy-load tile index (fast — ~800 hex parses + Lab conversion)
        self.tile_library.load_index()

        # mm -> pixels conversion (interpreting CSV coordinates as millimeters)
        px_per_mm = dpi / 25.4
        scale = final_mm / orig_mm if orig_mm > 0 else 1.0

        xmin, ymin, xmax, ymax = self.bbox  # type: ignore[misc]
        canvas_w = max(1, math.ceil((xmax - xmin) * px_per_mm))
        canvas_h = max(1, math.ceil((ymax - ymin) * px_per_mm))

        self.status.config(
            text=f"rendering {len(self.coords_list)} polygons at {dpi} DPI -> "
                 f"{canvas_w}x{canvas_h} px (orig {(xmax-xmin):.1f} x {(ymax-ymin):.1f} mm, scale x{scale:.3f})..."
        )
        self.root.update()

        bg_rgb = tuple(int(self.bg_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        canvas_img = Image.new("RGB", (canvas_w, canvas_h), bg_rgb)
        rng = np.random.default_rng(seed=0)  # deterministic rotation angles per run

        n = len(self.coords_list)
        update_every = max(1, n // 20)
        for idx, (poly_mm, color) in enumerate(zip(self.coords_list, self.colors_list)):
            # 1. polygon RGB (ignore alpha for print)
            r, g, b = color[0], color[1], color[2]

            # 2. find nearest tile in Lab space
            hex_str = self.tile_library.nearest_hex((r, g, b))
            tile = self.tile_library.get_tile(hex_str)  # 500x500 RGB

            # 3. convert polygon to canvas pixel coords (shift to canvas origin)
            poly_px = [((x - xmin) * px_per_mm, (y - ymin) * px_per_mm) for x, y in poly_mm]
            xs = [p[0] for p in poly_px]
            ys = [p[1] for p in poly_px]
            bx0, by0 = math.floor(min(xs)), math.floor(min(ys))
            bx1, by1 = math.ceil(max(xs)), math.ceil(max(ys))
            bw, bh = bx1 - bx0, by1 - by0
            if bw <= 0 or bh <= 0:
                continue

            # 4. rotate the tile by a random deterministic angle
            angle = float(rng.uniform(0, 360))
            rotated = tile.rotate(angle, resample=Image.BILINEAR, expand=True,
                                  fillcolor=(int(r * 255), int(g * 255), int(b * 255)))
            rW, rH = rotated.size

            # 5. fit the bbox: crop from rotated tile, or upscale if polygon is huge
            if bw > rW or bh > rH:
                crop = rotated.resize((bw, bh), Image.LANCZOS)
            else:
                cx = (rW - bw) // 2
                cy = (rH - bh) // 2
                crop = rotated.crop((cx, cy, cx + bw, cy + bh))

            # 6. create polygon-shaped mask (bbox-local coords)
            mask = Image.new("L", (bw, bh), 0)
            ImageDraw.Draw(mask).polygon(
                [(x - bx0, y - by0) for x, y in poly_px], fill=255,
            )

            # 7. paste the masked crop onto the canvas
            canvas_img.paste(crop, (bx0, by0), mask=mask)

            if (idx + 1) % update_every == 0:
                self.status.config(
                    text=f"rendering... {idx + 1}/{n} polygons"
                )
                self.root.update()

        # Apply final scale (the ratio between desired and original tile size).
        # Polygons that extend beyond the nominal tile will also be scaled —
        # so the resulting image's physical size is the bbox scaled, not exactly the requested tile size.
        if abs(scale - 1.0) > 1e-6:
            self.status.config(text=f"scaling output by x{scale:.3f}...")
            self.root.update()
            new_w = max(1, int(round(canvas_w * scale)))
            new_h = max(1, int(round(canvas_h * scale)))
            canvas_img = canvas_img.resize((new_w, new_h), Image.LANCZOS)
            final_w, final_h = new_w, new_h
        else:
            final_w, final_h = canvas_w, canvas_h

        self.status.config(text=f"saving {final_w}x{final_h} PNG...")
        self.root.update()
        canvas_img.save(out_path, "PNG", dpi=(dpi, dpi))

        # Physical sizes
        orig_mm_w = xmax - xmin
        orig_mm_h = ymax - ymin
        final_mm_w = orig_mm_w * scale
        final_mm_h = orig_mm_h * scale

        self.status.config(
            text=f"saved: {out_path.name}  |  {final_w}x{final_h} px @ {dpi} DPI  |  "
                 f"physical: {final_mm_w:.1f} x {final_mm_h:.1f} mm"
        )
        scale_line = ""
        if abs(scale - 1.0) > 1e-6:
            scale_line = (
                f"Scale: x{scale:.3f}  (requested tile {orig_mm} → {final_mm} mm)\n"
                f"Note: polygons extending beyond the tile boundary scale too,\n"
                f"      so the actual image is {final_mm_w:.1f} x {final_mm_h:.1f} mm,\n"
                f"      not exactly {final_mm} x {final_mm} mm.\n\n"
            )
        messagebox.showinfo(
            "Image created",
            f"Saved {out_path.name}\n\n"
            f"Resolution: {final_w} x {final_h} px @ {dpi} DPI\n"
            f"Physical size when printed: {final_mm_w:.1f} x {final_mm_h:.1f} mm\n"
            + scale_line +
            f"Polygons rendered: {n}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", help="optional CSV file to open at startup")
    args = parser.parse_args()

    initial = Path(args.csv) if args.csv else None
    if initial and not initial.exists():
        print(f"ERROR: file not found: {initial}")
        return 1

    root = tk.Tk()
    PolygonViewer(root, initial_csv=initial)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
