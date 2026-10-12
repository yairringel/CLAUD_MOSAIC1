"""Mosaic Vectorizer: load a mosaic photo, detect each stone with Segment Anything (SAM),
fix mistakes by clicking, and export one vector shape per stone (SVG / DXF / JSON)."""
import math
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk

from sam_engine import CHECKPOINTS, SamEngine, StoneMap
from vectorize import stone_polygon, to_dxf, to_json, to_svg


def make_palette(n=4096):
    pal = np.zeros((n, 3), np.uint8)
    for i in range(n):
        h = (i * 137.508) % 360 / 60
        x = 1 - abs(h % 2 - 1)
        r, g, b = [(1, x, 0), (x, 1, 0), (0, 1, x), (0, x, 1), (x, 0, 1), (1, 0, x)][int(h) % 6]
        pal[i] = [int(40 + 200 * v) for v in (r, g, b)]
    return pal


PALETTE = make_palette()
PROCESS_SIZES = ['1024', '1536', '2048', '3072', 'Full']


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Mosaic Vectorizer (SAM)')
        self.geometry('1400x900')
        self.minsize(900, 600)

        self.engine = SamEngine()
        self.smap = None
        self.source = None          # original PIL image
        self.photo = None           # work-resolution RGB numpy array
        self.overlay = None         # work-resolution RGBA numpy array
        self.scale = 1.0            # original px per work px
        self.image_name = 'mosaic'
        self.busy = False
        self.last_click = None
        self.view = [1.0, 0.0, 0.0]  # zoom, x offset, y offset (screen = work * zoom + offset)
        self.drag = None
        self.render_pending = False
        self.tk_img = None
        self.events = queue.Queue()

        self._build_ui()
        self.after(50, self._poll_events)

    # ---------------- UI ----------------

    def _build_ui(self):
        side = ttk.Frame(self, padding=10, width=320)
        side.pack(side='left', fill='y')
        side.pack_propagate(False)
        main = ttk.Frame(self)
        main.pack(side='left', fill='both', expand=True)

        def section(title):
            ttk.Label(side, text=title, font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(10, 4))

        section('1 · Image')
        row = ttk.Frame(side)
        row.pack(fill='x')
        ttk.Button(row, text='Open image…', command=self.open_image).pack(side='left')
        ttk.Label(row, text='  Process at').pack(side='left')
        self.proc_size = tk.StringVar(value='2048')
        cb = ttk.Combobox(row, textvariable=self.proc_size, values=PROCESS_SIZES, width=6, state='readonly')
        cb.pack(side='left', padx=4)
        cb.bind('<<ComboboxSelected>>', lambda e: self.reload_image())
        self.img_info = ttk.Label(side, text='No image loaded', foreground='gray', wraplength=300)
        self.img_info.pack(anchor='w')

        section('2 · Model')
        row = ttk.Frame(side)
        row.pack(fill='x')
        self.model_type = tk.StringVar(value='vit_b')
        ttk.Combobox(row, textvariable=self.model_type, values=list(CHECKPOINTS), width=8,
                     state='readonly').pack(side='left')
        self.load_btn = ttk.Button(row, text='Load model', command=self.load_model)
        self.load_btn.pack(side='left', padx=6)
        self.model_info = ttk.Label(side, wraplength=300, foreground='gray',
                                    text='vit_b 375 MB · vit_l 1.2 GB · vit_h 2.5 GB (downloaded once)')
        self.model_info.pack(anchor='w')

        section('3 · Detect stones')
        self.tiles = self._slider(side, 'Tiles (more = more detail for small stones)', 1, 5, 2, '{0:.0f} × {0:.0f}', step=1)
        self.pps = self._slider(side, 'Points per side (per tile)', 8, 96, 32, '{:.0f}', step=4)
        self.iou = self._slider(side, 'Min predicted quality (IoU)', 0.5, 0.99, 0.80, '{:.2f}')
        self.stab = self._slider(side, 'Min stability', 0.5, 0.99, 0.85, '{:.2f}')
        self.min_area = self._slider(side, 'Min stone area (px)', 0, 2000, 40, '{:.0f}', step=10)
        self.max_area = self._slider(side, 'Max stone area (% of image)', 0.1, 25, 3, '{:.1f}%')
        row = ttk.Frame(side)
        row.pack(fill='x', pady=4)
        self.run_btn = ttk.Button(row, text='Detect stones', command=self.run_auto)
        self.run_btn.pack(side='left')
        self.stop_btn = ttk.Button(row, text='Stop', command=lambda: setattr(self, 'stop_requested', True))
        self.stop_btn.pack(side='left', padx=6)
        self.clear_btn = ttk.Button(row, text='Clear all', command=self.clear_all)
        self.clear_btn.pack(side='left')
        self.count_lbl = ttk.Label(side, text='0 stones')
        self.count_lbl.pack(anchor='w')
        self.progress = ttk.Progressbar(side, mode='determinate', maximum=100)
        self.progress.pack(fill='x')
        self.stop_requested = False

        section('4 · Edit')
        ttk.Label(side, foreground='gray', justify='left', wraplength=300, text=(
            'Click a stone to add / re-segment it\n'
            'Click again on the same spot to cycle SAM\'s 3 masks\n'
            'Right-click or Shift+click to delete a stone\n'
            'Drag to pan · wheel to zoom · F to fit')).pack(anchor='w')

        section('5 · View & vector')
        self.mode = tk.StringVar(value='overlay')
        row = ttk.Frame(side)
        row.pack(fill='x')
        for text, val in [('Photo', 'image'), ('Masks', 'overlay'), ('Outlines', 'outline'), ('Vector', 'vector')]:
            ttk.Radiobutton(row, text=text, value=val, variable=self.mode,
                            command=self.request_render).pack(side='left')
        self.tol = self._slider(side, 'Simplify tolerance (px)', 0, 6, 1, '{:.2f}', on_change=self.invalidate_polys)
        self.smooth = self._slider(side, 'Corner smoothing', 0, 3, 0, '{:.0f}', step=1, on_change=self.invalidate_polys)

        section('6 · Export')
        row = ttk.Frame(side)
        row.pack(fill='x')
        ttk.Label(row, text='Real width (mm, 0 = pixels)').pack(side='left')
        self.width_mm = tk.StringVar(value='0')
        ttk.Entry(row, textvariable=self.width_mm, width=8).pack(side='left', padx=6)
        self.fill_var = tk.BooleanVar(value=True)
        self.stroke_var = tk.BooleanVar(value=True)
        self.embed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text='Fill stones with their colour', variable=self.fill_var).pack(anchor='w')
        ttk.Checkbutton(side, text='Draw outlines', variable=self.stroke_var).pack(anchor='w')
        ttk.Checkbutton(side, text='Embed photo as background layer', variable=self.embed_var).pack(anchor='w')
        row = ttk.Frame(side)
        row.pack(fill='x', pady=4)
        self.export_btns = [ttk.Button(row, text=t, command=lambda f=t: self.export(f)) for t in ('SVG', 'DXF', 'JSON')]
        for b in self.export_btns:
            b.pack(side='left', padx=(0, 6))

        self.canvas = tk.Canvas(main, bg='#2b2b2b', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        self.status = ttk.Label(main, text='Open an image and load a model.', anchor='w', padding=(8, 4))
        self.status.pack(fill='x')

        c = self.canvas
        c.bind('<Configure>', lambda e: self.request_render())
        c.bind('<ButtonPress-1>', self.on_press)
        c.bind('<B1-Motion>', self.on_drag)
        c.bind('<ButtonRelease-1>', self.on_release)
        c.bind('<ButtonRelease-3>', lambda e: self.on_click(e, delete=True))
        c.bind('<MouseWheel>', self.on_wheel)
        c.bind('<Button-4>', lambda e: self.on_wheel(e, 120))
        c.bind('<Button-5>', lambda e: self.on_wheel(e, -120))
        self.bind('<KeyPress-f>', lambda e: self.fit_view())
        self.bind('<KeyPress-F>', lambda e: self.fit_view())
        self.refresh_buttons()

    def _slider(self, parent, text, lo, hi, value, fmt, step=None, on_change=None):
        var = tk.DoubleVar(value=value)
        frame = ttk.Frame(parent)
        frame.pack(fill='x')
        ttk.Label(frame, text=text).pack(side='left')
        val_lbl = ttk.Label(frame, foreground='gray')
        val_lbl.pack(side='right')

        def changed(_=None):
            if step:
                var.set(round(var.get() / step) * step)
            val_lbl.config(text=fmt.format(var.get()))
            if on_change:
                on_change()

        ttk.Scale(parent, from_=lo, to=hi, variable=var, command=changed).pack(fill='x')
        val_lbl.config(text=fmt.format(value))
        return var

    def set_status(self, text):
        self.status.config(text=text)

    def refresh_buttons(self):
        ready = self.smap is not None and self.engine.sam is not None
        has_stones = self.smap is not None and len(self.smap.stones) > 0
        self.run_btn.state(['!disabled'] if ready and not self.busy else ['disabled'])
        self.stop_btn.state(['!disabled'] if self.busy else ['disabled'])
        self.load_btn.state(['disabled'] if self.busy else ['!disabled'])
        self.clear_btn.state(['!disabled'] if has_stones and not self.busy else ['disabled'])
        for b in self.export_btns:
            b.state(['!disabled'] if has_stones and not self.busy else ['disabled'])
        self.count_lbl.config(text=f'{len(self.smap.stones) if self.smap else 0} stones')

    # ---------------- background work ----------------

    def run_task(self, work, done, busy_text):
        """Run work(report) on a thread; done(result) runs on the UI thread afterwards."""
        self.busy = True
        self.stop_requested = False
        self.refresh_buttons()
        self.set_status(busy_text)
        self.progress.config(mode='indeterminate')
        self.progress.start(12)
        report = lambda msg, frac=None: self.events.put(('status', msg, frac))

        def target():
            try:
                self.events.put(('done', done, work(report)))
            except Exception as e:  # surfaced to the user in the status bar
                import traceback
                traceback.print_exc()
                self.events.put(('error', e))

        threading.Thread(target=target, daemon=True).start()

    def _poll_events(self):
        try:
            while True:
                ev = self.events.get_nowait()
                if ev[0] == 'status':
                    self.set_status(ev[1])
                    if ev[2] is not None:
                        self.progress.stop()
                        self.progress.config(mode='determinate', value=ev[2] * 100)
                    continue
                self.busy = False
                self.progress.stop()
                self.progress.config(mode='determinate', value=0)
                if ev[0] == 'done':
                    ev[1](ev[2])
                else:
                    self.set_status(f'Error: {ev[1]}')
                    messagebox.showerror('Error', str(ev[1]))
                self.sync_overlay()
                self.refresh_buttons()
        except queue.Empty:
            pass
        if self.busy:
            self.sync_overlay()
        self.after(100, self._poll_events)

    # ---------------- image / model ----------------

    def open_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp'),
                                                     ('All files', '*.*')])
        if not path:
            return
        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        except Exception as e:
            messagebox.showerror('Error', f'Could not open image:\n{e}')
            return
        self.source = img
        self.image_name = os.path.splitext(os.path.basename(path))[0]
        self.reload_image(confirm=False)

    def reload_image(self, confirm=True):
        if self.source is None or self.busy:
            return
        if confirm and self.smap and self.smap.stones and not messagebox.askyesno(
                'Change processing size', 'This clears the detected stones. Continue?'):
            return
        W, H = self.source.size
        size = self.proc_size.get()
        k = 1.0 if size == 'Full' else min(1.0, int(size) / max(W, H))
        work = self.source if k == 1.0 else self.source.resize((round(W * k), round(H * k)), Image.LANCZOS)
        self.photo = np.asarray(work).copy()
        self.scale = W / work.width
        self.smap = StoneMap(self.photo)
        self.overlay = np.zeros((work.height, work.width, 4), np.uint8)
        self.last_click = None
        note = f' (processing at {work.width} × {work.height})' if k < 1 else ''
        self.img_info.config(text=f'{self.image_name} — {W} × {H}px{note}')
        self.fit_view()
        self.refresh_buttons()
        self.set_status('Image loaded. ' + ('Press “Detect stones”.' if self.engine.sam else 'Now load a model.'))

    def load_model(self):
        mt = self.model_type.get()

        def work(report):
            last = [0]

            def progress(f):
                pct = int(f * 100)
                if pct != last[0]:
                    last[0] = pct
                    report(f'Downloading {mt} checkpoint… {pct}%')

            self.engine.load(mt, progress)
            return mt

        def done(_):
            self.model_info.config(text=f'Loaded SAM {mt} on {self.engine.device.upper()}')
            self.set_status('Model ready.' + (' Press “Detect stones” or click stones.' if self.smap else ' Open an image.'))

        self.run_task(work, done, f'Loading SAM {mt}…')

    # ---------------- segmentation ----------------

    def run_auto(self):
        smap, t0 = self.smap, time.time()
        params = dict(points_per_side=int(self.pps.get()), tiles=int(self.tiles.get()),
                      pred_iou_thresh=self.iou.get(), stability_thresh=self.stab.get(),
                      min_area=self.min_area.get() / self.scale ** 2, max_area_frac=self.max_area.get() / 100)

        def done(added):
            verb = 'Stopped' if self.stop_requested else 'Done'
            self.set_status(f'{verb}: {added} new stones in {time.time() - t0:.0f}s. Click missed stones to add them.')

        self.run_task(lambda report: self.engine.auto_segment(
            smap, report=report, should_stop=lambda: self.stop_requested, **params), done, 'Detecting stones…')

    def clear_all(self):
        if self.smap and messagebox.askyesno('Clear', f'Remove all {len(self.smap.stones)} stones?'):
            self.smap.clear()
            self.last_click = None
            self.sync_overlay()
            self.refresh_buttons()

    def on_click(self, event, delete=False):
        if self.smap is None or self.busy:
            return
        z, ox, oy = self.view
        x, y = int((event.x - ox) // z), int((event.y - oy) // z)
        if not (0 <= x < self.smap.W and 0 <= y < self.smap.H):
            return
        if delete or event.state & 0x0001:  # right button or Shift
            sid = self.smap.stone_at(x, y)
            if sid:
                self.smap.remove(sid)
                self.set_status(f'Deleted stone #{sid}.')
            self.last_click = None
            self.sync_overlay()
            self.refresh_buttons()
            return
        if self.engine.sam is None:
            self.set_status('Load a model first.')
            return
        here = self.smap.stone_at(x, y)
        lc = self.last_click
        cycle = lc and here and here == lc['id'] and math.hypot(x - lc['x'], y - lc['y']) < 6
        mask_index = lc['k'] + 1 if cycle else None
        smap, max_frac, tiles = self.smap, self.max_area.get() / 100, int(self.tiles.get())
        first = not self.engine.has_embedding(smap, self.engine.tile_for(smap, x, y, tiles)['crop'])

        def done(r):
            self.last_click = {'x': x, 'y': y, 'id': r['id'], 'k': r['k']} if r['id'] else None
            self.set_status(f'Stone #{r["id"]}: mask {r["k"] + 1}/3 (score {r["score"]:.2f}). '
                            'Click again to try another mask.' if r['id'] else 'SAM found nothing there.')

        self.run_task(lambda report: self.engine.click(smap, x, y, mask_index, max_frac, tiles), done,
                      'Computing image embedding for this tile…' if first else 'Segmenting…')

    # ---------------- view ----------------

    def fit_view(self):
        if self.smap is None:
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        z = min(cw / self.smap.W, ch / self.smap.H) * 0.96
        self.view = [z, (cw - self.smap.W * z) / 2, (ch - self.smap.H * z) / 2]
        self.request_render()

    def on_press(self, e):
        self.drag = {'x': e.x, 'y': e.y, 'ox': self.view[1], 'oy': self.view[2], 'moved': False}

    def on_drag(self, e):
        d = self.drag
        if not d:
            return
        if not d['moved'] and math.hypot(e.x - d['x'], e.y - d['y']) < 4:
            return
        d['moved'] = True
        self.view[1] = d['ox'] + e.x - d['x']
        self.view[2] = d['oy'] + e.y - d['y']
        self.request_render()

    def on_release(self, e):
        d, self.drag = self.drag, None
        if d and not d['moved']:
            self.on_click(e)

    def on_wheel(self, e, delta=None):
        if self.smap is None:
            return
        delta = delta if delta is not None else e.delta
        z, ox, oy = self.view
        nz = min(40.0, max(0.02, z * math.exp(delta / 120 * 0.18)))
        self.view = [nz, e.x - (e.x - ox) * nz / z, e.y - (e.y - oy) * nz / z]
        self.request_render()

    def invalidate_polys(self):
        if self.smap:
            for s in list(self.smap.stones.values()):
                s['poly'] = None
        self.request_render()

    def sync_overlay(self):
        """Repaint the mask overlay where labels changed; stone borders get a stronger alpha."""
        if self.smap is None:
            return
        d = self.smap.take_dirty()
        if d:
            W, H = self.smap.W, self.smap.H
            x0, y0, x1, y1 = max(0, d[0] - 2), max(0, d[1] - 2), min(W, d[2] + 2), min(H, d[3] + 2)
            lab = self.smap.labels[y0:y1, x0:x1]
            edge = np.zeros(lab.shape, bool)
            edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
            edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
            edge[1:] |= lab[1:] != lab[:-1]
            edge[:-1] |= lab[:-1] != lab[1:]
            ov = self.overlay[y0:y1, x0:x1]
            ov[..., :3] = PALETTE[lab % len(PALETTE)]
            ov[..., 3] = np.where(lab == 0, 0, np.where(edge, 235, 105))
        self.count_lbl.config(text=f'{len(self.smap.stones)} stones')
        self.request_render()

    def request_render(self):
        if not self.render_pending:
            self.render_pending = True
            self.after_idle(self.render)

    def render(self):
        self.render_pending = False
        c = self.canvas
        c.delete('all')
        if self.smap is None:
            c.create_text(c.winfo_width() // 2, c.winfo_height() // 2, text='Open a mosaic image to begin',
                          fill='#aaa', font=('Segoe UI', 14))
            return
        z, ox, oy = self.view
        cw, ch = c.winfo_width(), c.winfo_height()
        W, H = self.smap.W, self.smap.H
        x0, y0 = max(0, int((0 - ox) // z)), max(0, int((0 - oy) // z))
        x1, y1 = min(W, int(math.ceil((cw - ox) / z))), min(H, int(math.ceil((ch - oy) / z)))
        if x1 <= x0 or y1 <= y0:
            return
        dw, dh = max(1, round((x1 - x0) * z)), max(1, round((y1 - y0) * z))
        resample = Image.NEAREST if z >= 2 else Image.BILINEAR
        mode = self.mode.get()
        if mode == 'vector':
            img = Image.new('RGBA', (dw, dh), 'white')
        else:
            img = Image.fromarray(self.photo[y0:y1, x0:x1]).resize((dw, dh), resample).convert('RGBA')
        if mode == 'overlay':
            ov = Image.fromarray(self.overlay[y0:y1, x0:x1], 'RGBA').resize((dw, dh), resample)
            img = Image.alpha_composite(img, ov)
        if mode in ('outline', 'vector'):
            draw = ImageDraw.Draw(img)
            tol, smooth = self.tol.get(), int(self.smooth.get())
            for sid, s in list(self.smap.stones.items()):
                bx0, by0, bx1, by1 = s['bbox']
                if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
                    continue
                poly = s.get('poly')
                if poly is None:
                    poly = stone_polygon(self.smap, sid, tol, smooth)
                if poly is None:
                    continue
                pts = [((px - x0) * z, (py - y0) * z) for px, py in poly]
                if mode == 'vector':
                    draw.polygon(pts, fill=tuple(s['color']), outline=(0, 0, 0))
                else:
                    draw.polygon(pts, outline=(255, 45, 85), width=2 if z > 1 else 1)
        self.tk_img = ImageTk.PhotoImage(img)
        c.create_image(ox + x0 * z, oy + y0 * z, anchor='nw', image=self.tk_img)

    # ---------------- export ----------------

    def export(self, fmt):
        if not self.smap or not self.smap.stones:
            return
        try:
            width_mm = max(0.0, float(self.width_mm.get() or 0))
        except ValueError:
            messagebox.showerror('Export', 'Real width must be a number.')
            return
        ext = fmt.lower()
        path = filedialog.asksaveasfilename(defaultextension='.' + ext, initialfile=f'{self.image_name}-stones.{ext}',
                                            filetypes=[(fmt, '*.' + ext)])
        if not path:
            return
        args = dict(tolerance=self.tol.get(), smoothing=int(self.smooth.get()), width_mm=width_mm)
        size = self.source.size
        if fmt == 'SVG':
            text = to_svg(self.smap, self.scale, size, fill=self.fill_var.get(), stroke=self.stroke_var.get(),
                          source_image=self.source if self.embed_var.get() else None, **args)
        elif fmt == 'DXF':
            text = to_dxf(self.smap, self.scale, size, **args)
        else:
            text = to_json(self.smap, self.scale, size, **args)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        self.set_status(f'Saved {len(self.smap.stones)} stones to {path}')


if __name__ == '__main__':
    App().mainloop()
