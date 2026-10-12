// Mosaic stone segmentation with Segment Anything (SAM), run in the browser via transformers.js,
// plus raster -> vector conversion and SVG / DXF / JSON export.
import { SamModel, AutoProcessor, RawImage, Tensor, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';

env.allowLocalModels = false;

export const MODELS = [
  { id: 'Xenova/slimsam-77-uniform', label: 'SlimSAM-77 — fast, small download' },
  { id: 'Xenova/sam-vit-base', label: 'SAM ViT-B — slower, more accurate' },
];

const INPUT = 1024; // SAM input side (longest image side is resized to this)

export async function detectDevice() {
  try {
    if (navigator.gpu && await navigator.gpu.requestAdapter()) return 'webgpu';
  } catch { /* fall through */ }
  return 'wasm';
}

const yieldUI = () => new Promise(r => setTimeout(r, 0));

export class MosaicSegmenter {
  constructor() {
    this.model = null;
    this.processor = null;
    this.modelId = null;
    this.device = null;
    this.batch = 16; // prompts per decoder call; drops to 1 if the ONNX export rejects batching
    this.stones = new Map();
    this.nextId = 1;
    this.tiles = [];
    this.tileCount = 0;
    this.dirty = null;
  }

  // ---------- model ----------

  async loadModel(modelId, device, onProgress) {
    const progress_callback = p => {
      if (p.status === 'progress' && onProgress) onProgress(p.file, p.progress);
    };
    const dtype = { vision_encoder: device === 'webgpu' ? 'fp32' : 'q8', prompt_encoder_mask_decoder: 'fp32' };
    let model;
    try {
      model = await SamModel.from_pretrained(modelId, { device, dtype, progress_callback });
    } catch (e) {
      console.warn('Model load with explicit dtype failed, retrying with defaults', e);
      model = await SamModel.from_pretrained(modelId, { device, progress_callback });
    }
    this.processor = await AutoProcessor.from_pretrained(modelId);
    this.model = model;
    this.modelId = modelId;
    this.device = device;
    this.batch = 16;
    for (const t of this.tiles) t.emb = null;
  }

  // ---------- image & tiles ----------

  // The label map ("work" resolution) is the image scaled so its longest side is <= maxWorkSide.
  setImage(source, maxWorkSide = 3072) {
    const W = source.naturalWidth || source.width, H = source.naturalHeight || source.height;
    this.W = W; this.H = H;
    this.src = document.createElement('canvas');
    this.src.width = W; this.src.height = H;
    this.src.getContext('2d').drawImage(source, 0, 0);

    const g = Math.min(1, maxWorkSide / Math.max(W, H));
    this.GW = Math.max(1, Math.round(W * g));
    this.GH = Math.max(1, Math.round(H * g));
    this.kx = this.GW / W; // work px per original px
    this.ky = this.GH / H;
    this.work = document.createElement('canvas');
    this.work.width = this.GW; this.work.height = this.GH;
    const wctx = this.work.getContext('2d', { willReadFrequently: true });
    wctx.imageSmoothingQuality = 'high';
    wctx.drawImage(this.src, 0, 0, this.GW, this.GH);
    this.workData = wctx.getImageData(0, 0, this.GW, this.GH).data;

    this.labels = new Int32Array(this.GW * this.GH);
    this.stones.clear();
    this.nextId = 1;
    this.tiles = [];
    this.tileCount = 0;
    this.markDirty(0, 0, this.GW, this.GH);
  }

  // Split the image into T x T overlapping tiles; each tile gets its own SAM embedding,
  // which gives SAM more resolution per stone on large mosaics.
  configureTiles(T, overlap = 0.15) {
    if (T === this.tileCount) return;
    this.tileCount = T;
    this.tiles = [];
    const { W, H } = this, tw = W / T, th = H / T;
    const ox = Math.round(tw * overlap), oy = Math.round(th * overlap);
    for (let j = 0; j < T; j++) {
      for (let i = 0; i < T; i++) {
        const kx0 = Math.round(i * tw), kx1 = Math.round((i + 1) * tw);
        const ky0 = Math.round(j * th), ky1 = Math.round((j + 1) * th);
        this.tiles.push({
          kx0, ky0, kx1, ky1, // core (seed points live here)
          cx0: Math.max(0, kx0 - ox), cy0: Math.max(0, ky0 - oy),
          cx1: Math.min(W, kx1 + ox), cy1: Math.min(H, ky1 + oy), // crop sent to SAM
          emb: null,
        });
      }
    }
  }

  tileAt(ox, oy) {
    return this.tiles.find(t => ox >= t.kx0 && ox < t.kx1 && oy >= t.ky0 && oy < t.ky1)
      || this.tiles[this.tiles.length - 1];
  }

  async embedTile(t) {
    if (t.emb) return t.emb;
    const cw = t.cx1 - t.cx0, ch = t.cy1 - t.cy0;
    const k = Math.min(1, INPUT / Math.max(cw, ch));
    const w = Math.max(1, Math.round(cw * k)), h = Math.max(1, Math.round(ch * k));
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(this.src, t.cx0, t.cy0, cw, ch, 0, 0, w, h);
    const rgba = ctx.getImageData(0, 0, w, h).data;
    const rgb = new Uint8ClampedArray(w * h * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2];
    }
    const inputs = await this.processor(new RawImage(rgb, w, h, 3));
    const emb = await this.model.get_image_embeddings(inputs);
    const [rh, rw] = inputs.reshaped_input_sizes[0];
    t.emb = { emb, rw, rh, sx: rw / cw, sy: rh / ch };
    return t.emb;
  }

  // ---------- SAM decoding ----------

  // pts are in original-image pixels. Returns one result per point with 3 candidate masks.
  async decode(t, pts) {
    const { emb, sx, sy } = t.emb;
    const P = pts.length;
    const coords = new Float32Array(P * 2);
    pts.forEach(([ox, oy], i) => {
      coords[2 * i] = (ox - t.cx0) * sx;
      coords[2 * i + 1] = (oy - t.cy0) * sy;
    });
    const out = await this.model({
      ...emb,
      input_points: new Tensor('float32', coords, [1, P, 1, 2]),
      input_labels: new Tensor('int64', new BigInt64Array(P).fill(1n), [1, P, 1]),
    });
    const [, Pp, K, mh, mw] = out.pred_masks.dims;
    if (Pp !== P) throw new Error(`Decoder returned ${Pp} prompts, expected ${P}`);
    const iou = out.iou_scores.data, logits = out.pred_masks.data, n = K * mh * mw;
    const res = [];
    for (let i = 0; i < P; i++) {
      res.push({ K, mh, mw, iou: iou.subarray(i * K, (i + 1) * K), logits: logits.subarray(i * n, (i + 1) * n) });
    }
    return res;
  }

  async decodeMany(t, pts) {
    if (this.batch > 1 && pts.length > 1) {
      try {
        return await this.decode(t, pts);
      } catch (e) {
        console.warn('Batched prompts not supported, decoding one point at a time', e);
        this.batch = 1;
      }
    }
    const res = [];
    for (const p of pts) res.push(...await this.decode(t, [p]));
    return res;
  }

  // Bilinear sample of mask k's logits at an original-image point.
  sampleLogit(t, r, k, ox, oy) {
    const { mh, mw } = r, cell = INPUT / mw, off = k * mh * mw;
    const u = Math.min(mw - 1, Math.max(0, ((ox - t.cx0) * t.emb.sx) / cell - 0.5));
    const v = Math.min(mh - 1, Math.max(0, ((oy - t.cy0) * t.emb.sy) / cell - 0.5));
    const u0 = Math.min(Math.floor(u), mw - 2), v0 = Math.min(Math.floor(v), mh - 2);
    const fu = u - u0, fv = v - v0, L = r.logits, i = off + v0 * mw + u0;
    return (L[i] * (1 - fu) + L[i + 1] * fu) * (1 - fv) + (L[i + mw] * (1 - fu) + L[i + mw + 1] * fu) * fv;
  }

  // Statistics of mask k on SAM's low-res grid: positive-cell bbox, count, stability score.
  maskStats(t, r, k) {
    const { mh, mw } = r, L = r.logits, off = k * mh * mw, cell = INPUT / mw;
    const vw = Math.min(mw, Math.ceil(t.emb.rw / cell)), vh = Math.min(mh, Math.ceil(t.emb.rh / cell));
    let x0 = mw, y0 = mh, x1 = -1, y1 = -1, pos = 0, hi = 0, lo = 0;
    for (let y = 0; y < vh; y++) {
      const row = off + y * mw;
      for (let x = 0; x < vw; x++) {
        const v = L[row + x];
        if (v > -1) {
          lo++;
          if (v > 0) {
            pos++;
            if (v > 1) hi++;
            if (x < x0) x0 = x;
            if (x > x1) x1 = x;
            if (y < y0) y0 = y;
            if (y > y1) y1 = y;
          }
        }
      }
    }
    // area of one low-res cell in work pixels
    const cellArea = (cell / t.emb.sx * this.kx) * (cell / t.emb.sy * this.ky);
    return { k, iou: r.iou[k], stab: lo ? hi / lo : 0, pos, area: pos * cellArea, x0, y0, x1, y1, vw, vh };
  }

  // A mask cut off by a tile edge that is not an image edge is incomplete; a neighbouring tile will find it.
  touchesInnerEdge(t, st) {
    return (st.x0 <= 0 && t.cx0 > 0) || (st.y0 <= 0 && t.cy0 > 0)
      || (st.x1 >= st.vw - 1 && t.cx1 < this.W) || (st.y1 >= st.vh - 1 && t.cy1 < this.H);
  }

  // Upsample mask st.k into the work-resolution label grid; returns the covered pixel indices.
  rasterize(t, r, st) {
    const { mh, mw } = r, L = r.logits, off = st.k * mh * mw, cell = INPUT / mw;
    const { sx, sy } = t.emb, { kx, ky, GW, GH } = this;
    const gx0 = Math.max(0, Math.floor(t.cx0 * kx), Math.floor((t.cx0 + ((st.x0 - 1) * cell) / sx) * kx));
    const gx1 = Math.min(GW, Math.ceil(t.cx1 * kx), Math.ceil((t.cx0 + ((st.x1 + 2) * cell) / sx) * kx));
    const gy0 = Math.max(0, Math.floor(t.cy0 * ky), Math.floor((t.cy0 + ((st.y0 - 1) * cell) / sy) * ky));
    const gy1 = Math.min(GH, Math.ceil(t.cy1 * ky), Math.ceil((t.cy0 + ((st.y1 + 2) * cell) / sy) * ky));
    const nx = gx1 - gx0, pixels = [];
    if (nx <= 0 || gy1 <= gy0) return pixels;
    const u0 = new Int32Array(nx), fu = new Float32Array(nx);
    for (let i = 0; i < nx; i++) {
      const u = Math.min(mw - 1, Math.max(0, (((gx0 + i + 0.5) / kx - t.cx0) * sx) / cell - 0.5));
      u0[i] = Math.min(Math.floor(u), mw - 2);
      fu[i] = u - u0[i];
    }
    for (let gy = gy0; gy < gy1; gy++) {
      const v = Math.min(mh - 1, Math.max(0, (((gy + 0.5) / ky - t.cy0) * sy) / cell - 0.5));
      const v0 = Math.min(Math.floor(v), mh - 2), fv = v - v0;
      const row0 = off + v0 * mw, row1 = row0 + mw, base = gy * GW + gx0;
      for (let i = 0; i < nx; i++) {
        const a = row0 + u0[i], c = row1 + u0[i], f = fu[i];
        const val = (L[a] * (1 - f) + L[a + 1] * f) * (1 - fv) + (L[c] * (1 - f) + L[c + 1] * f) * fv;
        if (val > 0) pixels.push(base + i);
      }
    }
    return pixels;
  }

  // ---------- label map editing ----------

  markDirty(x0, y0, x1, y1) {
    const d = this.dirty;
    this.dirty = d
      ? { x0: Math.min(d.x0, x0), y0: Math.min(d.y0, y0), x1: Math.max(d.x1, x1), y1: Math.max(d.y1, y1) }
      : { x0, y0, x1, y1 };
  }

  takeDirty() {
    const d = this.dirty;
    this.dirty = null;
    return d;
  }

  // Add a stone from a pixel list. Without `steal` it only claims free pixels and is rejected
  // when it overlaps existing stones too much; with `steal` it takes pixels from neighbours.
  commit(pixels, { maxOverlap = 0.3, steal = false, minArea = 0 } = {}) {
    if (!pixels.length) return 0;
    const lab = this.labels, GW = this.GW;
    if (!steal) {
      let overlap = 0;
      for (const p of pixels) if (lab[p]) overlap++;
      if (overlap > maxOverlap * pixels.length) return 0;
    }
    const id = this.nextId++, touched = new Set();
    let x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1;
    for (const p of pixels) {
      const o = lab[p];
      if (o && !steal) continue;
      if (o) touched.add(o);
      lab[p] = id;
      const x = p % GW, y = (p / GW) | 0;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
    if (x1 < 0) return 0;
    this.stones.set(id, { id, x0, y0, x1: x1 + 1, y1: y1 + 1, area: 0, color: [0, 0, 0], poly: null });
    this.cleanupStone(id);
    for (const o of touched) this.cleanupStone(o);
    const s = this.stones.get(id);
    if (!s || s.area < minArea) {
      if (s) this.removeStone(id);
      return 0;
    }
    return id;
  }

  // Keep the largest 4-connected component, fill enclosed holes, recompute bbox/area/colour.
  cleanupStone(id) {
    const s = this.stones.get(id);
    if (!s) return;
    const { GW, labels: lab, workData: img } = this;
    const bx = s.x0, by = s.y0, w = s.x1 - bx, h = s.y1 - by, n = w * h;
    this.markDirty(s.x0, s.y0, s.x1, s.y1);
    const gi = i => (by + ((i / w) | 0)) * GW + bx + (i % w);

    const comp = new Int32Array(n), queue = new Int32Array(n);
    let nComp = 0, best = 0, bestSize = 0;
    for (let i = 0; i < n; i++) {
      if (comp[i] || lab[gi(i)] !== id) continue;
      const c = ++nComp;
      let head = 0, tail = 0, size = 0;
      comp[i] = c; queue[tail++] = i;
      while (head < tail) {
        const q = queue[head++], x = q % w;
        size++;
        const nb = [x > 0 ? q - 1 : -1, x < w - 1 ? q + 1 : -1, q - w, q + w];
        for (const m of nb) {
          if (m < 0 || m >= n || comp[m] || lab[gi(m)] !== id) continue;
          comp[m] = c; queue[tail++] = m;
        }
      }
      if (size > bestSize) { bestSize = size; best = c; }
    }
    if (!best) { this.stones.delete(id); return; }

    // drop the smaller fragments
    for (let i = 0; i < n; i++) if (comp[i] && comp[i] !== best) lab[gi(i)] = 0;

    // fill holes: flood "outside" from the bbox border through non-stone cells
    const outside = new Uint8Array(n);
    let head = 0, tail = 0;
    const seed = i => { if (!outside[i] && comp[i] !== best) { outside[i] = 1; queue[tail++] = i; } };
    for (let x = 0; x < w; x++) { seed(x); seed((h - 1) * w + x); }
    for (let y = 0; y < h; y++) { seed(y * w); seed(y * w + w - 1); }
    while (head < tail) {
      const q = queue[head++], x = q % w;
      if (x > 0) seed(q - 1);
      if (x < w - 1) seed(q + 1);
      if (q >= w) seed(q - w);
      if (q < n - w) seed(q + w);
    }

    let x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1, area = 0, r = 0, g = 0, b = 0;
    for (let i = 0; i < n; i++) {
      const p = gi(i);
      if (comp[i] !== best) {
        if (outside[i] || lab[p] !== 0) continue;
        lab[p] = id; // enclosed free pixel
      }
      const x = bx + (i % w), y = by + ((i / w) | 0);
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      area++;
      r += img[p * 4]; g += img[p * 4 + 1]; b += img[p * 4 + 2];
    }
    Object.assign(s, {
      x0, y0, x1: x1 + 1, y1: y1 + 1, area,
      color: [Math.round(r / area), Math.round(g / area), Math.round(b / area)], poly: null,
    });
  }

  removeStone(id) {
    const s = this.stones.get(id);
    if (!s) return;
    const { GW, labels: lab } = this;
    for (let y = s.y0; y < s.y1; y++) {
      for (let x = s.x0; x < s.x1; x++) if (lab[y * GW + x] === id) lab[y * GW + x] = 0;
    }
    this.stones.delete(id);
    this.markDirty(s.x0, s.y0, s.x1, s.y1);
  }

  clearStones() {
    this.labels.fill(0);
    this.stones.clear();
    this.markDirty(0, 0, this.GW, this.GH);
  }

  stoneAt(gx, gy) {
    if (gx < 0 || gy < 0 || gx >= this.GW || gy >= this.GH) return 0;
    return this.labels[gy * this.GW + gx];
  }

  workIndex(ox, oy) {
    const x = Math.min(this.GW - 1, Math.floor(ox * this.kx)), y = Math.min(this.GH - 1, Math.floor(oy * this.ky));
    return y * this.GW + x;
  }

  // ---------- automatic segmentation ----------

  // Prompts SAM with a grid of points per tile (skipping points already inside a stone)
  // and keeps the best-scoring mask per point that passes the quality filters.
  async autoSegment(opts, onProgress = () => {}, shouldStop = () => false) {
    const { pointsPerSide: n, iouThresh, stabThresh, minArea, maxAreaFrac, maxOverlap } = opts;
    const pxArea = this.kx * this.ky;
    const minA = minArea * pxArea, maxA = maxAreaFrac * this.GW * this.GH;
    const total = this.tiles.length * n * n;
    let done = 0, added = 0;

    for (const [ti, t] of this.tiles.entries()) {
      onProgress({ phase: 'embed', tile: ti + 1, tiles: this.tiles.length, done, total, added });
      await yieldUI();
      await this.embedTile(t);
      if (shouldStop()) return added;

      const seeds = [];
      for (let b = 0; b < n; b++) {
        for (let a = 0; a < n; a++) {
          seeds.push([t.kx0 + ((a + 0.5) * (t.kx1 - t.kx0)) / n, t.ky0 + ((b + 0.5) * (t.ky1 - t.ky0)) / n]);
        }
      }
      let idx = 0;
      while (idx < seeds.length) {
        if (shouldStop()) return added;
        const batch = [];
        while (idx < seeds.length && batch.length < this.batch) {
          const p = seeds[idx++];
          done++;
          if (!this.labels[this.workIndex(p[0], p[1])]) batch.push(p);
        }
        if (!batch.length) continue;
        const results = await this.decodeMany(t, batch);
        results.forEach((r, i) => {
          const [ox, oy] = batch[i];
          if (this.labels[this.workIndex(ox, oy)]) return; // covered by a stone found in this batch
          let best = null;
          for (let k = 0; k < r.K; k++) {
            const st = this.maskStats(t, r, k);
            if (!st.pos || st.iou < iouThresh || st.stab < stabThresh) continue;
            if (st.area < minA || st.area > maxA || this.touchesInnerEdge(t, st)) continue;
            if (this.sampleLogit(t, r, k, ox, oy) <= 0) continue; // mask must contain its prompt
            if (!best || st.iou > best.iou) best = st;
          }
          if (best && this.commit(this.rasterize(t, r, best), { maxOverlap, minArea: minA })) added++;
        });
        onProgress({ phase: 'segment', tile: ti + 1, tiles: this.tiles.length, done, total, added });
        await yieldUI();
      }
    }
    return added;
  }

  // ---------- manual editing ----------

  // Segment the stone under a work-resolution point. maskIndex selects one of SAM's 3 candidate
  // masks (null = highest predicted IoU among masks no larger than maxAreaFrac of the image).
  // Replaces whatever stone was there; stone-sized masks take pixels from their neighbours.
  async addAt(gx, gy, maskIndex = null, maxAreaFrac = 1) {
    const ox = (gx + 0.5) / this.kx, oy = (gy + 0.5) / this.ky;
    const t = this.tileAt(ox, oy);
    await this.embedTile(t);
    const [r] = await this.decodeMany(t, [[ox, oy]]);
    const stats = Array.from({ length: r.K }, (_, k) => this.maskStats(t, r, k));
    const maxA = maxAreaFrac * this.GW * this.GH;
    let k = maskIndex;
    if (k == null) {
      const fits = stats.filter(s => s.pos && s.area <= maxA);
      const pool = fits.length ? fits : stats;
      k = pool.reduce((a, b) => (b.iou > a.iou ? b : a)).k;
    }
    const st = stats[k % r.K];
    const old = this.stoneAt(gx, gy);
    if (old) this.removeStone(old);
    if (!st.pos) return { id: 0, k: st.k, iou: st.iou };
    const id = this.commit(this.rasterize(t, r, st), { steal: st.area <= maxA, maxOverlap: 1 });
    return { id, k: st.k, iou: st.iou };
  }

  // ---------- vectorization ----------

  vectorize(id, tolerance = 1, smoothing = 0) {
    const s = this.stones.get(id);
    if (!s) return null;
    const w = s.x1 - s.x0 + 2, h = s.y1 - s.y0 + 2, m = new Uint8Array(w * h);
    for (let y = s.y0; y < s.y1; y++) {
      const row = y * this.GW, my = (y - s.y0 + 1) * w + 1 - s.x0;
      for (let x = s.x0; x < s.x1; x++) if (this.labels[row + x] === id) m[my + x] = 1;
    }
    let pts = traceOuter(m, w, h);
    if (!pts || pts.length < 3) { s.poly = null; return null; }
    pts = simplifyClosed(pts, tolerance);
    if (smoothing > 0) pts = chaikin(pts, smoothing);
    s.poly = pts.map(([x, y]) => [x + s.x0 - 1, y + s.y0 - 1]);
    return s.poly;
  }

  vectorizeAll(tolerance, smoothing) {
    for (const id of this.stones.keys()) this.vectorize(id, tolerance, smoothing);
  }

  // ---------- export (coordinates in original image pixels) ----------

  exportPolys(tolerance, smoothing) {
    const out = [];
    for (const s of this.stones.values()) {
      const poly = s.poly || this.vectorize(s.id, tolerance, smoothing);
      if (poly) out.push({ id: s.id, color: s.color, points: poly.map(([x, y]) => [x / this.kx, y / this.ky]) });
    }
    return out;
  }

  toSVG({ tolerance = 1, smoothing = 0, fill = true, stroke = true, embedImage = false, widthMM = 0 } = {}) {
    const { W, H } = this, f = fmt;
    const size = widthMM > 0 ? `width="${f(widthMM)}mm" height="${f((widthMM * H) / W)}mm"` : `width="${W}" height="${H}"`;
    const sw = f(Math.max(0.5, Math.max(W, H) / 1500));
    const lines = [
      '<?xml version="1.0" encoding="UTF-8"?>',
      `<svg xmlns="http://www.w3.org/2000/svg" ${size} viewBox="0 0 ${W} ${H}">`,
    ];
    if (embedImage) {
      lines.push(`  <g id="source"><image href="${this.src.toDataURL('image/jpeg', 0.9)}" x="0" y="0" width="${W}" height="${H}" opacity="0.5"/></g>`);
    }
    lines.push(`  <g id="stones" stroke-linejoin="round"${stroke ? ` stroke="#000" stroke-width="${sw}"` : ''}>`);
    for (const p of this.exportPolys(tolerance, smoothing)) {
      const d = 'M' + p.points.map(([x, y]) => `${f(x)} ${f(y)}`).join('L') + 'Z';
      lines.push(`    <path id="stone-${p.id}" d="${d}" fill="${fill ? hex(p.color) : 'none'}"/>`);
    }
    lines.push('  </g>', '</svg>');
    return lines.join('\n');
  }

  // AutoCAD R12 DXF: one closed POLYLINE per stone, layer STONES, Y axis pointing up.
  toDXF({ tolerance = 1, smoothing = 0, widthMM = 0 } = {}) {
    const k = widthMM > 0 ? widthMM / this.W : 1, H = this.H;
    const L = ['0', 'SECTION', '2', 'ENTITIES'];
    for (const p of this.exportPolys(tolerance, smoothing)) {
      L.push('0', 'POLYLINE', '8', 'STONES', '66', '1', '70', '1', '10', '0', '20', '0', '30', '0');
      for (const [x, y] of p.points) {
        L.push('0', 'VERTEX', '8', 'STONES', '10', fmt(x * k, 4), '20', fmt((H - y) * k, 4), '30', '0');
      }
      L.push('0', 'SEQEND', '8', 'STONES');
    }
    L.push('0', 'ENDSEC', '0', 'EOF');
    return L.join('\n');
  }

  toJSON({ tolerance = 1, smoothing = 0, widthMM = 0 } = {}) {
    const k = widthMM > 0 ? widthMM / this.W : 1;
    return JSON.stringify({
      width: this.W, height: this.H, units: widthMM > 0 ? 'mm' : 'px',
      stones: this.exportPolys(tolerance, smoothing).map(p => ({
        id: p.id, color: hex(p.color), points: p.points.map(([x, y]) => [+fmt(x * k, 3), +fmt(y * k, 3)]),
      })),
    });
  }
}

// ---------- geometry helpers ----------

function fmt(v, digits = 2) {
  return String(Math.round(v * 10 ** digits) / 10 ** digits);
}

function hex([r, g, b]) {
  return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
}

// Trace pixel-edge boundaries of a binary mask (1-px zero border required) and return the
// longest loop as corner vertices. Walks with the inside on the right; at diagonal pinch
// points it turns right so each loop hugs its pixels.
export function traceOuter(m, w, h) {
  const VW = w + 1, DX = [1, 0, -1, 0], DY = [0, 1, 0, -1];
  const es = [], ed = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      if (!m[i]) continue;
      if (!m[i - w]) { es.push(y * VW + x); ed.push(0); }
      if (!m[i + 1]) { es.push(y * VW + x + 1); ed.push(1); }
      if (!m[i + w]) { es.push((y + 1) * VW + x + 1); ed.push(2); }
      if (!m[i - 1]) { es.push((y + 1) * VW + x); ed.push(3); }
    }
  }
  const E = es.length;
  if (!E) return null;
  const outA = new Int32Array(VW * (h + 1)).fill(-1), outB = new Int32Array(VW * (h + 1)).fill(-1);
  for (let e = 0; e < E; e++) {
    if (outA[es[e]] < 0) outA[es[e]] = e; else outB[es[e]] = e;
  }
  const used = new Uint8Array(E);
  let best = null, bestLen = 0;
  for (let e0 = 0; e0 < E; e0++) {
    if (used[e0]) continue;
    const pts = [];
    let e = e0, prevDir = -1, len = 0;
    while (e >= 0 && !used[e]) {
      used[e] = 1; len++;
      const v = es[e], d = ed[e];
      if (d !== prevDir) pts.push([v % VW, (v / VW) | 0]);
      prevDir = d;
      const end = v + DX[d] + DY[d] * VW, a = outA[end], b = outB[end];
      if (b >= 0 && !used[a] && !used[b]) e = ed[b] === (d + 1) % 4 ? b : a;
      else if (a >= 0 && !used[a]) e = a;
      else if (b >= 0 && !used[b]) e = b;
      else e = -1;
    }
    if (len > bestLen) { bestLen = len; best = pts; }
  }
  return best;
}

function douglasPeucker(pts, tol) {
  const n = pts.length;
  if (n < 3) return pts.slice();
  const keep = new Uint8Array(n);
  keep[0] = keep[n - 1] = 1;
  const stack = [[0, n - 1]], t2 = tol * tol;
  while (stack.length) {
    const [a, b] = stack.pop();
    const [ax, ay] = pts[a], [bx, by] = pts[b], dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
    let far = -1, fd = t2;
    for (let i = a + 1; i < b; i++) {
      const [px, py] = pts[i];
      let d;
      if (L2 === 0) d = (px - ax) ** 2 + (py - ay) ** 2;
      else { const c = dx * (py - ay) - dy * (px - ax); d = (c * c) / L2; }
      if (d > fd) { fd = d; far = i; }
    }
    if (far >= 0) { keep[far] = 1; stack.push([a, far], [far, b]); }
  }
  return pts.filter((_, i) => keep[i]);
}

export function simplifyClosed(pts, tol) {
  if (pts.length < 4 || tol <= 0) return pts;
  let far = 0, fd = -1;
  for (let i = 1; i < pts.length; i++) {
    const d = (pts[i][0] - pts[0][0]) ** 2 + (pts[i][1] - pts[0][1]) ** 2;
    if (d > fd) { fd = d; far = i; }
  }
  const a = douglasPeucker(pts.slice(0, far + 1), tol);
  const b = douglasPeucker(pts.slice(far).concat([pts[0]]), tol);
  const out = a.slice(0, -1).concat(b.slice(0, -1));
  return out.length >= 3 ? out : pts;
}

export function chaikin(pts, iterations) {
  for (let it = 0; it < iterations; it++) {
    const out = [], n = pts.length;
    for (let i = 0; i < n; i++) {
      const [px, py] = pts[i], [qx, qy] = pts[(i + 1) % n];
      out.push([0.75 * px + 0.25 * qx, 0.75 * py + 0.25 * qy], [0.25 * px + 0.75 * qx, 0.25 * py + 0.75 * qy]);
    }
    pts = out;
  }
  return pts;
}
