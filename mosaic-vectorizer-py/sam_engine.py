"""SAM model loading, automatic stone detection and click-to-segment, on top of a stone label map."""
import os
import urllib.request

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

CHECKPOINTS = {
    'vit_b': ('sam_vit_b_01ec64.pth', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', '375 MB'),
    'vit_l': ('sam_vit_l_0b3195.pth', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth', '1.2 GB'),
    'vit_h': ('sam_vit_h_4b8939.pth', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', '2.5 GB'),
}
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def ensure_checkpoint(model_type, progress=None):
    """Download the SAM checkpoint on first use. progress(fraction) is called while downloading."""
    name, url, _ = CHECKPOINTS[model_type]
    path = os.path.join(CHECKPOINT_DIR, name)
    if os.path.exists(path):
        return path
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tmp = path + '.part'

    def hook(blocks, block_size, total):
        if progress and total > 0:
            progress(min(1.0, blocks * block_size / total))

    urllib.request.urlretrieve(url, tmp, hook)
    os.replace(tmp, path)
    return path


class StoneMap:
    """Per-pixel stone ids (0 = grout/background) plus per-stone bbox, area, mean colour and cached polygon."""

    def __init__(self, image_rgb):
        self.image = image_rgb
        self.H, self.W = image_rgb.shape[:2]
        self.labels = np.zeros((self.H, self.W), np.int32)
        self.stones = {}
        self.next_id = 1
        self.dirty = None

    def mark_dirty(self, x0, y0, x1, y1):
        d = self.dirty
        self.dirty = (x0, y0, x1, y1) if d is None else (min(d[0], x0), min(d[1], y0), max(d[2], x1), max(d[3], y1))

    def take_dirty(self):
        d, self.dirty = self.dirty, None
        return d

    def stone_at(self, x, y):
        if 0 <= x < self.W and 0 <= y < self.H:
            return int(self.labels[y, x])
        return 0

    def commit(self, crop, x0, y0, steal=False, max_overlap=0.3, min_area=0):
        """Add a stone from a boolean mask crop placed at (x0, y0). Without `steal` it only claims free
        pixels and is rejected when it overlaps existing stones too much; with `steal` it takes
        pixels from its neighbours. Returns the new id, or 0 if rejected."""
        n = int(crop.sum())
        if not n:
            return 0
        h, w = crop.shape
        region = self.labels[y0:y0 + h, x0:x0 + w]
        if not steal and int((region[crop] != 0).sum()) > max_overlap * n:
            return 0
        sid = self.next_id
        self.next_id += 1
        take = crop if steal else crop & (region == 0)
        touched = set(np.unique(region[take]).tolist()) - {0}
        region[take] = sid
        self.stones[sid] = {'bbox': (x0, y0, x0 + w, y0 + h)}
        self._cleanup(sid)
        for t in touched:
            self._cleanup(t)
        s = self.stones.get(sid)
        if s is None or s['area'] < min_area:
            if s is not None:
                self.remove(sid)
            return 0
        return sid

    def _cleanup(self, sid):
        """Keep the largest 4-connected piece, fill enclosed free holes, recompute bbox/area/colour."""
        s = self.stones.get(sid)
        if s is None:
            return
        x0, y0, x1, y1 = s['bbox']
        self.mark_dirty(x0, y0, x1, y1)
        region = self.labels[y0:y1, x0:x1]
        m = (region == sid).astype(np.uint8)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        if n <= 1:
            del self.stones[sid]
            return
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        region[(m == 1) & (cc != best)] = 0
        keep = (cc == best).astype(np.uint8)
        contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(keep)
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
        region[(filled == 1) & (keep == 0) & (region == 0)] = sid

        mask = region == sid
        ys, xs = np.nonzero(mask)
        color = self.image[y0:y1, x0:x1][mask].mean(axis=0)
        s.update(
            bbox=(x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1),
            area=int(mask.sum()),
            color=tuple(int(round(c)) for c in color),
            poly=None,
        )

    def remove(self, sid):
        s = self.stones.pop(sid, None)
        if s is None:
            return
        x0, y0, x1, y1 = s['bbox']
        region = self.labels[y0:y1, x0:x1]
        region[region == sid] = 0
        self.mark_dirty(x0, y0, x1, y1)

    def clear(self):
        self.labels[:] = 0
        self.stones.clear()
        self.mark_dirty(0, 0, self.W, self.H)


def make_tiles(W, H, n, overlap=0.15):
    """n x n overlapping tiles: seed points live in the core, SAM sees the (larger) crop."""
    tw, th = W / n, H / n
    ox, oy = round(tw * overlap), round(th * overlap)
    tiles = []
    for j in range(n):
        for i in range(n):
            kx0, kx1, ky0, ky1 = round(i * tw), round((i + 1) * tw), round(j * th), round((j + 1) * th)
            tiles.append({'core': (kx0, ky0, kx1, ky1),
                          'crop': (max(0, kx0 - ox), max(0, ky0 - oy), min(W, kx1 + ox), min(H, ky1 + oy))})
    return tiles


class SamEngine:
    def __init__(self):
        self.sam = None
        self.model_type = None
        self.device = None
        self.predictor = None
        self._emb_map = None
        self._emb_cache = {}  # tile crop -> (features, original_size, input_size)

    def load(self, model_type, progress=None):
        path = ensure_checkpoint(model_type, progress)
        device = pick_device()
        sam = sam_model_registry[model_type](checkpoint=path)
        sam.to(device)
        sam.eval()
        self.sam, self.model_type, self.device = sam, model_type, device
        self.predictor = SamPredictor(sam)
        self._emb_map, self._emb_cache = None, {}

    def has_embedding(self, smap, crop):
        return self._emb_map is smap and crop in self._emb_cache

    def _use_tile(self, smap, crop):
        """Point the predictor at a tile crop, computing its image embedding once per image."""
        if self._emb_map is not smap:
            self._emb_map, self._emb_cache = smap, {}
        p = self.predictor
        if crop in self._emb_cache:
            p.features, p.original_size, p.input_size = self._emb_cache[crop]
            p.is_image_set = True
        else:
            x0, y0, x1, y1 = crop
            p.set_image(np.ascontiguousarray(smap.image[y0:y1, x0:x1]))
            self._emb_cache[crop] = (p.features, p.original_size, p.input_size)

    def _predict(self, points):
        """points: Nx2 in tile-crop pixels -> (logits N x 3 x h x w, iou scores N x 3)."""
        p = self.predictor
        coords = torch.as_tensor(p.transform.apply_coords(points, p.original_size),
                                 dtype=torch.float, device=self.device)[:, None, :]
        labels = torch.ones((len(points), 1), dtype=torch.int, device=self.device)
        logits, iou, _ = p.predict_torch(coords, labels, multimask_output=True, return_logits=True)
        return logits, iou

    def auto_segment(self, smap, points_per_side=32, tiles=2, pred_iou_thresh=0.80, stability_thresh=0.85,
                     min_area=30, max_area_frac=0.03, max_overlap=0.3, batch=16, report=None, should_stop=None):
        """Prompt SAM with a grid of points per tile, skipping points already inside a stone. For each
        point keep the best of SAM's 3 masks that contains the point and passes the quality and size
        filters; masks cut off by an inner tile edge are left for the neighbouring tile.
        Returns the number of stones added."""
        max_area = max_area_frac * smap.W * smap.H
        tile_list = make_tiles(smap.W, smap.H, tiles)
        total, done, added = len(tile_list) * points_per_side ** 2, 0, 0
        for ti, tile in enumerate(tile_list):
            if report:
                report(f'Tile {ti + 1}/{len(tile_list)}: computing image embedding…', done / total)
            cx0, cy0, cx1, cy1 = tile['crop']
            kx0, ky0, kx1, ky1 = tile['core']
            with torch.inference_mode():
                self._use_tile(smap, tile['crop'])
            n = points_per_side
            seeds = [(kx0 + (a + 0.5) * (kx1 - kx0) / n, ky0 + (b + 0.5) * (ky1 - ky0) / n)
                     for b in range(n) for a in range(n)]
            inner = (cx0 > 0, cy0 > 0, cx1 < smap.W, cy1 < smap.H)  # which crop edges are not image edges
            idx = 0
            while idx < len(seeds):
                if should_stop and should_stop():
                    return added
                pts = []
                while idx < len(seeds) and len(pts) < batch:
                    sx, sy = seeds[idx]
                    idx += 1
                    done += 1
                    if not smap.labels[int(sy), int(sx)]:
                        pts.append((sx, sy))
                if not pts:
                    continue
                local = np.array([(x - cx0, y - cy0) for x, y in pts])
                with torch.inference_mode():
                    logits, iou = self._predict(local)
                    pos = logits > 0
                    areas = pos.flatten(2).sum(-1)
                    stab = ((logits > 1).flatten(2).sum(-1) / (logits > -1).flatten(2).sum(-1).clamp(min=1))
                    edges = torch.stack([pos[..., :, 0].any(-1), pos[..., 0, :].any(-1),
                                         pos[..., :, -1].any(-1), pos[..., -1, :].any(-1)], -1)
                areas, stab, iou, edges = areas.cpu().numpy(), stab.cpu().numpy(), iou.cpu().numpy(), edges.cpu().numpy()
                for i, (sx, sy) in enumerate(pts):
                    if smap.labels[int(sy), int(sx)]:
                        continue  # covered by a stone found earlier in this batch
                    lx, ly = int(sx - cx0), int(sy - cy0)
                    best = None
                    for k in range(iou.shape[1]):
                        if iou[i, k] < pred_iou_thresh or stab[i, k] < stability_thresh:
                            continue
                        if not (min_area <= areas[i, k] <= max_area):
                            continue
                        if any(e and inner[j] for j, e in enumerate(edges[i, k])):
                            continue
                        if not pos[i, k, ly, lx]:
                            continue  # mask must contain its prompt point
                        if best is None or iou[i, k] > iou[i, best]:
                            best = k
                    if best is None:
                        continue
                    m = pos[i, best].cpu().numpy()
                    ys, xs = np.nonzero(m)
                    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                    if smap.commit(m[y0:y1, x0:x1], int(cx0 + x0), int(cy0 + y0),
                                   max_overlap=max_overlap, min_area=min_area):
                        added += 1
                if report:
                    report(f'Tile {ti + 1}/{len(tile_list)}: {added} stones found', done / total)
        return added

    @staticmethod
    def tile_for(smap, x, y, tiles):
        for t in make_tiles(smap.W, smap.H, tiles):
            kx0, ky0, kx1, ky1 = t['core']
            if kx0 <= x < kx1 and ky0 <= y < ky1:
                return t
        return t

    def click(self, smap, x, y, mask_index=None, max_area_frac=1.0, tiles=2):
        """Segment the stone under (x, y), replacing any stone there. mask_index picks one of SAM's
        3 candidate masks; None = best-scoring mask no larger than max_area_frac of the image."""
        tile = self.tile_for(smap, x, y, tiles)
        cx0, cy0 = tile['crop'][:2]
        with torch.inference_mode():
            self._use_tile(smap, tile['crop'])
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([[x + 0.5 - cx0, y + 0.5 - cy0]]), point_labels=np.array([1]),
                multimask_output=True)
        full = np.zeros((len(masks), smap.H, smap.W), bool)
        full[:, cy0:cy0 + masks.shape[1], cx0:cx0 + masks.shape[2]] = masks
        masks = full
        areas = masks.reshape(len(masks), -1).sum(axis=1)
        max_area = max_area_frac * smap.W * smap.H
        if mask_index is None:
            fits = [k for k in range(len(masks)) if 0 < areas[k] <= max_area] or list(range(len(masks)))
            k = max(fits, key=lambda i: scores[i])
        else:
            k = mask_index % len(masks)
        old = smap.stone_at(x, y)
        if old:
            smap.remove(old)
        ys, xs = np.nonzero(masks[k])
        if not len(xs):
            return {'id': 0, 'k': k, 'score': float(scores[k])}
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        sid = smap.commit(masks[k][y0:y1, x0:x1], int(x0), int(y0), steal=areas[k] <= max_area, max_overlap=1)
        return {'id': sid, 'k': k, 'score': float(scores[k])}
