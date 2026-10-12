"""Stone label map -> polygons, and SVG / DXF / JSON export."""
import base64
import io
import json

import cv2
import numpy as np


def chaikin(pts, iterations):
    for _ in range(iterations):
        nxt = np.roll(pts, -1, axis=0)
        pts = np.stack([0.75 * pts + 0.25 * nxt, 0.25 * pts + 0.75 * nxt], axis=1).reshape(-1, 2)
    return pts


def stone_polygon(smap, sid, tolerance=1.0, smoothing=0):
    """Outer outline of one stone in work-image pixel coordinates (cached on the stone)."""
    s = smap.stones.get(sid)
    if s is None:
        return None
    x0, y0, x1, y1 = s['bbox']
    m = np.pad((smap.labels[y0:y1, x0:x1] == sid).astype(np.uint8), 1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        s['poly'] = None
        return None
    c = max(contours, key=len)
    if tolerance > 0:
        c = cv2.approxPolyDP(c, tolerance, True)
    pts = c.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        s['poly'] = None
        return None
    pts = chaikin(pts, smoothing)
    pts += (x0 - 1 + 0.5, y0 - 1 + 0.5)  # undo padding; contour points are pixel centres
    s['poly'] = pts
    return pts


def export_polygons(smap, scale, tolerance, smoothing):
    """[(id, (r, g, b), Nx2 points in original-image pixels)]"""
    out = []
    for sid, s in list(smap.stones.items()):
        poly = s.get('poly')
        if poly is None:
            poly = stone_polygon(smap, sid, tolerance, smoothing)
        if poly is not None:
            out.append((sid, s['color'], poly * scale))
    return out


def _f(v, digits=2):
    return f'{round(float(v), digits):g}'


def _hex(c):
    return '#%02x%02x%02x' % tuple(c)


def to_svg(smap, scale, size, tolerance=1.0, smoothing=0, fill=True, stroke=True, source_image=None, width_mm=0):
    W, H = size
    dims = f'width="{_f(width_mm)}mm" height="{_f(width_mm * H / W)}mm"' if width_mm > 0 else f'width="{W}" height="{H}"'
    sw = _f(max(0.5, max(W, H) / 1500))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<svg xmlns="http://www.w3.org/2000/svg" {dims} viewBox="0 0 {W} {H}">']
    if source_image is not None:
        buf = io.BytesIO()
        source_image.convert('RGB').save(buf, 'JPEG', quality=90)
        data = base64.b64encode(buf.getvalue()).decode()
        lines.append(f'  <g id="source"><image href="data:image/jpeg;base64,{data}" x="0" y="0" '
                     f'width="{W}" height="{H}" opacity="0.5"/></g>')
    stroke_attr = f' stroke="#000" stroke-width="{sw}"' if stroke else ''
    lines.append(f'  <g id="stones" stroke-linejoin="round"{stroke_attr}>')
    for sid, color, pts in export_polygons(smap, scale, tolerance, smoothing):
        d = 'M' + 'L'.join(f'{_f(x)} {_f(y)}' for x, y in pts) + 'Z'
        lines.append(f'    <path id="stone-{sid}" d="{d}" fill="{_hex(color) if fill else "none"}"/>')
    lines += ['  </g>', '</svg>']
    return '\n'.join(lines)


def to_dxf(smap, scale, size, tolerance=1.0, smoothing=0, width_mm=0):
    """AutoCAD R12 DXF: one closed POLYLINE per stone on layer STONES, Y axis up."""
    W, H = size
    k = width_mm / W if width_mm > 0 else 1.0
    out = ['0', 'SECTION', '2', 'ENTITIES']
    for _, _, pts in export_polygons(smap, scale, tolerance, smoothing):
        out += ['0', 'POLYLINE', '8', 'STONES', '66', '1', '70', '1', '10', '0', '20', '0', '30', '0']
        for x, y in pts:
            out += ['0', 'VERTEX', '8', 'STONES', '10', _f(x * k, 4), '20', _f((H - y) * k, 4), '30', '0']
        out += ['0', 'SEQEND', '8', 'STONES']
    out += ['0', 'ENDSEC', '0', 'EOF']
    return '\n'.join(out)


def to_json(smap, scale, size, tolerance=1.0, smoothing=0, width_mm=0):
    W, H = size
    k = width_mm / W if width_mm > 0 else 1.0
    return json.dumps({
        'width': W, 'height': H, 'units': 'mm' if width_mm > 0 else 'px',
        'stones': [{'id': sid, 'color': _hex(color), 'points': [[round(x * k, 3), round(y * k, 3)] for x, y in pts]}
                   for sid, color, pts in export_polygons(smap, scale, tolerance, smoothing)],
    })
