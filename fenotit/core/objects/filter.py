"""Filtro de objetos (filter): ruido, borde de la foto, tamaño, objetos excluidos y
destacados por el usuario; renumera los objetos."""
import cv2
import numpy as np

from fenotit.core.pipeline.base import step


@step("filter", "processor", requires=("labels",), provides=("labels",), params=[
    {"key": "area_min", "type": "int", "default": -1, "min": -1, "max": 10_000_000},
    {"key": "area_max", "type": "int", "default": 0, "min": 0, "max": 1_000_000_000},
    {"key": "width_min", "type": "int", "default": 0, "min": 0, "max": 1_000_000},
    {"key": "width_max", "type": "int", "default": 0, "min": 0, "max": 1_000_000},
    {"key": "length_min", "type": "int", "default": 0, "min": 0, "max": 1_000_000},
    {"key": "length_max", "type": "int", "default": 0, "min": 0, "max": 1_000_000},
    {"key": "ar_max", "type": "float", "default": 1000.0, "min": 1.0, "max": 1000.0},
    {"key": "exclude_border", "type": "bool", "default": True},
    {"key": "drop_points", "type": "list", "default": []},
    {"key": "mark_points", "type": "list", "default": []},
    {"key": "relative_noise", "type": "bool", "default": False},
])
def filter_objects(ctx, p):
    """Filtra objetos (tamaños en px) y renumera 1..n por filas, de izquierda a derecha.
    area_min = -1: solo quita ruido (< 0.01 % de la foto y < 5 % del objeto típico; con
    relative_noise, todo lo que mida < 5 % del objeto típico).
    Máximos en 0 = sin límite."""
    from scipy.ndimage import find_objects
    labels = ctx.labels
    h, w = labels.shape
    if p["area_min"] < 0:
        areas = np.sort(np.bincount(labels.ravel())[1:])
        areas = areas[areas > 0]
        # objeto típico = mediana ponderada por área: muchos puntitos de ruido no la bajan
        typical = float(areas[np.searchsorted(np.cumsum(areas), areas.sum() / 2)]) if len(areas) else 0.0
        floor = 0.05 * typical if p.get("relative_noise") else min(1e-4 * h * w, 0.05 * typical)
        p = {**p, "area_min": floor if typical else 0}
    inf = float("inf")
    a_max, w_max, l_max = (p[k] or inf for k in ("area_max", "width_max", "length_max"))
    keep = []
    for oid, sl in enumerate(find_objects(labels), 1):
        if sl is None:
            continue
        ys, xs = np.nonzero(labels[sl] == oid)
        area = len(xs)
        if not (p["area_min"] <= area <= a_max):
            continue
        y0, x0 = sl[0].start, sl[1].start
        if p["exclude_border"] and (x0 == 0 or y0 == 0 or sl[1].stop == w or sl[0].stop == h):
            continue
        (_, _), (a, b), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        width, length = min(a, b), max(a, b)
        if not (p["width_min"] <= width <= w_max and p["length_min"] <= length <= l_max):
            continue
        if width > 0 and length / width > p["ar_max"]:
            continue
        keep.append((y0 + ys.mean(), x0 + xs.mean(), sl[0].stop - sl[0].start, oid, sl))
    out = np.zeros_like(labels)
    row_h = float(np.median([k[2] for k in keep])) if keep else 1.0
    for new_id, (_, _, _, oid, sl) in enumerate(sorted(keep, key=lambda k: (round(k[0] / row_h), k[1])), 1):
        region = out[sl]
        region[labels[sl] == oid] = new_id
    ctx.labels = out
    ctx.excluded = _picked(out, p.get("drop_points") or [])
    ctx.highlighted = _picked(out, p.get("mark_points") or []) - ctx.excluded


def _picked(labels, points) -> set[int]:
    """Objetos que el usuario excluyó: los que contienen cada punto (x, y en 0-1). Si el
    punto cae justo fuera (la segmentación cambió un poco), el objeto más cercano a ≤ 15 px.
    Siguen en la numeración; las mediciones no los cuentan ni los miden."""
    h, w = labels.shape
    out = set()
    for xf, yf in points:
        x, y = min(w - 1, max(0, int(xf * w))), min(h - 1, max(0, int(yf * h)))
        oid = int(labels[y, x])
        if not oid:
            win = labels[max(0, y - 15):y + 16, max(0, x - 15):x + 16]
            ys, xs = np.nonzero(win)
            if len(xs):
                k = int(np.argmin((ys - min(15, y)) ** 2 + (xs - min(15, x)) ** 2))
                oid = int(win[ys[k], xs[k]])
        if oid:
            out.add(oid)
    return out
