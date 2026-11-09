"""Objetos a partir de las cajas de un detector (boxes): área de análisis, borde, tamaño,
exclusiones del usuario y numeración por filas."""
import numpy as np

from fenotit.core.pipeline.base import step


def rect(d: dict) -> np.ndarray:
    """Las 4 esquinas de una caja como contorno (px)."""
    x0, y0, x1, y1 = d["x0"], d["y0"], d["x1"], d["y1"]
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)


def center(d: dict) -> tuple[float, float]:
    return (d["x0"] + d["x1"]) / 2, (d["y0"] + d["y1"]) / 2


def _picked(dets: list[dict], points, w: int, h: int) -> set[int]:
    """Cajas que contienen cada punto (x, y en 0-1): la más pequeña; si ninguna lo contiene,
    la de centro más cercano a ≤ 15 px."""
    out = set()
    for xf, yf in points:
        x, y = xf * w, yf * h
        inside = [d for d in dets if d["x0"] <= x <= d["x1"] and d["y0"] <= y <= d["y1"]]
        if inside:
            out.add(min(inside, key=lambda d: (d["x1"] - d["x0"]) * (d["y1"] - d["y0"]))["id"])
            continue
        near = [(np.hypot(center(d)[0] - x, center(d)[1] - y), d["id"]) for d in dets]
        if near and min(near)[0] <= 15:
            out.add(min(near)[1])
    return out


@step("boxes", "processor", requires=("mask",), provides=("labels",), params=[
    {"key": "exclude_border", "type": "bool", "default": False},
    {"key": "width_min", "type": "int", "default": 0, "min": 0, "max": 1_000_000},
    {"key": "width_max", "type": "int", "default": 0, "min": 0, "max": 1_000_000},
    {"key": "length_min", "type": "int", "default": 0, "min": 0, "max": 1_000_000},
    {"key": "length_max", "type": "int", "default": 0, "min": 0, "max": 1_000_000},
    {"key": "drop_points", "type": "list", "default": []},
    {"key": "mark_points", "type": "list", "default": []},
])
def boxes(ctx, p):
    """Conserva las cajas cuyo centro está en el área de análisis (fuera de las zonas
    excluidas) y que cumplen el tamaño (largo = lado mayor, ancho = lado menor; 0 = sin
    límite); las numera 1..n por filas. ctx.labels = las cajas rellenas (para elegir un
    objeto con un clic); objetos excluidos o destacados por el usuario con sus puntos."""
    from fenotit.core.segmentation.detect import boxes_labels
    h, w = ctx.image.shape[:2]
    region = ctx.analysis_region(ctx.image.shape)
    inf = float("inf")
    kept = []
    for d in ctx.detections or []:
        cx, cy = center(d)
        if region is not None and not region[min(h - 1, int(cy)), min(w - 1, int(cx))]:
            continue
        if p["exclude_border"] and (d["x0"] <= 0 or d["y0"] <= 0 or d["x1"] >= w or d["y1"] >= h):
            continue
        bw, bh = d["x1"] - d["x0"], d["y1"] - d["y0"]
        width, length = min(bw, bh), max(bw, bh)
        if not (p["width_min"] <= width <= (p["width_max"] or inf)
                and p["length_min"] <= length <= (p["length_max"] or inf)):
            continue
        kept.append(dict(d))
    row_h = float(np.median([d["y1"] - d["y0"] for d in kept])) if kept else 1.0
    kept.sort(key=lambda d: (round(center(d)[1] / row_h), center(d)[0]))
    for i, d in enumerate(kept, 1):
        d["id"] = i
    ctx.detections = kept
    ctx.labels = boxes_labels(ctx.image.shape, kept)
    ctx.excluded = _picked(kept, p.get("drop_points") or [], w, h)
    ctx.highlighted = _picked(kept, p.get("mark_points") or [], w, h) - ctx.excluded
