"""Objetos a partir de las cajas de un detector (boxes): área de análisis, borde, tamaño,
exclusiones del usuario y numeración por filas."""
import numpy as np

from fenotit.core.pipeline.base import step


def rect(box: dict) -> np.ndarray:
    """Las 4 esquinas de una caja como contorno (px)."""
    x0, y0, x1, y1 = box["x0"], box["y0"], box["x1"], box["y1"]
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)


def center(box: dict) -> tuple[float, float]:
    return (box["x0"] + box["x1"]) / 2, (box["y0"] + box["y1"]) / 2


def _picked(dets: list[dict], points, image_width: int, image_height: int) -> set[int]:
    """Cajas que contienen cada punto (x, y en 0-1): la más pequeña; si ninguna lo contiene,
    la de centro más cercano a ≤ 15 px."""
    out = set()
    for x_fraction, y_fraction in points:
        x, y = x_fraction * image_width, y_fraction * image_height
        inside = [box for box in dets if box["x0"] <= x <= box["x1"] and box["y0"] <= y <= box["y1"]]
        if inside:
            out.add(min(inside, key=lambda box: (box["x1"] - box["x0"]) * (box["y1"] - box["y0"]))["id"])
            continue
        near = [(np.hypot(center(box)[0] - x, center(box)[1] - y), box["id"]) for box in dets]
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
def boxes(ctx, params):
    """Conserva las cajas cuyo centro está en el área de análisis (fuera de las zonas
    excluidas) y que cumplen el tamaño (largo = lado mayor, ancho = lado menor; 0 = sin
    límite); las numera 1..n por filas. ctx.labels = las cajas rellenas (para elegir un
    objeto con un clic); objetos excluidos o destacados por el usuario con sus puntos."""
    from fenotit.core.segmentation.detect import boxes_labels
    image_height, image_width = ctx.image.shape[:2]
    region = ctx.analysis_region(ctx.image.shape)
    inf = float("inf")
    kept = []
    for box in ctx.detections or []:
        cx, cy = center(box)
        if region is not None and not region[min(image_height - 1, int(cy)), min(image_width - 1, int(cx))]:
            continue
        if params["exclude_border"] and (box["x0"] <= 0 or box["y0"] <= 0 or box["x1"] >= image_width or box["y1"] >= image_height):
            continue
        box_width, box_height = box["x1"] - box["x0"], box["y1"] - box["y0"]
        width, length = min(box_width, box_height), max(box_width, box_height)
        if not (params["width_min"] <= width <= (params["width_max"] or inf)
                and params["length_min"] <= length <= (params["length_max"] or inf)):
            continue
        kept.append(dict(box))
    row_h = float(np.median([box["y1"] - box["y0"] for box in kept])) if kept else 1.0
    kept.sort(key=lambda box: (round(center(box)[1] / row_h), center(box)[0]))
    for i, box in enumerate(kept, 1):
        box["id"] = i
    ctx.detections = kept
    ctx.labels = boxes_labels(ctx.image.shape, kept)
    ctx.excluded = _picked(kept, params.get("drop_points") or [], image_width, image_height)
    ctx.highlighted = _picked(kept, params.get("mark_points") or [], image_width, image_height) - ctx.excluded
