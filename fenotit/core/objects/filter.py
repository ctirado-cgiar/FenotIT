"""Filtro de objetos (filter): ruido, borde de la foto, tamaño, objetos excluidos y
destacados por el usuario; renumera los objetos."""
import cv2
import numpy as np

# El decorador step registra el filtro como pieza de la cadena.
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
def filter_objects(ctx, params):
    """Filtra objetos (tamaños en px) y renumera 1..n por filas, de izquierda a derecha.
    area_min = -1: solo quita ruido (< 0.01 % de la foto y < 5 % del objeto típico; con
    relative_noise, todo lo que mida < 5 % del objeto típico).
    Máximos en 0 = sin límite."""
    from scipy.ndimage import find_objects
    labels = ctx.labels
    image_height, image_width = labels.shape
    if params["area_min"] < 0:
        areas = np.sort(np.bincount(labels.ravel())[1:])
        areas = areas[areas > 0]
        # objeto típico = mediana ponderada por área: muchos puntitos de ruido no la bajan
        typical = float(areas[np.searchsorted(np.cumsum(areas), areas.sum() / 2)]) if len(areas) else 0.0
        floor = 0.05 * typical if params.get("relative_noise") else min(1e-4 * image_height * image_width, 0.05 * typical)
        params = {**params, "area_min": floor if typical else 0}
    inf = float("inf")
    a_max, w_max, l_max = (params[entry] or inf for entry in ("area_max", "width_max", "length_max"))
    keep = []
    for oid, bounds in enumerate(find_objects(labels), 1):
        if bounds is None:
            continue
        pixel_y, pixel_x = np.nonzero(labels[bounds] == oid)
        area = len(pixel_x)
        if not (params["area_min"] <= area <= a_max):
            continue
        y0, x0 = bounds[0].start, bounds[1].start
        if params["exclude_border"] and (x0 == 0 or y0 == 0 or bounds[1].stop == image_width or bounds[0].stop == image_height):
            continue
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(np.column_stack([pixel_x, pixel_y]).astype(np.float32))
        width, length = min(side_a, side_b), max(side_a, side_b)
        if not (params["width_min"] <= width <= w_max and params["length_min"] <= length <= l_max):
            continue
        if width > 0 and length / width > params["ar_max"]:
            continue
        keep.append((y0 + pixel_y.mean(), x0 + pixel_x.mean(), bounds[0].stop - bounds[0].start, oid, bounds))
    out = np.zeros_like(labels)
    row_h = float(np.median([entry[2] for entry in keep])) if keep else 1.0
    for new_id, (_, _, _, oid, bounds) in enumerate(sorted(keep, key=lambda entry: (round(entry[0] / row_h), entry[1])), 1):
        region = out[bounds]
        region[labels[bounds] == oid] = new_id
    ctx.labels = out
    ctx.excluded = _picked(out, params.get("drop_points") or [])
    ctx.highlighted = _picked(out, params.get("mark_points") or []) - ctx.excluded


def _picked(labels, points) -> set[int]:
    """Objetos que el usuario excluyó: los que contienen cada punto (x, y en 0-1). Si el
    punto cae justo fuera (la segmentación cambió un poco), el objeto más cercano a ≤ 15 px.
    Siguen en la numeración; las mediciones no los cuentan ni los miden."""
    image_height, image_width = labels.shape
    out = set()
    for x_fraction, y_fraction in points:
        x, y = min(image_width - 1, max(0, int(x_fraction * image_width))), min(image_height - 1, max(0, int(y_fraction * image_height)))
        oid = int(labels[y, x])
        if not oid:
            win = labels[max(0, y - 15):y + 16, max(0, x - 15):x + 16]
            pixel_y, pixel_x = np.nonzero(win)
            if len(pixel_x):
                nearest = int(np.argmin((pixel_y - min(15, y)) ** 2 + (pixel_x - min(15, x)) ** 2))
                oid = int(win[pixel_y[nearest], pixel_x[nearest]])
        if oid:
            out.add(oid)
    return out
