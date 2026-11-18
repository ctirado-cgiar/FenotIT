"""Mediciones con cajas de un detector: conteo por clase (count_boxes, lo llama la pieza
count) y tamaño de cada caja (box_size)."""
import cv2
import numpy as np

# slug convierte el nombre de una clase en un nombre de columna (n_<clase>).
from fenotit.core.measurements.classes import slug
# Centro y rectángulo de cada caja, para dibujarla y numerarla.
from fenotit.core.objects.boxes import center, rect
# El decorador step registra estas funciones como piezas de la cadena.
from fenotit.core.pipeline.base import step
# add_overlay guarda las marcas (cajas, números) que la ventana dibuja sobre la foto.
from fenotit.core.pipeline.views import add_overlay


def _rows(ctx) -> dict[int, dict]:
    """Filas de la tabla de objetos con la clase, el puntaje y el centro de cada caja."""
    rows = ctx.object_rows()
    for box in ctx.detections:
        cx, cy = center(box)
        rows[box["id"]].update({"class": box.get("class", ""), "score": round(box["score"], 3),
                              "centroid_x_px": int(round(cx)), "centroid_y_px": int(round(cy))})
    return rows


def _outlines(ctx) -> list:
    return [(rect(box), box["id"] not in ctx.excluded) for box in ctx.detections]


def count_boxes(ctx, params):
    """Número de cajas (sin las excluidas) y, con más de una clase, cuántas de cada clase.
    Vista: el contorno de cada caja con un punto y su número; con varias clases, una vista
    Clases con las cajas del color de su clase."""
    from fenotit.core.pipeline import overlay
    dets = ctx.detections
    used = [box for box in dets if box["id"] not in ctx.excluded]
    img = ctx.image_row()
    img.update({"n_objects": len(used), "n_touching": 0})
    if ctx.excluded:
        img["n_excluded_manual"] = len(ctx.excluded)
    if ctx.highlighted:
        img["n_highlighted"] = len(ctx.highlighted)
    rows = _rows(ctx)
    marks = overlay.empty()
    marks["outlines"] = _outlines(ctx)
    for box in dets:
        x, y = center(box)
        out = box["id"] in ctx.excluded
        rows[box["id"]]["status"] = "excluded" if out else ("highlighted" if box["id"] in ctx.highlighted else "ok")
        if not out:
            marks["dots"].append((x, y))
        if params.get("numbers", True):
            marks["labels"].append((x, y, str(box["id"]), not out))
    add_overlay(ctx, "count", marks)
    names = list(ctx.extra.get("class_names") or [])
    present = sorted({box["cls"] for box in dets})
    if len(names) < 2 and len(present) < 2:
        return
    colors = list(ctx.extra.get("class_colors") or [])[1:]
    table = []
    for class_index, name in enumerate(names or [f"class_{i + 1}" for i in range(max(present) + 1)]):
        count = sum(1 for box in used if box["cls"] == class_index)
        img[f"n_{slug(name)}"] = count
        table.append({"class": name, "n": count, "pct": round(100 * count / max(len(used), 1), 2),
                      "color": colors[class_index] if class_index < len(colors) else ""})
    ctx.tables["classes"] = table
    view = (ctx.image * 0.55).astype(np.uint8)
    t = max(2, int(round(max(view.shape[:2]) / 400)))
    for box in used:
        hex_color = (table[box["cls"]]["color"] or "#FFFFFF").lstrip("#") if box["cls"] < len(table) else "FFFFFF"
        bgr = (int(hex_color[4:6], 16), int(hex_color[2:4], 16), int(hex_color[0:2], 16))
        cv2.rectangle(view, (int(box["x0"]), int(box["y0"])), (int(box["x1"]), int(box["y1"])), bgr, t)
    ctx.images["classes"] = view


@step("box_size", "measurement", requires=("labels",))
def box_size(ctx, params):
    """Ancho (horizontal), alto (vertical) y área de cada caja, en la unidad de la escala.
    Es el tamaño de la caja, no del objeto: un objeto inclinado ocupa menos que su caja."""
    from fenotit.core.pipeline import overlay
    unit, scale = ctx.unit, ctx.scale
    rows = _rows(ctx)
    for box in ctx.detections:
        if box["id"] in ctx.excluded:
            continue
        box_width, box_height = box["x1"] - box["x0"], box["y1"] - box["y0"]
        rows[box["id"]].update({f"box_width_{unit}": round(box_width * scale, ctx.digits(1)),
                              f"box_height_{unit}": round(box_height * scale, ctx.digits(1)),
                              f"box_area_{unit}2": round(box_width * box_height * scale * scale, ctx.digits(2))})
    marks = overlay.empty()
    marks["outlines"] = _outlines(ctx)
    add_overlay(ctx, "box_size", marks)
    ctx.images["box_size"] = ctx.image
