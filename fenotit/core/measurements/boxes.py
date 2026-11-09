"""Mediciones con cajas de un detector: conteo por clase (count_boxes, lo llama la pieza
count) y tamaño de cada caja (box_size)."""
import cv2
import numpy as np

from fenotit.core.measurements.classes import slug
from fenotit.core.objects.boxes import center, rect
from fenotit.core.pipeline.base import step
from fenotit.core.pipeline.views import add_overlay


def _rows(ctx) -> dict[int, dict]:
    """Filas de la tabla de objetos con la clase, el puntaje y el centro de cada caja."""
    rows = ctx.object_rows()
    for d in ctx.detections:
        cx, cy = center(d)
        rows[d["id"]].update({"class": d.get("class", ""), "score": round(d["score"], 3),
                              "centroid_x_px": int(round(cx)), "centroid_y_px": int(round(cy))})
    return rows


def _outlines(ctx) -> list:
    return [(rect(d), d["id"] not in ctx.excluded) for d in ctx.detections]


def count_boxes(ctx, p):
    """Número de cajas (sin las excluidas) y, con más de una clase, cuántas de cada clase.
    Vista: el contorno de cada caja con un punto y su número; con varias clases, una vista
    Clases con las cajas del color de su clase."""
    from fenotit.core.pipeline import overlay
    dets = ctx.detections
    used = [d for d in dets if d["id"] not in ctx.excluded]
    img = ctx.image_row()
    img.update({"n_objects": len(used), "n_touching": 0})
    if ctx.excluded:
        img["n_excluded_manual"] = len(ctx.excluded)
    if ctx.highlighted:
        img["n_highlighted"] = len(ctx.highlighted)
    rows = _rows(ctx)
    ov = overlay.empty()
    ov["outlines"] = _outlines(ctx)
    for d in dets:
        x, y = center(d)
        out = d["id"] in ctx.excluded
        rows[d["id"]]["status"] = "excluded" if out else ("highlighted" if d["id"] in ctx.highlighted else "ok")
        if not out:
            ov["dots"].append((x, y))
        if p.get("numbers", True):
            ov["labels"].append((x, y, str(d["id"]), not out))
    add_overlay(ctx, "count", ov)
    names = list(ctx.extra.get("class_names") or [])
    present = sorted({d["cls"] for d in dets})
    if len(names) < 2 and len(present) < 2:
        return
    colors = list(ctx.extra.get("class_colors") or [])[1:]
    table = []
    for c, name in enumerate(names or [f"class_{i + 1}" for i in range(max(present) + 1)]):
        n = sum(1 for d in used if d["cls"] == c)
        img[f"n_{slug(name)}"] = n
        table.append({"class": name, "n": n, "pct": round(100 * n / max(len(used), 1), 2),
                      "color": colors[c] if c < len(colors) else ""})
    ctx.tables["classes"] = table
    view = (ctx.image * 0.55).astype(np.uint8)
    t = max(2, int(round(max(view.shape[:2]) / 400)))
    for d in used:
        hx = (table[d["cls"]]["color"] or "#FFFFFF").lstrip("#") if d["cls"] < len(table) else "FFFFFF"
        bgr = (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
        cv2.rectangle(view, (int(d["x0"]), int(d["y0"])), (int(d["x1"]), int(d["y1"])), bgr, t)
    ctx.images["classes"] = view


@step("box_size", "measurement", requires=("labels",))
def box_size(ctx, p):
    """Ancho (horizontal), alto (vertical) y área de cada caja, en la unidad de la escala.
    Es el tamaño de la caja, no del objeto: un objeto inclinado ocupa menos que su caja."""
    from fenotit.core.pipeline import overlay
    u, f = ctx.unit, ctx.scale
    rows = _rows(ctx)
    for d in ctx.detections:
        if d["id"] in ctx.excluded:
            continue
        bw, bh = d["x1"] - d["x0"], d["y1"] - d["y0"]
        rows[d["id"]].update({f"box_width_{u}": round(bw * f, ctx.digits(1)),
                              f"box_height_{u}": round(bh * f, ctx.digits(1)),
                              f"box_area_{u}2": round(bw * bh * f * f, ctx.digits(2))})
    ov = overlay.empty()
    ov["outlines"] = _outlines(ctx)
    add_overlay(ctx, "box_size", ov)
    ctx.images["box_size"] = ctx.image
