"""Detecciones por objeto (object_detections): las cajas de un detector se asignan al objeto
segmentado que las contiene, y se cuentan por clase en cada objeto y en la foto."""
import cv2
import numpy as np

from fenotit.core.measurements.classes import slug
from fenotit.core.objects.boxes import center
from fenotit.core.pipeline.base import step
from fenotit.core.pipeline.views import add_overlay, object_geometry

DIMMED_PHOTO = 0.55                  # brillo de la foto bajo las cajas en la vista


def assign(labels: np.ndarray, detections: list[dict]) -> list[int]:
    """Objeto (etiqueta) de cada caja: el que contiene su centro; si el centro cae en el
    fondo (p. ej. un agujero en el borde de la semilla), el objeto con más píxeles dentro de
    la caja; 0 si la caja no toca ningún objeto."""
    height, width = labels.shape[:2]
    owners = []
    for detection in detections:
        x, y = center(detection)
        column = min(width - 1, max(0, int(x)))
        row = min(height - 1, max(0, int(y)))
        owner = int(labels[row, column])
        if owner == 0:
            x0, y0 = max(0, int(detection["x0"])), max(0, int(detection["y0"]))
            x1, y1 = min(width, int(np.ceil(detection["x1"]))), min(height, int(np.ceil(detection["y1"])))
            inside = labels[y0:y1, x0:x1]
            inside = inside[inside > 0]
            owner = int(np.bincount(inside).argmax()) if inside.size else 0
        owners.append(owner)
    return owners


@step("object_detections", "measurement", requires=("labels",),
      params=[{"key": "model", "default": ""}, {"key": "conf", "default": 0.25}, {"key": "use", "default": []},
              {"key": "classes", "default": []}, {"key": "colors", "default": []}])
def object_detections(ctx, params):
    """Corre el detector en la foto y cuenta, en cada objeto, las cajas de cada clase que le
    corresponden (`assign`; las que no tocan ningún objeto no se cuentan). Tablas: `n_<clase>` por
    objeto; en la foto, el total de cada clase y la media por objeto; tabla `detections`
    con cada caja y su objeto. Vista: la foto atenuada con las cajas del color de su clase."""
    from fenotit.core.pipeline import overlay
    from fenotit.core.segmentation import detect, models
    model = models.get(params["model"])
    if model is None or model["output"] != "boxes":
        raise FileNotFoundError(f"Detection model not found: {params['model']}")
    names = list(params.get("classes") or model.get("classes") or [])
    colors = list(params.get("colors") or [])
    use = {int(index) for index in params.get("use") or []}
    found = [detection for detection in detect.run(ctx.image, model, float(params["conf"]))
             if not use or detection["cls"] in use]
    owners = assign(ctx.labels, found)
    counted = [index for index in sorted(use) if index < len(names)] if use else list(range(len(names)))
    columns = {index: f"n_{slug(names[index])}" for index in counted}
    rows = ctx.object_rows()
    measured = [object_id for object_id in rows if object_id not in ctx.excluded]
    for object_id in measured:
        rows[object_id].update({column: 0 for column in columns.values()})
    table = []
    for detection, owner in zip(found, owners):
        inside = owner in rows and owner not in ctx.excluded
        if inside and detection["cls"] in columns:
            rows[owner][columns[detection["cls"]]] += 1
        table.append({"object_id": owner if inside else "", "class": names[detection["cls"]]
                      if detection["cls"] < len(names) else f"class_{detection['cls'] + 1}",
                      "score": round(detection["score"], 3),
                      "x0_px": int(round(detection["x0"])), "y0_px": int(round(detection["y0"])),
                      "x1_px": int(round(detection["x1"])), "y1_px": int(round(detection["y1"]))})
    ctx.tables["detections"] = table
    image_row = ctx.image_row()
    for column in columns.values():
        total = sum(rows[object_id][column] for object_id in measured)
        image_row[column] = total
        image_row[f"mean_{column[2:]}_per_object"] = round(total / len(measured), 3) if measured else 0
    marks = overlay.empty()
    marks["outlines"] = [(contour, object_id not in ctx.excluded)
                         for object_id, contour in object_geometry(ctx.labels).items()]
    marks["on_view"] = True                     # los contornos van sobre la vista con las cajas
    add_overlay(ctx, "detections", marks)
    view = (ctx.image * DIMMED_PHOTO).astype(np.uint8)
    thickness = max(2, int(round(max(view.shape[:2]) / 400)))
    for detection, owner in zip(found, owners):
        if owner not in rows or owner in ctx.excluded:
            continue
        hex_color = colors[detection["cls"]] if detection["cls"] < len(colors) else overlay.UNNAMED_CLASS_COLOR
        cv2.rectangle(view, (int(detection["x0"]), int(detection["y0"])), (int(detection["x1"]), int(detection["y1"])),
                      overlay.hex_to_bgr(hex_color), thickness)
    ctx.images["detections"] = view
    ctx.extra["detection_classes"] = [{"class": names[index], "column": columns[index],
                                       "color": colors[index] if index < len(colors) else ""} for index in counted]
