"""Conteo (count): número de objetos y marca de cada objeto contado (con cajas de un
detector, `measurements.boxes.count_boxes`)."""
import numpy as np
from scipy.ndimage import find_objects

from fenotit.core.pipeline.base import step
from fenotit.core.pipeline.views import regions, inside_point, add_overlay


@step("count", "measurement", requires=("labels",), params=[
    {"key": "numbers", "type": "bool", "default": True},
])
def count(ctx, params):
    if ctx.detections is not None:
        from fenotit.core.measurements.boxes import count_boxes
        count_boxes(ctx, params)
        return
    ids = {i for i, bounds in enumerate(find_objects(ctx.labels), 1) if bounds is not None} - ctx.excluded
    touching = ctx.touching_ids()
    ctx.image_row().update({"n_objects": len(ids), "n_touching": len(touching & ids)})
    if ctx.excluded:
        ctx.image_row()["n_excluded_manual"] = len(ctx.excluded)
    if ctx.highlighted:
        ctx.image_row()["n_highlighted"] = len(ctx.highlighted)
    rows = ctx.object_rows()
    # vista de conteo: un punto por objeto contado (dos puntos en una semilla = partida;
    # una semilla sin punto = no contada)
    from fenotit.core.pipeline import overlay
    marks = overlay.empty()
    for oid, bounds, mask in regions(ctx.labels):
        row = rows[oid]
        row.setdefault("touching", int(oid in touching))
        if "centroid_x_px" not in row:
            pixel_y, pixel_x = np.nonzero(mask)
            row["centroid_x_px"] = int(round(pixel_x.mean() + bounds[1].start))
            row["centroid_y_px"] = int(round(pixel_y.mean() + bounds[0].start))
        x, y = inside_point(mask)
        x, y = x + bounds[1].start, y + bounds[0].start
        out = oid in ctx.excluded
        row["status"] = "excluded" if out else ("highlighted" if oid in ctx.highlighted else "ok")
        if not out:                            # excluido: conserva su número, en gris y sin punto
            marks["dots"].append((x, y))
        if params["numbers"]:
            marks["labels"].append((x, y, str(oid), not out))
    add_overlay(ctx, "count", marks)
