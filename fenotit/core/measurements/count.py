"""Conteo (count): número de objetos y marca de cada objeto contado (con cajas de un
detector, `measurements.boxes.count_boxes`)."""
import numpy as np
from scipy.ndimage import find_objects

from fenotit.core.pipeline.base import step
from fenotit.core.pipeline.views import regions, inside_point, add_overlay


@step("count", "measurement", requires=("labels",), params=[
    {"key": "numbers", "type": "bool", "default": True},
])
def count(ctx, p):
    if ctx.detections is not None:
        from fenotit.core.measurements.boxes import count_boxes
        count_boxes(ctx, p)
        return
    ids = {i for i, s in enumerate(find_objects(ctx.labels), 1) if s is not None} - ctx.excluded
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
    ov = overlay.empty()
    for oid, sl, m in regions(ctx.labels):
        row = rows[oid]
        row.setdefault("touching", int(oid in touching))
        if "centroid_x_px" not in row:
            ys, xs = np.nonzero(m)
            row["centroid_x_px"] = int(round(xs.mean() + sl[1].start))
            row["centroid_y_px"] = int(round(ys.mean() + sl[0].start))
        x, y = inside_point(m)
        x, y = x + sl[1].start, y + sl[0].start
        out = oid in ctx.excluded
        row["status"] = "excluded" if out else ("highlighted" if oid in ctx.highlighted else "ok")
        if not out:                            # excluido: conserva su número, en gris y sin punto
            ov["dots"].append((x, y))
        if p["numbers"]:
            ov["labels"].append((x, y, str(oid), not out))
    add_overlay(ctx, "count", ov)
