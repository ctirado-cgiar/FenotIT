"""Forma: coeficientes elípticos de Fourier por objeto (extracción).
Alinear, promediar y comparar formas es análisis posterior (fenotit.core.stats.shape)."""
import cv2
import numpy as np

from fenotit.core import efd
from fenotit.core.pipeline.base import step
from fenotit.core.pipeline.views import included_view


@step("shape", "measurement", requires=("labels",), params=[
    {"key": "harmonics", "type": "int", "default": 20, "min": 1, "max": 50},
    {"key": "isolated_only", "type": "bool", "default": True},
])
def shape(ctx, params):
    """Coeficientes normalizados (tamaño, rotación e inicio) por objeto: tabla object_shape."""
    from scipy.ndimage import find_objects
    order = int(params["harmonics"])
    touching = ctx.touching_ids() if params["isolated_only"] else set()
    rows, smooth = [], {}
    for oid, bounds in enumerate(find_objects(ctx.labels), 1):
        if bounds is None or oid in touching or oid in ctx.excluded:
            continue
        mask = (ctx.labels[bounds] == oid).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
        if len(cnt) >= 2 * order + 2:
            coeffs = efd.efd(cnt, order)
            rows.append({"object_id": oid, **efd.columns(efd.normalize(coeffs))})
            pts = efd.contour_points(coeffs, 120)       # la forma que describen los coeficientes
            smooth[oid] = pts + (cnt.mean(0) - pts.mean(0)) + (bounds[1].start, bounds[0].start)
    ctx.tables["object_shape"] = rows
    included_view(ctx, "shape", set(smooth))
    marks = ctx.extra["overlays"]["shape"]
    marks["outlines"] = [outline for outline in marks["outlines"] if not outline[1]] + [(contour, True) for contour in smooth.values()]
