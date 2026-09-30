"""Forma: coeficientes elípticos de Fourier por objeto (extracción).
Alinear, promediar y comparar formas es análisis posterior (fenotit.core.stats.shape)."""
import cv2
import numpy as np

from fenotit.core import efd
from fenotit.core.pipeline.base import step


@step("shape", "measurement", requires=("labels",), params=[
    {"key": "harmonics", "type": "int", "default": 20, "min": 1, "max": 50},
    {"key": "isolated_only", "type": "bool", "default": True},
])
def shape(ctx, p):
    """Coeficientes normalizados (tamaño, rotación e inicio) por objeto: tabla object_shape."""
    from scipy.ndimage import find_objects
    order = int(p["harmonics"])
    touching = ctx.touching_ids() if p["isolated_only"] else set()
    rows = []
    for oid, sl in enumerate(find_objects(ctx.labels), 1):
        if sl is None or oid in touching:
            continue
        m = (ctx.labels[sl] == oid).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
        if len(cnt) >= 2 * order + 2:
            rows.append({"object_id": oid, **efd.columns(efd.normalize(efd.efd(cnt, order)))})
    ctx.tables["object_shape"] = rows
