"""La máscara antes de formar objetos: recorte al área de análisis (roi) y limpieza
(clean)."""
import cv2
import numpy as np

from fenotit.core.pipeline.base import step


def _fit(mask, shape):
    return mask if mask.shape[:2] == shape[:2] else cv2.resize(
        mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


@step("roi", "processor", requires=("mask",), provides=("mask",))
def roi(ctx, p):
    mask = ctx.mask
    if ctx.roi is not None:
        mask = cv2.bitwise_and(mask, _fit(ctx.roi, mask.shape))
    if ctx.exclusions:
        mask = mask.copy()
        for pts, _color in ctx.exclusions:
            cv2.fillPoly(mask, [np.asarray(pts, np.int32)], 0)
    ctx.mask = mask


@step("clean", "processor", requires=("mask",), provides=("mask",), params=[
    {"key": "open_iterations", "type": "int", "default": 0, "min": 0, "max": 10},
    {"key": "fill_holes", "type": "bool", "default": True},
])
def clean(ctx, p):
    mask = ctx.mask
    if p["open_iterations"]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
                                iterations=int(p["open_iterations"]))
    if p["fill_holes"]:
        mask = _fill_small_holes(mask)
    ctx.mask = mask


def _fill_small_holes(mask, frac=0.2):
    """Rellena huecos menores que frac × el área típica de un objeto. Los huecos
    grandes suelen ser fondo encerrado por un anillo de objetos que se tocan."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n < 2:
        return mask
    typical = float(np.median(stats[1:, cv2.CC_STAT_AREA]))
    inv = (mask == 0).astype(np.uint8)
    k, lab, hs, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = mask.shape
    out = mask.copy()
    for i in range(1, k):
        x, y, bw, bh, a = hs[i]
        if x == 0 or y == 0 or x + bw == w or y + bh == h:
            continue
        if a < frac * typical:
            out[lab == i] = 255
    return out
