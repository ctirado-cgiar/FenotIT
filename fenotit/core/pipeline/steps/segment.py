"""Segmentadores: producen ctx.mask."""
import cv2
import numpy as np

from fenotit.core import colorspaces
from fenotit.core.pipeline.base import step

_SPACE = {"key": "color_space", "type": "choice", "choices": list(colorspaces.CHANNELS), "default": "BGR"}
_CHANNEL = {"key": "channel", "type": "int", "default": 0, "min": 0, "max": 2}


@step("threshold", "segmenter", provides=("mask",), params=[
    _SPACE, _CHANNEL,
    {"key": "min_val", "type": "int", "default": 0, "min": 0, "max": 255},
    {"key": "max_val", "type": "int", "default": 255, "min": 0, "max": 255},
])
def threshold(ctx, p):
    ch = colorspaces.channel(ctx.image, p["color_space"], p["channel"])
    mask = ((ch >= int(p["min_val"])) & (ch <= int(p["max_val"]))).astype(np.uint8) * 255
    ctx.mask = mask
    ctx.images["mask"] = mask


@step("whole", "segmenter", provides=("mask",))
def whole(ctx, p):
    """Toda la imagen (luego "roi" aplica el área de análisis y las exclusiones)."""
    ctx.mask = np.full(ctx.image.shape[:2], 255, np.uint8)


@step("otsu", "segmenter", provides=("mask",), params=[
    _SPACE, _CHANNEL,
    {"key": "invert", "type": "bool", "default": False},
    {"key": "auto_polarity", "type": "bool", "default": True},
])
def otsu(ctx, p):
    """Umbral automático (Otsu) calculado solo con los píxeles del área de análisis, sin
    las zonas excluidas (si las hay). auto_polarity: si la máscara cubre la mayor parte
    del borde de esa zona (o de la foto), se invierte: el fondo suele tocar el borde, los
    objetos no."""
    ch = colorspaces.channel(ctx.image, p["color_space"], p["channel"])
    region = _region(ctx, ch.shape)
    vals = ch[region > 0] if region is not None and region.any() else ch.ravel()
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inverted = bool(p["invert"])
    mask = ((ch > t) != inverted).astype(np.uint8) * 255
    if p["auto_polarity"]:
        if region is not None and region.any():
            edge = (region > 0) & (cv2.erode(region, np.ones((3, 3), np.uint8)) == 0)
            border = mask[edge]
        else:
            border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
        if (border > 0).mean() > 0.5:
            mask = 255 - mask
            inverted = not inverted
    ctx.mask = mask
    ctx.images["mask"] = mask
    ctx.extra["otsu_threshold"] = float(t)
    ctx.extra["otsu_range"] = (0, int(t)) if inverted else (int(t) + 1, 255)   # objeto = este rango


def _region(ctx, shape):
    """Área de análisis menos zonas excluidas (uint8 0/1), o None si no hay áreas."""
    if ctx.roi is None and not ctx.exclusions:
        return None
    if ctx.roi is not None:
        roi = ctx.roi if ctx.roi.shape[:2] == shape[:2] else cv2.resize(
            ctx.roi, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        region = (roi > 0).astype(np.uint8)
    else:
        region = np.ones(shape[:2], np.uint8)
    for pts, _color in ctx.exclusions or []:
        cv2.fillPoly(region, [np.asarray(pts, np.int32)], 0)
    return region
