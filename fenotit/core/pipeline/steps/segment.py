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


@step("otsu", "segmenter", provides=("mask",), params=[
    _SPACE, _CHANNEL,
    {"key": "invert", "type": "bool", "default": False},
    {"key": "auto_polarity", "type": "bool", "default": True},
])
def otsu(ctx, p):
    """Umbral automático. auto_polarity: si la máscara cubre la mayor parte del borde
    de la foto, se invierte (el fondo suele tocar el borde, los objetos no)."""
    ch = colorspaces.channel(ctx.image, p["color_space"], p["channel"])
    mode = cv2.THRESH_BINARY_INV if p["invert"] else cv2.THRESH_BINARY
    t, mask = cv2.threshold(ch, 0, 255, mode | cv2.THRESH_OTSU)
    if p["auto_polarity"]:
        border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
        if (border > 0).mean() > 0.5:
            mask = 255 - mask
    ctx.mask = mask
    ctx.images["mask"] = mask
    ctx.extra["otsu_threshold"] = float(t)
