"""Umbral en un canal de color: rango Mín-Máx (threshold), Otsu (otsu) y la foto
completa como un solo objeto (whole)."""
import cv2
import numpy as np

# Conversión de la foto a cada espacio de color y lectura de un canal.
from fenotit.core import colorspaces
# El decorador step registra el umbral y Otsu como piezas de la cadena.
from fenotit.core.pipeline.base import step


_SPACE = {"key": "color_space", "type": "choice", "choices": list(colorspaces.CHANNELS), "default": "BGR"}


_CHANNEL = {"key": "channel", "type": "int", "default": 0, "min": 0, "max": 2}


@step("threshold", "segmenter", provides=("mask",), params=[
    _SPACE, _CHANNEL,
    {"key": "min_val", "type": "int", "default": 0, "min": 0, "max": 255},
    {"key": "max_val", "type": "int", "default": 255, "min": 0, "max": 255},
])
def threshold(ctx, params):
    channel = colorspaces.channel(ctx.image, params["color_space"], params["channel"])
    mask = ((channel >= int(params["min_val"])) & (channel <= int(params["max_val"]))).astype(np.uint8) * 255
    ctx.mask = mask
    ctx.images["mask"] = mask


@step("whole", "segmenter", provides=("mask",))
def whole(ctx, params):
    """Toda la imagen (luego "roi" aplica el área de análisis y las exclusiones)."""
    ctx.mask = np.full(ctx.image.shape[:2], 255, np.uint8)


@step("otsu", "segmenter", provides=("mask",), params=[
    _SPACE, _CHANNEL,
    {"key": "invert", "type": "bool", "default": False},
    {"key": "auto_polarity", "type": "bool", "default": True},
])
def otsu(ctx, params):
    """Umbral automático (Otsu) calculado solo con los píxeles del área de análisis, sin
    las zonas excluidas (si las hay). auto_polarity: si la máscara cubre la mayor parte
    del borde de esa zona (o de la foto), se invierte: el fondo suele tocar el borde, los
    objetos no."""
    channel = colorspaces.channel(ctx.image, params["color_space"], params["channel"])
    region = ctx.analysis_region(channel.shape)
    vals = channel[region > 0] if region is not None and region.any() else channel.ravel()
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inverted = bool(params["invert"])
    mask = ((channel > t) != inverted).astype(np.uint8) * 255
    if params["auto_polarity"]:
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
