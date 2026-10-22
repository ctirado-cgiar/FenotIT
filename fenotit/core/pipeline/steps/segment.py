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


@step("classify", "segmenter", provides=("mask", "classes"), params=[
    {"key": "samples", "type": "str", "default": ""},
    {"key": "hash", "type": "str", "default": ""},
    {"key": "objects", "type": "list", "default": [1]},
    {"key": "colors", "type": "list", "default": []},
    {"key": "detail", "type": "float", "default": 1.0},
])
def classify(ctx, p):
    """Segmentación por ejemplos: cada píxel recibe la clase que el bosque aleatorio
    aprendió de los trazos; la máscara = las clases marcadas como objeto."""
    from fenotit.core.segmentation import classifier as C
    if not p["samples"]:
        raise ValueError("no examples: paint at least two classes")
    model = C.load_model(p["samples"], p["hash"])
    small = C.work_copy(ctx.image, float(p["detail"]))
    full = _region(ctx, ctx.image.shape)
    region = None if full is None else cv2.resize(full, (small.shape[1], small.shape[0]),
                                                  interpolation=cv2.INTER_NEAREST) > 0
    classes = C.predict(model, C.features(small), ctx.image.shape, region)
    if full is not None:
        classes[full == 0] = 0
    ctx.classes = classes
    ctx.mask = np.isin(classes, [int(c) for c in p["objects"]]).astype(np.uint8) * 255
    ctx.images["mask"] = ctx.mask
    if p["colors"]:
        ctx.images["classes"] = C.colorize(classes, list(p["colors"]))
    ctx.extra["class_colors"] = list(p["colors"])


@step("model_segment", "segmenter", provides=("mask",), params=[
    {"key": "model", "type": "str", "default": "mobilesam"},
    {"key": "mode", "type": "choice", "choices": ["auto", "points"], "default": "auto"},
    {"key": "grid", "type": "int", "default": 24, "min": 8, "max": 64},
    {"key": "points", "type": "list", "default": []},
])
def model_segment(ctx, p):
    """Segmentación con un modelo de IA (MobileSAM). Cada objeto sale ya separado: las
    etiquetas quedan en ctx.extra["instances"] y el paso de objetos las respeta."""
    from fenotit.core.segmentation import sam
    if p["mode"] == "points":
        labels = sam.from_points(ctx.image, list(p["points"] or []))
    else:
        labels = sam.automatic(ctx.image, grid=int(p["grid"]))
    ctx.extra["instances"] = labels
    ctx.mask = (labels > 0).astype(np.uint8) * 255
    ctx.images["mask"] = ctx.mask
