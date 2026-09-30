"""Análisis de objetos: una segmentación y las mediciones elegidas, con la cadena
de piezas de fenotit.core.pipeline. Adapta el resultado al formato de la interfaz."""
from __future__ import annotations

import cv2
import numpy as np

from fenotit.core import pipeline
from fenotit.core.analysis.registry import AnalysisResult, register
from fenotit.i18n import t

STEP_FOLDERS = {"mask": "mask", "distance": "mask", "morphometry": "morphometry",
                "shape": "shape", "color": "color", "count": "count"}


def _area_px(params: dict, key: str, default: float) -> int:
    """Las áreas del panel vienen en mm² cuando hay escala (area_unit = "mm")."""
    value = float(params.get(key, default))
    mpp = params.get("mm_per_pixel")
    if params.get("area_unit") == "mm" and mpp:
        value /= mpp * mpp
    return int(round(value))


def build_chain(params: dict) -> list[dict]:
    space = params.get("color_space", "BGR")
    channel = int(params.get("channel_idx", 0))
    if params.get("auto_threshold"):
        seg = {"step": "otsu", "params": {"color_space": space, "channel": channel}}
    else:
        seg = {"step": "threshold", "params": {"color_space": space, "channel": channel,
                                               "min_val": int(params.get("min_val", 0)),
                                               "max_val": int(params.get("max_val", 255))}}
    chain = [seg, {"step": "roi"}, {"step": "clean"},
             {"step": "separate"} if params.get("touching") else {"step": "label"},
             {"step": "filter", "params": {"area_min": _area_px(params, "area_min", 1000),
                                           "area_max": _area_px(params, "area_max", 500_000),
                                           "exclude_border": bool(params.get("exclude_border", True))}}]
    if params.get("measure_size", True):
        chain.append({"step": "morphometry"})
    if params.get("measure_shape"):
        chain.append({"step": "shape", "params": {"harmonics": int(params.get("harmonics", 20))}})
    if params.get("measure_color"):
        chain.append({"step": "color", "params": {"n_colors": int(params.get("n_colors", 3))}})
    chain.append({"step": "count"})
    return chain


def run(image: np.ndarray, params: dict) -> AnalysisResult:
    h, w = image.shape[:2]
    exclusions = [((np.asarray(pts, float) * [w, h]).astype(np.int32), color)
                  for pts, color in params.get("exclusions_norm") or []]
    ctx = pipeline.run(image, build_chain(params), mm_per_px=params.get("mm_per_pixel"),
                       roi=params.get("roi_mask"), exclusions=exclusions)
    res = AnalysisResult()
    names = {}
    for key in STEP_FOLDERS:          # el último (conteo) es el que se muestra primero
        img = ctx.images.get(key)
        if img is not None:
            label = t(f"step.{key}", key)
            res.step_images[label] = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            names[label] = STEP_FOLDERS[key]
    res.measurements = ctx.tables.get("objects", [])
    res.stats = dict(ctx.image_row())
    res.extra["tables"] = {k: v for k, v in ctx.tables.items() if v}
    res.extra["step_folders"] = names
    res.extra["chain"] = build_chain(params)
    return res


register(
    name="Análisis de objetos",
    func=run,
    description="Segmenta una vez y mide tamaño y forma, forma (Fourier), color y conteo",
    icon="shapes",
    params_schema=[
        {"key": "touching", "label": "Los objetos se tocan", "type": "bool", "default": False},
        {"key": "auto_threshold", "label": "Umbral automático (Otsu)", "type": "bool", "default": False},
        {"key": "measure_size", "label": "Morfometría", "type": "bool", "default": True},
        {"key": "measure_shape", "label": "Forma", "type": "bool", "default": False},
        {"key": "measure_color", "label": "Color", "type": "bool", "default": False},
        {"key": "area_min", "label": "Área mínima ({unit})", "type": "float", "default": 1000,
         "min": 0, "max": 100_000_000, "step": 1, "unit": "area", "advanced": True},
        {"key": "area_max", "label": "Área máxima ({unit})", "type": "float", "default": 500_000,
         "min": 0, "max": 1_000_000_000, "step": 1, "unit": "area", "advanced": True},
        {"key": "exclude_border", "label": "Excluir objetos del borde", "type": "bool", "default": True,
         "advanced": True},
        {"key": "harmonics", "label": "Armónicos", "type": "int", "default": 20, "min": 1, "max": 50,
         "requires": "measure_shape", "advanced": True},
        {"key": "n_colors", "label": "Colores por objeto", "type": "int", "default": 3, "min": 1, "max": 10,
         "requires": "measure_color", "advanced": True},
],
)
