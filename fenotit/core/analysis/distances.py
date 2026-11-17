"""Distancias entre objetos: los objetos salen de la segmentación (como en "Objetos") o de un
modelo de detección, y se miden distancias euclidianas entre vecinos (k más cercanos o red de
Voronoi), de borde a borde o de centro a centro."""
from __future__ import annotations

import cv2
import numpy as np

from fenotit.core.analysis import objects     # módulo (no nombres): evita el import circular
from fenotit.core.analysis.registry import AnalysisResult, register
from fenotit.core.pipeline import overlay
from fenotit.core.pipeline.views import add_stars, object_geometry
from fenotit.core.units import display, mean_sd
from fenotit.i18n import t


def from_detector(params: dict) -> dict:
    """Los parámetros con el detector como segmentación de cajas (si se eligió «Objetos de un
    modelo de detección»)."""
    if not params.get("from_detector"):
        return params
    from fenotit.core.analysis import detection
    detector = params.get("detector") or {}
    if not detector.get("name"):
        raise FileNotFoundError("No detection model chosen")
    return {**params, "touching": False, "segmentation": detection.segmentation(detector)}


def build_chain(params: dict) -> list[dict]:
    params = from_detector(params)
    return objects.object_chain(params) + [{"step": "distances", "params": {
        "neighbors": params.get("neighbors", "knn"), "k": int(params.get("k", 5)),
        "measure": params.get("measure", "edge_line")}}, {"step": "count"}]


NEIGHBOR_COLOR_BGR = (255, 0, 255)      # punto de la leyenda junto al número de objetos
Z_95 = 1.96                             # |z| menor = patrón que no se distingue del azar (95 %)


def _pattern(image_row: dict) -> str:
    """Agrupado / al azar / regular según Clark-Evans (z con 95 %)."""
    ratio, z_score = image_row.get("clark_evans_R"), image_row.get("clark_evans_z")
    if ratio is None or z_score is None:
        return ""
    pattern = "random" if abs(z_score) < Z_95 else ("clustered" if ratio < 1 else "regular")
    return t(f"legend.pattern.{pattern}")


def run(image: np.ndarray, params: dict) -> AnalysisResult:
    from fenotit.core import roi as areas
    height, width = image.shape[:2]
    roi_mask, exclusions = areas.masks(params.get("roi_shapes"), width, height)
    params = from_detector(params)
    chain = build_chain(params)
    ctx = objects._CACHE.run(image, chain, mm_per_px=params.get("mm_per_pixel"),
                             unit=params.get("length_unit", "mm"), roi=roi_mask, exclusions=exclusions)
    ctx.image_row().update(objects._image_info(image, params))
    image_row = ctx.image_row()
    unit = ctx.unit
    result = AnalysisResult()
    view, mask_view = t("step.distances"), t("step.mask")
    mask = ctx.images.get("mask")
    if mask is not None:
        result.step_images[mask_view] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result.step_images[view] = ctx.images["distances"]

    nearest_mean, nearest_sd = image_row.get(f"mean_nearest_{unit}"), image_row.get(f"sd_nearest_{unit}")
    legend_rows = [(f"n = {image_row.get('n_objects', 0)}", NEIGHBOR_COLOR_BGR)]
    if nearest_mean is not None:
        legend_rows.append((f"{t('legend.nearest')}  {mean_sd(nearest_mean, nearest_sd or 0)} {display(unit)}", None))
    if image_row.get("clark_evans_R") is not None:
        legend_rows.append((f"Clark-Evans R = {image_row['clark_evans_R']:.2f}  ({_pattern(image_row)})", None))
    result.extra["legends"] = {view: {"title": view, "rows": legend_rows}}

    result.extra["overlays"] = {view: ctx.extra.get("overlays", {}).get("distances")}
    result.extra["geometry"] = object_geometry(ctx.labels, ctx.detections)     # elegir, resaltar y recortar un objeto
    result.extra["excluded"] = sorted(ctx.excluded)
    result.extra["highlighted"] = sorted(ctx.highlighted)
    if ctx.highlighted:                                     # ★ en todas las vistas con marcas
        result.extra["overlays"] = add_stars(result.extra["overlays"], result.extra["geometry"], ctx.highlighted,
                                             list(result.step_images))
    result.extra["base_image"] = ctx.image
    result.extra["contrast"] = overlay.contrast_color(ctx.image, ctx.labels > 0)
    result.extra["view_tables"] = {view: ctx.tables.get("distances", []), mask_view: ctx.tables.get("image", [])}
    result.extra["step_folders"] = {view: "distances", mask_view: "mask"}
    result.extra["first_view"] = view
    result.extra["tables"] = {name: rows for name, rows in ctx.tables.items() if rows}
    result.extra["chain"] = chain
    result.stats = dict(image_row)
    return result


register(
    key="distances",
    func=run,
    icon="ruler",
    params_schema=[
        {"key": "hdr_objects", "type": "header"},
        {"key": "from_detector", "type": "bool", "default": False},
        {"key": "touching", "type": "bool", "default": False},
        {"key": "exclude_border", "type": "bool", "default": False},
        {"key": "hdr_measure", "type": "header"},
        {"key": "neighbors", "type": "choice", "choices": ["knn", "voronoi"], "default": "knn"},
        {"key": "k", "type": "int", "default": 5, "min": 1, "max": 50},
        {"key": "measure", "type": "choice", "choices": ["edge_line", "edge_nearest", "center"],
         "default": "edge_line"},
    ],
    optional_steps={"detect": "from_detector"},
    replaced_steps={"detect": "segment"},
)
