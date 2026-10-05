"""Distancias entre objetos: segmenta como "Objetos" y mide distancias euclidianas entre
vecinos (k más cercanos o red de Voronoi), de borde a borde o de centro a centro."""
from __future__ import annotations

import cv2
import numpy as np

from fenotit.core import pipeline
from fenotit.core.analysis import objects     # módulo (no nombres): evita el import circular
from fenotit.core.analysis.registry import AnalysisResult, register
from fenotit.core.units import display, mean_sd
from fenotit.i18n import t


def build_chain(params: dict) -> list[dict]:
    return objects.object_chain(params) + [{"step": "distances", "params": {
        "neighbors": params.get("neighbors", "knn"), "k": int(params.get("k", 5)),
        "measure": params.get("measure", "edge_line")}}, {"step": "count"}]


def _pattern(row: dict) -> str:
    """Agrupado / al azar / regular según Clark-Evans (z con 95 %)."""
    r, z = row.get("clark_evans_R"), row.get("clark_evans_z")
    if r is None or z is None:
        return ""
    key = "random" if abs(z) < 1.96 else ("clustered" if r < 1 else "regular")
    return t(f"legend.pattern.{key}")


def run(image: np.ndarray, params: dict) -> AnalysisResult:
    from fenotit.core import roi as areas
    h, w = image.shape[:2]
    roi_mask, exclusions = areas.masks(params.get("roi_shapes"), w, h)
    ctx = objects._CACHE.run(image, build_chain(params), mm_per_px=params.get("mm_per_pixel"),
                             unit=params.get("length_unit", "mm"),
                     roi=roi_mask, exclusions=exclusions)
    ctx.image_row().update(objects._image_info(image, params))
    row = ctx.image_row()
    unit = ctx.unit
    res = AnalysisResult()
    label = t("step.distances")
    mask = ctx.images.get("mask")
    if mask is not None:
        res.step_images[t("step.mask")] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    res.step_images[label] = ctx.images["distances"]
    mean, sd = row.get(f"mean_nearest_{unit}"), row.get(f"sd_nearest_{unit}")
    rows = [(f"n = {row.get('n_objects', 0)}", (255, 0, 255))]
    if mean is not None:
        rows.append((f"{t('legend.nearest')}  {mean_sd(mean, sd or 0)} {display(unit)}", None))
    if row.get("clark_evans_R") is not None:
        rows.append((f"Clark-Evans R = {row['clark_evans_R']:.2f}  ({_pattern(row)})", None))
    res.extra["legends"] = {label: {"title": t("step.distances"), "rows": rows}}
    res.extra["overlays"] = {label: ctx.extra.get("overlays", {}).get("distances")}
    from fenotit.core.pipeline.views import object_geometry
    res.extra["geometry"] = object_geometry(ctx.labels)     # elegir, resaltar y recortar un objeto
    res.extra["excluded"] = sorted(ctx.excluded)
    res.extra["highlighted"] = sorted(ctx.highlighted)
    if ctx.highlighted:                                     # ★ en todas las vistas con marcas
        from fenotit.core.pipeline.views import add_stars
        res.extra["overlays"] = add_stars(res.extra.get("overlays") or {}, res.extra["geometry"], ctx.highlighted,
                                          list(res.step_images))
    res.extra["base_image"] = ctx.image
    res.extra["contrast"] = __import__("fenotit.core.pipeline.overlay", fromlist=["x"]).contrast_color(
        ctx.image, ctx.labels > 0)
    res.extra["view_tables"] = {label: ctx.tables.get("distances", []), t("step.mask"): ctx.tables.get("image", [])}
    res.extra["step_folders"] = {label: "distances", t("step.mask"): "mask"}
    res.extra["first_view"] = label
    res.extra["tables"] = {k: v for k, v in ctx.tables.items() if v}
    res.extra["chain"] = build_chain(params)
    res.stats = dict(row)
    return res


register(
    name="Distancias",
    func=run,
    description="Distancias entre objetos vecinos y si están agrupados, al azar o espaciados",
    icon="ruler",
    params_schema=[
        {"key": "hdr_objects", "label": "OBJETOS", "type": "header"},
        {"key": "touching", "label": "Los objetos se tocan", "type": "bool", "default": False},
        {"key": "exclude_border", "label": "Excluir objetos del borde", "type": "bool", "default": True},
        {"key": "hdr_measure", "label": "DISTANCIAS", "type": "header"},
        {"key": "neighbors", "label": "Vecinos", "type": "choice", "choices": ["knn", "voronoi"], "default": "knn"},
        {"key": "k", "label": "Número de vecinos", "type": "int", "default": 5, "min": 1, "max": 50},
        {"key": "measure", "label": "Medir", "type": "choice", "choices": ["edge_line", "edge_nearest", "center"],
         "default": "edge_line"},
    ],
)
