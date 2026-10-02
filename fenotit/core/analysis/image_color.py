"""Color de la imagen: colores dominantes (KMeans) de toda la foto o del área de
análisis, sin segmentar. Las zonas excluidas no cuentan."""
from __future__ import annotations

import numpy as np

from fenotit.core import pipeline
from fenotit.core.analysis.registry import AnalysisResult, register
from fenotit.i18n import t

_CACHE = pipeline.ChainCache(size=2)


def build_chain(params: dict) -> list[dict]:
    return [{"step": "whole"}, {"step": "roi"}, {"step": "label", "params": {"single": True}},
            {"step": "color", "params": {"n_colors": int(params.get("n_colors", 5)), "edge_trim": 0}}]


def run(image: np.ndarray, params: dict) -> AnalysisResult:
    from fenotit.core import roi as areas
    h, w = image.shape[:2]
    roi_mask, exclusions = areas.masks(params.get("roi_shapes"), w, h)
    ctx = _CACHE.run(image, build_chain(params), roi=roi_mask, exclusions=exclusions)
    res = AnalysisResult()
    colors = ctx.tables.get("image_colors", [])
    label = t("step.color")
    if "color" in ctx.images:
        res.step_images[label] = ctx.images["color"]
        res.extra["legends"] = {label: {"title": t("legend.image_colors"), "colors": colors}}
        res.extra["view_tables"] = {label: colors}
        res.extra["step_folders"] = {label: "color"}
        res.extra["first_view"] = label
    row = ctx.object_rows().get(1, {})
    res.stats = {k: row[k] for k in ("mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b") if k in row}
    res.extra["tables"] = {"image_colors": colors,
                           "image": [{**res.stats, "pixels": int((ctx.labels > 0).sum())}]}
    res.extra["chain"] = build_chain(params)
    return res


register(
    name="Color de imagen",
    func=run,
    description="Colores dominantes de toda la foto o del área de análisis, sin segmentar",
    icon="palette",
    segmentation=None,
    params_schema=[
        {"key": "n_colors", "label": "Número de colores", "type": "int", "default": 5, "min": 1, "max": 20},
    ],
)
