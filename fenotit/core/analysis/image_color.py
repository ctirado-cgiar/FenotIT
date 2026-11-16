"""Color de la imagen: colores dominantes (KMeans) de toda la foto o del área de
análisis, sin segmentar. Las zonas excluidas no cuentan."""
from __future__ import annotations

import numpy as np

from fenotit.core import pipeline
from fenotit.core.analysis.registry import AnalysisResult, register
from fenotit.i18n import t

_CACHE = pipeline.ChainCache(size=2)
MEAN_COLOR_COLUMNS = ("mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b")


def build_chain(params: dict) -> list[dict]:
    return [{"step": "whole"}, {"step": "roi"}, {"step": "label", "params": {"single": True}},
            {"step": "color", "params": {"n_colors": int(params.get("n_colors", 5)), "edge_trim": 0,
                                         "auto": bool(params.get("n_colors_auto", False))}}]


def run(image: np.ndarray, params: dict) -> AnalysisResult:
    from fenotit.core import roi as areas
    height, width = image.shape[:2]
    roi_mask, exclusions = areas.masks(params.get("roi_shapes"), width, height)
    chain = build_chain(params)
    ctx = _CACHE.run(image, chain, roi=roi_mask, exclusions=exclusions)
    result = AnalysisResult()
    colors = ctx.tables.get("image_colors", [])
    view = t("step.color")
    if "color" in ctx.images:
        result.step_images[view] = ctx.images["color"]
        result.extra["legends"] = {view: {"title": t("legend.image_colors"), "colors": colors}}
        result.extra["view_tables"] = {view: colors}
        result.extra["step_folders"] = {view: "color"}
        result.extra["first_view"] = view
    whole_image_row = ctx.object_rows().get(1, {})          # la foto entera es el único «objeto»
    result.stats = {column: whole_image_row[column] for column in MEAN_COLOR_COLUMNS if column in whole_image_row}
    result.extra["tables"] = {"image_colors": colors,
                              "image": [{"image_width_px": width, "image_height_px": height,
                                         "mm_per_px": params.get("mm_per_pixel") or "",
                                         **result.stats, "pixels": int((ctx.labels > 0).sum())}]}
    result.extra["chain"] = chain
    return result


register(
    key="image_color",
    func=run,
    icon="palette",
    segmentation=None,
    params_schema=[
        {"key": "n_colors", "type": "int", "default": 5, "min": 1, "max": 20},
        {"key": "n_colors_auto", "type": "bool", "default": False},
    ],
)
