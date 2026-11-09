"""Clasificar áreas: cada píxel del área de análisis se asigna a una clase marcada con
ejemplos (segmentación por ejemplos) y se reporta el % y el área de cada clase, sin
separar objetos."""
from __future__ import annotations

import numpy as np

from fenotit.core import pipeline
from fenotit.core import segmentation as S
from fenotit.core.analysis.registry import AnalysisResult, register
from fenotit.i18n import t

_CACHE = pipeline.ChainCache(size=2)


def build_chain(params: dict) -> list[dict]:
    seg = params.get("segmentation") or {}
    c = seg.get("classifier") or {}
    return [*S.chain(dict(seg, method="classifier")),
            {"step": "cover", "params": {"names": list(c.get("names", [])), "colors": list(c.get("colors", []))}}]


def _legend(rows: list[dict]) -> dict | None:
    out = []
    for r in rows:
        if r["pct"] < 0.05:
            continue
        h = (r.get("color") or "#888888").lstrip("#")
        out.append((f"{r['class']}  {r['pct']:.1f} %", (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))))
    return {"title": t("step.cover"), "rows": out} if out else None


def run(image: np.ndarray, params: dict) -> AnalysisResult:
    from fenotit.core import roi as areas
    h, w = image.shape[:2]
    roi_mask, exclusions = areas.masks(params.get("roi_shapes"), w, h)
    chain = build_chain(params)
    ctx = _CACHE.run(image, chain, mm_per_px=params.get("mm_per_pixel"), unit=params.get("length_unit", "mm"),
                     roi=roi_mask, exclusions=exclusions)
    res = AnalysisResult()
    rows = ctx.tables.get("classes", [])
    label = t("step.cover")
    res.step_images[label] = ctx.images["cover"]
    res.extra["step_folders"] = {label: "cover"}
    res.extra["view_tables"] = {label: rows}
    legend = _legend(rows)
    if legend:
        res.extra["legends"] = {label: legend}
    res.extra["first_view"] = label
    ctx.image_row().update({"image_width_px": w, "image_height_px": h,
                            "mm_per_px": params.get("mm_per_pixel") or ""})
    res.stats = dict(ctx.image_row())
    res.extra["tables"] = {"classes": rows, "image": [dict(ctx.image_row())]}
    res.extra["base_image"] = ctx.image
    res.extra["chain"] = chain
    return res


register(
    name="Clasificar áreas",
    func=run,
    description="Porcentaje y área de cada clase marcada con ejemplos, sin separar objetos",
    icon="layers",
    segmentation="classifier",
    steps=("segment",),
    seg_methods=("classifier",),
    objects=False,
    params_schema=[],
)
