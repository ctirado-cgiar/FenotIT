"""Clasificar áreas: cada píxel del área de análisis se asigna a una clase marcada con
ejemplos (segmentación por ejemplos) y se reporta el % y el área de cada clase, sin
separar objetos."""
from __future__ import annotations

import numpy as np

# pipeline arma y corre la cadena de piezas; aquí se usa su caché (ChainCache) para no volver a
# clasificar la foto si solo cambia el resumen.
from fenotit.core import pipeline
# segmentation traduce la configuración del panel en piezas; este análisis le pide las de «Por
# ejemplos».
from fenotit.core import segmentation
# register pone este análisis en las tarjetas del panel derecho; AnalysisResult es lo que se
# devuelve a la ventana (tablas, vistas y leyendas).
from fenotit.core.analysis.registry import AnalysisResult, register
# overlay tiene los colores de las marcas; aquí convierte el color de cada clase para la leyenda.
from fenotit.core.pipeline import overlay
# t devuelve un texto en el idioma elegido (nombre de la vista y de la leyenda).
from fenotit.i18n import t

_CACHE = pipeline.ChainCache(size=2)


def build_chain(params: dict) -> list[dict]:
    settings = params.get("segmentation") or {}
    classifier = settings.get("classifier") or {}
    return [*segmentation.chain(dict(settings, method="classifier")),
            {"step": "cover", "params": {"names": list(classifier.get("names", [])),
                                         "colors": list(classifier.get("colors", []))}}]


def _legend(class_rows: list[dict]) -> dict | None:
    """Una línea por clase con al menos 0.05 % del área: nombre, porcentaje y su color."""
    legend_rows = [(f"{row['class']}  {row['pct']:.1f} %", overlay.hex_to_bgr(row.get("color") or overlay.UNNAMED_CLASS_COLOR))
                   for row in class_rows if row["pct"] >= 0.05]
    return {"title": t("step.cover"), "rows": legend_rows} if legend_rows else None


def run(image: np.ndarray, params: dict) -> AnalysisResult:
    from fenotit.core import roi as areas
    height, width = image.shape[:2]
    roi_mask, exclusions = areas.masks(params.get("roi_shapes"), width, height)
    chain = build_chain(params)
    ctx = _CACHE.run(image, chain, mm_per_px=params.get("mm_per_pixel"), unit=params.get("length_unit", "mm"),
                     roi=roi_mask, exclusions=exclusions)
    result = AnalysisResult()
    class_rows = ctx.tables.get("classes", [])
    view = t("step.cover")
    result.step_images[view] = ctx.images["cover"]
    result.extra["step_folders"] = {view: "cover"}
    result.extra["view_tables"] = {view: class_rows}
    legend = _legend(class_rows)
    if legend:
        result.extra["legends"] = {view: legend}
    result.extra["first_view"] = view
    ctx.image_row().update({"image_width_px": width, "image_height_px": height,
                            "mm_per_px": params.get("mm_per_pixel") or ""})
    result.stats = dict(ctx.image_row())
    result.extra["tables"] = {"classes": class_rows, "image": [dict(ctx.image_row())]}
    result.extra["base_image"] = ctx.image
    result.extra["chain"] = chain
    return result


register(
    key="cover",
    func=run,
    icon="layers",
    segmentation="classifier",
    steps=("segment",),
    seg_methods=("classifier",),
    objects=False,
    params_schema=[],
)
