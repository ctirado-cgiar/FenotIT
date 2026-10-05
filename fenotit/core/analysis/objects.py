"""Análisis de objetos: una segmentación y las mediciones elegidas, con la cadena
de piezas de fenotit.core.pipeline. Adapta el resultado al formato de la interfaz."""
from __future__ import annotations

import cv2
import numpy as np

from fenotit.core import pipeline
from fenotit.core.analysis.registry import AnalysisResult, register
from fenotit.core.units import ascii_name, display, mean_sd, per_px
from fenotit.i18n import t

STEP_FOLDERS = {"mask": "mask", "distance": "mask", "count": "count", "morphometry": "morphometry",
                "shape": "shape", "color": "color"}
# Últimas imágenes en memoria: al cambiar solo una medición no se vuelve a segmentar
# (cada imagen guardada ocupa ~100-200 MB en fotos de 8 MP; por eso solo 2).
_CACHE = pipeline.ChainCache(size=2)
MEASUREMENTS = {"count": "measure_count", "morphometry": "measure_size",
                "shape": "measure_shape", "color": "measure_color"}


def _size_filter(params: dict) -> dict:
    """Largo y ancho mín/máx del panel (en la unidad de la escala; sin escala, px)
    llevados a px. Si el panel está en unidades reales y esta foto no tiene escala, no se
    filtra por tamaño."""
    from fenotit.core.units import TO_MM
    if not params.get("size_filter"):
        return {}
    mpp = params.get("mm_per_pixel")
    unit = params.get("area_unit", "px")
    if unit in TO_MM:
        if not mpp:
            return {}
        k = TO_MM[unit] / mpp                      # px por unidad
    else:
        k = 1.0
    out = {}
    for key in ("length_min", "length_max", "width_min", "width_max"):
        v = float(params.get(key, 0) or 0)
        if v > 0:
            out[key] = int(round(v * k))
    return out


def object_chain(params: dict) -> list[dict]:
    """Segmentación y objetos (lo común a los análisis que trabajan con objetos)."""
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
             {"step": "filter", "params": {"exclude_border": bool(params.get("exclude_border", True)),
                                           "drop_points": [list(map(float, pt)) for pt in params.get("drop_points") or []],
                                           **_size_filter(params)}}]
    return chain


def build_chain(params: dict) -> list[dict]:
    chain = object_chain(params)
    if params.get("measure_size", True):
        chain.append({"step": "morphometry",
                      "params": {"isolated_only": not params.get("measure_touching", False)}})
    if params.get("measure_shape"):
        chain.append({"step": "shape", "params": {"harmonics": int(params.get("harmonics", 20))}})
    if params.get("measure_color"):
        chain.append({"step": "color", "params": {"n_colors": int(params.get("n_colors", 3)),
                                                  "mode": params.get("color_mode", "object")}})
    chain.append({"step": "count"})
    return chain


def run(image: np.ndarray, params: dict) -> AnalysisResult:
    from fenotit.core import roi as areas
    h, w = image.shape[:2]
    roi_mask, exclusions = areas.masks(params.get("roi_shapes"), w, h)
    ctx = _CACHE.run(image, build_chain(params), mm_per_px=params.get("mm_per_pixel"),
                     unit=params.get("length_unit", "mm"),
                     roi=roi_mask, exclusions=exclusions)
    res = AnalysisResult()
    chosen = [v for v, key in MEASUREMENTS.items() if params.get(key, v in ("count", "morphometry"))]
    ctx.image_row().update(_image_info(image, params))
    tables = _view_tables(ctx, params.get("color_mode", "object"))
    names, view_tables, legends, overlays = {}, {}, {}, {}
    marks = ctx.extra.get("overlays", {})
    for key in STEP_FOLDERS:
        img = ctx.images.get(key)
        if img is None or (key in MEASUREMENTS and key not in chosen):
            continue
        label = t(f"step.{key}", key)
        res.step_images[label] = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        spec = _legend_spec(key, ctx)
        if spec:
            legends[label] = spec
        names[label] = STEP_FOLDERS[key]
        if key in marks:
            overlays[label] = marks[key]
        view_tables[label] = tables.get(key, tables["image"])
    res.measurements = tables.get("morphometry", [])        # lo que se puede graficar
    res.stats = dict(ctx.image_row())
    res.extra["tables"] = {k: v for k, v in ctx.tables.items() if v}
    res.extra["step_folders"] = names
    res.extra["view_tables"] = view_tables
    res.extra["legends"] = legends
    res.extra["overlays"] = overlays          # marcas como datos: se dibujan con el estilo elegido
    from fenotit.core.pipeline.views import object_geometry
    res.extra["geometry"] = object_geometry(ctx.labels)     # elegir, resaltar y recortar un objeto
    res.extra["excluded"] = sorted(ctx.excluded)
    res.extra["base_image"] = ctx.image
    if ctx.labels is not None:
        from fenotit.core.pipeline import overlay
        res.extra["contrast"] = overlay.contrast_color(ctx.image, ctx.labels > 0)
    # vista a mostrar: la medición que se recalculó (si las demás se reutilizaron);
    # si se recalculó todo o nada, la interfaz conserva la vista que estaba viendo
    changed = [v for v in chosen if v not in _CACHE.hits]
    partial = "objects" in _CACHE.hits and 0 < len(changed) < len(chosen)
    res.extra["default_view"] = t(f"step.{changed[0]}", changed[0]) if partial else None
    res.extra["first_view"] = t(f"step.{chosen[0]}", chosen[0]) if chosen else None
    res.extra["chain"] = build_chain(params)
    res.extra["reused"] = list(_CACHE.hits)
    return res


def _legend_spec(key: str, ctx) -> dict | None:
    """Lo esencial de cada medición para la leyenda (la interfaz decide si se dibuja)."""
    objects = ctx.tables.get("objects", [])
    if key == "count":
        return {"rows": [(f"n = {ctx.image_row().get('n_objects', 0)}", (255, 0, 255))]}
    if key == "morphometry":
        measured = [r for r in objects if any(k.startswith("area_") for k in r)]
        if not measured:
            return None
        rows = [(f"n = {len(measured)}", (100, 220, 0))]
        for prefix, name in (("length_", "legend.length"), ("width_", "legend.width"), ("area_", "legend.area")):
            col = next((k for k in measured[0] if k.startswith(prefix)), None)
            if col:
                v = np.array([r[col] for r in measured], float)
                unit = display(col.split("_", 1)[1])
                rows.append((f"{t(name)}  {mean_sd(v.mean(), v.std(ddof=1) if len(v) > 1 else 0)} {unit}", None))
        return {"title": t("step.morphometry"), "rows": rows}
    if key == "color":
        return {"title": t("legend.image_colors"), "colors": ctx.tables.get("image_colors", [])}
    if key == "shape":
        shapes = ctx.tables.get("object_shape", [])
        if not shapes:
            return None
        from fenotit.core.efd import contour_points, from_row
        from fenotit.core.stats.shape import align
        coeffs = align([from_row(r) for r in shapes])
        return {"title": t("legend.mean_shape") if len(shapes) > 1 else t("step.shape"), "rows": [(f"n = {len(shapes)}", (255, 160, 0))],
                "shape": {"mean": contour_points(np.mean(coeffs, axis=0)),
                          "others": [contour_points(c, 80) for c in coeffs] if len(coeffs) > 1 else []}}
    return None


def _image_info(image, params) -> dict:
    """Resolución y escala de la foto (quedan en la tabla de imagen y en la exportación)."""
    h, w = image.shape[:2]
    mpp = params.get("mm_per_pixel")
    unit = params.get("length_unit", "mm")
    info = {"image_width_px": w, "image_height_px": h, "mm_per_px": mpp or ""}
    if mpp and ascii_name(unit) != "mm":           # también en la unidad de los resultados
        info[f"{ascii_name(unit)}_per_px"] = round(per_px(mpp, unit), 8)
    return info


def _view_tables(ctx, color_mode: str = "object") -> dict[str, list[dict]]:
    """La tabla que acompaña a cada vista: solo las columnas de esa medición."""
    objects = ctx.tables.get("objects", [])
    out = {"image": ctx.tables.get("image", []),
           "count": [{k: r.get(k) for k in ("object_id", "touching", "centroid_x_px", "centroid_y_px")}
                     for r in objects]}
    measured = [r for r in objects if any(k.startswith("area_") for k in r)]
    if measured:
        cols = [k for k in measured[0] if k not in ("touching", "centroid_x_px", "centroid_y_px")
                and not k.startswith("mean_")]
        out["morphometry"] = [{k: r.get(k) for k in cols} for r in measured]
    if ctx.tables.get("object_shape"):
        out["shape"] = ctx.tables["object_shape"]
    if ctx.tables.get("object_colors"):
        # todos juntos: solo los colores comunes (el % de cada objeto va en la exportación)
        out["color"] = (ctx.tables.get("image_colors", []) if color_mode == "pooled"
                        else ctx.tables["object_colors"])
    return out


register(
    name="Objetos",
    func=run,
    description="Segmenta una vez y mide tamaño y forma, forma (Fourier), color y conteo",
    icon="shapes",
    params_schema=[
        {"key": "hdr_objects", "label": "OBJETOS", "type": "header"},
        {"key": "touching", "label": "Los objetos se tocan", "type": "bool", "default": False},
        {"key": "exclude_border", "label": "Excluir objetos del borde", "type": "bool", "default": True},
        {"key": "size_filter", "label": "Filtrar por tamaño", "type": "bool", "default": False},
        {"key": "length_min", "label": "Largo mínimo ({unit})", "type": "float", "default": 0,
         "min": 0, "max": 10_000_000, "step": 1, "unit": "length", "group": "size_filter"},
        {"key": "length_max", "label": "Largo máximo ({unit})", "type": "float", "default": 0,
         "min": 0, "max": 10_000_000, "step": 1, "unit": "length", "group": "size_filter"},
        {"key": "width_min", "label": "Ancho mínimo ({unit})", "type": "float", "default": 0,
         "min": 0, "max": 10_000_000, "step": 1, "unit": "length", "group": "size_filter"},
        {"key": "width_max", "label": "Ancho máximo ({unit})", "type": "float", "default": 0,
         "min": 0, "max": 10_000_000, "step": 1, "unit": "length", "group": "size_filter"},
        {"key": "hdr_measure", "label": "MEDICIONES", "type": "header"},
        {"key": "measure_count", "label": "Conteo", "type": "bool", "default": True},
        {"key": "measure_size", "label": "Morfometría", "type": "bool", "default": True},
        {"key": "measure_touching", "label": "Medir también los que se tocan", "type": "bool",
         "default": False, "group": "measure_size"},
        {"key": "measure_shape", "label": "Forma", "type": "bool", "default": False},
        {"key": "harmonics", "label": "Armónicos", "type": "int", "default": 20, "min": 1, "max": 50,
         "group": "measure_shape"},
        {"key": "measure_color", "label": "Color", "type": "bool", "default": False},
        {"key": "color_mode", "label": "Colores", "type": "choice", "choices": ["object", "pooled"],
         "default": "object", "group": "measure_color"},
        {"key": "n_colors", "label": "Número de colores", "type": "int", "default": 3, "min": 1, "max": 20,
         "group": "measure_color"},
],
)
