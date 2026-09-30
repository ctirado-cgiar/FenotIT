"""Análisis de objetos: una segmentación y las mediciones elegidas, con la cadena
de piezas de fenotit.core.pipeline. Adapta el resultado al formato de la interfaz."""
from __future__ import annotations

import cv2
import numpy as np

from fenotit.core import pipeline
from fenotit.core.analysis.registry import AnalysisResult, register
from fenotit.i18n import t

STEP_FOLDERS = {"mask": "mask", "distance": "mask", "count": "count", "morphometry": "morphometry",
                "shape": "shape", "color": "color"}
# Últimas imágenes en memoria: al cambiar solo una medición no se vuelve a segmentar
# (cada imagen guardada ocupa ~100-200 MB en fotos de 8 MP; por eso solo 2).
_CACHE = pipeline.ChainCache(size=2)
MEASUREMENTS = {"count": "measure_count", "morphometry": "measure_size",
                "shape": "measure_shape", "color": "measure_color"}


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
        chain.append({"step": "morphometry",
                      "params": {"isolated_only": not params.get("measure_touching", False)}})
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
    ctx = _CACHE.run(image, build_chain(params), mm_per_px=params.get("mm_per_pixel"),
                     roi=params.get("roi_mask"), exclusions=exclusions)
    res = AnalysisResult()
    chosen = [v for v, key in MEASUREMENTS.items() if params.get(key, v in ("count", "morphometry"))]
    tables = _view_tables(ctx)
    names, view_tables, legends = {}, {}, {}
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
        view_tables[label] = tables.get(key, tables["image"])
    res.measurements = tables.get("morphometry", [])        # lo que se puede graficar
    res.stats = dict(ctx.image_row())
    res.extra["tables"] = {k: v for k, v in ctx.tables.items() if v}
    res.extra["step_folders"] = names
    res.extra["view_tables"] = view_tables
    res.extra["legends"] = legends
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
                unit = col.split("_", 1)[1].replace("2", "²")
                rows.append((f"{t(name)}  {v.mean():.1f} ± {v.std(ddof=1) if len(v) > 1 else 0:.1f} {unit}", None))
        return {"title": t("step.morphometry"), "rows": rows}
    if key == "color":
        return {"title": t("legend.image_colors"), "colors": ctx.tables.get("image_colors", [])}
    if key == "shape":
        shapes = ctx.tables.get("object_shape", [])
        if len(shapes) < 2:
            return None
        from fenotit.core.efd import contour_points, from_row
        from fenotit.core.stats.shape import align
        coeffs = align([from_row(r) for r in shapes])
        return {"title": t("legend.mean_shape"), "rows": [(f"n = {len(shapes)}", (255, 160, 0))],
                "shape": {"mean": contour_points(np.mean(coeffs, axis=0)),
                          "others": [contour_points(c, 80) for c in coeffs]}}
    return None


def _view_tables(ctx) -> dict[str, list[dict]]:
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
        out["color"] = ctx.tables["object_colors"]
    return out


register(
    name="Objetos",
    func=run,
    description="Segmenta una vez y mide tamaño y forma, forma (Fourier), color y conteo",
    icon="shapes",
    params_schema=[
        {"key": "hdr_objects", "label": "OBJETOS", "type": "header"},
        {"key": "touching", "label": "Los objetos se tocan", "type": "bool", "default": False},
        {"key": "filters", "label": "Filtros de objetos", "type": "section"},
        {"key": "area_min", "label": "Área mínima ({unit})", "type": "float", "default": 1000,
         "min": 0, "max": 100_000_000, "step": 1, "unit": "area", "group": "filters"},
        {"key": "area_max", "label": "Área máxima ({unit})", "type": "float", "default": 500_000,
         "min": 0, "max": 1_000_000_000, "step": 1, "unit": "area", "group": "filters"},
        {"key": "exclude_border", "label": "Excluir objetos del borde", "type": "bool", "default": True,
         "group": "filters"},
        {"key": "hdr_measure", "label": "MEDICIONES", "type": "header"},
        {"key": "measure_count", "label": "Conteo", "type": "bool", "default": True},
        {"key": "measure_size", "label": "Morfometría", "type": "bool", "default": True},
        {"key": "measure_touching", "label": "Medir también los que se tocan", "type": "bool",
         "default": False, "group": "measure_size"},
        {"key": "measure_shape", "label": "Forma", "type": "bool", "default": False},
        {"key": "harmonics", "label": "Armónicos", "type": "int", "default": 20, "min": 1, "max": 50,
         "group": "measure_shape"},
        {"key": "measure_color", "label": "Color", "type": "bool", "default": False},
        {"key": "n_colors", "label": "Colores por objeto", "type": "int", "default": 3, "min": 1, "max": 10,
         "group": "measure_color"},
],
)
