"""Análisis de objetos: una segmentación y las mediciones elegidas, con la cadena
de piezas de fenotit.core.pipeline. Adapta el resultado al formato de la interfaz."""
from __future__ import annotations

import cv2
import numpy as np

from fenotit.core import pipeline
from fenotit.core import roi as areas
from fenotit.core import segmentation as segmentation_methods
from fenotit.core.analysis.registry import AnalysisResult, register
from fenotit.core.efd import contour_points, from_row
from fenotit.core.pipeline import overlay
from fenotit.core.pipeline.base import BACKGROUND_COLOR
from fenotit.core.pipeline.views import add_stars, object_geometry
from fenotit.core.stats.shape import align
from fenotit.core.units import TO_MM, ascii_name, display, mean_sd, per_px
from fenotit.i18n import t

# Vista de cada pieza → carpeta de sus imágenes en la exportación
STEP_FOLDERS = {"mask": "mask", "distance": "mask", "count": "count", "morphometry": "morphometry",
                "box_size": "box_size", "shape": "shape", "color": "color", "classes": "classes",
                "detections": "detections"}
# Las 2 últimas imágenes quedan en memoria: al cambiar solo una medición no se vuelve a segmentar.
_CACHE = pipeline.ChainCache(size=2)
# Vista de cada medición → casilla del panel que la activa
MEASUREMENTS = {"count": "measure_count", "morphometry": "measure_size", "box_size": "measure_size",
                "shape": "measure_shape", "color": "measure_color", "classes": "measure_classes",
                "detections": "measure_detections"}
MEASURED_BY_DEFAULT = ("count", "morphometry", "box_size", "classes")
# Mediciones que necesitan el contorno de cada objeto (no se calculan con cajas)
NEEDS_MASKS = ("touching", "measure_shape", "measure_color", "measure_touching")
SIZE_LIMITS = ("length_min", "length_max", "width_min", "width_max")

# Colores (BGR) del punto que acompaña a «n = …» en la leyenda de cada vista
COUNT_LEGEND_BGR = (255, 0, 255)
SIZE_LEGEND_BGR = (100, 220, 0)
SHAPE_LEGEND_BGR = (255, 160, 0)
DETECTION_STEP = "object_detections"
MIN_LEGEND_PCT = 0.05                 # clases con menos área no van en la leyenda
SHAPE_LEGEND_POINTS = 80              # puntos del contorno de cada forma dibujada en la leyenda


def _size_filter(params: dict) -> dict:
    """Largo y ancho mín/máx del panel (en la unidad de la escala; sin escala, px)
    llevados a px. Si el panel está en unidades reales y esta foto no tiene escala, no se
    filtra por tamaño."""
    if not params.get("size_filter"):
        return {}
    mm_per_pixel = params.get("mm_per_pixel")
    panel_unit = params.get("area_unit", "px")
    if panel_unit in TO_MM:
        if not mm_per_pixel:
            return {}
        pixels_per_unit = TO_MM[panel_unit] / mm_per_pixel
    else:
        pixels_per_unit = 1.0
    limits_px = {}
    for limit in SIZE_LIMITS:
        value = float(params.get(limit, 0) or 0)
        if value > 0:
            limits_px[limit] = int(round(value * pixels_per_unit))
    return limits_px


def segmentation(params: dict) -> dict:
    """La segmentación elegida (paso 1). Proyectos y lotes anteriores traen claves sueltas."""
    return params.get("segmentation") or segmentation_methods.from_legacy(params)


def _user_points(params: dict, key: str) -> list[list[float]]:
    """Puntos (x, y en 0-1) con que el usuario excluyó (drop_points) o destacó (mark_points) objetos."""
    return [[float(coordinate) for coordinate in point] for point in params.get(key) or []]


def object_chain(params: dict) -> list[dict]:
    """Segmentación y objetos (lo común a los análisis que trabajan con objetos)."""
    settings = segmentation(params)
    selection = {"exclude_border": bool(params.get("exclude_border", False)),
                 "drop_points": _user_points(params, "drop_points"),
                 "mark_points": _user_points(params, "mark_points"),
                 **_size_filter(params)}
    if segmentation_methods.boxes_only(settings):
        return [*segmentation_methods.chain(settings), {"step": "boxes", "params": selection}]
    refine_steps = segmentation_methods.refine_steps(settings)
    # con "Afinar máscara" el usuario decide si se rellenan huecos (es una operación más)
    clean_step = {"step": "clean", "params": {"fill_holes": False}} if refine_steps else {"step": "clean"}
    label_step = {"step": "separate"} if params.get("touching") else {"step": "label"}
    method = settings.get("method")
    # regiones del modelo: las elige el usuario, sin quitar ruido
    noise = {"area_min": 0} if method == "model" else {}
    filter_step = {"step": "filter", "params": {"exclude_border": selection["exclude_border"],
                                                "relative_noise": method == "classifier",
                                                **noise,
                                                "drop_points": selection["drop_points"],
                                                "mark_points": selection["mark_points"],
                                                **_size_filter(params)}}
    return [*segmentation_methods.chain(settings), {"step": "roi"}, *refine_steps, clean_step, label_step,
            filter_step]


def _classes_step(names: list[str], colors: list[str]) -> dict:
    """Pieza `classes` para objetos enteros con clase (la clase 0 es el fondo, sin nombre)."""
    return {"step": "classes", "params": {"names": [""] + names, "colors": [BACKGROUND_COLOR] + colors,
                                          "whole": True}}


def build_chain(params: dict) -> list[dict]:
    chain = object_chain(params)
    settings = segmentation(params)
    if segmentation_methods.boxes_only(settings):        # cajas: tamaño de la caja y conteo (por clase)
        if params.get("measure_size", True):
            chain.append({"step": "box_size"})
        chain.append({"step": "count"})
        return chain
    if params.get("measure_size", True):
        chain.append({"step": "morphometry",
                      "params": {"isolated_only": not params.get("measure_touching", False)}})
    if params.get("measure_shape"):
        chain.append({"step": "shape", "params": {"harmonics": int(params.get("harmonics", 20))}})
    if params.get("measure_color"):
        chain.append({"step": "color", "params": {"n_colors": int(params.get("n_colors", 3)),
                                                  "auto": bool(params.get("n_colors_auto", False)),
                                                  "mode": params.get("color_mode", "object")}})
    method = settings.get("method")
    if method == "classifier" and params.get("measure_classes", True):
        classifier = settings.get("classifier") or {}
        chain.append({"step": "classes", "params": {"names": list(classifier.get("names", [])),
                                                    "colors": list(classifier.get("colors", []))}})
    model = settings.get("model") or {}
    if method == "model" and model.get("output") == "boxes":
        class_names = list(model.get("classes") or [])
        classes_used = [int(index) for index in model.get("use") or []] or list(range(len(class_names)))
        if len(classes_used) > 1:              # cajas pasadas a máscaras: cada objeto con su clase
            chain.append(_classes_step(class_names, list(model.get("colors") or [])))
    elif method == "model" and model.get("mode", "auto") == "auto" and len(model.get("categories") or []) > 1:
        categories = model["categories"]
        chain.append(_classes_step([category["name"] for category in categories],
                                   [category["color"] for category in categories]))
    detector = params.get("detector") or {}
    if params.get("measure_detections") and detector.get("name"):
        chain.append({"step": DETECTION_STEP, "params": detection_params(detector)})
    chain.append({"step": "count"})
    return chain


def detection_params(detector: dict) -> dict:
    """Parámetros de la pieza object_detections a partir del detector elegido."""
    return {"model": detector["name"], "conf": float(detector.get("conf", 0.25)),
            "use": [int(index) for index in detector.get("use") or []],
            "classes": list(detector.get("classes") or []), "colors": list(detector.get("colors") or [])}


def run(image: np.ndarray, params: dict) -> AnalysisResult:
    height, width = image.shape[:2]
    roi_mask, exclusions = areas.masks(params.get("roi_shapes"), width, height)
    chain = build_chain(params)
    ctx = _CACHE.run(image, chain, mm_per_px=params.get("mm_per_pixel"),
                     unit=params.get("length_unit", "mm"),
                     roi=roi_mask, exclusions=exclusions)
    result = AnalysisResult()
    steps_run = {step["step"] for step in chain} | ({"classes"} if ctx.tables.get("classes") else set())
    if DETECTION_STEP in steps_run:
        steps_run.add("detections")
    chosen = [measurement for measurement, checkbox in MEASUREMENTS.items()
              if params.get(checkbox, measurement in MEASURED_BY_DEFAULT) and measurement in steps_run]
    ctx.image_row().update(_image_info(image, params))
    _assign_areas(ctx, params.get("roi_shapes"))
    tables = _view_tables(ctx, params.get("color_mode", "object"))
    folders, view_tables, legends, overlays = {}, {}, {}, {}
    marks = ctx.extra.get("overlays", {})
    for step_key, folder in STEP_FOLDERS.items():
        step_image = ctx.images.get(step_key)
        if step_image is None or (step_key in MEASUREMENTS and step_key not in chosen):
            continue
        view = t(f"step.{step_key}", step_key)
        result.step_images[view] = step_image if step_image.ndim == 3 else cv2.cvtColor(step_image, cv2.COLOR_GRAY2BGR)
        legend = _legend_spec(step_key, ctx)
        if legend:
            legends[view] = legend
        folders[view] = folder
        if step_key in marks:
            overlays[view] = marks[step_key]
        view_tables[view] = tables.get(step_key, tables["image"])
    result.measurements = tables.get("morphometry") or tables.get("box_size", [])   # lo que se puede graficar
    result.stats = dict(ctx.image_row())
    result.extra["tables"] = {name: rows for name, rows in ctx.tables.items() if rows}
    result.extra["step_folders"] = folders
    result.extra["view_tables"] = view_tables
    result.extra["legends"] = legends
    result.extra["overlays"] = overlays          # marcas como datos: se dibujan con el estilo elegido
    result.extra["geometry"] = object_geometry(ctx.labels, ctx.detections)     # elegir, resaltar y recortar un objeto
    area_errors = [relative_sd for object_id, relative_sd in (ctx.extra.get("area_rel_sd") or {}).items()
                   if object_id not in ctx.excluded]
    if area_errors:                              # % de error del área por la discretización (metadatos)
        result.extra["area_uncertainty_pct"] = [float(np.median(area_errors)) * 100, float(np.max(area_errors)) * 100]
    result.extra["excluded"] = sorted(ctx.excluded)
    result.extra["highlighted"] = sorted(ctx.highlighted)
    if ctx.highlighted:                                     # ★ en todas las vistas con marcas
        result.extra["overlays"] = add_stars(overlays, result.extra["geometry"], ctx.highlighted,
                                             list(result.step_images))
    result.extra["base_image"] = ctx.image
    if ctx.labels is not None:
        result.extra["contrast"] = overlay.contrast_color(ctx.image, ctx.labels > 0)
    # vista a mostrar: la medición que se recalculó (si las demás se reutilizaron);
    # si se recalculó todo o nada, la interfaz conserva la vista que estaba viendo
    recalculated = [measurement for measurement in chosen if measurement not in _CACHE.hits]
    only_some = "objects" in _CACHE.hits and 0 < len(recalculated) < len(chosen)
    result.extra["default_view"] = t(f"step.{recalculated[0]}", recalculated[0]) if only_some else None
    result.extra["first_view"] = t(f"step.{chosen[0]}", chosen[0]) if chosen else None
    result.extra["chain"] = chain
    result.extra["reused"] = list(_CACHE.hits)
    return result


def _assign_areas(ctx, shapes: list[dict] | None) -> None:
    """Con áreas de análisis dibujadas: columna `analysis_area` en cada objeto (el área donde
    cae su centro; si el centro queda fuera, el área que más píxeles del objeto contiene) y
    tabla `areas` con el número de objetos y el tamaño de cada área. Sin áreas no agrega nada."""
    height, width = ctx.image.shape[:2]
    numbers = areas.area_map(shapes, width, height)
    object_rows = ctx.tables.get("objects", [])
    if numbers is None or not object_rows:
        return
    names = areas.area_names(shapes)
    counts = dict.fromkeys(names, 0)
    for row in object_rows:
        x = min(width - 1, max(0, int(row["centroid_x_px"])))
        y = min(height - 1, max(0, int(row["centroid_y_px"])))
        number = int(numbers[y, x])
        if number == 0 and ctx.labels is not None:
            inside = numbers[ctx.labels == row["object_id"]]
            inside = inside[inside > 0]
            number = int(np.bincount(inside).argmax()) if inside.size else 0
        row["analysis_area"] = names[number - 1] if number else ""
        if number and row.get("status", "ok") != "excluded":
            counts[names[number - 1]] += 1
    unit, scale = ctx.unit, ctx.scale
    ctx.tables["areas"] = [{"analysis_area": name, "n_objects": counts[name],
                            f"area_size_{unit}2": round(float((numbers == number).sum()) * scale * scale,
                                                        ctx.digits(2))}
                           for number, name in enumerate(names, 1)]


def _has_column(row: dict, prefix: str) -> bool:
    return any(column.startswith(prefix) for column in row)


def _mean_sd_row(measured: list[dict], column: str, title_key: str, unit: str) -> tuple[str, None]:
    """Línea de leyenda «título  media ± DE unidad» de una columna."""
    values = np.array([row[column] for row in measured], float)
    spread = values.std(ddof=1) if len(values) > 1 else 0
    return f"{t(title_key)}  {mean_sd(values.mean(), spread)} {unit}", None


def _legend_spec(step_key: str, ctx) -> dict | None:
    """Lo esencial de cada medición para la leyenda (la interfaz decide si se dibuja)."""
    object_rows = ctx.tables.get("objects", [])
    if step_key == "count":
        return {"rows": [(f"n = {ctx.image_row().get('n_objects', 0)}", COUNT_LEGEND_BGR)]}
    if step_key == "morphometry":
        measured = [row for row in object_rows if _has_column(row, "area_")]
        if not measured:
            return None
        legend_rows = [(f"n = {len(measured)}", SIZE_LEGEND_BGR)]
        for prefix, title_key in (("length_", "legend.length"), ("width_", "legend.width"), ("area_", "legend.area")):
            column = next((name for name in measured[0] if name.startswith(prefix)), None)
            if column:
                legend_rows.append(_mean_sd_row(measured, column, title_key, display(column.split("_", 1)[1])))
        return {"title": t("step.morphometry"), "rows": legend_rows}
    if step_key == "box_size":
        measured = [row for row in object_rows if _has_column(row, "box_width_")]
        if not measured:
            return None
        legend_rows = [(f"n = {len(measured)}", SIZE_LEGEND_BGR)]
        for prefix, title_key in (("box_width_", "legend.box_width"), ("box_height_", "legend.box_height")):
            column = next(name for name in measured[0] if name.startswith(prefix))
            legend_rows.append(_mean_sd_row(measured, column, title_key, display(column[len(prefix):])))
        return {"title": t("step.box_size"), "rows": legend_rows}
    if step_key == "color":
        return {"title": t("legend.image_colors"), "colors": ctx.tables.get("image_colors", [])}
    if step_key == "detections":
        image_row = ctx.image_row()
        legend_rows = [(f"{item['class']}  {image_row.get(item['column'], 0)}",
                        overlay.hex_to_bgr(item["color"] or overlay.UNNAMED_CLASS_COLOR))
                       for item in ctx.extra.get("detection_classes", [])]
        return {"title": t("step.detections"), "rows": legend_rows} if legend_rows else None
    if step_key == "classes":
        legend_rows = []
        for row in ctx.tables.get("classes", []):
            if row["pct"] < MIN_LEGEND_PCT:
                continue                                   # el fondo dentro de los objetos no va en la leyenda
            share = f"{row['n']}  ({row['pct']:.1f} %)" if "n" in row else f"{row['pct']:.1f} %"
            legend_rows.append((f"{row['class']}  {share}", overlay.hex_to_bgr(row.get("color") or overlay.UNNAMED_CLASS_COLOR)))
        return {"title": t("step.classes"), "rows": legend_rows} if legend_rows else None
    if step_key == "shape":
        shape_rows = ctx.tables.get("object_shape", [])
        if not shape_rows:
            return None
        coefficients = align([from_row(row) for row in shape_rows])
        several = len(shape_rows) > 1
        return {"title": t("legend.mean_shape") if several else t("step.shape"),
                "rows": [(f"n = {len(shape_rows)}", SHAPE_LEGEND_BGR)],
                "shape": {"mean": contour_points(np.mean(coefficients, axis=0)),
                          "others": [contour_points(one, SHAPE_LEGEND_POINTS) for one in coefficients] if several
                          else []}}
    return None


def _image_info(image, params) -> dict:
    """Resolución y escala de la foto (quedan en la tabla de imagen y en la exportación)."""
    height, width = image.shape[:2]
    mm_per_pixel = params.get("mm_per_pixel")
    unit = params.get("length_unit", "mm")
    info = {"image_width_px": width, "image_height_px": height, "mm_per_px": mm_per_pixel or ""}
    if mm_per_pixel and ascii_name(unit) != "mm":           # también en la unidad de los resultados
        info[f"{ascii_name(unit)}_per_px"] = round(per_px(mm_per_pixel, unit), 8)
    return info


def _columns(row: dict, keep) -> dict:
    """Las columnas de `row` para las que `keep(nombre)` es verdadero."""
    return {column: value for column, value in row.items() if keep(column)}


def _view_tables(ctx, color_mode: str = "object") -> dict[str, list[dict]]:
    """La tabla que acompaña a cada vista: solo las columnas de esa medición."""
    object_rows = ctx.tables.get("objects", [])
    from_boxes = ctx.detections is not None
    count_columns = ("object_id", "class", "score") if from_boxes else ("object_id", "touching")
    if any("analysis_area" in row for row in object_rows):
        count_columns += ("analysis_area",)
    count_columns += ("centroid_x_px", "centroid_y_px")
    if any(row.get("status", "ok") != "ok" for row in object_rows):
        count_columns += ("status",)
    tables = {"image": ctx.tables.get("image", []),
              "count": [{column: row.get(column) for column in count_columns} for row in object_rows]}
    if from_boxes:
        tables["box_size"] = [_columns(row, lambda column: column in ("object_id", "class") or column.startswith("box_"))
                              for row in object_rows if _has_column(row, "box_")]
        if ctx.tables.get("classes"):
            tables["classes"] = [_columns(row, lambda column: column != "color") for row in ctx.tables["classes"]]
        return tables
    measured = [row for row in object_rows if _has_column(row, "area_")]
    if measured:
        skipped = ("object_id", "analysis_area", "touching", "centroid_x_px", "centroid_y_px")
        size_columns = [column for column in ("object_id", "analysis_area") if column in measured[0]]
        size_columns += [column for column in measured[0] if column not in skipped and not column.startswith("mean_")]
        tables["morphometry"] = [{column: row.get(column) for column in size_columns} for row in measured]
    if ctx.tables.get("object_shape"):
        tables["shape"] = ctx.tables["object_shape"]
    if ctx.tables.get("classes"):
        class_columns = [column for column in (object_rows[0] if object_rows else {})
                         if column == "category" or column.endswith("_pct") or "_area_" in column]
        tables["classes"] = [{"object_id": row["object_id"], **{column: row.get(column) for column in class_columns}}
                             for row in object_rows if row.get("status", "ok") != "excluded"]
    detection_columns = [item["column"] for item in ctx.extra.get("detection_classes", [])]
    if detection_columns:
        tables["detections"] = [{"object_id": row["object_id"], **{column: row.get(column, 0) for column in detection_columns}}
                                for row in object_rows if row.get("status", "ok") != "excluded"]
    if ctx.tables.get("object_colors"):
        # todos juntos: solo los colores comunes (el % de cada objeto va en la exportación)
        tables["color"] = (ctx.tables.get("image_colors", []) if color_mode == "pooled"
                           else ctx.tables["object_colors"])
    return tables


PARAMS_SCHEMA = [
    {"key": "hdr_objects", "type": "header"},
    {"key": "touching", "type": "bool", "default": False},
    {"key": "exclude_border", "type": "bool", "default": False},
    {"key": "size_filter", "type": "bool", "default": False},
    {"key": "length_min", "type": "float", "default": 0,
     "min": 0, "max": 10_000_000, "step": 1, "unit": "length", "group": "size_filter"},
    {"key": "length_max", "type": "float", "default": 0,
     "min": 0, "max": 10_000_000, "step": 1, "unit": "length", "group": "size_filter"},
    {"key": "width_min", "type": "float", "default": 0,
     "min": 0, "max": 10_000_000, "step": 1, "unit": "length", "group": "size_filter"},
    {"key": "width_max", "type": "float", "default": 0,
     "min": 0, "max": 10_000_000, "step": 1, "unit": "length", "group": "size_filter"},
    {"key": "hdr_measure", "type": "header"},
    {"key": "measure_count", "type": "bool", "default": True},
    {"key": "measure_size", "type": "bool", "default": True},
    {"key": "measure_touching", "type": "bool",
     "default": False, "group": "measure_size"},
    {"key": "measure_shape", "type": "bool", "default": False},
    {"key": "harmonics", "type": "int", "default": 20, "min": 1, "max": 50,
     "group": "measure_shape"},
    {"key": "measure_color", "type": "bool", "default": False},
    {"key": "color_mode", "type": "choice", "choices": ["object", "pooled"],
     "default": "object", "group": "measure_color"},
    {"key": "n_colors", "type": "int", "default": 3, "min": 1, "max": 20,
     "group": "measure_color"},
    {"key": "n_colors_auto", "type": "bool", "default": False,
     "group": "measure_color"},
    {"key": "measure_classes", "type": "bool", "default": True,
     "segmentation": "classifier"},
    {"key": "measure_detections", "type": "bool", "default": False},
]


register(
    key="objects",
    func=run,
    icon="shapes",
    params_schema=PARAMS_SCHEMA,
    optional_steps={"detect": "measure_detections"},
)
