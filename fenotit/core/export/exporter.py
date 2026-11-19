"""
Exportación única y explícita. Por cada análisis, una carpeta:

  CARPETA/<analysis>/
  ├── objects.csv, image.csv, object_colors.csv...   una tabla por nivel (Image_ID + Image_name)
  ├── image.csv        una fila por foto: estado, mediciones de la foto, escala, áreas
  ├── metadata.csv     versión, fecha, análisis, parámetros, segmentación, correcciones
  ├── <analysis>.xlsx  lo mismo, una hoja por tabla
  └── views/<vista>/   imágenes de cada vista (opcional; se recalculan, ver core.batch)

Todo en inglés; CSV utf-8-sig. Image_ID = posición de la foto en la lista del proyecto,
igual en todos los análisis.
"""
from __future__ import annotations

import csv
import datetime
import platform
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Nombre y versión del programa, que se escriben en metadata.csv, y el registro de mensajes.
from fenotit import APP_NAME, __version__, log

_log = log.get("export")

try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    XLSX_OK = True
except ImportError:
    XLSX_OK = False

SKIP_PARAMS = ("roi_shapes", "mm_per_pixel", "color_space_code")


def save_image(folder: Path, name: str, img: np.ndarray):
    """Guarda imagen (rutas Unicode)."""
    folder.mkdir(parents=True, exist_ok=True)
    ext = Path(name).suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        name, ext = Path(name).stem + ".jpg", ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise OSError(f"no se pudo codificar {name}")
    (folder / name).write_bytes(buf.tobytes())


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, dict)):
        return str(value)
    return "" if value is None else value


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(column for row in rows for column in row))
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({column: _plain(value) for column, value in row.items()} for row in rows)


def _params_of(result) -> dict:
    params = result.extra.get("params") or {}
    return {key: value for key, value in sorted(params.items()) if key not in SKIP_PARAMS}


def collect(analysis: str, results: dict[str, Any], images: list[str],
            image_info: dict[str, dict] | None = None, skipped: set | None = None,
            board: set | None = None, ids: dict[str, int] | None = None,
            meta: dict[str, dict] | None = None) -> dict[str, list[dict]]:
    """Tablas del análisis. `image` lleva una fila por foto de `images` (también las no
    analizadas): estado, sus mediciones de imagen y escala/áreas. `ids` = Image_ID de cada
    foto (su posición en el proyecto) cuando se exporta solo una parte. `meta` = columnas de
    la tabla del usuario por foto (genotipo, rep…): van en todas las tablas junto a Image_name."""
    image_info = image_info or {}
    skipped = skipped or set()
    tables: dict[str, list[dict]] = {"image": []}
    params = [_params_of(result) for result in results.values() if result.status == "ok"]
    varied = any(other != params[0] for other in params[1:]) if params else False
    for position, path in enumerate(images, 1):
        image_id = (ids or {}).get(path, position)
        name = Path(path).name
        result = results.get(path)
        status = ("calibration" if path in (board or ()) else "skipped" if path in skipped else "not_analyzed") \
            if result is None else \
            ("ok" if result.status == "ok" else "error")
        user = (meta or {}).get(path) or {}
        head = {"Image_ID": image_id, "Image_name": name, **user}
        tail = dict(image_info.get(path, {}))
        if result is not None and result.status != "ok":
            tail["error"] = result.error
        if result is not None and result.extra.get("processing_s") is not None:
            tail["processing_s"] = result.extra["processing_s"]
        if varied and result is not None and result.status == "ok":
            tail["params"] = "; ".join(f"{param}={value}" for param, value in _params_of(result).items())
        own = (result.extra.get("tables") or {}).get("image") if result is not None and result.status == "ok" else None
        tables["image"].extend(_merge(head, user, {"status": status}, x, tail) for x in (own or [{}]))
        if result is None or result.status != "ok":
            continue
        for table, rows in (result.extra.get("tables") or {}).items():
            if table != "image":
                tables.setdefault(table, []).extend(
                    _merge(head, user, x) for x in rows)
    return tables


def _merge(head: dict, user: dict, *parts: dict) -> dict:
    """Une columnas; si una columna del usuario (`user`, ya en `head`) se llama como una
    nuestra, la suya pasa a llamarse user_<nombre> (no se pisan)."""
    out, theirs = dict(head), set(user)
    for part in parts:
        for column, value in part.items():
            if column in theirs:
                out = {(f"user_{existing}" if existing == column else existing): x for existing, x in out.items()}
                theirs.discard(column)
            out[column] = value
    return out


def preset_rows(project) -> list[tuple]:
    """Preset aplicado al proyecto y si se cambió algo después."""
    from fenotit.core import preset
    name = (getattr(project, "preset", None) or {}).get("name")
    if not name:
        return [("preset", "none")]
    return [("preset", name), ("preset_modified", "yes" if preset.modified(project) else "no")]


def metadata(analysis: str, results: dict[str, Any], project=None, extra: dict | None = None) -> list[dict]:
    """Filas clave/valor: lo necesario para repetir y citar el análisis."""
    ok = [result for result in results.values() if result.status == "ok"]
    rows = [("software", APP_NAME), ("version", __version__),
            ("exported", datetime.datetime.now().isoformat(timespec="seconds")),
            ("python", platform.python_version()), ("system", f"{platform.system()} {platform.release()}"),
            ("analysis", analysis), ("n_images_ok", len(ok)),
            ("n_images_error", sum(1 for result in results.values() if result.status != "ok"))]
    secs = [result.extra["processing_s"] for result in results.values() if result.extra.get("processing_s") is not None]
    if secs:
        rows += [("processing_s_total", round(sum(secs), 2)),
                 ("processing_s_per_image_mean", round(sum(secs) / len(secs), 3))]
    if project is not None:
        corr = project.corrections
        rows += [("project", str(project.file or "")), ("length_unit", project.unit),
                 ("scale_all_images_mm_per_px", project.scale.mm_per_pixel or ""),
                 ("scale_source", project.scale.source),
                 ("images_with_own_scale", len(project.image_scale)),
                 ("analysis_areas_all_images", len(project.roi)),
                 ("images_with_own_areas", len(project.image_roi)),
                 ("objects_excluded_manually", sum(len(value) for value in getattr(project, "excluded", {}).values())),
                 ("objects_highlighted", sum(len(value) for value in getattr(project, "highlighted", {}).values())),
                 *segmentation_rows(project),
                 ("corrections", ", ".join(corr.active()) or "none"),
                 ("max_megapixels", project.max_mpx or "full"),
                 *preset_rows(project)]
    unc = [result.extra["area_uncertainty_pct"] for result in ok if result.extra.get("area_uncertainty_pct")]
    if unc:
        # error relativo del área por la discretización en píxeles: sqrt(P/12)/A (docs/methods)
        rows += [("area_uncertainty_pct_median", round(float(np.median([bounds[0] for bounds in unc])), 3)),
                 ("area_uncertainty_pct_max", round(float(max(bounds[1] for bounds in unc)), 3)),
                 ("area_uncertainty_method", "pixel discretization: sd(A) = sqrt(P/12) px2; "
                                             "relative to A; scale error not included")]
    if ok:
        params = [_params_of(result) for result in ok]
        if any(other != params[0] for other in params[1:]):
            rows.append(("params", "varied between images: see the image table"))
        else:
            rows += [(f"param.{key}", value) for key, value in params[0].items()]
    rows += list((extra or {}).items())
    return [{"key": key, "value": _plain(value)} for key, value in rows]


def model_rows(model_id: str) -> list[tuple[str, Any]]:
    """La ficha del modelo de IA (nombre, para qué sirve, propietario, licencia, fuente, cita)."""
    from fenotit.core.segmentation import models
    model = models.get(model_id) if model_id else None
    if model is None:
        return [("model", model_id or "")]
    rows = [("model", model["name"]), ("model_output", model["output"]), ("model_format", model["format"])]
    rows += [(f"model_{field}", model[field]) for field in models.INFO if model.get(field)]
    if model["output"] == "boxes":
        rows.append(("model_classes", "; ".join(model["classes"])))
    return rows


def segmentation_rows(project) -> list[tuple[str, Any]]:
    """La segmentación (paso 1) en palabras: método, ajustes y operaciones de afinado."""
    seg = project.segmentation
    method = getattr(seg, "method", "threshold")
    rows = [("segmentation_method", method)]
    if method == "classifier":
        names = [class_info.get("name", "") for class_info in seg.classes]
        obj = [names[i] for i in seg.object_classes()]
        strokes = getattr(project, "strokes", {}) or {}
        rows += [("segmentation", "pixel classifier (random forest, examples painted by the user)"),
                 ("classes", "; ".join(names)), ("object_classes", "; ".join(obj)),
                 ("examples_images", sum(1 for image_strokes in strokes.values() if image_strokes)),
                 ("examples_strokes", sum(len(image_strokes) for image_strokes in strokes.values())),
                 ("classifier_detail_mpx", seg.detail)]
    elif method == "model" and (getattr(seg, "model", {}) or {}).get("output") == "boxes":
        model = seg.model
        names = list(model.get("classes") or [])
        use = [int(class_info) for class_info in model.get("use") or []] or list(range(len(names)))
        rows += [("segmentation", "object detector (boxes)" + (", boxes to masks with MobileSAM (Zhang et al. 2023)"
                                                               if model.get("to_masks") else "")),
                 *model_rows(model.get("name", "")),
                 ("detector_min_confidence", model.get("conf", 0.25)),
                 ("detector_max_overlap_iou", model.get("iou", 0.5)),
                 ("detector_overlap_across_classes", bool(model.get("one_per_object"))),
                 ("detector_classes_used", "; ".join(names[i] for i in use if i < len(names)))]
    elif method == "model":
        model = getattr(seg, "model", {}) or {}
        regions = getattr(project, "regions", {}) or {}
        rows += [*model_rows(model.get("name", "mobilesam")), ("segmentation", "MobileSAM (Zhang et al. 2023), "
                  + ("automatic regions" if model.get("mode", "auto") == "auto" else "clicks on objects"))]
        if model.get("mode", "auto") == "auto":
            rows += [("model_grid", model.get("grid", 16)),
                     ("region_categories", "; ".join(class_info.get("name", "") for class_info in model.get("categories") or [])),
                     ("images_with_classified_regions", len(regions)),
                     ("other_images", "regions classified by a random forest trained on the classified ones"
                      if regions else "all regions used")]
    else:
        rows.append(("segmentation", f"{seg.color_space} channel {seg.channel} "
                     + ("Otsu" if seg.auto else f"{seg.min_val}-{seg.max_val}")))
    refine = getattr(seg, "refine", []) or []
    rows.append(("refine_mask", "; ".join(
        operation["op"] + (f" {operation.get('size', 3)}px x{operation.get('iter', 1)}" if operation["op"] != "fill_holes" else "")
        for operation in refine) or "none"))
    return rows


def write_tables(folder: Path, tables: dict[str, list[dict]], meta: list[dict], excel: bool = True,
                 excel_name: str = "results.xlsx") -> list[Path]:
    """Un CSV por tabla, metadata.csv y (opcional) un Excel con una hoja por tabla."""
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for table, rows in tables.items():
        if rows:
            write_csv(folder / f"{table}.csv", rows)
            written.append(folder / f"{table}.csv")
    write_csv(folder / "metadata.csv", meta)
    written.append(folder / "metadata.csv")
    if excel and XLSX_OK:
        path = folder / excel_name
        _excel(path, {**{table_name: table_rows for table_name, table_rows in tables.items() if table_rows}, "metadata": meta})
        written.append(path)
    return written


def _excel(path: Path, tables: dict[str, list[dict]]):
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    head_fill = PatternFill("solid", fgColor="2166AC")
    head_font = Font(color="FFFFFF", bold=True, size=9)
    cell_font = Font(size=9)
    border = Border(*(Side(style="thin", color="CCCCCC"),) * 4)
    for table, rows in tables.items():
        sheet = workbook.create_sheet(table[:31])
        headers = list(dict.fromkeys(column_name for table_row in rows for column_name in table_row))
        for column_index, header in enumerate(headers, 1):
            cell = sheet.cell(row=1, column=column_index, value=header)
            cell.font, cell.fill, cell.border = head_font, head_fill, border
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row_index, row in enumerate(rows, 2):
            for column_index, header in enumerate(headers, 1):
                val = _plain(row.get(header, ""))
                cell = sheet.cell(row=row_index, column=column_index, value=val)
                cell.font, cell.border = cell_font, border
                cell.alignment = Alignment(horizontal="right" if isinstance(val, (int, float)) else "left")
        for column_index, header in enumerate(headers, 1):
            width = max([len(str(header))] + [len(str(_plain(table_row.get(header, "")))) for table_row in rows[:500]])
            sheet.column_dimensions[get_column_letter(column_index)].width = min(width + 2, 40 if table == "metadata" else 25)
        sheet.freeze_panes = "A2"
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def save_views(result, folder: Path, image_name: str, display: dict) -> int:
    """Imágenes de cada vista (con marcas y leyenda según `display`) en folder/<vista>/."""
    from fenotit.core.pipeline.views import decorate
    draw = decorate(result, display, Path(image_name).stem)
    saved = 0
    for view, sub in (result.extra.get("step_folders") or {}).items():
        img = result.step_images.get(view)
        if img is not None:
            save_image(folder / sub, image_name, draw(view, img))
            saved += 1
    return saved
