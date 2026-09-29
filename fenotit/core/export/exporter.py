"""
export/exporter.py  v2.0
Exportación simplificada — carpetas por tipo, sin JSONs, sin carpetas por imagen.

Estructura de salida:
  RUTA/
  ├── Binarizadas/              imágenes binarias, nombre original
  ├── CanalUmbralizado/         canal en escala de grises, nombre original
  ├── Segmentadas/              máscara segmentada, nombre original
  ├── Colorimetria/             resultado KMeans con leyenda, nombre original
  ├── resultados/
  │   ├── morfometria.csv       una fila por objeto
  │   ├── morfometria_resumen.csv  una fila por imagen (medias)
  │   ├── color.csv             una fila por imagen
  │   ├── conteo.csv            una fila por imagen
  │   └── resultados_completos.xlsx  Excel con pestañas
  └── (sin JSONs, sin carpetas por imagen)

Codificación: UTF-8 con BOM para compatibilidad con Excel en español.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import openpyxl
    from openpyxl.styles import (Font, PatternFill, Alignment,
                                  Border, Side)
    from openpyxl.utils import get_column_letter
    XLSX_OK = True
except ImportError:
    XLSX_OK = False


# ── Carpetas de imágenes por tipo de paso ─────────────────────────────────────

STEP_FOLDERS = {
    "máscara binaria":        "Binarizadas",
    "canal seleccionado":     "CanalUmbralizado",
    "canal umbralizado":      "CanalUmbralizado",
    "máscara objetos":        "Binarizadas",
    "objetos segmentados":    "Segmentadas",
    "colores KMeans":         "Colorimetria",
    "resultado con leyenda":  "Colorimetria",
    "contornos detectados":   "Segmentadas",
    "semillas detectadas":    "Conteo",
}

# Nombre de imagen que se guarda por cada paso
STEP_SAVE = {
    "Morfometría":          ["máscara binaria", "contornos detectados"],
    "Color KMeans":         ["objetos segmentados", "resultado con leyenda"],
    "Contador de semillas": ["semillas detectadas"],
}


def _stem(image_name: str) -> str:
    """Nombre sin extensión y sin prefijo in_row="""
    s = Path(image_name).stem
    if s.lower().startswith("in_row="):
        s = s[7:]
    return s


def _save_img(folder: str, name: str, img: np.ndarray):
    """Guarda imagen Unicode-safe."""
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        ext = Path(name).suffix.lower() or ".jpg"
        out = Path(folder) / name
        ok, buf = cv2.imencode(ext, img)
        if ok:
            out.write_bytes(buf.tobytes())
    except Exception as e:
        print(f"[exporter] Error guardando {name}: {e}")


def _write_csv(path: str, rows: list[dict], encoding="utf-8-sig"):
    """
    Escribe CSV con encoding utf-8-sig (BOM) para compatibilidad con Excel.
    Sin caracteres problemáticos en headers.
    """
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: str, rows: list[dict], encoding="utf-8-sig"):
    """Agrega filas a CSV existente o lo crea."""
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    exists = Path(path).exists()
    with open(path, "a", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


# ── Formateadores por tipo de análisis ────────────────────────────────────────

def _format_morphometry_row(image_name: str,
                             measurement: dict) -> dict:
    """
    Formatea una fila de morfometría al formato del pipeline.
    Columnas: Imagen, Total_seeds, seed, W, L, P, A, AR, Circ, ...
    """
    row = {
        "Imagen":       image_name,
        "Total_seeds":  measurement.get("_total", ""),
        "seed":         measurement.get("index", ""),
    }
    # Métricas principales — nombres limpios sin unidades en el header
    metric_map = {
        "W":           [f"ancho (mm)", f"ancho (px)", "ancho"],
        "L":           [f"largo (mm)", f"largo (px)", "largo"],
        "P":           [f"perímetro (mm)", f"perímetro (px)", "perímetro"],
        "A":           [f"área (mm²)", f"área (px²)", "área"],
        "AR":          ["AR"],
        "Circ":        ["circularidad"],
        "Solid":       ["solidez"],
        "Centroid_X":  ["centroid_x"],
        "Centroid_Y":  ["centroid_y"],
        "Radius_min":  [f"radio_min (mm)", f"radio_min (px)", "radio_min"],
        "Radius_mean": [f"radio_mean (mm)", f"radio_mean (px)", "radio_mean"],
        "Radius_max":  [f"radio_max (mm)", f"radio_max (px)", "radio_max"],
        "Radius_ratio":["radio_ratio"],
        "Diam_min":    [f"diam_min (mm)", f"diam_min (px)", "diam_min"],
        "Diam_mean":   [f"diam_mean (mm)", f"diam_mean (px)", "diam_mean"],
        "Diam_max":    [f"diam_max (mm)", f"diam_max (px)", "diam_max"],
        "Major_axis":  [f"eje_mayor (mm)", f"eje_mayor (px)", "eje_mayor"],
        "Minor_axis":  [f"eje_menor (mm)", f"eje_menor (px)", "eje_menor"],
        "Eccentricity":["excentricidad"],
        "Form_factor": ["Form_factor"],
        "Narrow_factor":["Narrow_factor"],
        "Rectangularity":["rectangularidad"],
        "Elongation":  ["elongacion"],
        "Convexity":   ["convexidad"],
    }
    for col, possible_keys in metric_map.items():
        val = ""
        for k in possible_keys:
            if k in measurement:
                val = measurement[k]
                break
        row[col] = val
    return row


def _format_color_row(image_name: str,
                       measurements: list[dict]) -> dict:
    """
    Formato: Image, RGB1, Hex1, Luminosidad1, 1%, RGB2, Hex2, ...
    """
    row = {"Image": image_name}
    for i, m in enumerate(measurements, 1):
        row[f"RGB{i}"]         = m.get("RGB", "")
        row[f"Hex{i}"]         = m.get("HEX", "")
        row[f"Luminosidad{i}"] = m.get("luminancia", "")
        row[f"{i}%"]           = m.get("% area", m.get("% área", ""))
        if m.get("L* (CIELab)") is not None:
            row[f"L*{i}"]  = m.get("L* (CIELab)", "")
            row[f"a*{i}"]  = m.get("a* (CIELab)", "")
            row[f"b*{i}"]  = m.get("b* (CIELab)", "")
    return row


def _format_count_row(image_name: str,
                       stats: dict) -> dict:
    return {
        "imagen":      image_name,
        "ConteoTotal": stats.get("total semillas",
                       stats.get("total", "")),
    }


# ── Clase principal ───────────────────────────────────────────────────────────

class Exporter:
    """
    Gestiona la exportación de resultados de FenotIT.
    Acumula resultados en sesión y exporta cuando el usuario lo pide.
    """

    def __init__(self, output_root: str):
        self.output_root = Path(output_root)
        self.results_dir = self.output_root / "resultados"

        # Acumuladores en memoria por análisis
        # {analysis_name: {image_stem: [measurements]}}
        self._data: dict[str, dict[str, Any]] = {}

    # ── Guardar resultado (se llama en cada _on_result) ───────────────────────

    def save_result(self, analysis_name: str,
                    image_name: str,
                    result,
                    save_step_images: bool = True):
        """
        Guarda un resultado:
          - Imágenes de pasos en carpetas por tipo
          - Acumula datos para CSV/Excel posterior
        """
        stem = _stem(image_name)

        # Guardar imágenes de pasos
        if save_step_images and result.step_images:
            steps_to_save = STEP_SAVE.get(analysis_name,
                                          list(result.step_images.keys()))
            for step_name in steps_to_save:
                if step_name in result.step_images:
                    folder_name = STEP_FOLDERS.get(
                        step_name, "Procesadas")
                    folder = self.output_root / folder_name
                    _save_img(str(folder), image_name,
                              result.step_images[step_name])

        # Acumular datos
        if analysis_name not in self._data:
            self._data[analysis_name] = {}
        self._data[analysis_name][stem] = {
            "image_name":   image_name,
            "measurements": result.measurements,
            "stats":        result.stats,
        }

    # ── Exportar CSV en tiempo real (append) ──────────────────────────────────

    def append_to_csv(self, analysis_name: str,
                      image_name: str, result):
        """
        Agrega datos al CSV correspondiente inmediatamente.
        Se llama desde _cache_result para tener el CSV actualizado
        mientras corre el lote.
        """
        stem = _stem(image_name)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if analysis_name == "Morfometría":
            rows = []
            n = len(result.measurements)
            for m in result.measurements:
                m2 = dict(m)
                m2["_total"] = n
                rows.append(_format_morphometry_row(stem, m2))
            if rows:
                _append_csv(
                    str(self.results_dir / "morfometria.csv"),
                    rows)

        elif analysis_name == "Color KMeans":
            row = _format_color_row(stem, result.measurements)
            _append_csv(
                str(self.results_dir / "color.csv"),
                [row])

        elif analysis_name == "Contador de semillas":
            row = _format_count_row(stem, result.stats)
            _append_csv(
                str(self.results_dir / "conteo.csv"),
                [row])

    # ── Exportar todo (llamado desde diálogo) ─────────────────────────────────

    def export_all(self,
                   analyses_to_include: list[str] | None = None,
                   generate_excel: bool = True) -> dict[str, str]:
        """
        Genera CSVs + Excel con todos los resultados acumulados.
        analyses_to_include: None = todos, o lista de nombres específicos.
        Retorna dict con rutas generadas.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)
        generated = {}

        names = analyses_to_include or list(self._data.keys())

        for name in names:
            if name not in self._data:
                continue
            data = self._data[name]

            if name == "Morfometría":
                # CSV completo (una fila por objeto)
                rows_full = []
                rows_sum  = []
                for stem, d in data.items():
                    n = len(d["measurements"])
                    for m in d["measurements"]:
                        m2 = dict(m)
                        m2["_total"] = n
                        rows_full.append(
                            _format_morphometry_row(stem, m2))

                    # Resumen: medias por imagen
                    rows_sum.append(
                        self._morph_summary_row(stem, d))

                csv_full = str(
                    self.results_dir / "morfometria.csv")
                csv_sum  = str(
                    self.results_dir / "morfometria_resumen.csv")
                _write_csv(csv_full, rows_full)
                _write_csv(csv_sum,  rows_sum)
                generated["morfometria"]         = csv_full
                generated["morfometria_resumen"] = csv_sum

            elif name == "Color KMeans":
                rows = [_format_color_row(stem, d["measurements"])
                        for stem, d in data.items()]
                csv_path = str(self.results_dir / "color.csv")
                _write_csv(csv_path, rows)
                generated["color"] = csv_path

            elif name == "Contador de semillas":
                rows = [_format_count_row(stem, d["stats"])
                        for stem, d in data.items()]
                csv_path = str(self.results_dir / "conteo.csv")
                _write_csv(csv_path, rows)
                generated["conteo"] = csv_path

        # Excel con pestañas
        if generate_excel and XLSX_OK:
            xlsx_path = str(
                self.results_dir / "resultados_completos.xlsx")
            self._generate_excel(xlsx_path, names)
            generated["excel"] = xlsx_path
        elif generate_excel and not XLSX_OK:
            generated["excel_error"] = \
                "openpyxl no instalado — pip install openpyxl"

        return generated

    # ── Generador de Excel ────────────────────────────────────────────────────

    def _generate_excel(self, path: str, names: list[str]):
        """Genera Excel con una pestaña por análisis + resumen general."""
        wb = openpyxl.Workbook()
        wb.remove(wb.active)   # quitar hoja vacía por defecto

        header_fill = PatternFill("solid", fgColor="2166AC")
        header_font = Font(color="FFFFFF", bold=True, size=9)
        header_align = Alignment(horizontal="center",
                                  vertical="center", wrap_text=True)
        cell_font  = Font(size=9)
        thin_border = Border(
            left=Side(style="thin", color="CCCCCC"),
            right=Side(style="thin", color="CCCCCC"),
            top=Side(style="thin", color="CCCCCC"),
            bottom=Side(style="thin", color="CCCCCC"))

        def _style_sheet(ws, rows_data: list[dict]):
            if not rows_data:
                return
            headers = list(rows_data[0].keys())
            # Headers
            for col_idx, h in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col_idx, value=h)
                cell.font   = header_font
                cell.fill   = header_fill
                cell.alignment = header_align
                cell.border = thin_border
            # Datos
            for row_idx, row in enumerate(rows_data, 2):
                for col_idx, h in enumerate(headers, 1):
                    val = row.get(h, "")
                    # Convertir a float si es posible
                    if isinstance(val, str):
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            pass
                    cell = ws.cell(row=row_idx,
                                   column=col_idx, value=val)
                    cell.font   = cell_font
                    cell.border = thin_border
                    cell.alignment = Alignment(horizontal="right"
                        if isinstance(val, (int, float)) else "left")
            # Ajustar ancho de columnas
            for col_idx, h in enumerate(headers, 1):
                col_letter = get_column_letter(col_idx)
                max_len = max(len(str(h)),
                    max((len(str(r.get(h,""))) for r in rows_data),
                        default=0))
                ws.column_dimensions[col_letter].width = \
                    min(max_len + 2, 25)
            # Freeze header row
            ws.freeze_panes = "A2"

        summary_rows = []   # para pestaña resumen general

        for name in names:
            if name not in self._data:
                continue
            data = self._data[name]

            if name == "Morfometría":
                # Pestaña completa
                ws1 = wb.create_sheet("Morfometría")
                rows_full = []
                rows_sum  = []
                for stem, d in data.items():
                    n = len(d["measurements"])
                    for m in d["measurements"]:
                        m2 = dict(m)
                        m2["_total"] = n
                        rows_full.append(
                            _format_morphometry_row(stem, m2))
                    rows_sum.append(
                        self._morph_summary_row(stem, d))
                _style_sheet(ws1, rows_full)

                # Pestaña resumen
                ws2 = wb.create_sheet("Morfometría_resumen")
                _style_sheet(ws2, rows_sum)
                summary_rows.extend(rows_sum)

            elif name == "Color KMeans":
                ws = wb.create_sheet("Color")
                rows = [_format_color_row(stem, d["measurements"])
                        for stem, d in data.items()]
                _style_sheet(ws, rows)
                # Agregar al resumen general
                for stem, d in data.items():
                    summary_rows.append(
                        {"Imagen": stem,
                         "color_dominante": d["stats"].get(
                             "color dominante", "")})

            elif name == "Contador de semillas":
                ws = wb.create_sheet("Conteo")
                rows = [_format_count_row(stem, d["stats"])
                        for stem, d in data.items()]
                _style_sheet(ws, rows)

        # Pestaña resumen general — unión por imagen
        if summary_rows:
            ws_sum = wb.create_sheet("Resumen_general", 0)
            merged = self._merge_summary(summary_rows)
            _style_sheet(ws_sum, merged)

        wb.save(path)

    def _morph_summary_row(self, stem: str,
                            data: dict) -> dict:
        """Genera fila de resumen (medias) para morfometría."""
        measurements = data["measurements"]
        if not measurements:
            return {"Imagen": stem}

        row = {"Imagen": stem,
               "Total_seeds": len(measurements)}

        # Calcular media de todas las columnas numéricas
        numeric_keys = [k for k, v in measurements[0].items()
                        if isinstance(v, (int, float))
                        and k not in ("index", "_total",
                                      "centroid_x", "centroid_y")]
        for key in numeric_keys:
            vals = [m[key] for m in measurements
                    if isinstance(m.get(key), (int, float))]
            if vals:
                # Mapear nombre con sufijo _mean
                clean = key.split(" (")[0]   # quitar unidad
                row[f"{clean}_mean"] = round(
                    sum(vals) / len(vals), 4)

        return row

    def _merge_summary(self,
                        rows: list[dict]) -> list[dict]:
        """Une filas del resumen general por imagen."""
        merged: dict[str, dict] = {}
        for row in rows:
            img = row.get("Imagen", "")
            if img not in merged:
                merged[img] = {}
            merged[img].update(row)
        result = list(merged.values())
        # Asegurar que Imagen es la primera columna
        for r in result:
            r.pop("Imagen", None)
        return [{"Imagen": k, **v} for k, v in merged.items()]


# ── Exportación rápida de un solo análisis ────────────────────────────────────

def quick_export_csv(analysis_name: str,
                     image_name: str,
                     result,
                     output_root: str):
    """
    Exporta directamente a CSV sin acumular en memoria.
    Útil para exportación en tiempo real durante el lote.
    """
    exp = Exporter(output_root)
    exp.append_to_csv(analysis_name, image_name, result)
