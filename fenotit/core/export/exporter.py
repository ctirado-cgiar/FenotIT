"""
Exportación de resultados: una tabla por nivel (objects, image, object_colors,
image_colors, object_shape...) y las imágenes de cada vista en su carpeta.

  RUTA/
  ├── mask/  count/  morphometry/  shape/  color/     imágenes, nombre original
  └── results/
      ├── objects.csv, image.csv, object_colors.csv...  (Image_ID + Image_name + columnas)
      └── results.xlsx                                  una hoja por tabla

Todo en inglés, CSV utf-8-sig. La exportación completa con metadatos es la etapa 6.
"""
import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fenotit import log

_log = log.get("export")

try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    XLSX_OK = True
except ImportError:
    XLSX_OK = False


def _save_img(folder: Path, name: str, img: np.ndarray):
    """Guarda imagen (rutas Unicode)."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        ext = Path(name).suffix.lower() or ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if ok:
            (folder / name).write_bytes(buf.tobytes())
    except Exception:
        _log.exception("Error guardando %s", name)


def _write_csv(path: Path, rows: list[dict], append: bool = False):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    exists = append and path.exists()
    with open(path, "a" if append else "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)


class Exporter:
    """Acumula los resultados de la sesión y los exporta cuando el usuario lo pide."""

    def __init__(self, output_root: str):
        self.output_root = Path(output_root)
        self.results_dir = self.output_root / "results"
        # {análisis: {nombre_imagen: {"tables": {...}}}} en orden de llegada
        self._data: dict[str, dict[str, Any]] = {}

    def save_result(self, analysis_name: str, image_name: str, result, save_step_images: bool = True,
                    decorate=None):
        """decorate(nombre_vista, imagen) -> imagen: p. ej. la leyenda, si está activada."""
        folders = result.extra.get("step_folders") or {}
        if save_step_images:
            for step, folder in folders.items():
                img = result.step_images.get(step)
                if img is not None:
                    _save_img(self.output_root / folder, image_name, decorate(step, img) if decorate else img)
        self._data.setdefault(analysis_name, {})[image_name] = {
            "tables": result.extra.get("tables") or {}}

    def append_to_csv(self, analysis_name: str, image_name: str, result):
        """CSV al día mientras corre un lote (una tabla por nivel)."""
        image_id = list(self._data.get(analysis_name, {})).index(image_name) + 1 \
            if image_name in self._data.get(analysis_name, {}) else None
        for table, rows in (result.extra.get("tables") or {}).items():
            _write_csv(self.results_dir / f"{table}.csv",
                       [{"Image_ID": image_id, "Image_name": image_name, **r} for r in rows], append=True)

    def _combined(self, names: list[str]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for name in names:
            for image_id, (image_name, d) in enumerate(self._data.get(name, {}).items(), 1):
                for table, rows in d["tables"].items():
                    out.setdefault(table, []).extend(
                        {"Image_ID": image_id, "Image_name": image_name, **r} for r in rows)
        return out

    def export_all(self, analyses_to_include: list[str] | None = None,
                   generate_excel: bool = True) -> dict[str, str]:
        """Escribe un CSV por tabla (y un Excel con una hoja por tabla). Devuelve las rutas."""
        names = analyses_to_include or list(self._data)
        tables = self._combined(names)
        generated = {}
        for table, rows in tables.items():
            path = self.results_dir / f"{table}.csv"
            _write_csv(path, rows)
            generated[table] = str(path)
        if generate_excel and tables:
            if XLSX_OK:
                path = self.results_dir / "results.xlsx"
                self._excel(path, tables)
                generated["excel"] = str(path)
            else:
                generated["excel_error"] = "openpyxl not installed"
        return generated

    @staticmethod
    def _excel(path: Path, tables: dict[str, list[dict]]):
        wb = openpyxl.Workbook()
        wb.remove(wb.active)
        header_fill = PatternFill("solid", fgColor="2166AC")
        header_font = Font(color="FFFFFF", bold=True, size=9)
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell_font = Font(size=9)
        thin_border = Border(*(Side(style="thin", color="CCCCCC"),) * 4)

        def _style_sheet(ws, rows_data: list[dict]):
            if not rows_data:
                return
            headers = list(dict.fromkeys(k for r in rows_data for k in r))
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
                    if isinstance(val, np.generic):
                        val = val.item()
                    if val is None:
                        val = ""
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


        for table, rows in tables.items():
            _style_sheet(wb.create_sheet(table[:31]), rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(path)
