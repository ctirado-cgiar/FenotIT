"""Tabla de datos del usuario (CSV o Excel) que relaciona cada foto con genotipo,
repetición, tratamiento... Sus columnas sirven para agrupar los análisis."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def read_table(path: str | Path) -> tuple[list[str], list[dict]]:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook
        sheet = load_workbook(path, read_only=True, data_only=True).active
        row_iter = sheet.iter_rows(values_only=True)
        cols = [str(column).strip() if column is not None else f"col{i + 1}" for i, column in enumerate(next(row_iter))]
        rows = [dict(zip(cols, row)) for row in row_iter if any(value not in (None, "") for value in row)]
        return cols, rows
    text = path.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    cols = [column.strip() for column in reader.fieldnames or []]
    rows = [{column.strip(): (value.strip() if isinstance(value, str) else value) for column, value in row.items() if column}
            for row in reader]
    return cols, [row for row in rows if any(value not in (None, "") for value in row.values())]


def key(value) -> str:
    """Normaliza un nombre para comparar: sin extensión, sin mayúsculas, 001 = 1 = 1.0."""
    text = str(value).strip()
    if Path(text).suffix.lower() in IMAGE_EXTS:
        text = Path(text).stem
    text = text.casefold()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


@dataclass
class Match:
    by_image: dict[str, dict] = field(default_factory=dict)   # nombre de imagen -> fila
    images_without_row: list[str] = field(default_factory=list)
    rows_without_image: list[str] = field(default_factory=list)
    duplicated_keys: list[str] = field(default_factory=list)


def match(image_names: list[str], rows: list[dict], key_column: str) -> Match:
    index, dup = {}, []
    for row in rows:
        photo_key = key(row.get(key_column, ""))
        if photo_key in index:
            dup.append(str(row.get(key_column)))
        index.setdefault(photo_key, row)
    result = Match(duplicated_keys=dup)
    used = set()
    for name in image_names:
        photo_key = key(name)
        if photo_key in index:
            result.by_image[name] = index[photo_key]
            used.add(photo_key)
        else:
            result.images_without_row.append(name)
    result.rows_without_image = [str(row.get(key_column)) for row in rows if key(row.get(key_column, "")) not in used]
    return result


def guess_key_column(columns: list[str], rows: list[dict], image_names: list[str]) -> str | None:
    """La columna que coincide con más nombres de imagen distintos."""
    names = {key(n_matches) for n_matches in image_names}
    best, best_n = None, 0
    for column in columns:
        n_matches = len({key(row.get(column, "")) for row in rows} & names)
        if n_matches > best_n:
            best, best_n = column, n_matches
    return best
