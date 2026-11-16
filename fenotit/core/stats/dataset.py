"""Unir resultados de varias imágenes y resumir por grupos."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def combine(results: list[tuple[str, dict[str, list[dict]]]], table: str = "objects",
            meta: dict[str, dict] | None = None, key_column: str | None = None) -> list[dict]:
    """[(nombre_imagen, ctx.tables), ...] -> filas de 'table' con Image_ID (1..n), Image_name
    y, si hay tabla del usuario (meta: nombre -> fila), sus columnas (genotipo, rep...)."""
    rows = []
    for image_id, (name, tables) in enumerate(results, 1):
        extra = {column: value for column, value in (meta or {}).get(name, {}).items() if column != key_column}
        for row in tables.get(table, []):
            rows.append({"Image_ID": image_id, "Image_name": name, **extra, **row})
    return rows


def groups(rows: list[dict], by: str | tuple[str, ...] | None) -> dict:
    """Agrupa filas. by=None -> un solo grupo 'all'."""
    if not by:
        return {"all": list(rows)}
    keys = (by,) if isinstance(by, str) else tuple(by)
    out = defaultdict(list)
    for row in rows:
        out[tuple(row.get(key) for key in keys) if len(keys) > 1 else row.get(keys[0])].append(row)
    return dict(out)


def summarize(rows: list[dict], by=None, columns: list[str] | None = None) -> list[dict]:
    """n, media y desviación estándar de columnas numéricas por grupo."""
    skip = {"object_id", "Image_ID", "touching", "centroid_x_px", "centroid_y_px"}
    out = []
    for key, grp in groups(rows, by).items():
        cols = columns or [name for name, value in grp[0].items()
                           if name not in skip and isinstance(value, (int, float, np.number)) and not isinstance(value, bool)]
        row = {"group": key, "n": len(grp)}
        for column in cols:
            vals = np.array([row[column] for row in grp if row.get(column) is not None], float)
            row[f"{column}_mean"] = round(float(vals.mean()), 4) if len(vals) else None
            row[f"{column}_sd"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else None
        out.append(row)
    return out
