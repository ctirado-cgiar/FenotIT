"""Nombres de columnas para la pantalla. Los CSV y el Excel conservan los nombres en inglés
con la unidad (area_mm2); en la tabla y la barra de estado se ven traducidos y con la unidad
entre paréntesis: «Área (mm²)»."""
from __future__ import annotations

import re

from fenotit.i18n import t

_UNITS = {"px": "px", "um": "µm", "mm": "mm", "cm": "cm", "m": "m"}
_UNIT_RE = re.compile(r"^(?P<base>.+?)_(?P<unit>px|um|mm|cm|m)(?P<sq>2?)$")


def column(col: str, aliases: dict | None = None) -> str:
    """area_mm2 → Área (mm²); sano_pct → Sano (%); lesion_area_mm2 → Área Lesión (mm²); n_lesion → N.º Lesión;
    efd_a3 → EFD a3; object_id → Objeto. `aliases`: nombre ASCII de cada clase → su nombre."""
    if col.startswith("efd_"):
        return "EFD " + col[4:]
    unit = ""
    base = col
    m = _UNIT_RE.match(col)
    if m and m.group("base") not in ("mm_per",):
        base, unit = m.group("base"), _UNITS[m.group("unit")] + ("²" if m.group("sq") else "")
    elif col.endswith("_pct"):
        base, unit = col[:-4], "%"
    elif col == "pct":
        base, unit = "share", "%"
    aliases = aliases or {}
    if base in aliases:
        return f"{aliases[base]} ({unit})" if unit else aliases[base]
    if base.endswith("_area") and base[:-5] in aliases:
        name = f"{t('col.area')} {aliases[base[:-5]]}"
        return f"{name} ({unit})" if unit else name
    if base.startswith("n_") and base[2:] in aliases:
        return f"{t('col.n')} {aliases[base[2:]]}"
    name = t(f"col.{base}", "")
    if not name:
        name = base.replace("_", " ")
        name = name[:1].upper() + name[1:]
    return f"{name} ({unit})" if unit else name
