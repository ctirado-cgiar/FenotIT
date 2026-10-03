"""Unidades de longitud. Internamente la escala se guarda en mm/px; los resultados se
reportan en la unidad con la que el usuario estableció la escala (µm, mm, cm o m).
En nombres de columna se usa la versión ASCII (`um`), en pantalla la de verdad (`µm`)."""
from __future__ import annotations

UNITS = ["µm", "mm", "cm", "m"]
TO_MM = {"µm": 0.001, "um": 0.001, "mm": 1.0, "cm": 10.0, "m": 1000.0}


def ascii_name(unit: str) -> str:
    return "um" if unit in ("µm", "um") else unit


def display(unit: str) -> str:
    """Para mostrar: um → µm, um2 → µm², mm2 → mm²."""
    u = "µm" + unit[2:] if unit.startswith("um") else unit
    return u.replace("2", "²")


def per_px(mm_per_px: float | None, unit: str) -> float | None:
    """Escala en la unidad pedida (unidad/px)."""
    return mm_per_px / TO_MM.get(unit, 1.0) if mm_per_px else None


def mean_sd(mean: float, sd: float) -> str:
    """Media ± DE con decimales según el tamaño (5.7 ± 0.4; 0.57 ± 0.04; 16193 ± 1958)."""
    import math
    m = abs(mean)
    d = 0 if m >= 1000 else 1 if m >= 1 or m == 0 else min(8, 1 - math.floor(math.log10(m)))
    return f"{mean:.{d}f} ± {sd:.{d}f}"
