"""Constantes y funciones pequeñas que usan varias partes de la ventana principal."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

import cv2
import numpy as np

from fenotit.core.analysis.registry import ANALYSES
from fenotit.gui.style.theme import COLORS, SPACE
from fenotit.i18n import t


CHANNEL_NAMES = {
    "BGR":   ["B", "G", "R"],
    "HSV":   ["H", "S", "V"],
    "LAB":   ["L", "A", "B"],
    "YCrCb": ["Y", "Cr", "Cb"],
    "HLS":   ["H", "L", "S"],
    "XYZ":   ["X", "Y", "Z"],
    "YUV":   ["Y", "U", "V"],
    "LUV":   ["L", "U", "V"],
}


CV2_CODES = {
    "HSV":   cv2.COLOR_BGR2HSV,
    "LAB":   cv2.COLOR_BGR2LAB,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
    "LUV":   cv2.COLOR_BGR2LUV,
    "HLS":   cv2.COLOR_BGR2HLS,
    "XYZ":   cv2.COLOR_BGR2XYZ,
    "YUV":   cv2.COLOR_BGR2YUV,
}


def _cell(v) -> str:
    """Texto de una celda: decimales recortados para que la tabla se lea."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e-3 and v != 0 else f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def _analysis_key(name: str) -> str:
    return ANALYSES[name].func.__module__.rsplit(".", 1)[-1]


def _row_objects(row: dict) -> list[int]:
    """Objetos de una fila de tabla: object_id, o los dos de un par (Distancias)."""
    ids = [row.get(k) for k in ("object_id", "object_a", "object_b")]
    return [int(i) for i in ids if isinstance(i, (int, np.integer))]


def _on_base(result, view) -> bool:
    """¿La vista se dibuja como foto + marcas (contornos, puntos)? Las de Color o Máscara
    tienen su propia imagen (y, a lo sumo, ★ encima)."""
    ov = (result.extra.get("overlays") or {}).get(view) if result is not None else None
    return ov is not None and not ov.get("on_view")


def _analysis_label(name: str) -> str:
    return t(f"analysis.{_analysis_key(name)}.name", name)


def _analysis_name(key: str | None) -> str | None:
    return next((n for n in ANALYSES if _analysis_key(n) == key), None)


def _assets() -> Path:
    return Path(__file__).parents[2] / "assets"


def _peek_tab(host, image, command, side: str):
    """Pestañita que se asoma en el borde cuando un panel está oculto (como la de una
    carpeta): clic = mostrar el panel. side = left | right | bottom."""
    tab = tk.Label(host, image=image, bg=COLORS["bg_panel"], cursor="hand2", padx=SPACE["sm"], pady=SPACE["sm"],
                   highlightthickness=1, highlightbackground=COLORS["border"])
    tab.bind("<Button-1>", lambda e: command())
    tab.bind("<Enter>", lambda e: tab.config(bg=COLORS["accent_light"]))
    tab.bind("<Leave>", lambda e: tab.config(bg=COLORS["bg_panel"]))
    place = {"left": dict(x=0, y=24, anchor="nw"),
             "right": dict(relx=1.0, x=0, y=24, anchor="ne"),
             "bottom": dict(relx=0.5, rely=1.0, y=0, anchor="s")}[side]
    tab.show = lambda: (tab.place(**place), tab.lift())
    return tab


_SUMMARY_SKIP = ("image_width_px", "image_height_px")


def _duration(seconds: float) -> str:
    """12.3 s, 2 min 05 s"""
    if seconds < 60:
        return f"{seconds:.1f} s" if seconds < 10 else f"{seconds:.0f} s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s:02d} s"


def _summary(stats: dict, limit: int = 6, aliases: dict | None = None) -> str:
    """Resumen de la fila de la imagen para la barra de estado. Sirve para cualquier
    análisis (conteos, distancias, color, modelos…): los valores que el análisis puso, sin
    vacíos ni la resolución, con pocos decimales."""
    aliases = aliases or {}
    hidden = {k for k in aliases if k.startswith("\0")}          # fondo de la segmentación por ejemplos
    items = [(k, v) for k, v in stats.items() if v not in (None, "") and k not in _SUMMARY_SKIP
             and not k.endswith("_per_px") and not any(k.startswith(h[1:] + "_") for h in hidden)
             and not (v == 0 and k.startswith("n_") and k[2:] in aliases)]     # clases sin objetos
    from fenotit.gui.widgets.labels import column
    text = ", ".join(f"{column(k, aliases)}: {_cell(v)}" for k, v in items[:limit])
    return text + (", …" if len(items) > limit else "")
