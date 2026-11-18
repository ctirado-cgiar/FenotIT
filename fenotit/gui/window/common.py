"""Constantes y funciones pequeñas que usan varias partes de la ventana principal."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

import cv2
import numpy as np

# ANALYSES tiene los análisis registrados (Objetos, Detección…) por clave.
from fenotit.core.analysis.registry import ANALYSES
# Valores del tema (fenotit/assets/themes): COLORS (colores); SPACE (márgenes).
from fenotit.gui.style.theme import COLORS, SPACE
# t devuelve cada texto de la ventana en el idioma elegido (fenotit/lang).
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


def _cell(value) -> str:
    """Texto de una celda: decimales recortados para que la tabla se lea."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) < 1e-3 and value != 0 else f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _row_objects(row: dict) -> list[int]:
    """Objeto de una fila de tabla (object_id), o ninguno."""
    object_id = row.get("object_id")
    return [int(object_id)] if isinstance(object_id, (int, np.integer)) else []


def _on_base(result, view) -> bool:
    """¿La vista se dibuja como foto + marcas (contornos, puntos)? Las de Color o Máscara
    tienen su propia imagen (y, a lo sumo, ★ encima)."""
    marks = (result.extra.get("overlays") or {}).get(view) if result is not None else None
    return marks is not None and not marks.get("on_view")


def _analysis_label(key: str) -> str:
    """Nombre del análisis en el idioma de la interfaz."""
    return t(f"analysis.{key}.name", key)


def _analysis_name(key: str | None) -> str | None:
    """La clave si corresponde a un análisis registrado (un proyecto guardado puede traer otra)."""
    return key if key in ANALYSES else None


def _assets() -> Path:
    return Path(__file__).parents[2] / "assets"


def _peek_tab(host, image, command, side: str):
    """Pestañita que se asoma en el borde cuando un panel está oculto (como la de una
    carpeta): clic = mostrar el panel. side = left | right | bottom."""
    tab = tk.Label(host, image=image, bg=COLORS["bg_panel"], cursor="hand2", padx=SPACE["sm"], pady=SPACE["sm"],
                   highlightthickness=1, highlightbackground=COLORS["border"])
    tab.bind("<Button-1>", lambda event: command())
    tab.bind("<Enter>", lambda event: tab.config(bg=COLORS["accent_light"]))
    tab.bind("<Leave>", lambda event: tab.config(bg=COLORS["bg_panel"]))
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
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes} min {secs:02d} s"


def _summary(stats: dict, limit: int = 6, aliases: dict | None = None) -> str:
    """Resumen de la fila de la imagen para la barra de estado. Sirve para cualquier
    análisis (conteos, color, modelos…): los valores que el análisis puso, sin
    vacíos ni la resolución, con pocos decimales."""
    aliases = aliases or {}
    hidden = {key for key in aliases if key.startswith("\0")}          # fondo de la segmentación por ejemplos
    items = [(key, value) for key, value in stats.items() if value not in (None, "") and key not in _SUMMARY_SKIP
             and not key.endswith("_per_px") and not any(key.startswith(hidden_key[1:] + "_") for hidden_key in hidden)
             and not (value == 0 and key.startswith("n_") and key[2:] in aliases)]     # clases sin objetos
    from fenotit.gui.widgets.labels import column
    text = ", ".join(f"{column(key, aliases)}: {_cell(value)}" for key, value in items[:limit])
    return text + (", …" if len(items) > limit else "")
