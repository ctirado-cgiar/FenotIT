"""Tema visual de la interfaz: colores, letras, tamaños y paletas, con nombre.

Los valores vienen de un archivo de tema (`fenotit/assets/themes/<nombre>.toml`, o la
carpeta `themes/` de los datos del usuario). El tema activo se elige en Preferencias y se
guarda en la configuración (`theme`); la variable de entorno FENOTIT_THEME lo cambia solo
para esa ejecución. Lo que el tema no define se toma de `classic.toml`.
Ningún archivo de la interfaz escribe un color ni un tipo de letra directamente.

COLORS        colores de la interfaz (#RRGGBB)
FONTS         tipos de letra: (familia, tamaño[, estilo])
SIZES         tamaños en px (filas de tabla, márgenes de botones…)
SPACE         escala de márgenes entre controles: xs, sm, md, lg, xl, xxl
BUTTONS       estilo de cada tipo de botón (primary, secondary, quiet, link)
CLASS_COLORS  colores que se asignan a las clases y categorías nuevas
CHART_COLORS  colores de las series en los gráficos
LOGO_TINTS    tonos del mosaico del logo (de oscuro a claro)
LOGO_EDGES    bordes de los píxeles decorativos (de oscuro a claro)
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

from fenotit import settings
from fenotit.log import get as _get_log

DEFAULT = "classic"
FOLDERS = (Path(__file__).resolve().parents[2] / "assets" / "themes", settings.data_dir() / "themes")

_log = _get_log("gui.style.theme")


def available() -> dict[str, dict]:
    """Temas instalados: nombre del archivo → sección [info]."""
    found = {}
    for folder in FOLDERS:
        for f in sorted(folder.glob("*.toml")):
            try:
                found[f.stem] = _read(f).get("info", {})
            except (OSError, tomllib.TOMLDecodeError):
                _log.warning("Tema ilegible: %s", f)
    return found


def _read(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _find(name: str) -> Path | None:
    for folder in reversed(FOLDERS):
        if (folder / f"{name}.toml").is_file():
            return folder / f"{name}.toml"
    return None


def load(name: str | None = None) -> dict:
    """Tema `name` (o el elegido en la configuración) completado con el clásico."""
    data = _read(_find(DEFAULT))
    name = name or os.environ.get("FENOTIT_THEME") or settings.get("theme") or DEFAULT
    path = _find(name)
    if name != DEFAULT and path:
        try:
            for section, values in _read(path).items():
                data.setdefault(section, {}).update(values)
        except (OSError, tomllib.TOMLDecodeError):
            _log.exception("Tema %s", name)
    elif name != DEFAULT:
        _log.warning("Tema no encontrado: %s", name)
    return data


def _font(spec: str, families: dict) -> tuple:
    family, size, *style = spec.split()
    return (families[family], int(size), *style)


_THEME = load()
_FAMILIES = {k: v[0] if isinstance(v, list) else v for k, v in _THEME["families"].items()}

COLORS: dict[str, str] = dict(_THEME["colors"])
FONTS: dict[str, tuple] = {k: _font(v, _FAMILIES) for k, v in _THEME["fonts"].items()}
SIZES: dict[str, int] = dict(_THEME["sizes"])
SPACE: dict[str, int] = dict(_THEME["spacing"])
BUTTONS: dict[str, dict] = {k: dict(v) for k, v in _THEME["buttons"].items()}
CLASS_COLORS = tuple(_THEME["palettes"]["classes"])
CHART_COLORS = tuple(_THEME["palettes"]["charts"])
LOGO_TINTS = tuple(_THEME["palettes"]["logo_tints"])
LOGO_EDGES = tuple(_THEME["palettes"]["logo_edges"])


def use_installed_fonts(root) -> None:
    """Cambia cada familia por la primera de su lista que está instalada en el equipo.
    Se llama una vez, con la ventana principal ya creada."""
    import tkinter.font as tkfont
    installed = {f.lower() for f in tkfont.families(root)}
    for key, names in _THEME["families"].items():
        names = names if isinstance(names, list) else [names]
        _FAMILIES[key] = next((n for n in names if n.lower() in installed), names[0])
    FONTS.update({k: _font(v, _FAMILIES) for k, v in _THEME["fonts"].items()})


def font(size: int, weight: str | None = None) -> tuple:
    """Tipo de letra de la interfaz en un tamaño calculado (p. ej. rótulos que cambian
    de tamaño con el zoom)."""
    return (_FAMILIES["sans"], int(size), weight) if weight else (_FAMILIES["sans"], int(size))
