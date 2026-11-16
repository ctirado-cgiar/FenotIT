"""Áreas de análisis y zonas excluidas de una foto, en coordenadas 0-1 (sirven para
cualquier resolución). Cada forma: {"kind": "include"|"exclude", "type": "rect"|"polygon",
"points": [[x, y], ...]} (rect = dos esquinas). Las áreas de análisis se numeran 1, 2, 3… en
el orden en que se dibujaron; con "name" llevan ese nombre en los resultados."""
from __future__ import annotations

import cv2
import numpy as np


def polygon(shape: dict) -> list[tuple[float, float]]:
    pts = shape["points"]
    if shape["type"] == "rect":
        (x0, y0), (x1, y1) = pts
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [tuple(p) for p in pts]


def masks(shapes: list[dict] | None, w: int, h: int):
    """(máscara de inclusión o None si no hay áreas, [(puntos px, color)] excluidas)."""
    include, exclusions = None, []
    for s in shapes or []:
        pts = np.round(np.asarray(polygon(s), float) * [w, h]).astype(np.int32)
        if s["kind"] == "exclude":
            exclusions.append((pts, (255, 255, 255)))
        else:
            if include is None:
                include = np.zeros((h, w), np.uint8)
            cv2.fillPoly(include, [pts], 255)
    return include, exclusions


def analysis_areas(shapes: list[dict] | None) -> list[dict]:
    """Las áreas de análisis (sin las zonas excluidas), en el orden en que se dibujaron."""
    return [shape for shape in shapes or [] if shape["kind"] != "exclude"]


def area_names(shapes: list[dict] | None) -> list[str]:
    """Nombre de cada área de análisis: el que le puso el usuario o su número (1, 2, 3…)."""
    return [str(shape.get("name") or number) for number, shape in enumerate(analysis_areas(shapes), 1)]


def area_map(shapes: list[dict] | None, width: int, height: int) -> np.ndarray | None:
    """Número del área de análisis de cada píxel (0 = fuera de todas); None si no hay áreas.
    Donde dos áreas se cruzan queda la que se dibujó después."""
    areas = analysis_areas(shapes)
    if not areas:
        return None
    numbers = np.zeros((height, width), np.uint8)
    for number, shape in enumerate(areas, 1):
        corners = np.round(np.asarray(polygon(shape), float) * [width, height]).astype(np.int32)
        cv2.fillPoly(numbers, [corners], number)
    return numbers


def from_legacy(data) -> list[dict]:
    """Formato anterior ({"inclusion": [...], "exclusions": [...]}) a la lista de formas."""
    if isinstance(data, list):
        return data
    out = []
    for op in (data or {}).get("inclusion", []):
        kind = "exclude" if op.get("type") == "hole" else "include"
        out.append({"kind": kind, "type": "rect" if op["type"] == "rect" else "polygon",
                    "points": op["points"]})
    for op in (data or {}).get("exclusions", []):
        out.append({"kind": "exclude", "type": "polygon", "points": op["points"]})
    return out
