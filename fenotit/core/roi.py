"""Áreas de análisis y zonas excluidas de una foto, en coordenadas 0-1 (sirven para
cualquier resolución). Cada forma: {"kind": "include"|"exclude", "type": "rect"|"polygon",
"points": [[x, y], ...]} (rect = dos esquinas)."""
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
