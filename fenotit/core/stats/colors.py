"""Puntos de color para graficar (2D/3D) a cuatro niveles: cada color KMeans de cada
objeto, el objeto (color medio + sus colores como torta), cada color de la imagen y
la imagen (color medio + torta). Posición en CIELab, RGB o HSV."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

LEVELS = ("object_colors", "objects", "image_colors", "images")
SPACES = {"Lab": ("L*", "a*", "b*"), "RGB": ("R", "G", "B"), "HSV": ("H", "S", "V")}


@dataclass
class ColorPoint:
    level: str
    path: str
    image_id: int
    name: str
    object_id: int | None
    rgb: tuple[float, float, float]                 # 0-1, color del punto
    pct: float                                      # % de área (tamaño del punto)
    sectors: list[tuple[tuple[float, float, float], float]] = field(default_factory=list)  # torta: (rgb, %)
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def label_id(self) -> str:
        return f"{self.image_id}-{self.object_id}" if self.object_id is not None else str(self.image_id)


def to_space(rgb: np.ndarray, space: str) -> np.ndarray:
    """rgb (n, 3) en 0-1 → coordenadas: Lab (L 0-100, a/b ±128), RGB 0-255, HSV (H 0-360, S/V 0-100)."""
    rgb = np.asarray(rgb, np.float32).reshape(-1, 1, 3)
    if space == "RGB":
        return rgb.reshape(-1, 3) * 255
    if space == "HSV":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        return hsv * np.array([1, 100, 100], np.float32)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def _mean_rgb(rows: list[dict]) -> tuple[float, float, float]:
    """Color medio ponderado por %, promediado en Lab (perceptual)."""
    rgb = np.array([[r["R"], r["G"], r["B"]] for r in rows], np.float32) / 255
    w = np.array([float(r.get("pct") or 0) for r in rows], np.float32)
    w = w / w.sum() if w.sum() > 0 else np.full(len(rows), 1 / len(rows), np.float32)
    lab = (to_space(rgb, "Lab") * w[:, None]).sum(axis=0)
    out = cv2.cvtColor(lab.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_Lab2RGB).reshape(3)
    return tuple(float(v) for v in np.clip(out, 0, 1))


def _rgb(r: dict) -> tuple[float, float, float]:
    return (r["R"] / 255, r["G"] / 255, r["B"] / 255)


def points(images: list[tuple[str, int, str, dict]], levels=LEVELS, space: str = "Lab") -> list[ColorPoint]:
    """images = [(ruta, Image_ID, nombre, tablas del resultado)] → puntos de los niveles pedidos."""
    out: list[ColorPoint] = []
    for path, image_id, name, tables in images:
        obj_rows = [r for r in tables.get("object_colors", []) if r.get("hex")]
        img_rows = [r for r in tables.get("image_colors", []) if r.get("hex")]
        by_obj: dict[int, list[dict]] = {}
        for r in obj_rows:
            by_obj.setdefault(int(r["object_id"]), []).append(r)
        if "object_colors" in levels:
            out += [ColorPoint("object_colors", path, image_id, name, int(r["object_id"]), _rgb(r),
                               float(r.get("pct") or 0)) for r in obj_rows]
        if "objects" in levels:
            for oid, rows in by_obj.items():
                out.append(ColorPoint("objects", path, image_id, name, oid, _mean_rgb(rows), 100.0,
                                      [(_rgb(r), float(r.get("pct") or 0)) for r in rows]))
        if "image_colors" in levels:
            out += [ColorPoint("image_colors", path, image_id, name, None, _rgb(r), float(r.get("pct") or 0))
                    for r in img_rows]
        if "images" in levels and img_rows:
            out.append(ColorPoint("images", path, image_id, name, None, _mean_rgb(img_rows), 100.0,
                                  [(_rgb(r), float(r.get("pct") or 0)) for r in img_rows]))
    if out:
        xyz = to_space(np.array([p.rgb for p in out]), space)
        for p, c in zip(out, xyz):
            p.xyz = tuple(float(v) for v in c)
    return out
