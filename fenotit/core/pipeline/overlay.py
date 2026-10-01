"""Marcas sobre los resultados (contornos, puntos, números) guardadas como datos.

Se dibujan al mostrar, a tamaño fijo de pantalla (se ven igual en una foto de 50 MP
que en una de 300 px, con cualquier zoom), y al exportar, a escala de la imagen.
Los colores salen de un estilo que el usuario elige; "auto" busca el color que más
contrasta con la foto. Toda línea lleva un halo oscuro para no camuflarse.
"""
from __future__ import annotations

import cv2
import numpy as np

PALETTE = {                      # BGR
    "green": (60, 200, 0), "magenta": (255, 0, 255), "yellow": (0, 230, 255),
    "cyan": (255, 230, 0), "red": (40, 40, 230), "orange": (0, 150, 255),
    "blue": (230, 120, 0), "white": (255, 255, 255), "black": (20, 20, 20),
}
ROLES = ("mask", "outline", "dot")
DEFAULT_STYLE = {"mask": "green", "outline": "auto", "dot": "magenta", "mask_alpha": 0.45}
_EXCLUDED = (150, 150, 150)
_AUTO_CANDIDATES = ("green", "magenta", "yellow", "cyan", "red", "orange", "blue")


def empty() -> dict:
    return {"outlines": [], "dots": [], "labels": [], "size": 0}


def typical_size(labels: np.ndarray) -> float:
    """Diámetro típico de los objetos (px de imagen): decide si caben los números."""
    areas = np.bincount(labels.ravel())[1:]
    areas = areas[areas > 0]
    return float(np.sqrt(np.median(areas))) if len(areas) else 0.0


def contrast_color(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, int, int]:
    """Color de la paleta más distinto (en Lab) del fondo y de los objetos."""
    small = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    refs = [np.median(lab, axis=0)]
    if mask is not None and mask.any():
        m = cv2.resize(mask.astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST).reshape(-1) > 0
        if m.any():
            refs += [np.median(lab[m], axis=0)]
        if (~m).any():
            refs += [np.median(lab[~m], axis=0)]
    best, best_d = _AUTO_CANDIDATES[0], -1.0
    for name in _AUTO_CANDIDATES:
        c = cv2.cvtColor(np.uint8([[PALETTE[name]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        d = min(np.linalg.norm(c - r) for r in refs)
        if d > best_d:
            best, best_d = name, d
    return PALETTE[best]


def resolve(style: dict | None, image: np.ndarray | None = None, mask=None, auto=None) -> dict:
    """Colores BGR por rol; "auto" se resuelve con la imagen (o usa `auto` ya calculado)."""
    s = {**DEFAULT_STYLE, **(style or {})}
    out = {"mask_alpha": float(s["mask_alpha"])}
    for role in ROLES:
        name = s[role]
        if name == "auto":
            if auto is None:
                auto = contrast_color(image, mask) if image is not None else PALETTE["green"]
            out[role] = tuple(auto)
        else:
            out[role] = PALETTE.get(name, PALETTE["green"])
    return out


def _sizes_screen():
    return 2, 5, 0.42              # grosor de línea, radio de punto, letra (px de pantalla)


def _sizes_image(image):
    k = max(image.shape[:2]) / 1500
    return max(1, round(2 * k)), max(3, round(7 * k)), max(0.4, 0.6 * k)


def _text(out, text, x, y, scale, color):
    th = max(1, round(scale * 2))
    cv2.putText(out, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(out, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def draw(img: np.ndarray, ov: dict, colors: dict, origin=(0, 0), zoom: float = 1.0,
         screen: bool = False) -> np.ndarray:
    """Dibuja `ov` sobre `img` (BGR, se modifica). Las coordenadas de `ov` son de la
    imagen completa; `origin` y `zoom` las llevan al recorte que se está mostrando."""
    width, radius, scale = _sizes_screen() if screen else _sizes_image(img)
    if screen and ov.get("size"):              # el punto no tapa el objeto al alejar
        radius = int(np.clip(ov["size"] * zoom * 0.2, 2, radius))
    ox, oy = origin

    def tr(pts):
        return np.round((np.asarray(pts, float) - (ox, oy)) * zoom).astype(np.int32)

    for cnt, included in ov.get("outlines", []):
        p = tr(cnt).reshape(-1, 1, 2)
        if included:
            cv2.polylines(img, [p], True, (0, 0, 0), width + 2, cv2.LINE_AA)
            cv2.polylines(img, [p], True, colors["outline"], width, cv2.LINE_AA)
        else:
            cv2.polylines(img, [p], True, _EXCLUDED, 1, cv2.LINE_AA)
    if ov.get("dots"):
        for x, y in tr(ov["dots"]):
            cv2.circle(img, (int(x), int(y)), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), radius, colors["dot"], -1, cv2.LINE_AA)
    h, w = img.shape[:2]
    shift = radius if ov.get("dots") else 0
    if screen and ov.get("size", 0) * zoom < 22:       # objetos muy chicos en pantalla: sin números
        return img
    for (x, y, text, strong) in ov.get("labels", []):
        sx, sy = tr([(x, y)])[0]
        if -50 < sx < w + 50 and -20 < sy < h + 20:
            _text(img, text, int(sx) + shift, int(sy), scale,
                  (255, 255, 255) if strong else (190, 190, 190))
    return img


def tint(image: np.ndarray, mask: np.ndarray, colors: dict) -> np.ndarray:
    """Vista previa de la segmentación: color semitransparente y borde con halo."""
    out = image.copy()
    sel = mask > 0
    a = colors["mask_alpha"]
    out[sel] = (out[sel] * (1 - a) + np.array(colors["mask"], np.float32) * a).astype(np.uint8)
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    w = max(1, round(max(image.shape[:2]) / 1500))
    cv2.drawContours(out, cnts, -1, (0, 0, 0), w + 2, cv2.LINE_AA)
    cv2.drawContours(out, cnts, -1, colors["mask"], w, cv2.LINE_AA)
    return out
