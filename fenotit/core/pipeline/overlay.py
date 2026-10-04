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
DEFAULT_STYLE = {"mask": "green", "outline": "auto", "dot": "magenta", "mask_alpha": 0.45, "width": 1.5}
_EXCLUDED = (150, 150, 150)
_AUTO_CANDIDATES = ("green", "magenta", "yellow", "cyan", "red", "orange", "blue")


def empty() -> dict:
    return {"outlines": [], "dots": [], "labels": [], "lines": [], "size": 0}


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
    out = {"mask_alpha": float(s["mask_alpha"]), "width": float(s["width"])}
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
         screen: bool = False, keep=()) -> np.ndarray:
    """Dibuja `ov` sobre `img` (BGR, se modifica). Las coordenadas de `ov` son de la
    imagen completa; `origin` y `zoom` las llevan al recorte que se está mostrando.
    `keep`: rectángulos (x0, y0, x1, y1, en px de imagen) que no se tapan (la leyenda)."""
    saved = []
    for bx0, by0, bx1, by1 in keep:
        r = (slice(max(0, int((by0 - origin[1]) * zoom)), max(0, int((by1 - origin[1]) * zoom) + 1)),
             slice(max(0, int((bx0 - origin[0]) * zoom)), max(0, int((bx1 - origin[0]) * zoom) + 1)))
        saved.append((r, img[r].copy()))
    _draw(img, ov, colors, origin, zoom, screen)
    for r, patch in saved:
        img[r] = patch
    return img


def _lines(img, polys, color, width: float, halo: bool):
    """Contornos en una sola llamada. Menos de 1 px: línea de 1 px semitransparente."""
    if not polys:
        return
    if width < 1:
        layer = img.copy()
        cv2.polylines(layer, polys, True, color, 1, cv2.LINE_AA)
        a = max(0.35, width)
        cv2.addWeighted(layer, a, img, 1 - a, 0, dst=img)
        return
    w = int(round(width))
    if halo:
        cv2.polylines(img, polys, True, (0, 0, 0), w + 2, cv2.LINE_AA)
    cv2.polylines(img, polys, True, color, w, cv2.LINE_AA)


def _draw(img, ov, colors, origin, zoom, screen):
    _, radius, scale = _sizes_screen() if screen else _sizes_image(img)
    lw = float(colors.get("width", 1.5))            # grosor elegido, en px de pantalla
    seen = ov.get("size", 0) * zoom                 # tamaño típico del objeto en pantalla
    if screen:
        width = min(lw, max(0.5, seen * 0.05)) if seen else lw   # al alejar se afina
        if seen:
            radius = int(np.clip(seen * 0.2, 2, radius))
    else:
        width = max(1.0, lw * max(img.shape[:2]) / 1500)
    ox, oy = origin

    def tr(pts):
        return np.round((np.asarray(pts, float) - (ox, oy)) * zoom).astype(np.int32)

    inc = [tr(c).reshape(-1, 1, 2) for c, ok in ov.get("outlines", []) if ok]
    exc = [tr(c).reshape(-1, 1, 2) for c, ok in ov.get("outlines", []) if not ok]
    _lines(img, exc, _EXCLUDED, min(width, 1.0), False)
    _lines(img, inc, colors["outline"], width, halo=width >= 1.5)
    if ov.get("lines"):                             # segmentos sueltos con opacidad (p. ej. distancias)
        thin = ov.get("thin_lines")                 # líneas de referencia (largo/ancho): finas
        lw = (max(1, int(round(width * 0.5))) - 1) if thin else max(1, int(round(width)))
        bins = {}
        for p1, p2, strength in ov["lines"]:
            q = tr([p1, p2])
            bins.setdefault(min(3, int(strength * 4)), []).append(q)
        for b, segs in sorted(bins.items()):
            layer = img.copy()
            for q in segs:
                cv2.line(layer, tuple(map(int, q[0])), tuple(map(int, q[1])), colors["dot"], lw + 1, cv2.LINE_AA)
            a = 0.3 + 0.7 * (b + 1) / 4
            cv2.addWeighted(layer, a, img, 1 - a, 0, dst=img)
    if ov.get("dots"):
        for x, y in tr(ov["dots"]):
            cv2.circle(img, (int(x), int(y)), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), radius, colors["dot"], -1, cv2.LINE_AA)
    h, w = img.shape[:2]
    shift = radius if ov.get("dots") else 0
    if screen and seen:
        if seen < 12:                               # no caben: se ven al acercar
            return img
        scale = float(np.clip(seen / 70, 0.28, 0.45))
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
