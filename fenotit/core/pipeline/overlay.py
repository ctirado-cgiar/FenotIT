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


UNNAMED_CLASS_COLOR = "#888888"       # color de una clase sin color asignado


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """"#RRGGBB" → (B, G, R)."""
    digits = hex_color.lstrip("#")
    return int(digits[4:6], 16), int(digits[2:4], 16), int(digits[0:2], 16)


def empty() -> dict:
    return {"outlines": [], "dots": [], "labels": [], "lines": [], "stars": [], "size": 0}


STAR = (0, 200, 255)             # BGR, amarillo


def star(img, x: float, y: float, radius: float):
    """Estrella de 5 puntas (objeto destacado) con borde oscuro."""
    angles = np.deg2rad(np.arange(10) * 36 - 90)
    rad = np.where(np.arange(10) % 2 == 0, radius, radius * 0.45)
    pts = np.column_stack([x + rad * np.cos(angles), y + rad * np.sin(angles)]).round().astype(np.int32)
    cv2.fillPoly(img, [pts], STAR, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (30, 30, 30), 1, cv2.LINE_AA)


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
        selected = cv2.resize(mask.astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST).reshape(-1) > 0
        if selected.any():
            refs += [np.median(lab[selected], axis=0)]
        if (~selected).any():
            refs += [np.median(lab[~selected], axis=0)]
    best, best_d = _AUTO_CANDIDATES[0], -1.0
    for name in _AUTO_CANDIDATES:
        candidate = cv2.cvtColor(np.uint8([[PALETTE[name]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        distance = min(np.linalg.norm(candidate - reference) for reference in refs)
        if distance > best_d:
            best, best_d = name, distance
    return PALETTE[best]


def resolve(style: dict | None, image: np.ndarray | None = None, mask=None, auto=None) -> dict:
    """Colores BGR por rol; "auto" se resuelve con la imagen (o usa `auto` ya calculado)."""
    merged = {**DEFAULT_STYLE, **(style or {})}
    out = {"mask_alpha": float(merged["mask_alpha"]), "width": float(merged["width"])}
    for role in ROLES:
        name = merged[role]
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
    factor = max(image.shape[:2]) / 1500
    return max(1, round(2 * factor)), max(3, round(7 * factor)), max(0.4, 0.6 * factor)


def _text(out, text, x, y, scale, color):
    thickness = max(1, round(scale * 2))
    cv2.putText(out, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw(img: np.ndarray, marks: dict, colors: dict, origin=(0, 0), zoom: float = 1.0,
         screen: bool = False, keep=()) -> np.ndarray:
    """Dibuja `ov` sobre `img` (BGR, se modifica). Las coordenadas de `ov` son de la
    imagen completa; `origin` y `zoom` las llevan al recorte que se está mostrando.
    `keep`: rectángulos (x0, y0, x1, y1, en px de imagen) que no se tapan (la leyenda)."""
    saved = []
    for bx0, by0, bx1, by1 in keep:
        region = (slice(max(0, int((by0 - origin[1]) * zoom)), max(0, int((by1 - origin[1]) * zoom) + 1)),
             slice(max(0, int((bx0 - origin[0]) * zoom)), max(0, int((bx1 - origin[0]) * zoom) + 1)))
        saved.append((region, img[region].copy()))
    _draw(img, marks, colors, origin, zoom, screen)
    for region, patch in saved:
        img[region] = patch
    return img


def _lines(img, polys, color, width: float, halo: bool):
    """Contornos en una sola llamada. Menos de 1 px: línea de 1 px semitransparente."""
    if not polys:
        return
    if width < 1:
        layer = img.copy()
        cv2.polylines(layer, polys, True, color, 1, cv2.LINE_AA)
        opacity = max(0.35, width)
        cv2.addWeighted(layer, opacity, img, 1 - opacity, 0, dst=img)
        return
    thickness = int(round(width))
    if halo:
        cv2.polylines(img, polys, True, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.polylines(img, polys, True, color, thickness, cv2.LINE_AA)


def _draw(img, marks, colors, origin, zoom, screen):
    _, radius, scale = _sizes_screen() if screen else _sizes_image(img)
    line_width = float(colors.get("width", 1.5))            # grosor elegido, en px de pantalla
    seen = marks.get("size", 0) * zoom                 # tamaño típico del objeto en pantalla
    if screen:
        width = min(line_width, max(0.5, seen * 0.05)) if seen else line_width   # al alejar se afina
        if seen:
            radius = int(np.clip(seen * 0.2, 2, radius))
    else:
        width = max(1.0, line_width * max(img.shape[:2]) / 1500)
    origin_x, origin_y = origin

    def tr(pts):
        return np.round((np.asarray(pts, float) - (origin_x, origin_y)) * zoom).astype(np.int32)

    inc = [tr(contour).reshape(-1, 1, 2) for contour, ok in marks.get("outlines", []) if ok]
    exc = [tr(contour).reshape(-1, 1, 2) for contour, ok in marks.get("outlines", []) if not ok]
    _lines(img, exc, _EXCLUDED, min(width, 1.0), False)
    _lines(img, inc, colors["outline"], width, halo=width >= 1.5)
    if marks.get("lines"):                             # segmentos sueltos con opacidad (p. ej. distancias)
        thin = marks.get("thin_lines")                 # líneas de referencia (largo/ancho): finas
        line_width = (max(1, int(round(width * 0.5))) - 1) if thin else max(1, int(round(width)))
        bins = {}
        for start, end, strength in marks["lines"]:
            segment = tr([start, end])
            bins.setdefault(min(3, int(strength * 4)), []).append(segment)
        for level, segs in sorted(bins.items()):
            layer = img.copy()
            for segment in segs:
                cv2.line(layer, tuple(map(int, segment[0])), tuple(map(int, segment[1])), colors["dot"], line_width + 1, cv2.LINE_AA)
            opacity = 0.3 + 0.7 * (level + 1) / 4
            cv2.addWeighted(layer, opacity, img, 1 - opacity, 0, dst=img)
    if marks.get("dots"):
        for x, y in tr(marks["dots"]):
            cv2.circle(img, (int(x), int(y)), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), radius, colors["dot"], -1, cv2.LINE_AA)
    if marks.get("stars"):
        star_radius = max(5.0, radius * 1.8)
        for x, y in tr(marks["stars"]):
            star(img, float(x), float(y), star_radius)
    image_height, image_width = img.shape[:2]
    shift = radius if marks.get("dots") else 0
    if screen and seen:
        if seen < 12:                               # no caben: se ven al acercar
            return img
        scale = float(np.clip(seen / 70, 0.28, 0.45))
    for (x, y, text, strong) in marks.get("labels", []):
        screen_x, screen_y = tr([(x, y)])[0]
        if -50 < screen_x < image_width + 50 and -20 < screen_y < image_height + 20:
            _text(img, text, int(screen_x) + shift, int(screen_y), scale,
                  (255, 255, 255) if strong else (190, 190, 190))
    return img


def highlight(img: np.ndarray, contour, origin=(0, 0), zoom: float = 1.0) -> np.ndarray:
    """Resalta un objeto (el elegido en el inspector) como la segmentación: relleno claro
    semitransparente y un borde fino."""
    pts = np.round((np.asarray(contour, float) - origin) * zoom).astype(np.int32).reshape(-1, 1, 2)
    x, y, box_width, box_height = cv2.boundingRect(pts)
    image_height, image_width = img.shape[:2]
    x0, y0, x1, y1 = max(0, x - 2), max(0, y - 2), min(image_width, x + box_width + 2), min(image_height, y + box_height + 2)
    if x1 <= x0 or y1 <= y0:
        return img
    roi = img[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts - (x0, y0)], 255)
    tinted = cv2.addWeighted(roi, 0.55, np.full_like(roi, 255), 0.45, 0)
    roi[mask > 0] = tinted[mask > 0]
    cv2.polylines(img, [pts], True, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def tint(image: np.ndarray, mask: np.ndarray, colors: dict) -> np.ndarray:
    """Vista previa de la segmentación: color semitransparente y borde con halo."""
    out = image.copy()
    sel = mask > 0
    opacity = colors["mask_alpha"]
    out[sel] = (out[sel] * (1 - opacity) + np.array(colors["mask"], np.float32) * opacity).astype(np.uint8)
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(1, round(max(image.shape[:2]) / 1500))
    cv2.drawContours(out, cnts, -1, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.drawContours(out, cnts, -1, colors["mask"], thickness, cv2.LINE_AA)
    return out
