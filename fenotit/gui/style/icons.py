"""Íconos de la interfaz, por nombre. Cada ícono es de un solo color: `icon(nombre,
tamaño, color)` lo devuelve como imagen RGBA de ese color.

Si existe fenotit/assets/icons/<nombre>.png, se usa ese archivo (su transparencia es la
forma del ícono y se pinta del color pedido); si no, el ícono se dibuja con las funciones
de este archivo. Los .png salen de los .svg de fenotit/assets/icons/svg con
docs/design/icons_svg_to_png.py. La lista de nombres está en NAMES."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from fenotit.gui.style.theme import COLORS

ICON_DIR = Path(__file__).resolve().parents[2] / "assets" / "icons"

_SUPERSAMPLING = 4          # sobre-muestreo para bordes suaves


def _magnifier(draw: ImageDraw.ImageDraw, size: int, stroke: int, color, sign: str | None):
    radius = size * 0.30
    cx = cy = size * 0.40
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=stroke)
    handle = radius * 0.7071
    draw.line((cx + handle, cy + handle, size * 0.90, size * 0.90), fill=color, width=int(stroke * 1.6))
    sign_size = radius * 0.5
    if sign:
        draw.line((cx - sign_size, cy, cx + sign_size, cy), fill=color, width=stroke)
    if sign == "+":
        draw.line((cx, cy - sign_size, cx, cy + sign_size), fill=color, width=stroke)


def _hand(draw: ImageDraw.ImageDraw, size: int, stroke: int, color):
    """Mano abierta rellena."""
    finger_width, gap = size * 0.13, size * 0.03
    for i, top in enumerate((0.20, 0.08, 0.11, 0.22)):
        x = size * 0.27 + i * (finger_width + gap)
        draw.rounded_rectangle((x, size * top, x + finger_width, size * 0.60), radius=finger_width / 2, fill=color)
    draw.rounded_rectangle((size * 0.27, size * 0.42, size * 0.27 + 4 * finger_width + 3 * gap, size * 0.94),
                        radius=size * 0.17, fill=color)
    draw.line((size * 0.30, size * 0.74, size * 0.10, size * 0.44), fill=color, width=int(finger_width))
    draw.ellipse((size * 0.10 - finger_width / 2, size * 0.44 - finger_width / 2, size * 0.10 + finger_width / 2, size * 0.44 + finger_width / 2), fill=color)


def _fit(draw: ImageDraw.ImageDraw, size: int, stroke: int, color):
    margin, corner = size * 0.12, size * 0.30
    for x, y, dx, dy in ((margin, margin, 1, 1), (size - margin, margin, -1, 1), (margin, size - margin, 1, -1), (size - margin, size - margin, -1, -1)):
        draw.line((x, y, x + dx * corner, y), fill=color, width=stroke)
        draw.line((x, y, x, y + dy * corner), fill=color, width=stroke)


def _zoom_area(draw: ImageDraw.ImageDraw, size: int, stroke: int, color):
    x0, y0, x1, y1 = size * 0.08, size * 0.08, size * 0.70, size * 0.62
    dash = size * 0.10
    for start, end, horiz, fixed in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
        position = start
        while position < end:
            other = min(position + dash, end)
            draw.line((position, fixed, other, fixed) if horiz else (fixed, position, fixed, other), fill=color, width=stroke)
            position += dash * 1.8
    radius = size * 0.17
    cx, cy = size * 0.62, size * 0.60
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=stroke, fill=None)
    draw.line((cx + radius * 0.7, cy + radius * 0.7, size * 0.94, size * 0.94), fill=color, width=int(stroke * 1.6))
    handle = radius * 0.55
    draw.line((cx - handle, cy, cx + handle, cy), fill=color, width=max(1, int(stroke * 0.8)))
    draw.line((cx, cy - handle, cx, cy + handle), fill=color, width=max(1, int(stroke * 0.8)))


def _palette(draw: ImageDraw.ImageDraw, size: int, stroke: int, color):
    """Paleta de pintor rellena, con muesca, hueco para el pulgar y manchas caladas."""
    clear = (0, 0, 0, 0)
    draw.ellipse((size * 0.06, size * 0.14, size * 0.94, size * 0.86), fill=color)
    draw.ellipse((size * 0.62, size * 0.50, size * 1.02, size * 0.84), fill=clear)          # muesca
    draw.ellipse((size * 0.50, size * 0.52, size * 0.66, size * 0.68), fill=clear)          # pulgar
    radius = size * 0.075
    for cx, cy in ((0.24, 0.42), (0.40, 0.28), (0.60, 0.27), (0.76, 0.38), (0.30, 0.64)):
        draw.ellipse((size * cx - radius, size * cy - radius, size * cx + radius, size * cy + radius), fill=clear)


def _shape_pts(size, kind):
    if kind == "rect":
        return [(size * 0.14, size * 0.22), (size * 0.86, size * 0.22), (size * 0.86, size * 0.78), (size * 0.14, size * 0.78)]
    return [(size * 0.50, size * 0.10), (size * 0.90, size * 0.40), (size * 0.74, size * 0.88),
            (size * 0.26, size * 0.88), (size * 0.10, size * 0.40)]


def _area(draw, size, stroke, color, kind, exclude):
    pts = _shape_pts(size, kind)
    if exclude:                                       # zona excluida: rayada
        for offset in range(-4, 6):
            x = size * (0.12 + offset * 0.16)
            draw.line((x, size * 0.92, x + size * 0.8, size * 0.12), fill=color, width=max(1, stroke // 2))
        clear = (0, 0, 0, 0)
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        draw._image.paste(clear, mask=Image.eval(mask, lambda value: 255 - value))
    draw.line(pts + [pts[0]], fill=color, width=stroke, joint="curve")


def _select(draw, size, stroke, color):
    x0, y0, x1, y1 = size * 0.08, size * 0.08, size * 0.66, size * 0.62
    dash = size * 0.09
    for start, end, horiz, fixed in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
        other = start
        while other < end:
            event = min(other + dash, end)
            draw.line((other, fixed, event, fixed) if horiz else (fixed, other, fixed, event), fill=color, width=max(1, stroke // 2 + 1))
            other += dash * 1.9
    arrow = [(size * 0.45, size * 0.40), (size * 0.45, size * 0.95), (size * 0.58, size * 0.80), (size * 0.68, size * 0.98),
             (size * 0.78, size * 0.93), (size * 0.68, size * 0.76), (size * 0.86, size * 0.74)]
    draw.polygon(arrow, fill=color)


def _photo(draw, size, stroke, color):
    draw.rounded_rectangle((size * 0.08, size * 0.18, size * 0.92, size * 0.82), radius=size * 0.08, outline=color, width=stroke)
    draw.polygon([(size * 0.16, size * 0.74), (size * 0.40, size * 0.44), (size * 0.56, size * 0.62), (size * 0.66, size * 0.52),
               (size * 0.84, size * 0.74)], fill=color)
    draw.ellipse((size * 0.62, size * 0.28, size * 0.74, size * 0.40), fill=color)


def _folder(draw, size, stroke, color):
    draw.polygon([(size * 0.08, size * 0.22), (size * 0.38, size * 0.22), (size * 0.46, size * 0.32), (size * 0.92, size * 0.32),
               (size * 0.92, size * 0.80), (size * 0.08, size * 0.80)], outline=color, width=stroke)
    draw.line((size * 0.08, size * 0.42, size * 0.92, size * 0.42), fill=color, width=stroke)


def _project(draw, size, stroke, color):
    """Documento con esquina doblada (abrir un proyecto guardado)."""
    draw.polygon([(size * 0.22, size * 0.08), (size * 0.60, size * 0.08), (size * 0.80, size * 0.28), (size * 0.80, size * 0.92),
               (size * 0.22, size * 0.92)], outline=color, width=stroke)
    draw.line((size * 0.60, size * 0.08, size * 0.60, size * 0.28, size * 0.80, size * 0.28), fill=color, width=stroke)
    for y in (0.48, 0.62, 0.76):
        draw.line((size * 0.34, size * y, size * 0.68, size * y), fill=color, width=stroke)


def _eye(draw, size, stroke, color, off=False):
    draw.arc((size * 0.04, size * 0.22, size * 0.96, size * 1.30), 200, 340, fill=color, width=stroke)     # párpado de arriba
    draw.arc((size * 0.04, size * -0.30, size * 0.96, size * 0.78), 20, 160, fill=color, width=stroke)     # de abajo
    draw.ellipse((size * 0.34, size * 0.34, size * 0.66, size * 0.66), outline=color, width=stroke)
    draw.ellipse((size * 0.44, size * 0.44, size * 0.56, size * 0.56), fill=color)
    if off:
        draw.line((size * 0.12, size * 0.88, size * 0.88, size * 0.12), fill=color, width=int(stroke * 1.3))


def _list(draw, size, stroke, color):
    for y in (0.24, 0.50, 0.76):
        draw.ellipse((size * 0.08, size * y - size * 0.06, size * 0.20, size * y + size * 0.06), fill=color)
        draw.line((size * 0.32, size * y, size * 0.92, size * y), fill=color, width=stroke)


def _grid(draw, size, stroke, color):
    cell, gap = size * 0.36, size * 0.10
    for x in (size * 0.09, size * 0.09 + cell + gap):
        for y in (size * 0.09, size * 0.09 + cell + gap):
            draw.rounded_rectangle((x, y, x + cell, y + cell), radius=size * 0.05, outline=color, width=stroke)


def _panel(draw, size, stroke, color, side="left"):
    """Ventana con una franja a un lado: mostrar/ocultar el panel de ese lado."""
    x0, y0, x1, y1 = size * 0.08, size * 0.16, size * 0.92, size * 0.84
    draw.rounded_rectangle((x0, y0, x1, y1), radius=size * 0.08, outline=color, width=stroke)
    if side == "bottom":
        draw.rectangle((x0, y1 - size * 0.24, x1, y1), fill=color)
        return
    box_x = (x0, x0 + size * 0.30) if side == "left" else (x1 - size * 0.30, x1)
    draw.rectangle((box_x[0], y0, box_x[1], y1), fill=color)


def _tune(draw, size, stroke, color):
    """Ajustes: tres líneas con su perilla (opciones de una medición)."""
    for y, knob_x in ((0.26, 0.66), (0.50, 0.34), (0.74, 0.58)):
        draw.line((size * 0.08, size * y, size * 0.92, size * y), fill=color, width=stroke)
        radius = size * 0.11
        draw.ellipse((size * knob_x - radius, size * y - radius, size * knob_x + radius, size * y + radius), fill=color)


def _inspect(draw, size, stroke, color):
    """Inspeccionar: mira con un punto al centro (elegir un objeto)."""
    center, radius = size * 0.5, size * 0.30
    draw.ellipse((center - radius, center - radius, center + radius, center + radius), outline=color, width=stroke)
    for start, end in (((center, size * 0.04), (center, size * 0.26)), ((center, size * 0.74), (center, size * 0.96)),
                 ((size * 0.04, center), (size * 0.26, center)), ((size * 0.74, center), (size * 0.96, center))):
        draw.line((start, end), fill=color, width=stroke)
    dot = size * 0.08
    draw.ellipse((center - dot, center - dot, center + dot, center + dot), fill=color)


def _dock_up(draw, size, stroke, color):
    """Ventana con flecha hacia arriba: llevar el gráfico al lugar de una imagen."""
    x0, y0, x1, y1 = size * 0.08, size * 0.16, size * 0.92, size * 0.84
    draw.rounded_rectangle((x0, y0, x1, y1), radius=size * 0.08, outline=color, width=stroke)
    cx = size * 0.5
    draw.line((cx, size * 0.72, cx, size * 0.34), fill=color, width=stroke)
    draw.polygon([(cx, size * 0.26), (cx - size * 0.17, size * 0.45), (cx + size * 0.17, size * 0.45)], fill=color)


def _expand(draw, size, stroke, color):
    """Dos flechas en diagonal hacia las esquinas: pantalla completa."""
    near, far, head = size * 0.14, size * 0.86, size * 0.26
    for (x0, y0, x1, y1, dir_x, dir_y) in ((size * 0.56, size * 0.44, far, near, -1, 1), (size * 0.44, size * 0.56, near, far, 1, -1)):
        draw.line((x0, y0, x1, y1), fill=color, width=stroke)
        draw.line((x1, y1, x1 + dir_x * head, y1), fill=color, width=stroke)
        draw.line((x1, y1, x1, y1 + dir_y * head), fill=color, width=stroke)


def _home(draw, size, stroke, color):
    """Casa: techo y cuerpo con puerta."""
    margin = size * 0.16
    draw.line((margin, size * 0.5, size / 2, margin, size - margin, size * 0.5), fill=color, width=stroke, joint="curve")
    draw.line((size * 0.26, size * 0.42, size * 0.26, size - margin, size * 0.74, size - margin, size * 0.74, size * 0.42), fill=color, width=stroke)
    draw.line((size * 0.44, size - margin, size * 0.44, size * 0.64, size * 0.56, size * 0.64, size * 0.56, size - margin), fill=color, width=stroke)


_DRAW = {"home": _home, "palette": _palette, "dock_up": _dock_up, "expand": _expand, "inspect": _inspect, "list": _list, "grid": _grid, "tune": _tune,
         "panel_left": lambda draw, size, stroke, color: _panel(draw, size, stroke, color, "left"),
         "panel_right": lambda draw, size, stroke, color: _panel(draw, size, stroke, color, "right"),
         "panel_bottom": lambda draw, size, stroke, color: _panel(draw, size, stroke, color, "bottom"), "select": _select,
         "eye": _eye, "eye_off": lambda draw, size, stroke, color: _eye(draw, size, stroke, color, off=True), "photo": _photo, "folder": _folder, "project": _project,
         "area_rect": lambda draw, size, stroke, color: _area(draw, size, stroke, color, "rect", False),
         "area_poly": lambda draw, size, stroke, color: _area(draw, size, stroke, color, "poly", False),
         "excl_rect": lambda draw, size, stroke, color: _area(draw, size, stroke, color, "rect", True),
         "excl_poly": lambda draw, size, stroke, color: _area(draw, size, stroke, color, "poly", True), "zoom_in": lambda draw, size, stroke, color: _magnifier(draw, size, stroke, color, "+"),
         "zoom_out": lambda draw, size, stroke, color: _magnifier(draw, size, stroke, color, "-"),
         "pan": _hand, "fit": _fit, "zoom_area": _zoom_area}


NAMES = tuple(_DRAW)
_FILES: dict[str, Image.Image | None] = {}


def _file_mask(name: str) -> Image.Image | None:
    """Forma del ícono desde su .png (canal alfa), o None si no hay archivo."""
    if name not in _FILES:
        path = ICON_DIR / f"{name}.png"
        _FILES[name] = Image.open(path).convert("RGBA").getchannel("A") if path.exists() else None
    return _FILES[name]


def icon(name: str, size: int, color: str = COLORS["on_accent"]) -> Image.Image:
    mask = _file_mask(name)
    if mask is not None:
        out = Image.new("RGBA", (size, size), color)
        out.putalpha(mask.resize((size, size), Image.LANCZOS))
        return out
    canvas_size = size * _SUPERSAMPLING
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    _DRAW[name](ImageDraw.Draw(img), canvas_size, max(_SUPERSAMPLING, int(canvas_size * 0.085)), color)
    return img.resize((size, size), Image.LANCZOS)


