"""Íconos de la interfaz, por nombre. Cada ícono es de un solo color: `icon(nombre,
tamaño, color)` lo devuelve como imagen RGBA de ese color.

Si existe fenotit/assets/icons/<nombre>.png, se usa ese archivo (su transparencia es la
forma del ícono y se pinta del color pedido); si no, el ícono se dibuja con las funciones
de este archivo. Los .png salen de los .svg de fenotit/assets/icons/svg con
docs/design/icons_svg_to_png.py. La lista de nombres está en NAMES."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from fenotit.gui.theme import COLORS

ICON_DIR = Path(__file__).resolve().parents[1] / "assets" / "icons"

_SS = 4          # sobre-muestreo para bordes suaves


def _magnifier(d: ImageDraw.ImageDraw, s: int, w: int, color, sign: str | None):
    r = s * 0.30
    cx = cy = s * 0.40
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=w)
    a = r * 0.7071
    d.line((cx + a, cy + a, s * 0.90, s * 0.90), fill=color, width=int(w * 1.6))
    k = r * 0.5
    if sign:
        d.line((cx - k, cy, cx + k, cy), fill=color, width=w)
    if sign == "+":
        d.line((cx, cy - k, cx, cy + k), fill=color, width=w)


def _hand(d: ImageDraw.ImageDraw, s: int, w: int, color):
    """Mano abierta rellena."""
    fw, gap = s * 0.13, s * 0.03
    for i, top in enumerate((0.20, 0.08, 0.11, 0.22)):
        x = s * 0.27 + i * (fw + gap)
        d.rounded_rectangle((x, s * top, x + fw, s * 0.60), radius=fw / 2, fill=color)
    d.rounded_rectangle((s * 0.27, s * 0.42, s * 0.27 + 4 * fw + 3 * gap, s * 0.94),
                        radius=s * 0.17, fill=color)
    d.line((s * 0.30, s * 0.74, s * 0.10, s * 0.44), fill=color, width=int(fw))
    d.ellipse((s * 0.10 - fw / 2, s * 0.44 - fw / 2, s * 0.10 + fw / 2, s * 0.44 + fw / 2), fill=color)


def _fit(d: ImageDraw.ImageDraw, s: int, w: int, color):
    m, k = s * 0.12, s * 0.30
    for x, y, dx, dy in ((m, m, 1, 1), (s - m, m, -1, 1), (m, s - m, 1, -1), (s - m, s - m, -1, -1)):
        d.line((x, y, x + dx * k, y), fill=color, width=w)
        d.line((x, y, x, y + dy * k), fill=color, width=w)


def _zoom_area(d: ImageDraw.ImageDraw, s: int, w: int, color):
    x0, y0, x1, y1 = s * 0.08, s * 0.08, s * 0.70, s * 0.62
    dash = s * 0.10
    for a, b, horiz, fixed in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
        p = a
        while p < b:
            q = min(p + dash, b)
            d.line((p, fixed, q, fixed) if horiz else (fixed, p, fixed, q), fill=color, width=w)
            p += dash * 1.8
    r = s * 0.17
    cx, cy = s * 0.62, s * 0.60
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=w, fill=None)
    d.line((cx + r * 0.7, cy + r * 0.7, s * 0.94, s * 0.94), fill=color, width=int(w * 1.6))
    k = r * 0.55
    d.line((cx - k, cy, cx + k, cy), fill=color, width=max(1, int(w * 0.8)))
    d.line((cx, cy - k, cx, cy + k), fill=color, width=max(1, int(w * 0.8)))


def _palette(d: ImageDraw.ImageDraw, s: int, w: int, color):
    """Paleta de pintor rellena, con muesca, hueco para el pulgar y manchas caladas."""
    clear = (0, 0, 0, 0)
    d.ellipse((s * 0.06, s * 0.14, s * 0.94, s * 0.86), fill=color)
    d.ellipse((s * 0.62, s * 0.50, s * 1.02, s * 0.84), fill=clear)          # muesca
    d.ellipse((s * 0.50, s * 0.52, s * 0.66, s * 0.68), fill=clear)          # pulgar
    r = s * 0.075
    for cx, cy in ((0.24, 0.42), (0.40, 0.28), (0.60, 0.27), (0.76, 0.38), (0.30, 0.64)):
        d.ellipse((s * cx - r, s * cy - r, s * cx + r, s * cy + r), fill=clear)


def _shape_pts(s, kind):
    if kind == "rect":
        return [(s * 0.14, s * 0.22), (s * 0.86, s * 0.22), (s * 0.86, s * 0.78), (s * 0.14, s * 0.78)]
    return [(s * 0.50, s * 0.10), (s * 0.90, s * 0.40), (s * 0.74, s * 0.88),
            (s * 0.26, s * 0.88), (s * 0.10, s * 0.40)]


def _area(d, s, w, color, kind, exclude):
    pts = _shape_pts(s, kind)
    if exclude:                                       # zona excluida: rayada
        for k in range(-4, 6):
            x = s * (0.12 + k * 0.16)
            d.line((x, s * 0.92, x + s * 0.8, s * 0.12), fill=color, width=max(1, w // 2))
        clear = (0, 0, 0, 0)
        mask = Image.new("L", (s, s), 0)
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        d._image.paste(clear, mask=Image.eval(mask, lambda v: 255 - v))
    d.line(pts + [pts[0]], fill=color, width=w, joint="curve")


def _select(d, s, w, color):
    x0, y0, x1, y1 = s * 0.08, s * 0.08, s * 0.66, s * 0.62
    dash = s * 0.09
    for a, b, horiz, fixed in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
        q = a
        while q < b:
            e = min(q + dash, b)
            d.line((q, fixed, e, fixed) if horiz else (fixed, q, fixed, e), fill=color, width=max(1, w // 2 + 1))
            q += dash * 1.9
    arrow = [(s * 0.45, s * 0.40), (s * 0.45, s * 0.95), (s * 0.58, s * 0.80), (s * 0.68, s * 0.98),
             (s * 0.78, s * 0.93), (s * 0.68, s * 0.76), (s * 0.86, s * 0.74)]
    d.polygon(arrow, fill=color)


def _photo(d, s, w, color):
    d.rounded_rectangle((s * 0.08, s * 0.18, s * 0.92, s * 0.82), radius=s * 0.08, outline=color, width=w)
    d.polygon([(s * 0.16, s * 0.74), (s * 0.40, s * 0.44), (s * 0.56, s * 0.62), (s * 0.66, s * 0.52),
               (s * 0.84, s * 0.74)], fill=color)
    d.ellipse((s * 0.62, s * 0.28, s * 0.74, s * 0.40), fill=color)


def _folder(d, s, w, color):
    d.polygon([(s * 0.08, s * 0.22), (s * 0.38, s * 0.22), (s * 0.46, s * 0.32), (s * 0.92, s * 0.32),
               (s * 0.92, s * 0.80), (s * 0.08, s * 0.80)], outline=color, width=w)
    d.line((s * 0.08, s * 0.42, s * 0.92, s * 0.42), fill=color, width=w)


def _project(d, s, w, color):
    """Documento con esquina doblada (abrir un proyecto guardado)."""
    d.polygon([(s * 0.22, s * 0.08), (s * 0.60, s * 0.08), (s * 0.80, s * 0.28), (s * 0.80, s * 0.92),
               (s * 0.22, s * 0.92)], outline=color, width=w)
    d.line((s * 0.60, s * 0.08, s * 0.60, s * 0.28, s * 0.80, s * 0.28), fill=color, width=w)
    for y in (0.48, 0.62, 0.76):
        d.line((s * 0.34, s * y, s * 0.68, s * y), fill=color, width=w)


def _eye(d, s, w, color, off=False):
    d.arc((s * 0.04, s * 0.22, s * 0.96, s * 1.30), 200, 340, fill=color, width=w)     # párpado de arriba
    d.arc((s * 0.04, s * -0.30, s * 0.96, s * 0.78), 20, 160, fill=color, width=w)     # de abajo
    d.ellipse((s * 0.34, s * 0.34, s * 0.66, s * 0.66), outline=color, width=w)
    d.ellipse((s * 0.44, s * 0.44, s * 0.56, s * 0.56), fill=color)
    if off:
        d.line((s * 0.12, s * 0.88, s * 0.88, s * 0.12), fill=color, width=int(w * 1.3))


def _list(d, s, w, color):
    for y in (0.24, 0.50, 0.76):
        d.ellipse((s * 0.08, s * y - s * 0.06, s * 0.20, s * y + s * 0.06), fill=color)
        d.line((s * 0.32, s * y, s * 0.92, s * y), fill=color, width=w)


def _grid(d, s, w, color):
    k, g = s * 0.36, s * 0.10
    for x in (s * 0.09, s * 0.09 + k + g):
        for y in (s * 0.09, s * 0.09 + k + g):
            d.rounded_rectangle((x, y, x + k, y + k), radius=s * 0.05, outline=color, width=w)


def _panel(d, s, w, color, side="left"):
    """Ventana con una franja a un lado: mostrar/ocultar el panel de ese lado."""
    x0, y0, x1, y1 = s * 0.08, s * 0.16, s * 0.92, s * 0.84
    d.rounded_rectangle((x0, y0, x1, y1), radius=s * 0.08, outline=color, width=w)
    if side == "bottom":
        d.rectangle((x0, y1 - s * 0.24, x1, y1), fill=color)
        return
    bx = (x0, x0 + s * 0.30) if side == "left" else (x1 - s * 0.30, x1)
    d.rectangle((bx[0], y0, bx[1], y1), fill=color)


def _tune(d, s, w, color):
    """Ajustes: tres líneas con su perilla (opciones de una medición)."""
    for y, kx in ((0.26, 0.66), (0.50, 0.34), (0.74, 0.58)):
        d.line((s * 0.08, s * y, s * 0.92, s * y), fill=color, width=w)
        r = s * 0.11
        d.ellipse((s * kx - r, s * y - r, s * kx + r, s * y + r), fill=color)


def _inspect(d, s, w, color):
    """Inspeccionar: mira con un punto al centro (elegir un objeto)."""
    c, r = s * 0.5, s * 0.30
    d.ellipse((c - r, c - r, c + r, c + r), outline=color, width=w)
    for a, b in (((c, s * 0.04), (c, s * 0.26)), ((c, s * 0.74), (c, s * 0.96)),
                 ((s * 0.04, c), (s * 0.26, c)), ((s * 0.74, c), (s * 0.96, c))):
        d.line((a, b), fill=color, width=w)
    q = s * 0.08
    d.ellipse((c - q, c - q, c + q, c + q), fill=color)


def _dock_up(d, s, w, color):
    """Ventana con flecha hacia arriba: llevar el gráfico al lugar de una imagen."""
    x0, y0, x1, y1 = s * 0.08, s * 0.16, s * 0.92, s * 0.84
    d.rounded_rectangle((x0, y0, x1, y1), radius=s * 0.08, outline=color, width=w)
    cx = s * 0.5
    d.line((cx, s * 0.72, cx, s * 0.34), fill=color, width=w)
    d.polygon([(cx, s * 0.26), (cx - s * 0.17, s * 0.45), (cx + s * 0.17, s * 0.45)], fill=color)


def _expand(d, s, w, color):
    """Dos flechas en diagonal hacia las esquinas: pantalla completa."""
    a, b, h = s * 0.14, s * 0.86, s * 0.26
    for (x0, y0, x1, y1, sx, sy) in ((s * 0.56, s * 0.44, b, a, -1, 1), (s * 0.44, s * 0.56, a, b, 1, -1)):
        d.line((x0, y0, x1, y1), fill=color, width=w)
        d.line((x1, y1, x1 + sx * h, y1), fill=color, width=w)
        d.line((x1, y1, x1, y1 + sy * h), fill=color, width=w)


def _home(d, s, w, color):
    """Casa: techo y cuerpo con puerta."""
    m = s * 0.16
    d.line((m, s * 0.5, s / 2, m, s - m, s * 0.5), fill=color, width=w, joint="curve")
    d.line((s * 0.26, s * 0.42, s * 0.26, s - m, s * 0.74, s - m, s * 0.74, s * 0.42), fill=color, width=w)
    d.line((s * 0.44, s - m, s * 0.44, s * 0.64, s * 0.56, s * 0.64, s * 0.56, s - m), fill=color, width=w)


_DRAW = {"home": _home, "palette": _palette, "dock_up": _dock_up, "expand": _expand, "inspect": _inspect, "list": _list, "grid": _grid, "tune": _tune,
         "panel_left": lambda d, s, w, c: _panel(d, s, w, c, "left"),
         "panel_right": lambda d, s, w, c: _panel(d, s, w, c, "right"),
         "panel_bottom": lambda d, s, w, c: _panel(d, s, w, c, "bottom"), "select": _select,
         "eye": _eye, "eye_off": lambda d, s, w, c: _eye(d, s, w, c, off=True), "photo": _photo, "folder": _folder, "project": _project,
         "area_rect": lambda d, s, w, c: _area(d, s, w, c, "rect", False),
         "area_poly": lambda d, s, w, c: _area(d, s, w, c, "poly", False),
         "excl_rect": lambda d, s, w, c: _area(d, s, w, c, "rect", True),
         "excl_poly": lambda d, s, w, c: _area(d, s, w, c, "poly", True), "zoom_in": lambda d, s, w, c: _magnifier(d, s, w, c, "+"),
         "zoom_out": lambda d, s, w, c: _magnifier(d, s, w, c, "-"),
         "pan": _hand, "fit": _fit, "zoom_area": _zoom_area}


NAMES = tuple(_DRAW)
_FILES: dict[str, Image.Image | None] = {}


def _file_mask(name: str) -> Image.Image | None:
    """Forma del ícono desde su .png (canal alfa), o None si no hay archivo."""
    if name not in _FILES:
        p = ICON_DIR / f"{name}.png"
        _FILES[name] = Image.open(p).convert("RGBA").getchannel("A") if p.exists() else None
    return _FILES[name]


def icon(name: str, size: int, color: str = COLORS["on_accent"]) -> Image.Image:
    mask = _file_mask(name)
    if mask is not None:
        out = Image.new("RGBA", (size, size), color)
        out.putalpha(mask.resize((size, size), Image.LANCZOS))
        return out
    s = size * _SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    _DRAW[name](ImageDraw.Draw(img), s, max(_SS, int(s * 0.085)), color)
    return img.resize((size, size), Image.LANCZOS)


