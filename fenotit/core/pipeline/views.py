"""Imágenes de resultado: qué objetos entraron en cada medición."""
import cv2
import numpy as np
from scipy.ndimage import find_objects


def regions(labels):
    for oid, sl in enumerate(find_objects(labels), 1):
        if sl is not None:
            yield oid, sl, (labels[sl] == oid).astype(np.uint8)


def inside_point(m: np.ndarray) -> tuple[int, int]:
    """Punto dentro del objeto: el centroide si cae adentro; si no (formas cóncavas,
    flores), el punto más alejado del borde."""
    ys, xs = np.nonzero(m)
    cx, cy = int(round(xs.mean())), int(round(ys.mean()))
    if m[cy, cx]:
        return cx, cy
    d = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(d.argmax()), d.shape)
    return int(x), int(y)


def sizes(image):
    """Grosor de línea, radio de punto y tamaño de letra según la resolución."""
    k = max(image.shape[:2]) / 1500
    return max(1, round(2 * k)), max(3, round(7 * k)), max(0.4, 0.6 * k)


def _outline(out, m, sl, color, width):
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                               offset=(sl[1].start, sl[0].start))
    cv2.drawContours(out, cnts, -1, color, width, cv2.LINE_AA)


def label_text(out, text, x, y, scale, color=(255, 255, 255)):
    th = max(1, round(scale * 2))
    cv2.putText(out, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(out, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def included_view(ctx, name, included, color):
    """Objetos incluidos en una medición: contorno de color y número; los demás
    (se tocaban) en gris, para ver qué entró y qué no."""
    out = ctx.image.copy()
    width, _, scale = sizes(out)
    for oid, sl, m in regions(ctx.labels):
        ok = oid in included
        _outline(out, m, sl, color if ok else (150, 150, 150), width if ok else 1)
        x, y = inside_point(m)
        label_text(out, str(oid), x + sl[1].start, y + sl[0].start, scale,
               (255, 255, 255) if ok else (190, 190, 190))
    ctx.images[name] = out


_FONTS = {False: ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"),
          True: ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")}


def _font(size: int, bold: bool = False):
    """Fuente del sistema con tildes y símbolos (²); la de Pillow no los trae."""
    from PIL import ImageFont
    for name in _FONTS[bold]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def legend(image: np.ndarray, rows: list[tuple[str, tuple | None]], title: str = "",
           inset: np.ndarray | None = None) -> np.ndarray:
    """Recuadro de leyenda arriba a la izquierda: título, filas de texto (con muestra de
    color BGR opcional) e imagen pequeña opcional (p. ej. forma media)."""
    from PIL import Image, ImageDraw
    h, w = image.shape[:2]
    size = max(12, round(max(h, w) / 90))
    font, bold = _font(size), _font(round(size * 1.1), bold=True)
    pad, gap, sw = size // 2, size // 3, size
    lines = ([(title, None, bold)] if title else []) + [(txt, col, font) for txt, col in rows]
    widths = [f.getbbox(txt)[2] + (sw + gap if col is not None else 0) for txt, col, f in lines]
    line_h = size + gap
    box_w = max(widths + [0]) + 2 * pad
    box_h = len(lines) * line_h + 2 * pad
    if inset is not None:
        box_w = max(box_w, inset.shape[1] + 2 * pad)
        box_h += inset.shape[0] + pad
    box_w, box_h = min(box_w, w), min(box_h, h)
    out = image.copy()
    region = out[:box_h, :box_w]
    region[:] = (region * 0.15 + 255 * 0.85).astype(np.uint8)
    pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    y = pad
    for txt, col, f in lines:
        x = pad
        if col is not None:
            draw.rectangle([x, y + 2, x + sw, y + sw], fill=tuple(int(c) for c in col[::-1]), outline=(60, 60, 60))
            x += sw + gap
        draw.text((x, y), txt, fill=(20, 20, 20), font=f)
        y += line_h
    region[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    if inset is not None:
        ih, iw = inset.shape[:2]
        y0 = min(y + pad // 2, box_h - ih)
        if y0 >= 0 and iw <= box_w:
            out[y0:y0 + ih, pad:pad + iw] = inset
    cv2.rectangle(out, (0, 0), (box_w - 1, box_h - 1), (120, 120, 120), 1)
    return out
