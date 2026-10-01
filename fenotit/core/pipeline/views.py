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


def included_view(ctx, name, included, color=None):
    """Objetos incluidos en una medición: contorno de color y número; los demás
    (se tocaban) en gris, para ver qué entró y qué no. Las marcas quedan como datos
    (ctx.extra["overlays"]) para dibujarlas al mostrar con el estilo del usuario."""
    from fenotit.core.pipeline import overlay
    ov = overlay.empty()
    for oid, sl, m in regions(ctx.labels):
        ok = oid in included
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                   offset=(sl[1].start, sl[0].start))
        for c in cnts:
            ov["outlines"].append((c.reshape(-1, 2), ok))
        x, y = inside_point(m)
        ov["labels"].append((x + sl[1].start, y + sl[0].start, str(oid), ok))
    add_overlay(ctx, name, ov)


def add_overlay(ctx, name, ov):
    """Guarda las marcas y una imagen ya dibujada con el estilo por defecto."""
    from fenotit.core.pipeline import overlay
    ov["size"] = overlay.typical_size(ctx.labels)
    ctx.extra.setdefault("overlays", {})[name] = ov
    colors = overlay.resolve(None, ctx.image, ctx.labels > 0)
    ctx.images[name] = overlay.draw(ctx.image.copy(), ov, colors)


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


COLOR_FORMATS = ("RGB", "Lab", "HEX")


def _color_text(c: dict, fmt: str) -> str:
    if fmt == "Lab":
        return f"[{c['L']:.0f}, {c['a']:.0f}, {c['b']:.0f}]"
    if fmt == "HEX":
        return c["hex"]
    return f"[{c['R']}, {c['G']}, {c['B']}]"


def render_legend(image: np.ndarray, spec: dict, color_format: str = "RGB", scale: float = 1.0,
                  box: list | None = None) -> np.ndarray:
    """Recuadro semitransparente arriba a la izquierda, sobrio y con pocos datos.

    spec = {"title": str, "rows": [(texto, color_bgr|None)], "colors": [{R,G,B,L,a,b,hex,pct}],
            "shape": {"mean": Nx2, "others": [Nx2, ...]}}  (todas las claves opcionales)
    """
    from PIL import Image, ImageDraw
    h, w = image.shape[:2]
    size = max(10, round(max(h, w) / 75 * scale))
    font, bold = _font(size), _font(size, bold=True)
    pad, gap, sw = round(size * 0.8), round(size * 0.45), round(size * 0.9)
    lines = [(txt, col) for txt, col in spec.get("rows", [])]
    for c in spec.get("colors", []):
        lines.append((f"{_color_text(c, color_format)}   {c['pct']:.1f} %", (c["B"], c["G"], c["R"])))
    title = spec.get("title", "")
    if spec.get("colors"):
        title = f"{title}  ·  {color_format}" if title else color_format
    shape = spec.get("shape")
    line_h = size + gap
    text_w = max([bold.getlength(title)] + [font.getlength(t) + (sw + gap if col else 0) for t, col in lines])
    shape_h = round(size * 8) if shape else 0
    shape_w = 0
    if shape:                                          # ancho según la proporción de la forma (de pie)
        mean = np.asarray(shape["mean"], float)
        ext = mean.max(0) - mean.min(0)
        shape_w = shape_h * min(1.5, max(0.3, ext[1] / max(ext[0], 1e-9)))
    box_w = int(max(text_w, shape_w) + 2 * pad)
    box_h = int(pad * 2 + (line_h if title else 0) + len(lines) * line_h + (shape_h + gap if shape else 0))
    m = pad                                            # margen con el borde de la foto
    box_w, box_h = min(box_w, w - 2 * m), min(box_h, h - 2 * m)
    if box_w <= 0 or box_h <= 0:
        return image

    if box is not None:                                # zona de la leyenda: las marcas no la tapan
        box.append((0, 0, m + box_w + 2, m + box_h + 2))
    base = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([m, m, m + box_w, m + box_h], radius=round(size * 0.6),
                        fill=(255, 255, 255, 200), outline=(0, 0, 0, 40), width=1)
    x0, y = m + pad, m + pad
    if title:
        d.text((x0, y), title, fill=(30, 30, 30, 255), font=bold)
        y += line_h
    for txt, col in lines:
        x = x0
        if col is not None:
            r = sw // 2
            cy = y + size // 2 + 1
            d.ellipse([x, cy - r, x + sw, cy + r], fill=tuple(int(v) for v in col[::-1]) + (255,),
                      outline=(0, 0, 0, 90))
            x += sw + gap
        d.text((x, y), txt, fill=(45, 45, 45, 255), font=font)
        y += line_h
    if shape:
        _draw_shape(d, shape, (x0, y + gap // 2, box_w - 2 * pad, shape_h), size)
    out = Image.alpha_composite(base, layer).convert("RGB")
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)


def _draw_shape(d, shape: dict, box: tuple, size: int):
    """Forma media de pie (eje largo vertical), con las formas individuales tenues."""
    x, y, bw, bh = box
    bw = min(bw, bh * 1.5)
    mean = np.asarray(shape["mean"], float)
    rot = lambda p: np.column_stack([p[:, 1], -p[:, 0]])           # 90°: queda de pie
    others = shape.get("others", [])
    step = max(1, len(others) // 40)                   # a lo sumo ~40 formas de fondo
    pts = [rot(np.asarray(o, float)) for o in others[::step]] + [rot(mean)]
    allp = np.vstack(pts)
    lo, hi = allp.min(0), allp.max(0)
    scale = min(bw / max(hi[0] - lo[0], 1e-9), bh / max(hi[1] - lo[1], 1e-9))
    off = np.array([x + (bw - (hi[0] - lo[0]) * scale) / 2, y + (bh - (hi[1] - lo[1]) * scale) / 2])
    to_xy = lambda p: [tuple(v) for v in ((p - lo) * scale + off)]
    for p in pts[:-1]:
        d.line(to_xy(p) + to_xy(p)[:1], fill=(140, 140, 140, 45), width=1)
    mean_xy = to_xy(pts[-1]) + to_xy(pts[-1])[:1]
    lw = max(2, size // 5)
    d.line(mean_xy, fill=(255, 255, 255, 255), width=lw + 3)           # halo para que resalte
    d.line(mean_xy, fill=(215, 30, 30, 255), width=lw)
