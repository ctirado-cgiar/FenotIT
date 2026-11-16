"""Imágenes de resultado: qué objetos entraron en cada medición."""
import cv2
import numpy as np
from scipy.ndimage import find_objects


def regions(labels):
    for oid, bounds in enumerate(find_objects(labels), 1):
        if bounds is not None:
            yield oid, bounds, (labels[bounds] == oid).astype(np.uint8)


def inside_point(mask: np.ndarray) -> tuple[int, int]:
    """Punto dentro del objeto: el centroide si cae adentro; si no (formas cóncavas,
    flores), el punto más alejado del borde."""
    rows, columns = np.nonzero(mask)
    cx, cy = int(round(columns.mean())), int(round(rows.mean()))
    if mask[cy, cx]:
        return cx, cy
    depth = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(depth.argmax()), depth.shape)
    return int(x), int(y)


def sizes(image):
    """Grosor de línea, radio de punto y tamaño de letra según la resolución."""
    factor = max(image.shape[:2]) / 1500
    return max(1, round(2 * factor)), max(3, round(7 * factor)), max(0.4, 0.6 * factor)


def _outline(out, mask, bounds, color, width):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                               offset=(bounds[1].start, bounds[0].start))
    cv2.drawContours(out, cnts, -1, color, width, cv2.LINE_AA)


def object_geometry(labels, detections=None) -> dict[int, np.ndarray]:
    """Contorno de cada objeto (px de la imagen, simplificado): para elegir un objeto con
    un clic, resaltarlo y recortarlo, también en resultados livianos (sin imágenes). Con
    cajas de un detector, el rectángulo de cada caja."""
    if detections is not None:
        from fenotit.core.objects.boxes import rect
        return {detection["id"]: np.round(rect(detection)).astype(np.int32) for detection in detections}
    out = {}
    if labels is None:
        return out
    for oid, bounds, mask in regions(labels):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                   offset=(bounds[1].start, bounds[0].start))
        if cnts:
            out[oid] = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)
    return out


def add_stars(overlays: dict, geometry: dict, ids, views=()) -> dict:
    """Copia de las marcas con una ★ arriba a la derecha de cada objeto destacado (sin
    tocar las del caché). Las vistas sin marcas (Color, Máscara) reciben solo las ★,
    dibujadas sobre su propia imagen (`on_view`)."""
    from fenotit.core.pipeline import overlay
    stars = []
    for oid in ids:
        contour = geometry.get(oid)
        if contour is not None:
            stars.append((float(contour[:, 0].max()), float(contour[:, 1].min())))
    out = {name: ({**marks, "stars": stars} if marks is not None else marks) for name, marks in overlays.items()}
    sizes = [marks.get("size", 0) for marks in overlays.values() if marks]
    for view in views:
        if out.get(view) is None:
            out[view] = {**overlay.empty(), "stars": stars, "on_view": True, "size": max(sizes, default=0)}
    return out


def label_text(out, text, x, y, scale, color=(255, 255, 255)):
    thickness = max(1, round(scale * 2))
    cv2.putText(out, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def included_view(ctx, name, included, color=None):
    """Objetos incluidos en una medición: contorno de color y número; los demás
    (se tocaban) en gris, para ver qué entró y qué no. Las marcas quedan como datos
    (ctx.extra["overlays"]) para dibujarlas al mostrar con el estilo del usuario."""
    from fenotit.core.pipeline import overlay
    marks = overlay.empty()
    for oid, bounds, mask in regions(ctx.labels):
        ok = oid in included
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                   offset=(bounds[1].start, bounds[0].start))
        for contour in cnts:
            marks["outlines"].append((contour.reshape(-1, 2), ok))
        x, y = inside_point(mask)
        marks["labels"].append((x + bounds[1].start, y + bounds[0].start, str(oid), ok))
    add_overlay(ctx, name, marks)


def add_overlay(ctx, name, marks):
    """Guarda las marcas; la imagen de la vista es la foto misma (sin copiar)."""
    from fenotit.core.pipeline import overlay
    marks["size"] = overlay.typical_size(ctx.labels)
    ctx.extra.setdefault("overlays", {})[name] = marks
    # sin copia: la vista es la foto y las marcas se dibujan al mostrar o exportar
    # (overlay.draw); así cada resultado no guarda una imagen dibujada por vista
    ctx.images[name] = ctx.image


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


def _color_text(color: dict, fmt: str) -> str:
    if fmt == "Lab":
        return f"[{color['L']:.0f}, {color['a']:.0f}, {color['b']:.0f}]"
    if fmt == "HEX":
        return color["hex"]
    return f"[{color['R']}, {color['G']}, {color['B']}]"


LEGEND_POSITIONS = ("tl", "tc", "tr", "ml", "mr", "bl", "bc", "br")


def render_legend(image: np.ndarray, spec: dict, color_format: str = "RGB", scale: float = 1.0,
                  box: list | None = None, pos: str = "tl") -> np.ndarray:
    """Recuadro semitransparente arriba a la izquierda, sobrio y con pocos datos.

    spec = {"title": str, "rows": [(texto, color_bgr|None)], "colors": [{R,G,B,L,a,b,hex,pct}],
            "shape": {"mean": Nx2, "others": [Nx2, ...]}}  (todas las claves opcionales)
    """
    from PIL import Image, ImageDraw
    height, width = image.shape[:2]
    size = max(10, round(max(height, width) / 75 * scale))
    font, bold = _font(size), _font(size, bold=True)
    pad, gap, swatch = round(size * 0.8), round(size * 0.45), round(size * 0.9)
    lines = [(txt, col) for txt, col in spec.get("rows", [])]
    for color in spec.get("colors", []):
        lines.append((f"{_color_text(color, color_format)}   {color['pct']:.1f} %", (color["B"], color["G"], color["R"])))
    title = spec.get("title", "")
    if spec.get("colors"):
        title = f"{title} ({color_format})" if title else color_format
    shape = spec.get("shape")
    footer = spec.get("footer", "")                    # nombre de la foto, pequeño
    small = _font(max(8, round(size * 0.72)))
    line_h = size + gap
    text_w = max([bold.getlength(title), small.getlength(footer)]
                 + [font.getlength(t) + (swatch + gap if col else 0) for t, col in lines])
    shape_h = round(size * 8) if shape else 0
    shape_w = 0
    if shape:                                          # ancho según la proporción de la forma (de pie)
        mean = np.asarray(shape["mean"], float)
        ext = mean.max(0) - mean.min(0)
        shape_w = shape_h * min(1.5, max(0.3, ext[1] / max(ext[0], 1e-9)))
    box_w = int(max(text_w, shape_w) + 2 * pad)
    box_h = int(pad * 2 + (line_h if title else 0) + len(lines) * line_h + (shape_h + gap if shape else 0)
                + (round(size * 0.72) + gap if footer else 0))
    margin = pad                                            # margen con el borde de la foto
    box_w, box_h = min(box_w, width - 2 * margin), min(box_h, height - 2 * margin)
    if box_w <= 0 or box_h <= 0:
        return image

    # posición: t/m/b (arriba, centro, abajo) + l/c/r; siempre dentro de la foto
    vertical, horizontal = (pos or "tl")[0], (pos or "tl")[-1]
    x0 = {"l": margin, "c": (width - box_w) // 2, "r": width - box_w - margin}.get(horizontal, margin)
    y0 = {"t": margin, "m": (height - box_h) // 2, "b": height - box_h - margin}.get(vertical, margin)
    if box is not None:                                # zona de la leyenda: las marcas no la tapan
        box.append((x0 - 2, y0 - 2, x0 + box_w + 2, y0 + box_h + 2))
    base = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(layer)
    pen.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=round(size * 0.6),
                        fill=(255, 255, 255, 200), outline=(0, 0, 0, 40), width=1)
    x0, y = x0 + pad, y0 + pad
    if title:
        pen.text((x0, y), title, fill=(30, 30, 30, 255), font=bold)
        y += line_h
    for txt, col in lines:
        x = x0
        if col is not None:
            swatch_radius = swatch // 2
            cy = y + size // 2 + 1
            pen.ellipse([x, cy - swatch_radius, x + swatch, cy + swatch_radius],
                        fill=tuple(int(channel) for channel in col[::-1]) + (255,), outline=(0, 0, 0, 90))
            x += swatch + gap
        pen.text((x, y), txt, fill=(45, 45, 45, 255), font=font)
        y += line_h
    if shape:
        _draw_shape(pen, shape, (x0, y + gap // 2, box_w - 2 * pad, shape_h), size)
        y += shape_h + gap
    if footer:
        pen.text((x0, y), footer, fill=(110, 110, 110, 255), font=small)
    out = Image.alpha_composite(base, layer).convert("RGB")
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)


def _draw_shape(pen, shape: dict, box: tuple, size: int):
    """Forma media de pie (eje largo vertical), con las formas individuales tenues."""
    x, y, box_width, box_height = box
    box_width = min(box_width, box_height * 1.5)
    mean = np.asarray(shape["mean"], float)
    rot = lambda points: np.column_stack([points[:, 1], -points[:, 0]])           # 90°: queda de pie
    others = shape.get("others", [])
    step = max(1, len(others) // 40)                   # a lo sumo ~40 formas de fondo
    pts = [rot(np.asarray(outline, float)) for outline in others[::step]] + [rot(mean)]
    allp = np.vstack(pts)
    lowest, highest = allp.min(0), allp.max(0)
    scale = min(box_width / max(highest[0] - lowest[0], 1e-9), box_height / max(highest[1] - lowest[1], 1e-9))
    off = np.array([x + (box_width - (highest[0] - lowest[0]) * scale) / 2, y + (box_height - (highest[1] - lowest[1]) * scale) / 2])
    to_xy = lambda points: [tuple(vertex) for vertex in ((points - lowest) * scale + off)]
    for points in pts[:-1]:
        pen.line(to_xy(points) + to_xy(points)[:1], fill=(140, 140, 140, 45), width=1)
    mean_xy = to_xy(pts[-1]) + to_xy(pts[-1])[:1]
    line_width = max(2, size // 5)
    pen.line(mean_xy, fill=(255, 255, 255, 255), width=line_width + 3)           # halo para que resalte
    pen.line(mean_xy, fill=(215, 30, 30, 255), width=line_width)


def decorate(result, display: dict, image_name: str | None = None):
    """Función (vista, imagen) -> imagen con las marcas en el estilo elegido y la leyenda
    si está activada. La usan la pantalla y la exportación (sin tkinter)."""
    from fenotit.core.pipeline import overlay
    legends = result.extra.get("legends") or {}
    marks = result.extra.get("overlays") or {}
    base = result.extra.get("base_image")
    colors = overlay.resolve(display.get("style"), auto=result.extra.get("contrast"))

    def draw(name: str, img: np.ndarray, with_marks: bool = True, box: list | None = None) -> np.ndarray:
        if name in marks and marks[name] is not None:
            on_view = marks[name].get("on_view")          # marcas sobre la imagen de la vista (Color)
            if on_view or base is not None:
                img = img.copy() if on_view else base.copy()
                if with_marks:
                    img = overlay.draw(img, marks[name], colors)
        spec = legends.get(name)
        if not spec or not display.get("legend", True):
            return img
        if image_name:
            spec = {**spec, "footer": image_name}
        return render_legend(img, spec, display.get("color_format", "RGB"), display.get("legend_scale", 1.0),
                             box=box, pos=display.get("legend_pos", "tl"))
    return draw
