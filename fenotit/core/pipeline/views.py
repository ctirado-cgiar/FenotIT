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
