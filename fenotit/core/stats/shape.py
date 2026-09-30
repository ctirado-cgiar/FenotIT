"""Forma media y alineación de formas elípticas de Fourier dentro de un grupo."""
import cv2
import numpy as np

from fenotit.core.efd import columns, contour_points, efd, from_row, normalize
from fenotit.core.stats.dataset import groups


def _variants(coeffs):
    """La misma forma girada 180° y reflejada (una semilla vista por la otra cara)."""
    pts = contour_points(coeffs, 256)
    order = len(coeffs)
    out = []
    for sx, sy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        v = pts * [sx, sy]
        if sx * sy < 0:
            v = v[::-1]                        # la reflexión invierte el sentido del contorno
        out.append(normalize(efd(v, order)))
    return out


def align(coeffs_list, rounds=3):
    """Elige para cada objeto la variante (180°, espejo) más parecida a la forma media."""
    if not coeffs_list:
        return coeffs_list
    variants = [_variants(c) for c in coeffs_list]
    ref = variants[0][0]
    chosen = coeffs_list
    for _ in range(rounds):
        chosen = [min(v, key=lambda c: float(((c - ref) ** 2).sum())) for v in variants]
        ref = np.mean(chosen, axis=0)
    return chosen


def mean_shapes(shape_rows: list[dict], by=None, align_mirror: bool = True) -> list[dict]:
    """Forma media por grupo. La alineación (giro 180°, reflejo) se hace dentro de cada
    grupo, así la media depende de qué se compara: una foto, un lote o un genotipo."""
    out = []
    for key, grp in groups(shape_rows, by).items():
        coeffs = [from_row(r) for r in grp]
        if align_mirror:
            coeffs = align(coeffs)
        out.append({"group": key, "n": len(coeffs), **columns(np.mean(coeffs, axis=0))})
    return out


def draw(mean, coeffs_list, size=400):
    """Contornos del grupo en gris y la forma media en verde."""
    img = np.full((size, size, 3), 255, np.uint8)
    scale = size * 0.4

    def pts(c):
        return (contour_points(c) * scale + size / 2).astype(np.int32)
    for c in coeffs_list[:200]:
        cv2.polylines(img, [pts(c)], True, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.polylines(img, [pts(mean)], True, (0, 120, 0), 2, cv2.LINE_AA)
    return img
