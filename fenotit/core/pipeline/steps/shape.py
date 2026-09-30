"""Forma: descriptores elípticos de Fourier (Kuhl & Giardina, 1982)."""
import cv2
import numpy as np

from fenotit.core.pipeline.base import step


def efd(contour: np.ndarray, order: int) -> np.ndarray:
    """Coeficientes (order × 4: a, b, c, d) de un contorno cerrado Nx2."""
    d = np.diff(np.vstack([contour, contour[:1]]), axis=0).astype(float)
    dt = np.hypot(d[:, 0], d[:, 1])
    keep = dt > 0
    d, dt = d[keep], dt[keep]
    t = np.concatenate([[0.0], np.cumsum(dt)])
    T = t[-1]
    n = np.arange(1, order + 1)[:, None]
    phi = 2 * np.pi * t / T
    c = T / (2 * n ** 2 * np.pi ** 2)
    dcos = np.cos(n * phi[1:]) - np.cos(n * phi[:-1])
    dsin = np.sin(n * phi[1:]) - np.sin(n * phi[:-1])
    ux, uy = d[:, 0] / dt, d[:, 1] / dt
    return np.column_stack([c[:, 0] * (dcos @ ux), c[:, 0] * (dsin @ ux),
                            c[:, 0] * (dcos @ uy), c[:, 0] * (dsin @ uy)])


def normalize(coeffs: np.ndarray) -> np.ndarray:
    """Invariante a tamaño, rotación y punto de inicio (primer armónico = elipse unitaria)."""
    a1, b1, c1, d1 = coeffs[0]
    theta = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), a1 ** 2 - b1 ** 2 + c1 ** 2 - d1 ** 2)
    out = coeffs.copy()
    for i in range(len(out)):
        k = i + 1
        rot = np.array([[np.cos(k * theta), -np.sin(k * theta)], [np.sin(k * theta), np.cos(k * theta)]])
        out[i] = (np.array([[out[i, 0], out[i, 1]], [out[i, 2], out[i, 3]]]) @ rot).ravel()
    psi = np.arctan2(out[0, 2], out[0, 0])
    rot = np.array([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]])
    for i in range(len(out)):
        out[i] = (rot @ np.array([[out[i, 0], out[i, 1]], [out[i, 2], out[i, 3]]])).ravel()
    return out / abs(out[0, 0])


def contour_points(coeffs: np.ndarray, n: int = 200) -> np.ndarray:
    """Reconstruye el contorno (n puntos) desde los coeficientes."""
    t = np.linspace(0, 1, n, endpoint=False)
    k = np.arange(1, len(coeffs) + 1)[:, None]
    cos, sin = np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)
    x = (coeffs[:, 0:1] * cos + coeffs[:, 1:2] * sin).sum(0)
    y = (coeffs[:, 2:3] * cos + coeffs[:, 3:4] * sin).sum(0)
    return np.column_stack([x, y])


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


def _columns(coeffs):
    return {f"efd_{l}{i}": round(float(v), 6)
            for i, row in enumerate(coeffs, 1) for l, v in zip("abcd", row)}


@step("shape", "measurement", requires=("labels",), params=[
    {"key": "harmonics", "type": "int", "default": 20, "min": 1, "max": 50},
    {"key": "isolated_only", "type": "bool", "default": True},
    {"key": "align_mirror", "type": "bool", "default": True},
])
def shape(ctx, p):
    """Coeficientes normalizados por objeto (tabla object_shape) y forma media de
    la imagen (tabla image_shape + imagen 'mean_shape'). align_mirror alinea giros de
    180° y reflejos para que la media conserve rasgos como la curva del hilio."""
    from scipy.ndimage import find_objects
    order = int(p["harmonics"])
    touching = ctx.touching_ids() if p["isolated_only"] else set()
    ids, all_coeffs = [], []
    for oid, sl in enumerate(find_objects(ctx.labels), 1):
        if sl is None or oid in touching:
            continue
        m = (ctx.labels[sl] == oid).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
        if len(cnt) < 2 * order + 2:
            continue
        ids.append(oid)
        all_coeffs.append(normalize(efd(cnt, order)))
    if p["align_mirror"]:
        all_coeffs = align(all_coeffs)
    ctx.tables["object_shape"] = [{"object_id": oid, **_columns(c)} for oid, c in zip(ids, all_coeffs)]
    if all_coeffs:
        mean = np.mean(all_coeffs, axis=0)
        ctx.tables["image_shape"] = [{"n_objects": len(all_coeffs), **_columns(mean)}]
        ctx.images["mean_shape"] = _draw(mean, all_coeffs)


def _draw(mean, coeffs_list, size=400):
    img = np.full((size, size, 3), 255, np.uint8)
    scale = size * 0.4

    def pts(c):
        return (contour_points(c) * scale + size / 2).astype(np.int32)
    for c in coeffs_list[:200]:
        cv2.polylines(img, [pts(c)], True, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.polylines(img, [pts(mean)], True, (0, 120, 0), 2, cv2.LINE_AA)
    return img
