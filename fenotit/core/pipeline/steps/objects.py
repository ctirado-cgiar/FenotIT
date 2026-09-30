"""Procesadores: ROI, limpieza, etiquetado, separación y filtros de objetos."""
import cv2
import numpy as np

from fenotit.core.pipeline.base import step


def _fit(mask, shape):
    return mask if mask.shape[:2] == shape[:2] else cv2.resize(
        mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


@step("roi", "processor", requires=("mask",), provides=("mask",))
def roi(ctx, p):
    mask = ctx.mask
    if ctx.roi is not None:
        mask = cv2.bitwise_and(mask, _fit(ctx.roi, mask.shape))
    if ctx.exclusions:
        mask = mask.copy()
        for pts, _color in ctx.exclusions:
            cv2.fillPoly(mask, [np.asarray(pts, np.int32)], 0)
    ctx.mask = mask


@step("clean", "processor", requires=("mask",), provides=("mask",), params=[
    {"key": "open_iterations", "type": "int", "default": 0, "min": 0, "max": 10},
    {"key": "fill_holes", "type": "bool", "default": True},
])
def clean(ctx, p):
    mask = ctx.mask
    if p["open_iterations"]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
                                iterations=int(p["open_iterations"]))
    if p["fill_holes"]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, cnts, -1, 255, -1)
    ctx.mask = mask


@step("label", "processor", requires=("mask",), provides=("labels",))
def label(ctx, p):
    _, ctx.labels = cv2.connectedComponents((ctx.mask > 0).astype(np.uint8), connectivity=8)


@step("separate", "processor", requires=("mask",), provides=("labels",), params=[
    {"key": "depth", "type": "float", "default": 0.4, "min": 0.05, "max": 0.95},
])
def separate(ctx, p):
    """Separa objetos pegados: watershed con marcadores h-máximos de la distancia.

    depth: cuánto debe estrecharse el cuello entre dos objetos, como fracción del
    grosor máximo del grupo (0.4 = el cuello mide menos del 60 % del grosor).
    """
    from scipy.ndimage import find_objects
    from skimage.morphology import reconstruction
    from skimage.segmentation import watershed

    binary = (ctx.mask > 0).astype(np.uint8)
    _, comps = cv2.connectedComponents(binary, connectivity=8)
    out = np.zeros(binary.shape, np.int32)
    dist_img = np.zeros(binary.shape, np.float32)
    n = 0
    for cid, sl in enumerate(find_objects(comps), 1):
        if sl is None:
            continue
        sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
        m = (comps[sl] == cid).astype(np.uint8)
        d = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float64)
        dist_img[sl] = np.maximum(dist_img[sl], d)
        h = float(p["depth"]) * d.max()
        r = reconstruction(d - h, d, method="dilation")                 # h-máximos:
        peaks = ((r - reconstruction(r - 0.01, r, method="dilation")) > 0.005) & (m > 0)  # máximos regionales
        k, markers = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)
        if k <= 2:
            out[sl][m > 0] = n + 1
            n += 1
            continue
        ws = watershed(-d, markers, mask=m > 0)
        region = out[sl]
        region[ws > 0] = ws[ws > 0] + n
        n += k - 1
    ctx.labels = out
    ctx.images["distance"] = cv2.normalize(dist_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


@step("filter", "processor", requires=("labels",), provides=("labels",), params=[
    {"key": "area_min", "type": "int", "default": 1000, "min": 0, "max": 10_000_000},
    {"key": "area_max", "type": "int", "default": 500_000, "min": 1, "max": 100_000_000},
    {"key": "width_min", "type": "int", "default": 5, "min": 0, "max": 100_000},
    {"key": "width_max", "type": "int", "default": 99_999, "min": 1, "max": 100_000},
    {"key": "length_min", "type": "int", "default": 10, "min": 0, "max": 100_000},
    {"key": "length_max", "type": "int", "default": 99_999, "min": 1, "max": 100_000},
    {"key": "ar_max", "type": "float", "default": 10.0, "min": 1.0, "max": 100.0},
    {"key": "exclude_border", "type": "bool", "default": True},
])
def filter_objects(ctx, p):
    """Filtra objetos (tamaños en px) y renumera 1..n por filas, de izquierda a derecha."""
    from scipy.ndimage import find_objects
    labels = ctx.labels
    h, w = labels.shape
    keep = []
    for oid, sl in enumerate(find_objects(labels), 1):
        if sl is None:
            continue
        ys, xs = np.nonzero(labels[sl] == oid)
        area = len(xs)
        if not (p["area_min"] <= area <= p["area_max"]):
            continue
        y0, x0 = sl[0].start, sl[1].start
        if p["exclude_border"] and (x0 == 0 or y0 == 0 or sl[1].stop == w or sl[0].stop == h):
            continue
        (_, _), (a, b), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        width, length = min(a, b), max(a, b)
        if not (p["width_min"] <= width <= p["width_max"] and p["length_min"] <= length <= p["length_max"]):
            continue
        if width > 0 and length / width > p["ar_max"]:
            continue
        keep.append((y0 + ys.mean(), x0 + xs.mean(), sl[0].stop - sl[0].start, oid, sl))
    out = np.zeros_like(labels)
    row_h = float(np.median([k[2] for k in keep])) if keep else 1.0
    for new_id, (_, _, _, oid, sl) in enumerate(sorted(keep, key=lambda k: (round(k[0] / row_h), k[1])), 1):
        region = out[sl]
        region[labels[sl] == oid] = new_id
    ctx.labels = out
