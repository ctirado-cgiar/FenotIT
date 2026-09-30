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
    {"key": "min_distance", "type": "int", "default": 0, "min": 0, "max": 1000},
    {"key": "peak_threshold", "type": "float", "default": 0.2, "min": 0.05, "max": 0.95},
    {"key": "merge_ratio", "type": "float", "default": 0.95, "min": 0.5, "max": 1.0},
])
def separate(ctx, p):
    """Separa objetos pegados: un pico de la transformada de distancia por objeto + watershed.

    min_distance (px) entre centros; 0 = automático (~ radio típico de los objetos).
    merge_ratio: vuelve a unir dos partes si el cuello entre ellas es al menos esa
    fracción del grosor de la más delgada (evita cortar objetos alargados).
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    binary = (ctx.mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() == 0:
        ctx.labels = np.zeros(binary.shape, np.int32)
        return
    _, comps = cv2.connectedComponents(binary, connectivity=8)
    thr = float(p["peak_threshold"]) * dist.max()
    md = int(p["min_distance"])
    if md <= 0:
        rough = peak_local_max(dist, min_distance=5, threshold_abs=thr, labels=comps)
        md = max(3, int(0.9 * np.median(dist[tuple(rough.T)]))) if len(rough) else 5
    ctx.extra["separate_min_distance"] = md
    peaks = peak_local_max(dist, min_distance=md, threshold_abs=thr, labels=comps)
    markers = np.zeros(binary.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-dist, markers, mask=binary > 0).astype(np.int32)
    ctx.labels = _merge_flat_necks(labels, dist, float(p["merge_ratio"]))
    ctx.images["distance"] = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _merge_flat_necks(labels, dist, ratio):
    from scipy.ndimage import maximum
    if ratio >= 1.0 or labels.max() < 2:
        return labels
    saddle = {}
    for a, b, da, db in ((labels[:, :-1], labels[:, 1:], dist[:, :-1], dist[:, 1:]),
                         (labels[:-1, :], labels[1:, :], dist[:-1, :], dist[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        for x, y, v in zip(a[m], b[m], np.minimum(da[m], db[m])):
            k = (min(x, y), max(x, y))
            saddle[k] = max(saddle.get(k, 0.0), v)
    peak = maximum(dist, labels, index=np.arange(labels.max() + 1))
    parent = np.arange(labels.max() + 1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for (a, b), v in saddle.items():
        if v >= ratio * min(peak[a], peak[b]):
            parent[root(a)] = root(b)
    roots = np.array([root(i) for i in range(len(parent))])
    return roots[labels].astype(np.int32)


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
