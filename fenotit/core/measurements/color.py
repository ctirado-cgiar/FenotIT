"""Color (color): colores dominantes de cada objeto y de la foto (KMeans en CIELAB)."""
import cv2
import numpy as np

from fenotit.core.pipeline.base import step
from fenotit.core.pipeline.views import regions


AUTO_ERR, AUTO_SEP, AUTO_SHARE = 5.0, 10.0, 0.01


def _lab(rgb):
    L, a, b = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(int)
    return round(float(L) * 100 / 255, 2), int(a) - 128, int(b) - 128


def _kmeans(pixels_rgb, k, sample=3000):
    """KMeans (OpenCV, k-means++, 3 intentos, semilla fija) ajustado sobre una muestra
    (máx. `sample` píxeles) y aplicado a todos los píxeles. Devuelve [(color, fracción)]
    de mayor a menor y la etiqueta (rango) de cada píxel."""
    px = np.asarray(pixels_rgb)
    fit = px
    if len(px) > sample:
        fit = px[np.random.default_rng(134).choice(len(px), sample, replace=False)]
    packed = (fit[:, 0].astype(np.int64) << 16) | (fit[:, 1].astype(np.int64) << 8) | fit[:, 2]
    k = int(min(k, len(np.unique(packed))))
    cv2.setRNGSeed(134)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-3)
    _, _, centers = cv2.kmeans(fit.astype(np.float32), k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    pxf = px.astype(np.float32)
    best = np.full(len(px), np.inf, np.float32)
    labels = np.zeros(len(px), np.int64)
    for j, c in enumerate(centers):                    # el centro más cercano, sin matrices grandes
        d = ((pxf - c) ** 2).sum(axis=1)
        closer = d < best
        best[closer], labels[closer] = d[closer], j
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts, kind="stable")
    centers_out = [tuple(int(v) for v in centers[i]) for i in order]
    rank = np.empty(k, int)
    rank[order] = np.arange(k)
    return [(centers_out[j], counts[order[j]] / counts.sum()) for j in range(k)], rank[labels]


def _color_row(rgb, frac):
    L, a, b = _lab(rgb)
    return {"R": rgb[0], "G": rgb[1], "B": rgb[2], "hex": "#%02x%02x%02x" % rgb,
            "L": L, "a": a, "b": b, "pct": round(100 * float(frac), 2)}


def _core(m: np.ndarray, trim: int) -> np.ndarray:
    """Quita el borde del objeto: esos píxeles mezclan objeto y fondo. La mezcla depende
    del desenfoque de la foto (1-3 px), no del tamaño del objeto: en automático se quitan
    de 1 a 3 px aunque el objeto sea una hoja grande."""
    if trim < 0:
        trim = int(np.clip(round(0.015 * np.sqrt(m.sum())), 1, 3))
    if trim == 0:
        return m
    inner = cv2.erode(np.pad(m, 1), np.ones((3, 3), np.uint8), iterations=trim)[1:-1, 1:-1]
    return inner if inner.sum() >= 20 else m


def auto_k(pixels_rgb: np.ndarray, kmax: int = 10, sample: int = 1500) -> int:
    """Cuántos colores explican el objeto: el menor k cuyo error medio (ΔE00 de cada píxel
    a su color) es ≤ 5, sin pasar el k en que dos colores quedan a menos de 10 ΔE00 (ya no
    se distinguen a simple vista) o uno cubre menos del 1 %. Muestra fija (semilla 134)."""
    from skimage.color import deltaE_ciede2000
    px = np.asarray(pixels_rgb)
    if len(px) > sample:
        px = px[np.random.default_rng(134).choice(len(px), sample, replace=False)]
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.float32) / 255, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    best = 1
    for k in range(1, max(1, kmax) + 1):
        if k > len(np.unique(px, axis=0)):
            break
        cv2.setRNGSeed(134)
        _, lab_k, cen = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lab_k = lab_k.ravel()
        if k > 1:
            share = np.bincount(lab_k, minlength=k) / len(lab_k)
            sep = min(deltaE_ciede2000(cen[i], cen[j]) for i in range(k) for j in range(i + 1, k))
            if sep < AUTO_SEP or share.min() < AUTO_SHARE:
                break
        best = k
        if float(np.mean(deltaE_ciede2000(lab, cen[lab_k]))) <= AUTO_ERR:
            break
    return best


@step("color", "measurement", requires=("labels",), params=[
    {"key": "n_colors", "type": "int", "default": 3, "min": 1, "max": 20},
    {"key": "auto", "type": "bool", "default": False},
    {"key": "edge_trim", "type": "int", "default": -1, "min": -1, "max": 50},
    {"key": "mode", "type": "choice", "choices": ["object", "pooled"], "default": "object"},
])
def color(ctx, p):
    """Colores dominantes (KMeans). mode="object": cada objeto con su propia paleta;
    mode="pooled": una paleta común a todos los objetos y el % de cada color en cada
    objeto (comparables entre objetos). Sin el borde del objeto (edge_trim px; -1 = auto).
    auto: el número de colores lo decide cada objeto (auto_k; n_colors = máximo)."""
    rgb_img = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2RGB)
    rows = ctx.object_rows()
    k = int(p["n_colors"])
    objs = []
    for oid, sl, m in regions(ctx.labels):
        if oid in ctx.excluded:
            continue
        m = _core(m, int(p["edge_trim"]))
        px = rgb_img[sl][m > 0].reshape(-1, 3)
        objs.append((oid, sl, m, px))
        mean = tuple(int(v) for v in px.mean(axis=0))
        L, a, b = _lab(mean)
        rows[oid].update({"mean_R": mean[0], "mean_G": mean[1], "mean_B": mean[2],
                          "mean_L": L, "mean_a": a, "mean_b": b})
    view = (ctx.image * 0.35).astype(np.uint8)         # fondo atenuado
    per_obj = []
    if not objs:
        ctx.tables["object_colors"] = []
        ctx.images["color"] = view
        return
    all_px = np.concatenate([o[3] for o in objs])
    auto = bool(p.get("auto"))
    shared, shared_lab = _kmeans(all_px, auto_k(all_px, min(k, 10)) if auto else k)
    if auto:
        ctx.image_row()["n_colors"] = len(shared)
    ctx.tables["image_colors"] = [{"cluster": i, **_color_row(c, frac)}
                                  for i, (c, frac) in enumerate(shared, 1)]
    start = 0
    for oid, sl, m, px in objs:
        if p["mode"] == "pooled":
            lab = shared_lab[start:start + len(px)]
            counts = np.bincount(lab, minlength=len(shared))
            clusters = [(shared[j][0], counts[j] / max(len(px), 1)) for j in range(len(shared))]
            palette = np.array([c[::-1] for c, _ in shared], np.uint8)
        else:
            clusters, lab = _kmeans(px, auto_k(px, min(k, 10)) if auto else k)
            palette = np.array([c[::-1] for c, _ in clusters], np.uint8)
            if auto:
                rows[oid]["n_colors"] = len(clusters)
        start += len(px)
        for i, (c, frac) in enumerate(clusters, 1):
            per_obj.append({"object_id": oid, "cluster": i, **_color_row(c, frac)})
        view[sl][m > 0] = palette[lab]                 # cada píxel con su color KMeans
    ctx.tables["object_colors"] = per_obj
    ctx.images["color"] = view
