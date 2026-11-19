"""Color (color): colores dominantes de cada objeto y de la foto (KMeans en CIELAB)."""
import cv2
import numpy as np

# El decorador step registra esta medición como pieza de la cadena.
from fenotit.core.pipeline.base import step
# regions entrega cada objeto recortado con su máscara, para tomar sus píxeles.
from fenotit.core.pipeline.views import regions


AUTO_ERR, AUTO_SEP = 5.0, 10.0


def _lab(rgb):
    lightness, a_star, b_star = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(int)
    return round(float(lightness) * 100 / 255, 2), int(a_star) - 128, int(b_star) - 128


def _kmeans(pixels_rgb, n_clusters, sample=3000):
    """KMeans (OpenCV, k-means++, 3 intentos, semilla fija) ajustado sobre una muestra
    (máx. `sample` píxeles) y aplicado a todos los píxeles. Devuelve [(color, fracción)]
    de mayor a menor y la etiqueta (rango) de cada píxel."""
    pixels = np.asarray(pixels_rgb)
    fit = pixels
    if len(pixels) > sample:
        fit = pixels[np.random.default_rng(134).choice(len(pixels), sample, replace=False)]
    packed = (fit[:, 0].astype(np.int64) << 16) | (fit[:, 1].astype(np.int64) << 8) | fit[:, 2]
    n_clusters = int(min(n_clusters, len(np.unique(packed))))
    cv2.setRNGSeed(134)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-3)
    _, _, centers = cv2.kmeans(fit.astype(np.float32), n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    pxf = pixels.astype(np.float32)
    best = np.full(len(pixels), np.inf, np.float32)
    labels = np.zeros(len(pixels), np.int64)
    for j, center in enumerate(centers):                    # el centro más cercano, sin matrices grandes
        distance = ((pxf - center) ** 2).sum(axis=1)
        closer = distance < best
        best[closer], labels[closer] = distance[closer], j
    counts = np.bincount(labels, minlength=n_clusters)
    order = np.argsort(-counts, kind="stable")
    centers_out = [tuple(int(value) for value in centers[i]) for i in order]
    rank = np.empty(n_clusters, int)
    rank[order] = np.arange(n_clusters)
    return [(centers_out[j], counts[order[j]] / counts.sum()) for j in range(n_clusters)], rank[labels]


def _color_row(rgb, frac):
    lightness, a_star, b_star = _lab(rgb)
    return {"R": rgb[0], "G": rgb[1], "B": rgb[2], "hex": "#%02x%02x%02x" % rgb,
            "L": lightness, "a": a_star, "b": b_star, "pct": round(100 * float(frac), 2)}


def _core(mask: np.ndarray, trim: int) -> np.ndarray:
    """Quita el borde del objeto: esos píxeles mezclan objeto y fondo. La mezcla depende
    del desenfoque de la foto (1-3 px), no del tamaño del objeto: en automático se quitan
    de 1 a 3 px aunque el objeto sea una hoja grande."""
    if trim < 0:
        trim = int(np.clip(round(0.015 * np.sqrt(mask.sum())), 1, 3))
    if trim == 0:
        return mask
    inner = cv2.erode(np.pad(mask, 1), np.ones((3, 3), np.uint8), iterations=trim)[1:-1, 1:-1]
    return inner if inner.sum() >= 20 else mask


def auto_k(pixels_rgb: np.ndarray, kmax: int = 10, sample: int = 1500) -> int:
    """Cuántos colores explican el objeto: el menor k cuyo error medio (ΔE00 de cada píxel
    a su color) es ≤ 5, sin pasar el k en que dos colores quedan a menos de 10 ΔE00 (ya no
    se distinguen a simple vista). Muestra fija (semilla 134)."""
    from skimage.color import deltaE_ciede2000
    pixels = np.asarray(pixels_rgb)
    if len(pixels) > sample:
        pixels = pixels[np.random.default_rng(134).choice(len(pixels), sample, replace=False)]
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.float32) / 255, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    best = 1
    for n_clusters in range(1, max(1, kmax) + 1):
        if n_clusters > len(np.unique(pixels, axis=0)):
            break
        cv2.setRNGSeed(134)
        _, lab_k, cen = cv2.kmeans(lab, n_clusters, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lab_k = lab_k.ravel()
        if n_clusters > 1:
            sep = min(deltaE_ciede2000(cen[i], cen[j]) for i in range(n_clusters) for j in range(i + 1, n_clusters))
            if sep < AUTO_SEP:
                break
        best = n_clusters
        if float(np.mean(deltaE_ciede2000(lab, cen[lab_k]))) <= AUTO_ERR:
            break
    return best


@step("color", "measurement", requires=("labels",), params=[
    {"key": "n_colors", "type": "int", "default": 3, "min": 1, "max": 20},
    {"key": "auto", "type": "bool", "default": False},
    {"key": "edge_trim", "type": "int", "default": -1, "min": -1, "max": 50},
    {"key": "mode", "type": "choice", "choices": ["object", "pooled"], "default": "object"},
])
def color(ctx, params):
    """Colores dominantes (KMeans). mode="object": cada objeto con su propia paleta;
    mode="pooled": una paleta común a todos los objetos y el % de cada color en cada
    objeto (comparables entre objetos). Sin el borde del objeto (edge_trim px; -1 = auto).
    auto: el número de colores lo decide cada objeto (auto_k; n_colors = máximo)."""
    rgb_img = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2RGB)
    rows = ctx.object_rows()
    max_colors = int(params["n_colors"])
    objs = []
    for oid, bounds, mask in regions(ctx.labels):
        if oid in ctx.excluded:
            continue
        mask = _core(mask, int(params["edge_trim"]))
        pixels = rgb_img[bounds][mask > 0].reshape(-1, 3)
        objs.append((oid, bounds, mask, pixels))
        mean = tuple(int(value) for value in pixels.mean(axis=0))
        lightness, a_star, b_star = _lab(mean)
        rows[oid].update({"mean_R": mean[0], "mean_G": mean[1], "mean_B": mean[2],
                          "mean_L": lightness, "mean_a": a_star, "mean_b": b_star})
    view = (ctx.image * 0.35).astype(np.uint8)         # fondo atenuado
    per_obj = []
    if not objs:
        ctx.tables["object_colors"] = []
        ctx.images["color"] = view
        return
    all_px = np.concatenate([entry[3] for entry in objs])
    auto = bool(params.get("auto"))
    shared, shared_lab = _kmeans(all_px, auto_k(all_px, min(max_colors, 10)) if auto else max_colors)
    if auto:
        ctx.image_row()["n_colors"] = len(shared)
    ctx.tables["image_colors"] = [{"cluster": i, **_color_row(rgb, frac)}
                                  for i, (rgb, frac) in enumerate(shared, 1)]
    start = 0
    for oid, bounds, mask, pixels in objs:
        if params["mode"] == "pooled":
            lab = shared_lab[start:start + len(pixels)]
            counts = np.bincount(lab, minlength=len(shared))
            clusters = [(shared[j][0], counts[j] / max(len(pixels), 1)) for j in range(len(shared))]
            palette = np.array([rgb[::-1] for rgb, _ in shared], np.uint8)
        else:
            clusters, lab = _kmeans(pixels, auto_k(pixels, min(max_colors, 10)) if auto else max_colors)
            palette = np.array([rgb[::-1] for rgb, _ in clusters], np.uint8)
            if auto:
                rows[oid]["n_colors"] = len(clusters)
        start += len(pixels)
        for i, (rgb, frac) in enumerate(clusters, 1):
            per_obj.append({"object_id": oid, "cluster": i, **_color_row(rgb, frac)})
        view[bounds][mask > 0] = palette[lab]                 # cada píxel con su color KMeans
    ctx.tables["object_colors"] = per_obj
    ctx.images["color"] = view
