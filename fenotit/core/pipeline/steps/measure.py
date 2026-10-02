"""Mediciones: agregan columnas a las tablas 'objects' e 'image'."""
import cv2
import numpy as np
from scipy.ndimage import find_objects

from fenotit.core.pipeline.base import step
from fenotit.core.pipeline.views import regions, included_view, inside_point, add_overlay


@step("morphometry", "measurement", requires=("labels",), params=[
    {"key": "isolated_only", "type": "bool", "default": True},
])
def morphometry(ctx, p):
    """Tamaño y forma. Con isolated_only, los objetos que se tocaban con otro
    (desagrupados) se cuentan pero no se miden: su contorno no es confiable."""
    u, f = ctx.unit, ctx.scale
    rows = ctx.object_rows()
    touching = ctx.touching_ids()
    for oid, sl, m in regions(ctx.labels):
        rows[oid]["touching"] = int(oid in touching)
        if p["isolated_only"] and oid in touching:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        area = float(m.sum())                     # píxeles del objeto
        poly_area = cv2.contourArea(cnt)          # para razones de forma (consistente con el casco)
        perim = cv2.arcLength(cv2.approxPolyDP(cnt, 1.0, True), True)  # corrige el escalonado de píxeles
        hull = cv2.convexHull(cnt)
        hull_area, hull_perim = cv2.contourArea(hull), cv2.arcLength(hull, True)
        (_, _), (a, b), _ = cv2.minAreaRect(cnt)
        width, length = min(a, b), max(a, b)
        major, minor, ecc = length, width, 0.0
        if len(cnt) >= 5:
            (_, _), (ma, mi), _ = cv2.fitEllipse(cnt)
            major, minor = max(ma, mi), min(ma, mi)
            ecc = float(np.sqrt(1 - (minor / major) ** 2)) if major > 0 else 0.0
        ys, xs = np.nonzero(m)
        cx, cy = xs.mean(), ys.mean()
        pts = cnt.reshape(-1, 2).astype(float)
        r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        rows[oid].update({
            f"area_{u}2": round(area * f * f, 3),
            f"length_{u}": round(length * f, 3),
            f"width_{u}": round(width * f, 3),
            f"perimeter_{u}": round(perim * f, 3),
            "aspect_ratio": round(length / width, 4) if width else 0,
            "circularity": round(min(1.0, 4 * np.pi * poly_area / perim ** 2), 4) if perim else 0,
            "solidity": round(poly_area / hull_area, 4) if hull_area else 0,
            "convexity": round(min(1.0, hull_perim / perim), 4) if perim else 0,
            "elongation": round(1 - width / length, 4) if length else 0,
            "rectangularity": round(poly_area / (width * length), 4) if width * length else 0,
            "eccentricity": round(ecc, 4),
            f"major_axis_{u}": round(major * f, 3),
            f"minor_axis_{u}": round(minor * f, 3),
            f"radius_min_{u}": round(r.min() * f, 3),
            f"radius_mean_{u}": round(r.mean() * f, 3),
            f"radius_max_{u}": round(r.max() * f, 3),
            "radius_ratio": round(r.min() / r.max(), 4) if r.max() else 0,
            "centroid_x_px": int(round(cx + sl[1].start)),
            "centroid_y_px": int(round(cy + sl[0].start)),
        })
    measured = [r for r in rows.values() if f"area_{u}2" in r]
    ctx.image_row().update({
        "n_objects": len(rows),
        "n_touching": len(touching & set(rows)),
        "n_measured": len(measured),
    })
    included_view(ctx, "morphometry", {r["object_id"] for r in measured})


def _lab(rgb):
    L, a, b = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(int)
    return round(float(L) * 100 / 255, 2), int(a) - 128, int(b) - 128


def _kmeans(pixels_rgb, k, sample=3000):
    """KMeans (OpenCV, k-means++, 3 intentos, semilla fija) ajustado sobre una muestra
    (máx. `sample` píxeles) y aplicado a todos los píxeles. Devuelve [(color, fracción)]
    de mayor a menor y la etiqueta (rango) de cada píxel. OpenCV en vez de scikit-learn:
    mismo algoritmo, ~5× menos tiempo por objeto (importa con cientos de objetos)."""
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


@step("color", "measurement", requires=("labels",), params=[
    {"key": "n_colors", "type": "int", "default": 3, "min": 1, "max": 20},
    {"key": "edge_trim", "type": "int", "default": -1, "min": -1, "max": 50},
    {"key": "mode", "type": "choice", "choices": ["object", "pooled"], "default": "object"},
])
def color(ctx, p):
    """Colores dominantes (KMeans). mode="object": cada objeto con su propia paleta;
    mode="pooled": una paleta común a todos los objetos y el % de cada color en cada
    objeto (comparables entre objetos). Sin el borde del objeto (edge_trim px; -1 = auto)."""
    rgb_img = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2RGB)
    rows = ctx.object_rows()
    k = int(p["n_colors"])
    objs = []
    for oid, sl, m in regions(ctx.labels):
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
    shared, shared_lab = _kmeans(all_px, k)
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
            clusters, lab = _kmeans(px, k)
            palette = np.array([c[::-1] for c, _ in clusters], np.uint8)
        start += len(px)
        for i, (c, frac) in enumerate(clusters, 1):
            per_obj.append({"object_id": oid, "cluster": i, **_color_row(c, frac)})
        view[sl][m > 0] = palette[lab]                 # cada píxel con su color KMeans
    ctx.tables["object_colors"] = per_obj
    ctx.images["color"] = view


@step("count", "measurement", requires=("labels",), params=[
    {"key": "numbers", "type": "bool", "default": True},
])
def count(ctx, p):
    ids = {i for i, s in enumerate(find_objects(ctx.labels), 1) if s is not None}
    touching = ctx.touching_ids()
    ctx.image_row().update({"n_objects": len(ids), "n_touching": len(touching & ids)})
    rows = ctx.object_rows()
    # vista de conteo: un punto por objeto contado (dos puntos en una semilla = partida;
    # una semilla sin punto = no contada)
    from fenotit.core.pipeline import overlay
    ov = overlay.empty()
    for oid, sl, m in regions(ctx.labels):
        row = rows[oid]
        row.setdefault("touching", int(oid in touching))
        if "centroid_x_px" not in row:
            ys, xs = np.nonzero(m)
            row["centroid_x_px"] = int(round(xs.mean() + sl[1].start))
            row["centroid_y_px"] = int(round(ys.mean() + sl[0].start))
        x, y = inside_point(m)
        x, y = x + sl[1].start, y + sl[0].start
        ov["dots"].append((x, y))
        if p["numbers"]:
            ov["labels"].append((x, y, str(oid), True))
    add_overlay(ctx, "count", ov)
