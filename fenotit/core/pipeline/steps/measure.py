"""Mediciones: agregan columnas a las tablas 'objects' e 'image'."""
import cv2
import numpy as np
from scipy.ndimage import find_objects

from fenotit.core.pipeline.base import step
from fenotit.core.pipeline.views import regions, included_view, inside_point, label_text, sizes


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
    included_view(ctx, "morphometry", {r["object_id"] for r in measured}, (0, 220, 100))


def _lab(rgb):
    L, a, b = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(int)
    return round(L * 100 / 255, 2), a - 128, b - 128


def _kmeans(pixels_rgb, k):
    from sklearn.cluster import KMeans
    k = int(min(k, len(np.unique(pixels_rgb, axis=0))))
    km = KMeans(n_clusters=k, n_init=10, random_state=134).fit(pixels_rgb)
    counts = np.bincount(km.labels_, minlength=k)
    order = np.argsort(-counts)
    centers = [tuple(int(v) for v in km.cluster_centers_[i]) for i in order]
    rank = np.empty(k, int)
    rank[order] = np.arange(k)
    return [(centers[j], counts[order[j]] / counts.sum()) for j in range(k)], rank[km.labels_]


def _color_row(rgb, frac):
    L, a, b = _lab(rgb)
    return {"R": rgb[0], "G": rgb[1], "B": rgb[2], "hex": "#%02x%02x%02x" % rgb,
            "L": L, "a": a, "b": b, "pct": round(100 * frac, 2)}


def _core(m: np.ndarray, trim: int) -> np.ndarray:
    """Quita el borde del objeto: esos píxeles mezclan objeto y fondo."""
    if trim < 0:
        trim = max(1, round(0.04 * np.sqrt(m.sum())))
    if trim == 0:
        return m
    inner = cv2.erode(np.pad(m, 1), np.ones((3, 3), np.uint8), iterations=trim)[1:-1, 1:-1]
    return inner if inner.sum() >= 20 else m


@step("color", "measurement", requires=("labels",), params=[
    {"key": "n_colors", "type": "int", "default": 3, "min": 1, "max": 10},
    {"key": "edge_trim", "type": "int", "default": -1, "min": -1, "max": 50},
])
def color(ctx, p):
    """Colores dominantes (KMeans) por objeto y de todos los objetos juntos. Solo usa
    píxeles del objeto, sin el borde (edge_trim px; -1 = automático, ~4 % del tamaño)."""
    rgb_img = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2RGB)
    rows = ctx.object_rows()
    per_obj, all_px = [], []
    view = (ctx.image * 0.35).astype(np.uint8)         # fondo atenuado
    for oid, sl, m in regions(ctx.labels):
        m = _core(m, int(p["edge_trim"]))
        px = rgb_img[sl][m > 0].reshape(-1, 3)
        all_px.append(px)
        mean = tuple(int(v) for v in px.mean(axis=0))
        L, a, b = _lab(mean)
        rows[oid].update({"mean_R": mean[0], "mean_G": mean[1], "mean_B": mean[2],
                          "mean_L": L, "mean_a": a, "mean_b": b})
        clusters, lab = _kmeans(px, p["n_colors"])
        for i, (c, frac) in enumerate(clusters, 1):
            per_obj.append({"object_id": oid, "cluster": i, **_color_row(c, frac)})
        # vista: cada píxel del objeto pintado con su color KMeans
        palette = np.array([c[::-1] for c, _ in clusters], np.uint8)
        region = view[sl]
        region[m > 0] = palette[lab]
    ctx.tables["object_colors"] = per_obj
    ctx.images["color"] = view
    if all_px:
        px = np.concatenate(all_px)
        clusters, _ = _kmeans(px, p["n_colors"])
        ctx.tables["image_colors"] = [{"cluster": i, **_color_row(c, frac)}
                                      for i, (c, frac) in enumerate(clusters, 1)]


@step("count", "measurement", requires=("labels",), params=[
    {"key": "numbers", "type": "bool", "default": True},
])
def count(ctx, p):
    ids = {i for i, s in enumerate(find_objects(ctx.labels), 1) if s is not None}
    ctx.image_row().update({"n_objects": len(ids), "n_touching": len(ctx.touching_ids() & ids)})
    # vista de conteo: un punto por objeto contado (dos puntos en una semilla = partida;
    # una semilla sin punto = no contada)
    out = ctx.image.copy()
    _, radius, scale = sizes(out)
    for oid, sl, m in regions(ctx.labels):
        x, y = inside_point(m)
        x, y = x + sl[1].start, y + sl[0].start
        cv2.circle(out, (x, y), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), radius, (255, 0, 255), -1, cv2.LINE_AA)
        if p["numbers"]:
            label_text(out, str(oid), x + radius, y, scale)
    ctx.images["count"] = out
